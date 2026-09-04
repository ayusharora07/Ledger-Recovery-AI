from __future__ import annotations
import os
import io
import json
import uuid
import time
import hmac
import hashlib
import secrets
import asyncio
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List
import re
from difflib import SequenceMatcher
from agent_engine import process_buyer_message, validate_intent_action, ActionType, IntentType, ExtractedIntent, ValidationResult, AgentDecision, generate_firm_insight_narrative, classify_copilot_message, CopilotIntent, CopilotDecision

from fastapi import FastAPI, Depends, HTTPException, Request, BackgroundTasks, Header
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from pydantic import BaseModel

from database import get_db, engine, Base, SessionLocal
import models
from models import (Client, Invoice, PaymentRecord, AuditLog, InvoiceStatus, AuditStatus,
                       CollectionCase, CollectionCaseStatus, PaymentPromise, PromiseStatus,
                       CollectionAction, CollectionActionStatus, Communication, PaymentAllocation)
from collection_engine import (
    get_or_create_case, record_communication, refresh_case, create_promise,
    reconcile_promises, schedule_action, cancel_open_actions_for_invoice, collection_queue
)

from agent_engine import process_buyer_message, ActionType, IntentType
from razorpay_client import (
    create_payment_link,
    verify_webhook_signature,
    paise_to_rupees,
    rupees_to_paise,
    fetch_payment_link,
    cancel_payment_link,
    RazorpayClientError,
)
from invoice_pdf import generate_invoice_pdf_bytes, generate_receipt_pdf_bytes, generate_combined_invoices_pdf_bytes

logger = logging.getLogger("main")
logging.basicConfig(level=logging.INFO)

# Ensure tables exist even if someone skips the manual init step.
Base.metadata.create_all(bind=engine)

app = FastAPI(title="LedgerRecover AI")
templates = Jinja2Templates(directory="templates")

# --------------------------------------------------------------------------
# AUTH — single wholesaler login, signed session tokens.
#
# This app is single-tenant (one wholesaler's dashboard), so there's no user
# table — just one shared password from the environment. On success we hand
# back a signed, expiring token (HMAC-SHA256 over "expiry.signature", no new
# dependency like PyJWT needed). The dashboard stores it in localStorage and
# sends it as `Authorization: Bearer <token>` on every API call; `require_auth`
# below is attached to every data-bearing endpoint except the login route
# itself, /health, and the Razorpay webhook (which authenticates itself via
# its own signature, not our session token).
# --------------------------------------------------------------------------

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
SESSION_SECRET = os.getenv("SESSION_SECRET")
SESSION_TTL_SECONDS = 12 * 60 * 60  # 12 hours

if not ADMIN_PASSWORD:
    logger.warning(
        "ADMIN_PASSWORD is not set in .env — falling back to 'admin' for local dev only. "
        "Set a real ADMIN_PASSWORD before deploying this anywhere reachable."
    )
    ADMIN_PASSWORD = "admin"

if not SESSION_SECRET:
    logger.warning(
        "SESSION_SECRET is not set in .env — generating a random one for this process only, "
        "which means every login session is invalidated on restart. Set a fixed SESSION_SECRET "
        "in .env to avoid that."
    )
    SESSION_SECRET = secrets.token_hex(32)


def create_session_token() -> str:
    expiry = int(time.time()) + SESSION_TTL_SECONDS
    signature = hmac.new(SESSION_SECRET.encode(), str(expiry).encode(), hashlib.sha256).hexdigest()
    return f"{expiry}.{signature}"


def verify_session_token(token: str) -> bool:
    try:
        expiry_str, signature = token.split(".", 1)
        expected = hmac.new(SESSION_SECRET.encode(), expiry_str.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return False
        return int(expiry_str) > time.time()
    except (ValueError, AttributeError):
        return False


def require_auth(request: Request, authorization: Optional[str] = Header(None)):
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization[len("Bearer "):]
    elif request.query_params.get("token"):
        # Plain <a href="..."> download links (invoice/receipt/combined PDFs,
        # including ones rendered from markdown inside chat bubbles) can't
        # attach a custom Authorization header, so those URLs carry the
        # token as a query param instead. Same token, same validation.
        token = request.query_params.get("token")
    if not token or not verify_session_token(token):
        raise HTTPException(status_code=401, detail="Missing, expired, or invalid session — please log in again")


class LoginRequest(BaseModel):
    password: str


@app.post("/api/auth/login")
def login(payload: LoginRequest):
    if not hmac.compare_digest(payload.password, ADMIN_PASSWORD):
        raise HTTPException(status_code=401, detail="Incorrect password")
    return {"token": create_session_token(), "expires_in_seconds": SESSION_TTL_SECONDS}


# How often the background poller checks outstanding payment links.
POLL_INTERVAL_SECONDS = 12
# Razorpay rejects any expire_by under 15 minutes from now (error: "expire_by:
# timestamp must be atleast 15 minutes in future"). Keep meaningful headroom
# above that floor so a slow request round-trip can never push the computed
# timestamp back under Razorpay's minimum by the time it's received server-side.
PAYMENT_LINK_EXPIRY_MINUTES = 20
_DEAD_LINK_IDS: set[str] = set()  # payment_link_ids confirmed permanently gone (e.g. from a rotated/old
                                   # Razorpay key) — skipped on future poll cycles instead of re-fetched forever


# --------------------------------------------------------------------------
# REQUEST / RESPONSE SCHEMAS
# --------------------------------------------------------------------------
class ChatOption(BaseModel):
    id: str
    title: str
    payload: str

class SimulateMessageRequest(BaseModel):
    # Either is accepted: invoice_id targets one bill directly (legacy /
    # single-invoice flows), client_id targets a Firm and lets the backend
    # resolve which of that firm's invoices the message applies to (see
    # resolve_target_invoice) — this is what the Firm-based simulator uses.
    invoice_id: Optional[int] = None
    client_id: Optional[int] = None
    message: str

class SimulateMessageResponse(BaseModel):
    invoice_id: int
    incoming_message: str
    detected_intent: str
    confidence: float
    llm_reasoning: str
    action_taken: str
    allowed: bool
    guardrail_reason: str
    final_amount: Optional[float] = None
    payment_link_url: Optional[str] = None
    invoice_pdf_url: Optional[str] = None
    updated_invoice_status: Optional[str] = None
    audit_log_id: int
    bot_reply_text: str
    options: Optional[List[ChatOption]] = None


class CopilotMessageRequest(BaseModel):
    message: str
    # Plain [{"sender": "wholesaler"|"copilot", "text": "..."}] turns from
    # the frontend's own in-memory transcript — the Copilot itself has no
    # DB-persisted conversation, unlike the buyer-side chat which is
    # reconstructed from AuditLog. Gives the LLM classifier the same kind of
    # short-term memory the buyer-facing agent now has.
    history: List[dict] = []
    # Echoed back from the PREVIOUS response's pending_action, if any — lets
    # a bare "yes"/"no" reply be resolved deterministically against the
    # exact batch just proposed, instead of re-running the LLM classifier
    # on a one-word message with no idea what it's replying to.
    pending_action: Optional[dict] = None


class CopilotMessageResponse(BaseModel):
    reply_text: str
    intent: str
    firms: List[dict] = []
    reminders_sent: int = 0
    pdf_url: Optional[str] = None
    # Set whenever the Copilot proposes an action that needs confirmation
    # (currently just SEND_REMINDERS). The frontend must echo this back
    # unchanged as `pending_action` on the wholesaler's next message.
    pending_action: Optional[dict] = None


# --------------------------------------------------------------------------
# DETERMINISTIC LEDGER RECOMPUTATION (single source of truth)
# --------------------------------------------------------------------------

def recompute_invoice_ledger(invoice: Invoice) -> None:
    """
    Pure deterministic recalculation of balance_amount + status from
    total_amount and paid_amount. This is the ONLY place ledger numbers
    are derived — never set balance_amount or status directly elsewhere.
    """
    invoice.paid_amount = round(float(invoice.paid_amount), 2)
    remaining = round(float(invoice.total_amount) - float(invoice.paid_amount), 2)

    if remaining <= 0.01:
        invoice.balance_amount = 0.0
        invoice.status = InvoiceStatus.PAID
    elif invoice.paid_amount > 0:
        invoice.balance_amount = remaining
        invoice.status = InvoiceStatus.PARTIALLY_PAID
    else:
        invoice.balance_amount = remaining
        if invoice.status != InvoiceStatus.DISPUTED:
            invoice.status = InvoiceStatus.PENDING


def write_audit_log(
    db: Session,
    invoice_id: Optional[int],
    incoming_message: str,
    detected_intent: str,
    action_taken: str,
    status: AuditStatus,
    execution_payload: dict,
) -> AuditLog:
    log = AuditLog(
        invoice_id=invoice_id,
        incoming_message=incoming_message,
        detected_intent=detected_intent,
        action_taken=action_taken,
        status=status,
        execution_payload=json.dumps(execution_payload, default=str),
        timestamp=datetime.utcnow(),
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log

def resolve_target_invoice(db: Session, client: Client) -> Optional[Invoice]:
    """
    Firm-level conversations still resolve to a single Invoice under the
    hood, because the agent's validation/guardrail logic (agent_engine.py)
    is deliberately scoped to one invoice's balance at a time — that's what
    makes "never let the LLM invent a balance" enforceable. So when a buyer
    is messaged in the context of a Firm rather than a single bill, we pick
    the invoice a real collections agent would chase first: the oldest
    unpaid one (earliest due_date, excluding anything already PAID).
    Falls back to the most recently created invoice if everything is paid,
    and to None if the firm has no invoices at all.
    """
    outstanding = (
        db.query(Invoice)
        .filter(Invoice.client_id == client.id, Invoice.status != InvoiceStatus.PAID)
        .order_by(Invoice.due_date.asc())
        .first()
    )
    if outstanding:
        return outstanding
    return (
        db.query(Invoice)
        .filter(Invoice.client_id == client.id)
        .order_by(Invoice.created_at.desc())
        .first()
    )


def build_invoice_messages(db: Session, invoice: Invoice) -> list:
    """
    Reconstructs the buyer<->bot conversation thread for a single invoice
    from its audit trail. Shared by the per-invoice chat-history endpoint
    and the per-firm (merged) chat-history endpoint so both stay consistent.
    """
    logs = (
        db.query(AuditLog)
        .filter(AuditLog.invoice_id == invoice.id)
        .order_by(AuditLog.timestamp.asc())
        .all()
    )

    messages = []
    for log in logs:
        payload = json.loads(log.execution_payload) if log.execution_payload else {}

        if log.detected_intent == "OUTBOUND_COLLECTION_INIT":
            messages.append({
                "sender": "bot",
                "text": payload.get("dispatched_text", "Your invoice has been created."),
                "timestamp": log.timestamp.isoformat(),
            })
            continue

        if log.detected_intent in ("PAYMENT_RECEIVED", "AUTO_POLL_RECONCILIATION"):
            messages.append({
                "sender": "bot",
                "text": synthesize_bot_reply_text(log.detected_intent, log.action_taken, log.status.value, payload),
                "timestamp": log.timestamp.isoformat(),
                # NEW (Feature 5): so a "PAID & CLEARED" receipt bubble reconstructs
                # with its download button, same as at the moment it originally fired.
                "invoice_pdf_url": payload.get("invoice_pdf_url"),
            })
            continue

        if log.detected_intent == "AUTOMATED_REMINDER_FIRED":
            messages.append({
                "sender": "bot",
                "text": payload.get("reminder_text", "This is an automated reminder about your outstanding balance."),
                "timestamp": log.timestamp.isoformat(),
            })
            continue

        if log.detected_intent == "OVERDUE_AUTO_REMINDER":
            fallback_text = payload.get("overdue_text")
            if not fallback_text:
                # Payload predates/omits overdue_text (e.g. malformed or
                # seeded data) — rebuild a real reminder from the actual
                # invoice instead of showing a flat, contextless line.
                days = payload.get("days_overdue")
                days_part = f"and is now {days} day(s) overdue" if days is not None else "and is now overdue"
                fallback_text = (
                    f"⚠️ Hi {invoice.client.name}, Invoice {invoice.invoice_number} "
                    f"(₹{invoice.balance_amount:,.2f}) was due on {invoice.due_date.strftime('%d %b %Y')} "
                    f"{days_part}. Please arrange payment at the earliest, or let us know if there's an "
                    f"issue with this bill."
                )
            messages.append({
                "sender": "bot",
                "text": fallback_text,
                "timestamp": log.timestamp.isoformat(),
            })
            continue

        if log.detected_intent == "MANUAL_COPILOT_REMINDER":
            messages.append({
                "sender": "bot",
                "text": payload.get("reminder_text", "This is a reminder about your outstanding balance."),
                "timestamp": log.timestamp.isoformat(),
            })
            continue

        if log.detected_intent == "ESCALATION":
            # System-triggered (e.g. repeated collection attempts with no
            # response) — never something the buyer actually typed, so it
            # must never render as a fake incoming chat bubble.
            messages.append({
                "sender": "bot",
                "text": synthesize_bot_reply_text(log.detected_intent, log.action_taken, log.status.value, payload),
                "timestamp": log.timestamp.isoformat(),
            })
            continue

        if log.detected_intent == "AUTOMATED_PROMISE_FOLLOW_UP":
            # System-triggered when a payment promise's date arrives — never
            # something the buyer typed, so (like ESCALATION above) it must
            # render as a single bot message using the reminder_text that was
            # actually generated, not fall through to the generic buyer+fallback
            # path below (which is what was happening: it rendered as a fake
            # buyer bubble reading "SYSTEM: Payment promise follow-up fired",
            # and since neither this intent nor SEND_PROMISE_FOLLOW_UP is
            # recognized by synthesize_bot_reply_text, the "reply" collapsed
            # to the generic "I couldn't quite process that" fallback).
            messages.append({
                "sender": "bot",
                "text": payload.get(
                    "reminder_text",
                    "This is a follow-up on the payment you told us to expect.",
                ),
                "timestamp": log.timestamp.isoformat(),
            })
            continue

        if log.detected_intent in ("WEBHOOK_RECONCILIATION", "MANUAL_SYNC", "OVERPAYMENT_FLAG"):
            continue  # internal reconciliation noise, not part of the buyer-facing thread

        messages.append({"sender": "buyer", "text": log.incoming_message, "timestamp": log.timestamp.isoformat()})
        messages.append({
            "sender": "bot",
            "text": synthesize_bot_reply_text(log.detected_intent, log.action_taken, log.status.value, payload),
            "timestamp": log.timestamp.isoformat(),
            "invoice_pdf_url": payload.get("invoice_pdf_url"),
        })

    return messages


def synthesize_bot_reply_text(detected_intent: str, action_taken: str, status: str, execution_payload: dict) -> str:
    """
    Converts the agent's structured decision into a natural WhatsApp-style
    reply. Kept as one shared function so live replies and reconstructed
    chat history (from audit logs) always say the same thing for the same
    kind of event.
    """
    # NEW (Feature 1): deterministic greeting text was built once in
    # simulate_message (build_greeting_reply) and stashed in the payload —
    # this just replays it, identically, for both live replies and
    # reconstructed history.
    if action_taken == "GREETING_RESPONSE":
        return execution_payload.get("greeting_text", "Hi! How can I help you today?")

    # NEW (Feature 2): multi-invoice breakdown, reconstructed from the
    # itemized list stored at the time it was generated.
    if action_taken == "MULTI_INVOICE_BREAKDOWN":
        breakdown = execution_payload.get("invoice_breakdown", [])
        lines = [
            f"• {b['invoice_number']}: ₹{b['balance_amount']:,.2f} (due {b['due_date'][:10]})"
            for b in breakdown
        ]
        return (
            f"You have {len(breakdown)} open invoices:\n" + "\n".join(lines) +
            "\n\nWhich invoice would you like to clear, or reply 'PAY ALL' to generate "
            "a single payment link for the entire balance?"
        )

    # NEW: invoice-choice prompt ("specific invoice vs combined PDF"),
    # reconstructed from the exact prompt text stored when it first fired.
    if action_taken == "INVOICE_CHOICE_PROMPT":
        return execution_payload.get(
            "prompt_text",
            "Would you like a copy of a specific invoice, or a combined PDF with all of them?",
        )

    if detected_intent == "OUTBOUND_COLLECTION_INIT":
        return execution_payload.get("dispatched_text", "Your invoice has been created.")

    if detected_intent == "AUTOMATED_REMINDER_FIRED":
        return execution_payload.get("reminder_text", "This is an automated reminder about your outstanding balance.")

    if detected_intent == "OVERDUE_AUTO_REMINDER":
        return execution_payload.get("overdue_text", "This invoice is now overdue.")

    if detected_intent == "MANUAL_COPILOT_REMINDER":
        return execution_payload.get("reminder_text", "This is a reminder about your outstanding balance.")

    if detected_intent in ("PAYMENT_RECEIVED", "AUTO_POLL_RECONCILIATION"):
        amt = execution_payload.get("amount") or execution_payload.get("amount_paid") or 0
        new_status = execution_payload.get("new_status", "")
        # NEW (Feature 5): invoice_pdf_url is only ever populated by
        # apply_captured_payment when THIS payment is what crossed the
        # balance to zero — so its presence is the single source of truth
        # for whether to use the "PAID & CLEARED" receipt copy.
        if execution_payload.get("invoice_pdf_url"):
            inv_num = execution_payload.get("invoice_number", "your invoice")
            return (
                f"✅ Payment Received! ₹{amt:,.2f} credited towards {inv_num}. "
                f"Your bill is now PAID & CLEARED. 📄 Tap below to download your payment receipt."
            )
        inv_num = execution_payload.get("invoice_number")
        inv_part = f" towards {inv_num}" if inv_num else ""
        return f"✅ We've received your payment of ₹{amt:,.2f}{inv_part}. Your account is now {new_status}. Thank you!"

    if action_taken == "CREATE_PAYMENT_LINK":
        if status == "SUCCESS" and execution_payload.get("short_url"):
            amount_paise = execution_payload.get("amount_paise")
            url = execution_payload.get("short_url", "")
            amount_str = f"₹{amount_paise/100:,.2f}" if amount_paise else "the agreed amount"
            expires_in = execution_payload.get("expires_in_minutes")
            expiry_note = f" This link is valid for {expires_in} minutes." if expires_in else ""

            invoice_numbers = execution_payload.get("invoice_numbers")
            if invoice_numbers and len(invoice_numbers) > 1:
                return (
                    f"Sure! Here's a single payment link covering all {len(invoice_numbers)} "
                    f"outstanding invoices ({amount_str} total): {url}{expiry_note}"
                )
            invoice_number = execution_payload.get("invoice_number")
            if invoice_number:
                return f"Sure! Here's your payment link for {amount_str} towards invoice {invoice_number}: {url}{expiry_note}"
            return f"Sure! Here's your payment link for {amount_str}: {url}{expiry_note}"
        if status == "SUCCESS":
            # Marked SUCCESS but the link URL itself is missing (malformed or
            # legacy payload) — never claim a link exists without actually
            # showing one.
            return "Your payment request has been noted — I'll share the payment link shortly."
        # ... (unchanged failure handling below)
        # Surface our OWN guardrail messages verbatim (safe, actionable text
        # we authored ourselves) but keep Razorpay's raw internal error
        # strings hidden from the buyer — those aren't written for a
        # customer-facing audience and could leak implementation details.
        error_msg = execution_payload.get("error", "")
        if "exceeds Razorpay's default per-transaction limit" in error_msg:
            return error_msg
        if "test mode limit" in error_msg.lower():
            # This is a hard Razorpay account-level cap on how many payment
            # links a TEST-mode account can ever create (commonly 30) — it
            # is not a transient failure and retrying won't help. Saying
            # "our team will reach out" here would be actively misleading,
            # since nothing is actually wrong with the buyer's request.
            return (
                "Sorry, this business's Razorpay test account has hit its test-mode "
                "payment-link limit, so we can't issue a new payment link right now. "
                "Please contact us directly to arrange payment another way — the "
                "account owner needs to activate/verify their Razorpay account to lift this limit."
            )
        return "Sorry, I wasn't able to generate a payment link right now — our team will reach out shortly."
    if action_taken == "RESEND_INVOICE":
        if execution_payload.get("invoice_pdf_url"):
            return "Sure, here's a copy of your invoice — tap below to download the PDF."
        return "Sure, I've resent a copy of your invoice to this number."

    if action_taken == "SCHEDULE_REMINDER":
        reminder_date = execution_payload.get("reminder_scheduled_for", "")
        date_part = reminder_date[:10] if reminder_date else "the date you mentioned"
        return f"No problem, I've noted that and will remind you on {date_part}."

    if action_taken == "FLAG_DISPUTE":
        return "I'm sorry to hear that — I've flagged this for our team to review. We'll pause automated collection on this bill until it's resolved."

    if action_taken == "VERIFY_PAYMENT":
        ref = execution_payload.get("payment_reference")
        if ref:
            return f"Thanks — I've noted payment reference {ref}. I'll verify it before updating your balance."
        return "Thanks — I've noted your payment claim. I'll verify it before updating your balance."

    if detected_intent == "PAYMENT_FAILED":
        return "No problem. The payment doesn't appear to have gone through. I can issue a fresh payment link when you're ready."

    if action_taken == "ASK_CLARIFICATION":
        return execution_payload.get(
            "clarification_text",
            "I want to make sure I get this right — could you tell me the exact amount and date you can pay?",
        )

    if action_taken == "ESCALATE":
        return "I understand. I've sent your request to the account team for review. We'll get back to you about the payment arrangement."

    # NEW (Feature 1): UNKNOWN recovery now names the firm, when we have it,
    # instead of a raw generic error.
    business_name = execution_payload.get("business_name")
    if business_name:
        return (
            f"I'm not quite sure I understood that — I'm the automated collections assistant "
            f"for {business_name}. You can tell me how much & when you can pay, ask for your "
            f"bill copy, or let me know if something's wrong with the bill."
        )
    return "I couldn't quite process that — could you let me know how much and when you're able to pay?"


def build_greeting_reply(client: Client) -> str:
    """
    NEW (Feature 1): deterministic GREETING response. Never let the LLM
    author balance figures directly — this pulls the client's live
    unpaid-invoice set straight from the ORM, same discipline as every
    other balance-bearing reply in this file.
    """
    unpaid = [inv for inv in client.invoices if inv.status != InvoiceStatus.PAID]
    if not unpaid:
        return (
            f"Hi {client.name}! Good news — you have no outstanding balance with us "
            f"right now. Let us know if you need anything else."
        )
    total_outstanding = round(sum(inv.balance_amount for inv in unpaid), 2)
    count = len(unpaid)
    return (
        f"Hi {client.name}! You currently have {count} active bill{'s' if count != 1 else ''} "
        f"with us, totaling ₹{total_outstanding:,.2f} outstanding.\n\n"
        f"Quick actions — reply STATEMENT for a full breakdown, PAY to get a payment link, "
        f"or INVOICE to get a copy of your bill."
    )


def apply_captured_payment(
    db: Session,
    invoice: Invoice,
    razorpay_payment_id: str,
    razorpay_payment_link_id: str,
    amount_paid_rupees: float,
    allocation_invoices: Optional[List[Invoice]] = None,
) -> dict:
    """Reconcile one captured provider payment exactly once.

    For a normal invoice payment, the payment is allocated to that invoice.
    For a PAY ALL payment, one provider transaction is recorded once and a
    PaymentAllocation row distributes it across the affected invoices. This
    avoids double-counting a single Razorpay payment while preserving invoice
    level history.
    """
    existing = db.query(PaymentRecord).filter(
        PaymentRecord.razorpay_payment_id == razorpay_payment_id
    ).first()
    if existing:
        return {"applied": False, "invoice_pdf_url": None, "already_processed": True}

    invoices = allocation_invoices or [invoice]
    invoices = sorted([i for i in invoices if i.status != InvoiceStatus.PAID],
                      key=lambda i: (i.due_date, i.id))
    if not invoices:
        return {"applied": False, "invoice_pdf_url": None, "already_processed": False}

    remaining_payment = round(float(amount_paid_rupees), 2)
    allocations = []
    for inv in invoices:
        if remaining_payment <= 0:
            break
        balance = max(round(float(inv.balance_amount), 2), 0.0)
        applied = min(balance, remaining_payment)
        if applied <= 0:
            continue
        inv.paid_amount = round(float(inv.paid_amount) + applied, 2)
        recompute_invoice_ledger(inv)
        allocations.append((inv, round(applied, 2)))
        remaining_payment = round(remaining_payment - applied, 2)

    # Provider transaction is stored once, on the anchor invoice; allocations
    # are the authoritative invoice-level distribution for combined payments.
    payment = PaymentRecord(
        invoice_id=invoice.id,
        razorpay_payment_id=razorpay_payment_id,
        razorpay_payment_link_id=razorpay_payment_link_id or "unknown",
        amount_paid=round(float(amount_paid_rupees), 2),
        paid_at=datetime.utcnow(),
    )
    db.add(payment)
    db.flush()
    for inv, applied in allocations:
        db.add(PaymentAllocation(payment_id=payment.id, invoice_id=inv.id, amount_allocated=applied))
    db.commit()
    db.refresh(invoice)

    if remaining_payment > 0.01:
        write_audit_log(
            db=db, invoice_id=invoice.id,
            incoming_message="Payment exceeds all targeted outstanding balances",
            detected_intent="OVERPAYMENT_FLAG", action_taken="MANUAL_REVIEW_REQUIRED",
            status=AuditStatus.BLOCKED,
            execution_payload={
                "razorpay_payment_id": razorpay_payment_id,
                "captured_amount": amount_paid_rupees,
                "allocated_amount": round(amount_paid_rupees - remaining_payment, 2),
                "excess_amount": remaining_payment,
            },
        )

    # Promise reconciliation is performed for every affected customer/invoice.
    for inv, _ in allocations:
        reconcile_promises(db, inv.client)
        refresh_case(db, inv.client)

    cleared = [inv for inv, _ in allocations if inv.status == InvoiceStatus.PAID]
    receipt_url = f"/api/invoices/{invoice.id}/receipt-pdf" if len(cleared) == 1 and cleared[0].id == invoice.id else None
    return {
        "applied": bool(allocations),
        "invoice_pdf_url": receipt_url,
        "already_processed": False,
        "allocations": [{"invoice_id": inv.id, "invoice_number": inv.invoice_number, "amount": amt} for inv, amt in allocations],
        "excess_amount": remaining_payment,
    }


# --------------------------------------------------------------------------
# REMINDER SCHEDULING — fires an automated follow-up message when a
# buyer's promise-to-pay reminder time arrives (Feature 4).
#
# Implementation note: timers live in-process (threading.Timer), so they
# are lost on a server restart. To make that safe rather than silently
# dropping reminders, requeue_pending_reminders() runs at startup and
# re-schedules (or immediately fires, if the time already passed while
# the server was down) any SCHEDULE_REMINDER audit log that hasn't yet
# been followed by a matching AUTOMATED_REMINDER_FIRED log.
# --------------------------------------------------------------------------

_reminder_timers: Dict[int, threading.Timer] = {}

def cancel_stale_payment_links(db: Session, invoice: Invoice, exclude_link_id: str = None):
    prior_logs = (
        db.query(AuditLog)
        .filter(
            AuditLog.invoice_id == invoice.id,
            AuditLog.action_taken == "CREATE_PAYMENT_LINK",
            AuditLog.status == AuditStatus.SUCCESS,
        )
        .all()
    )
    logger.info(
        "cancel_stale_payment_links: scoped to invoice_id=%s (%s) — found %d prior link(s) to check",
        invoice.id, invoice.invoice_number, len(prior_logs),
    )
    for log in prior_logs:
        payload = json.loads(log.execution_payload) if log.execution_payload else {}
        link_id = payload.get("razorpay_payment_link_id")
        if not link_id or link_id == exclude_link_id:
            continue
        try:
            link_data = fetch_payment_link(link_id)
            if link_data.get("status") in ("created", "issued"):
                cancel_payment_link(link_id)
                logger.info(
                    "cancel_stale_payment_links: CANCELLED link_id=%s short_url=%s for invoice_id=%s (%s)",
                    link_id, link_data.get("short_url"), invoice.id, invoice.invoice_number,
                )
        except RazorpayClientError:
            continue


def get_or_create_payment_link(
    db: Session,
    invoice: Invoice,
    client: Client,
    amount_rupees: float,
    description: str = None,
    invoice_numbers: Optional[List[str]] = None,
) -> tuple[dict, dict]:
    """
    Single shared entry point for every place that needs a Razorpay payment
    link for an invoice — used by the main chat flow (step 8), the
    deterministic PAY fast path, and PAY ALL. Centralizing this means the
    expiry window, the reuse behaviour below, and the exec_payload shape are
    all defined exactly once instead of copy-pasted three times.

    Returns (link_res, exec_payload_fields) — merge exec_payload_fields into
    whichever exec_payload dict the caller is building; it already contains
    razorpay_payment_link_id / short_url / amount_paise / reference_id /
    invoice_number / expires_in_minutes / reused_existing_link.

    WHY REUSE: Razorpay test-mode accounts have a hard cap on the total
    number of payment links that can EVER be created (commonly 30) — see
    RazorpayClientError messages containing "test mode limit ... reached".
    Previously every single buyer message that implied a payment (even a
    repeat of the exact same amount, e.g. re-testing "pay 10000" a few times)
    created a brand-new link and cancelled the old one, burning through that
    fixed quota for no real benefit. Now, if the invoice's most recent link
    is still open (status created/issued) and was created for the SAME
    amount, we just hand that one back instead of minting a new one.
    """
    latest_log = (
        db.query(AuditLog)
        .filter(
            AuditLog.invoice_id == invoice.id,
            AuditLog.action_taken == "CREATE_PAYMENT_LINK",
            AuditLog.status == AuditStatus.SUCCESS,
        )
        .order_by(AuditLog.timestamp.desc())
        .first()
    )
    if latest_log:
        prior_payload = json.loads(latest_log.execution_payload) if latest_log.execution_payload else {}
        prior_link_id = prior_payload.get("razorpay_payment_link_id")
        prior_amount_paise = prior_payload.get("amount_paise")
        if prior_link_id and prior_amount_paise is not None:
            same_amount = abs(int(prior_amount_paise) - rupees_to_paise(amount_rupees)) < 1
            if same_amount:
                try:
                    link_data = fetch_payment_link(prior_link_id)
                    if link_data.get("status") in ("created", "issued"):
                        logger.info(
                            "get_or_create_payment_link: reusing still-open link_id=%s for invoice_id=%s (%s)",
                            prior_link_id, invoice.id, invoice.invoice_number,
                        )
                        exec_fields = {
                            "razorpay_payment_link_id": link_data["id"],
                            "short_url": link_data["short_url"],
                            "amount_paise": link_data.get("amount"),
                            "invoice_number": invoice.invoice_number,
                            "expires_in_minutes": PAYMENT_LINK_EXPIRY_MINUTES,
                            "reused_existing_link": True,
                        }
                        return link_data, exec_fields
                except RazorpayClientError:
                    pass  # couldn't confirm it's still open — fall through and create a fresh one

    ref_id = f"{invoice.invoice_number}-{uuid.uuid4().hex[:8]}"
    link_res = create_payment_link(
        amount_rupees=amount_rupees,
        customer_name=client.name,
        customer_contact=client.phone_number,
        invoice_number=invoice.invoice_number,
        description=description or f"Payment for {invoice.invoice_number}",
        reference_id=ref_id,
        # NOTE: must use datetime.now(timezone.utc), NOT datetime.utcnow().
        # utcnow() returns a naive datetime — correct in VALUE but with no
        # tzinfo attached — and calling .timestamp() on a naive datetime
        # makes Python assume it's in the server's LOCAL timezone, silently
        # converting it as if it were e.g. IST instead of UTC. On a machine
        # set to IST (UTC+5:30) that pushes the computed expire_by ~5.5
        # hours into the past, which is why bumping the minutes buffer
        # alone (10 -> 20) never fixed the "at least 15 minutes in future"
        # rejection — the real error was hours, not minutes. now(timezone.utc)
        # is timezone-AWARE, so .timestamp() converts correctly regardless
        # of what timezone the server happens to run in.
        expire_by=int((datetime.now(timezone.utc) + timedelta(minutes=PAYMENT_LINK_EXPIRY_MINUTES)).timestamp()),
        invoice_numbers=invoice_numbers,
    )
    cancel_stale_payment_links(db, invoice, exclude_link_id=link_res["id"])
    exec_fields = {
        "razorpay_payment_link_id": link_res["id"],
        "short_url": link_res["short_url"],
        "amount_paise": link_res.get("amount"),
        "reference_id": ref_id,
        "invoice_number": invoice.invoice_number,
        **({"invoice_numbers": invoice_numbers} if invoice_numbers else {}),
        "expires_in_minutes": PAYMENT_LINK_EXPIRY_MINUTES,
        "reused_existing_link": False,
    }
    return link_res, exec_fields


def fire_scheduled_reminder(invoice_id: int, schedule_audit_log_id: int) -> None:
    db = SessionLocal()
    try:
        invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
        if not invoice:
            return
        # Nothing to remind about if the buyer already fully paid up in the meantime.
        if invoice.status == InvoiceStatus.PAID:
            return

        reminder_text = (
            f"⏰ Hi {invoice.client.name}, just a friendly follow-up on Invoice "
            f"{invoice.invoice_number} — outstanding balance is ₹{invoice.balance_amount:,.2f}. "
            f"Let us know when you're able to make the payment, or reply here to arrange it now."
        )
        write_audit_log(
            db=db,
            invoice_id=invoice.id,
            incoming_message="SYSTEM: Scheduled reminder fired",
            detected_intent="AUTOMATED_REMINDER_FIRED",
            action_taken="SEND_REMINDER_MESSAGE",
            status=AuditStatus.SUCCESS,
            execution_payload={
                "reminder_text": reminder_text,
                "source_schedule_audit_log_id": schedule_audit_log_id,
            },
        )
        record_communication(db, invoice.client, invoice, "OUTBOUND", reminder_text, channel="SIMULATOR")
        refresh_case(db, invoice.client)
        db.commit()
        logger.info("Fired scheduled reminder for %s", invoice.invoice_number)
    except Exception as exc:
        logger.warning("Failed to fire scheduled reminder for invoice %s: %s", invoice_id, exc)
    finally:
        db.close()
        _reminder_timers.pop(schedule_audit_log_id, None)


def schedule_reminder_timer(invoice_id: int, reminder_dt: datetime, schedule_audit_log_id: int) -> None:
    delay_seconds = max((reminder_dt - datetime.utcnow()).total_seconds(), 1.0)
    timer = threading.Timer(delay_seconds, fire_scheduled_reminder, args=[invoice_id, schedule_audit_log_id])
    timer.daemon = True
    _reminder_timers[schedule_audit_log_id] = timer
    timer.start()
    logger.info(
        "Scheduled reminder for invoice %s in %.0fs (at %s)",
        invoice_id, delay_seconds, reminder_dt.isoformat(),
    )


def requeue_pending_reminders() -> None:
    """Re-arms in-memory reminder timers after a server restart."""
    db = SessionLocal()
    try:
        schedule_logs = (
            db.query(AuditLog)
            .filter(AuditLog.action_taken == "SCHEDULE_REMINDER", AuditLog.status == AuditStatus.SUCCESS)
            .order_by(AuditLog.timestamp.asc())
            .all()
        )
        for log in schedule_logs:
            payload = json.loads(log.execution_payload) if log.execution_payload else {}
            reminder_iso = payload.get("reminder_scheduled_for")
            # New promise reminders are persisted as CollectionAction rows and
            # executed by poll_collection_actions. Do not also arm the legacy
            # in-memory Timer, or one promise would generate duplicate reminders.
            if payload.get("scheduler") == "COLLECTION_ACTION":
                continue
            if not reminder_iso or not log.invoice_id:
                continue

            already_fired = (
                db.query(AuditLog)
                .filter(
                    AuditLog.invoice_id == log.invoice_id,
                    AuditLog.detected_intent.in_(["AUTOMATED_REMINDER_FIRED", "AUTOMATED_PROMISE_FOLLOW_UP"]),
                    AuditLog.timestamp > log.timestamp,
                )
                .first()
            )
            if already_fired:
                continue

            try:
                reminder_dt = datetime.fromisoformat(reminder_iso)
            except ValueError:
                continue

            if reminder_dt <= datetime.utcnow():
                fire_scheduled_reminder(log.invoice_id, log.id)
            else:
                schedule_reminder_timer(log.invoice_id, reminder_dt, log.id)
    finally:
        db.close()


# --------------------------------------------------------------------------
# BACKGROUND POLLER — genuinely automatic reconciliation, no ngrok required
# --------------------------------------------------------------------------

async def poll_outstanding_payment_links():
    """
    Runs forever in the background from app startup. Every POLL_INTERVAL_SECONDS,
    checks every invoice that isn't already fully PAID and asks Razorpay whether
    its MOST RECENT payment link has been paid, applying any newly captured
    payment through apply_captured_payment().

    Only the latest link per invoice is checked — cancel_stale_payment_links()
    already cancels every older link the instant a new one is created, so
    polling historical links is pure wasted API calls and was the direct
    cause of Razorpay's "Too many requests" rate-limit errors once enough
    test messages had accumulated a long link history per invoice.

    A small delay between invoices spaces out calls further, so even a large
    number of open invoices can't burst past Razorpay's rate limit in one cycle.
    """
    while True:
        try:
            db = SessionLocal()
            try:
                invoices = (
                    db.query(Invoice)
                    .filter(Invoice.status != InvoiceStatus.PAID)
                    .all()
                )
                for invoice in invoices:
                    latest_log = (
                        db.query(AuditLog)
                        .filter(
                            AuditLog.invoice_id == invoice.id,
                            AuditLog.action_taken == "CREATE_PAYMENT_LINK",
                            AuditLog.status == AuditStatus.SUCCESS,
                        )
                        .order_by(AuditLog.timestamp.desc())
                        .first()
                    )
                    if not latest_log:
                        continue

                    exec_payload = json.loads(latest_log.execution_payload) if latest_log.execution_payload else {}
                    link_id = exec_payload.get("razorpay_payment_link_id")
                    if not link_id:
                        continue

                    if link_id in _DEAD_LINK_IDS:
                        continue

                    try:
                        link_data = fetch_payment_link(link_id)
                    except RazorpayClientError as exc:
                        if "does not exist" in str(exc).lower():
                            _DEAD_LINK_IDS.add(link_id)
                            logger.info("Poller: %s no longer exists on Razorpay (likely a pre-key-rotation link) — will stop checking it.", link_id)
                        else:
                            logger.warning("Poller: fetch_payment_link failed for %s: %s", link_id, exc)
                        await asyncio.sleep(0.5)
                        continue

                    if link_data.get("status") == "paid":
                        link_notes = link_data.get("notes", {}) or {}
                        combined_numbers = [x.strip() for x in str(link_notes.get("invoice_numbers", "")).split(",") if x.strip()]
                        allocation_invoices = [invoice]
                        if combined_numbers:
                            allocation_invoices = (db.query(Invoice).filter(Invoice.client_id == invoice.client_id, Invoice.invoice_number.in_(combined_numbers)).all() or [invoice])
                        for p in link_data.get("payments", []):
                            if p.get("status") != "captured":
                                continue
                            amount_paid_rupees = paise_to_rupees(p["amount"])
                            result = apply_captured_payment(
                                db, invoice, p["payment_id"], link_id, amount_paid_rupees,
                                allocation_invoices=allocation_invoices,
                            )
                            if result["applied"]:
                                write_audit_log(
                                    db=db, invoice_id=invoice.id,
                                    incoming_message="Background poller detected a captured payment",
                                    detected_intent="AUTO_POLL_RECONCILIATION",
                                    action_taken="RECONCILE_PAYMENT",
                                    status=AuditStatus.SUCCESS,
                                    execution_payload={
                                        "payment_id": p["payment_id"],
                                        "payment_link_id": link_id,
                                        "amount": amount_paid_rupees,
                                        "new_paid_amount": invoice.paid_amount,
                                        "new_status": invoice.status.value,
                                        "invoice_number": invoice.invoice_number,
                                        "invoice_pdf_url": result["invoice_pdf_url"],
                                    },
                                )
                                logger.info(
                                    "Auto-reconciled %s: +₹%s -> paid=%s status=%s",
                                    invoice.invoice_number, amount_paid_rupees,
                                    invoice.paid_amount, invoice.status.value,
                                )

                    # Small pacing delay between invoices — spreads calls out so
                    # a large open-invoice count still can't burst past Razorpay's
                    # rate limit within a single poll cycle.
                    await asyncio.sleep(0.3)
            finally:
                db.close()
        except Exception as exc:
            logger.warning("Background poller iteration failed: %s", exc)

        await asyncio.sleep(POLL_INTERVAL_SECONDS)


OVERDUE_CHECK_INTERVAL_SECONDS = 60  # how often to scan for newly-overdue invoices


async def poll_overdue_invoices():
    """
    Runs forever in the background from app startup. Every
    OVERDUE_CHECK_INTERVAL_SECONDS, finds invoices whose due_date has passed
    and that aren't PAID, and — for any that haven't already gotten one —
    sends a single automated overdue reminder into that buyer's chat
    thread (an AuditLog entry with detected_intent=OVERDUE_AUTO_REMINDER,
    which build_invoice_messages/synthesize_bot_reply_text render as a bot
    message like any other). This also makes is_overdue true wherever it's
    exposed via the API (see _serialize_invoice / list_firms), which is
    what drives the 🚩 flag in the ledger and invoice list.

    Only fires ONCE per invoice — if the buyer disputes or pays afterward,
    we don't want another copy of the same reminder landing in their inbox
    on the next poll cycle.

    Also skips DISPUTED invoices outright (not just PAID ones) — automated
    collection must pause the moment a bill is disputed, matching every
    other "still open" scan in this file (see the DISPUTED exclusions in
    the manual-reminder and pay-all paths). Without this, a dispute raised
    before the next poll cycle would still get chased for payment.
    """
    while True:
        try:
            db = SessionLocal()
            try:
                now = datetime.utcnow()
                overdue_invoices = (
                    db.query(Invoice)
                    .filter(
                        Invoice.due_date < now,
                        Invoice.status != InvoiceStatus.PAID,
                        Invoice.status != InvoiceStatus.DISPUTED,
                    )
                    .all()
                )
                for inv in overdue_invoices:
                    already_flagged = (
                        db.query(AuditLog)
                        .filter(
                            AuditLog.invoice_id == inv.id,
                            AuditLog.detected_intent == "OVERDUE_AUTO_REMINDER",
                        )
                        .first()
                    )
                    if already_flagged:
                        continue

                    days_overdue = max((now - inv.due_date).days, 1)
                    overdue_text = (
                        f"⚠️ Hi {inv.client.name}, Invoice {inv.invoice_number} "
                        f"(₹{inv.balance_amount:,.2f}) was due on {inv.due_date.strftime('%d %b %Y')} "
                        f"and is now {days_overdue} day(s) overdue. Please arrange payment at the "
                        f"earliest, or let us know if there's an issue with this bill."
                    )
                    write_audit_log(
                        db=db,
                        invoice_id=inv.id,
                        incoming_message="SYSTEM: Invoice due date surpassed",
                        detected_intent="OVERDUE_AUTO_REMINDER",
                        action_taken="SEND_OVERDUE_REMINDER",
                        status=AuditStatus.SUCCESS,
                        execution_payload={"overdue_text": overdue_text, "days_overdue": days_overdue},
                    )
                    record_communication(db, inv.client, inv, "OUTBOUND", overdue_text, channel="SIMULATOR")
                    schedule_action(
                        db, inv.client, "OVERDUE_FOLLOW_UP", now + timedelta(hours=72), inv,
                        reason="Initial overdue reminder sent",
                        dedupe_key=f"overdue-followup:{inv.id}:{now.date().isoformat()}"
                    )
                    refresh_case(db, inv.client)
                    db.commit()
                    logger.info(
                        "Flagged %s as overdue (%s day(s)) and sent an automated reminder",
                        inv.invoice_number, days_overdue,
                    )
            finally:
                db.close()
        except Exception as exc:
            logger.warning("Overdue-invoice poller iteration failed: %s", exc)

        await asyncio.sleep(OVERDUE_CHECK_INTERVAL_SECONDS)


COLLECTION_ACTION_INTERVAL_SECONDS = 30


async def poll_collection_actions():
    """Execute due database-backed collection actions safely.

    This is intentionally deterministic: the LLM chooses/extracts bounded
    actions, while this loop checks current financial state before executing.
    """
    while True:
        try:
            db = SessionLocal()
            try:
                now = datetime.utcnow()
                actions = db.query(CollectionAction).filter(
                    CollectionAction.status == CollectionActionStatus.SCHEDULED,
                    CollectionAction.scheduled_at <= now,
                ).order_by(CollectionAction.scheduled_at.asc()).limit(50).all()
                for action in actions:
                    action.status = CollectionActionStatus.PROCESSING
                    action.attempt_count += 1
                    db.commit()
                    try:
                        client = db.query(Client).filter(Client.id == action.client_id).first()
                        invoice = db.query(Invoice).filter(Invoice.id == action.invoice_id).first() if action.invoice_id else None
                        if not client:
                            action.status = CollectionActionStatus.SKIPPED
                            action.result = "Client no longer exists"
                            db.commit()
                            continue
                        case = get_or_create_case(db, client)
                        if not case.autonomous_enabled:
                            action.status = CollectionActionStatus.CANCELLED
                            action.result = "Autonomous collections disabled for this firm"
                            db.commit()
                            continue
                        if invoice and invoice.status == InvoiceStatus.PAID:
                            action.status = CollectionActionStatus.SKIPPED
                            action.result = "Invoice already paid; automated action cancelled"
                            db.commit()
                            continue
                        # For a promise follow-up, the scheduled action itself is
                        # the reminder. Do not run reconcile_promises first, because
                        # that would mark a promise BROKEN at the exact scheduled
                        # timestamp before the buyer has had the rest of the day.
                        promise = None
                        if invoice:
                            promise = db.query(PaymentPromise).filter(
                                PaymentPromise.invoice_id == invoice.id,
                                PaymentPromise.status == PromiseStatus.ACTIVE,
                            ).order_by(PaymentPromise.promised_date.asc()).first()

                        if action.action_type == "FOLLOW_UP_PROMISE":
                            if promise and promise.promised_date > now:
                                action.status = CollectionActionStatus.SCHEDULED
                                action.scheduled_at = promise.promised_date
                                db.commit()
                                continue

                            if invoice and invoice.status == InvoiceStatus.PAID:
                                action.status = CollectionActionStatus.SKIPPED
                                action.result = "Promise fulfilled; no reminder sent"
                            elif invoice and invoice.status != InvoiceStatus.DISPUTED:
                                # Send the promised-date reminder first. The
                                # promise remains ACTIVE until the next-day
                                # broken-promise check.
                                reminder_text = (
                                    f"⏰ Hi {invoice.client.name}, just a friendly follow-up on "
                                    f"Invoice {invoice.invoice_number} — you had planned to make "
                                    f"the payment today. The current outstanding balance is "
                                    f"₹{invoice.balance_amount:,.2f}. Let us know once the payment "
                                    f"is made, or reply here if you need help."
                                )
                                write_audit_log(
                                    db=db,
                                    invoice_id=invoice.id,
                                    incoming_message="SYSTEM: Payment promise follow-up fired",
                                    detected_intent="AUTOMATED_PROMISE_FOLLOW_UP",
                                    action_taken="SEND_PROMISE_FOLLOW_UP",
                                    status=AuditStatus.SUCCESS,
                                    execution_payload={
                                        "reminder_text": reminder_text,
                                        "promise_id": promise.id if promise else None,
                                    },
                                )
                                record_communication(
                                    db, invoice.client, invoice, "OUTBOUND",
                                    reminder_text, channel="SIMULATOR"
                                )
                                action.status = CollectionActionStatus.COMPLETED
                                action.executed_at = now
                                action.result = "Promise-date reminder sent"

                                # Give the buyer until the following day before
                                # declaring the promise broken.
                                schedule_action(
                                    db, client, "BROKEN_PROMISE_FOLLOW_UP",
                                    now + timedelta(hours=24), invoice,
                                    reason="Check whether the promised payment arrived after the due-day reminder",
                                    dedupe_key=f"broken-followup:{invoice.id}:{(now + timedelta(hours=24)).date().isoformat()}",
                                )
                            else:
                                action.status = CollectionActionStatus.SKIPPED
                                action.result = "Invoice disputed or unavailable"
                        elif action.action_type == "OVERDUE_FOLLOW_UP":
                            if invoice and invoice.status != InvoiceStatus.PAID and invoice.status != InvoiceStatus.DISPUTED:
                                # 72h contact cadence with a hard safety stop at 4 attempts.
                                recent_contacts = db.query(Communication).filter(
                                    Communication.client_id == client.id,
                                    Communication.direction == "OUTBOUND",
                                    Communication.created_at >= now - timedelta(days=7),
                                ).count()
                                if recent_contacts >= 4:
                                    case = get_or_create_case(db, client)
                                    case.status = CollectionCaseStatus.ESCALATED
                                    case.escalation_reason = "No payment response after repeated automated collection attempts."
                                    action.status = CollectionActionStatus.SKIPPED
                                    action.result = "Escalated after repeated no-response contacts"
                                else:
                                    text = send_manual_reminder(db, invoice, is_overdue=True)
                                    action.status = CollectionActionStatus.COMPLETED
                                    action.executed_at = now
                                    action.result = "Overdue follow-up sent"
                                    schedule_action(
                                        db, client, "OVERDUE_FOLLOW_UP", now + timedelta(hours=72), invoice,
                                        reason="Continue overdue collection cadence",
                                        dedupe_key=f"overdue-followup:{invoice.id}:{(now + timedelta(hours=72)).date().isoformat()}"
                                    )
                            else:
                                action.status = CollectionActionStatus.SKIPPED
                                action.result = "Invoice no longer collectible automatically"
                        elif action.action_type == "BROKEN_PROMISE_FOLLOW_UP":
                            reconcile_promises(db, client)
                            db.flush()
                            if invoice and invoice.status != InvoiceStatus.PAID and invoice.status != InvoiceStatus.DISPUTED:
                                broken = db.query(PaymentPromise).filter(
                                    PaymentPromise.invoice_id == invoice.id,
                                    PaymentPromise.status == PromiseStatus.BROKEN,
                                ).order_by(PaymentPromise.broken_at.desc()).first()
                                if broken:
                                    reminder_text = send_manual_reminder(db, invoice, is_overdue=True)
                                    action.status = CollectionActionStatus.COMPLETED
                                    action.executed_at = now
                                    action.result = "Broken-promise follow-up sent"
                                else:
                                    action.status = CollectionActionStatus.SKIPPED
                                    action.result = "Promise fulfilled or still active; no broken-promise follow-up needed"
                            else:
                                action.status = CollectionActionStatus.SKIPPED
                                action.result = "No longer collectible automatically"
                        db.commit()
                    except Exception as exc:
                        action.status = CollectionActionStatus.FAILED
                        action.result = str(exc)[:2000]
                        db.commit()
            finally:
                db.close()
        except Exception as exc:
            logger.warning("Collection action poller failed: %s", exc)
        await asyncio.sleep(COLLECTION_ACTION_INTERVAL_SECONDS)


@app.on_event("startup")
async def start_background_poller():
    asyncio.create_task(poll_outstanding_payment_links())
    logger.info("Started background payment-link poller (every %ss)", POLL_INTERVAL_SECONDS)
    asyncio.create_task(poll_overdue_invoices())
    logger.info("Started background overdue-invoice poller (every %ss)", OVERDUE_CHECK_INTERVAL_SECONDS)
    asyncio.create_task(poll_collection_actions())
    logger.info("Started database-backed collection action poller (every %ss)", COLLECTION_ACTION_INTERVAL_SECONDS)
    requeue_pending_reminders()


# --------------------------------------------------------------------------
# COLLECTION COMMAND CENTER
# --------------------------------------------------------------------------

@app.get("/api/collections/queue")
def collections_queue(limit: int = 20, db: Session = Depends(get_db), _auth: None = Depends(require_auth)):
    return {"items": collection_queue(db, max(1, min(limit, 100)))}


@app.get("/api/collections/cases/{client_id}")
def get_collection_case(client_id: int, db: Session = Depends(get_db), _auth: None = Depends(require_auth)):
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Firm not found")
    reconcile_promises(db, client)
    case = refresh_case(db, client)
    db.commit()
    return {
        "client_id": client.id, "status": case.status.value,
        "priority_score": case.priority_score,
        "next_action_at": case.next_action_at.isoformat() if case.next_action_at else None,
        "autonomous_enabled": case.autonomous_enabled,
        "escalation_reason": case.escalation_reason,
    }


@app.post("/api/collections/cases/{client_id}/autonomy")
def set_collection_autonomy(client_id: int, enabled: bool, db: Session = Depends(get_db), _auth: None = Depends(require_auth)):
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Firm not found")
    case = get_or_create_case(db, client)
    case.autonomous_enabled = enabled
    if not enabled:
        for action in db.query(CollectionAction).filter(
            CollectionAction.client_id == client_id,
            CollectionAction.status == CollectionActionStatus.SCHEDULED,
        ).all():
            action.status = CollectionActionStatus.CANCELLED
    db.commit()
    return {"client_id": client_id, "autonomous_enabled": case.autonomous_enabled}


@app.get("/api/collections/summary")
def collections_summary(db: Session = Depends(get_db), _auth: None = Depends(require_auth)):
    now = datetime.utcnow()
    invoices = db.query(Invoice).all()
    outstanding = round(sum(float(i.balance_amount) for i in invoices if i.status != InvoiceStatus.PAID), 2)
    recovered = round(sum(float(p.amount_paid) for p in db.query(PaymentRecord).all()), 2)
    active_promises = db.query(PaymentPromise).filter(PaymentPromise.status == PromiseStatus.ACTIVE).count()
    broken_promises = db.query(PaymentPromise).filter(PaymentPromise.status == PromiseStatus.BROKEN).count()
    disputed = db.query(Invoice).filter(Invoice.status == InvoiceStatus.DISPUTED).count()
    due_today = sum(1 for i in invoices if i.status != InvoiceStatus.PAID and i.due_date.date() == now.date())
    overdue = sum(1 for i in invoices if i.status != InvoiceStatus.PAID and i.due_date < now)
    return {
        "outstanding": outstanding, "recovered_total": recovered,
        "active_promises": active_promises, "broken_promises": broken_promises,
        "disputed_invoices": disputed, "due_today": due_today, "overdue_invoices": overdue,
    }


@app.post("/api/collections/promises/{promise_id}/check")
def check_promise(promise_id: int, db: Session = Depends(get_db), _auth: None = Depends(require_auth)):
    promise = db.query(PaymentPromise).filter(PaymentPromise.id == promise_id).first()
    if not promise:
        raise HTTPException(status_code=404, detail="Promise not found")
    reconcile_promises(db, promise.client)
    refresh_case(db, promise.client)
    db.commit()
    return {"id": promise.id, "status": promise.status.value, "fulfilled_at": promise.fulfilled_at, "broken_at": promise.broken_at}


# --------------------------------------------------------------------------
# DASHBOARD PAGE
# --------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    invoices = db.query(Invoice).order_by(Invoice.due_date.asc()).all()
    audit_logs = db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(50).all()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "invoices": invoices, "audit_logs": audit_logs},
    )


# --------------------------------------------------------------------------
# READ-ONLY API ENDPOINTS (used by dashboard polling / JS)
# --------------------------------------------------------------------------



@app.get("/api/invoices")
def list_invoices(db: Session = Depends(get_db), _auth: None = Depends(require_auth)):
    invoices = db.query(Invoice).order_by(Invoice.due_date.asc()).all()
    result = []
    for inv in invoices:
        last_log = (
            db.query(AuditLog)
            .filter(AuditLog.invoice_id == inv.id)
            .order_by(AuditLog.timestamp.desc())
            .first()
        )
        result.append({
            "id": inv.id,
            "invoice_number": inv.invoice_number,
            "client_name": inv.client.name,
            "business_name": inv.client.business_name,
            "phone_number": inv.client.phone_number,
            "total_amount": inv.total_amount,
            "paid_amount": inv.paid_amount,
            "balance_amount": inv.balance_amount,
            "status": inv.status.value,
            "due_date": inv.due_date.isoformat(),
            "last_contacted": last_log.timestamp.isoformat() if last_log else None,
        })
    return {"invoices": result}

@app.get("/api/audit-logs")
def list_audit_logs(limit: int = 50, db: Session = Depends(get_db), _auth: None = Depends(require_auth)):
    logs = (
        db.query(AuditLog)
        .order_by(AuditLog.timestamp.desc())
        .limit(limit)
        .all()
    )
    result = []
    for log in logs:
        result.append({
            "id": log.id,
            "invoice_id": log.invoice_id,
            "incoming_message": log.incoming_message,
            "detected_intent": log.detected_intent,
            "action_taken": log.action_taken,
            "status": log.status.value,
            "execution_payload": json.loads(log.execution_payload) if log.execution_payload else None,
            "timestamp": log.timestamp.isoformat(),
        })
    return {"audit_logs": result}


@app.get("/api/clients")
def list_clients(db: Session = Depends(get_db), _auth: None = Depends(require_auth)):
    clients = db.query(Client).all()
    return {
        "clients": [
            {
                "id": c.id,
                "name": c.name,
                "business_name": c.business_name,
                "phone_number": c.phone_number,
            }
            for c in clients
        ]
    }


# --------------------------------------------------------------------------
# WHATSAPP SIMULATOR — the core agentic loop
# --------------------------------------------------------------------------

INVOICES_PER_PAGE = 5


def _format_invoice_breakdown_lines(invoices_page: List[Invoice]) -> str:
    lines = [
        f"• {inv.invoice_number}: ₹{inv.balance_amount:,.2f} (due {inv.due_date.strftime('%d %b %Y')})"
        for inv in invoices_page
    ]
    return "\n".join(lines)


def _handle_multi_invoice_breakdown(
    db: Session, client: Client, message: str, open_invoices: List[Invoice], decision, page: int = 1
) -> SimulateMessageResponse:
    """
    Buyer asked generically about their bill with more than one open invoice
    and no invoice_number named. Returns an itemized breakdown, paginated at
    INVOICES_PER_PAGE per page for firms with a large number of open bills,
    plus tappable ChatOptions: one per invoice (payload = its invoice number,
    which the existing fast-regex path in pre_process_user_intent/
    simulate_message already knows how to resolve — no new routing needed),
    a "Pay All" option, and Next/Previous page controls when applicable.
    """
    extracted = decision.extracted
    anchor_invoice = min(open_invoices, key=lambda inv: inv.due_date)

    sorted_invoices = sorted(open_invoices, key=lambda i: i.due_date)
    total_count = len(sorted_invoices)
    total_pages = max(1, (total_count + INVOICES_PER_PAGE - 1) // INVOICES_PER_PAGE)
    page = max(1, min(page, total_pages))

    start = (page - 1) * INVOICES_PER_PAGE
    page_invoices = sorted_invoices[start:start + INVOICES_PER_PAGE]

    page_note = f" (page {page} of {total_pages})" if total_pages > 1 else ""

    # Preserve the buyer's original intent when they tap an invoice. A bare
    # invoice number used to open the invoice menu and lose the original
    # promise/payment request. For payment-related clarification, re-submit
    # the original message with the chosen invoice appended so the same
    # deterministic/LLM flow can finish the action safely.
    intent_value = extracted.intent.value
    if intent_value == IntentType.PROMISE_TO_PAY.value:
        prompt_tail = "Which invoice should this promise apply to?"
        invoice_option_payload = lambda inv: f"{message} for {inv.invoice_number}"
        footer = "Choose the invoice this promised payment should apply to."
    elif intent_value == IntentType.PARTIAL_PAYMENT.value:
        prompt_tail = "Which invoice should the payment apply to?"
        invoice_option_payload = lambda inv: f"{message} for {inv.invoice_number}"
        footer = "Choose the invoice you want to make this payment against."
    elif intent_value == IntentType.FULL_PAYMENT.value:
        prompt_tail = "Which invoice would you like to pay in full?"
        invoice_option_payload = lambda inv: f"{message} for {inv.invoice_number}"
        footer = "Choose the invoice you want to clear."
    else:
        prompt_tail = "Which invoice would you like to work with?"
        invoice_option_payload = lambda inv: inv.invoice_number
        footer = "Choose an invoice to continue."

    bot_reply_text = (
        f"You have {total_count} open invoices{page_note}:\n"
        f"{_format_invoice_breakdown_lines(page_invoices)}\n\n"
        f"{prompt_tail}\n{footer}"
    )

    options = [
        ChatOption(
            id=f"OPT_{inv.invoice_number}",
            title=inv.invoice_number,
            payload=invoice_option_payload(inv),
        )
        for inv in page_invoices
    ]
    if intent_value != IntentType.PROMISE_TO_PAY.value:
        options.append(ChatOption(id="OPT_PAY_ALL", title="💳 Pay All", payload="PAY ALL"))
    if page > 1:
        options.append(ChatOption(id=f"OPT_PAGE_{page-1}", title="⬅️ Previous Page", payload=f"PAGE {page-1}"))
    if page < total_pages:
        options.append(ChatOption(id=f"OPT_PAGE_{page+1}", title="➡️ Next Page", payload=f"PAGE {page+1}"))

    execution_payload = {
        "validation_reason": (
            "Buyer's message was ambiguous across multiple open invoices; "
            "returned itemized breakdown instead of guessing which one."
        ),
        "page": page,
        "total_pages": total_pages,
        "invoice_breakdown": [
            {
                "invoice_number": inv.invoice_number,
                "balance_amount": inv.balance_amount,
                "due_date": inv.due_date.isoformat(),
            }
            for inv in page_invoices
        ],
    }

    log = write_audit_log(
        db=db, invoice_id=anchor_invoice.id,
        incoming_message=message,
        detected_intent=extracted.intent.value, action_taken="MULTI_INVOICE_BREAKDOWN",
        status=AuditStatus.SUCCESS,
        execution_payload=execution_payload,
    )

    return SimulateMessageResponse(
        invoice_id=anchor_invoice.id,
        incoming_message=message,
        detected_intent=extracted.intent.value,
        confidence=extracted.confidence,
        llm_reasoning=extracted.reasoning,
        action_taken="MULTI_INVOICE_BREAKDOWN",
        allowed=True,
        guardrail_reason=execution_payload["validation_reason"],
        final_amount=None,
        payment_link_url=None,
        invoice_pdf_url=None,
        updated_invoice_status=anchor_invoice.status.value,
        audit_log_id=log.id,
        bot_reply_text=bot_reply_text,
        options=options,
    )


def _handle_all_invoices_pdf(
    db: Session, client: Client, message: str, open_invoices: List[Invoice]
) -> SimulateMessageResponse:
    """
    Buyer explicitly asked for ALL/BOTH open invoices as one combined
    document (e.g. "give me invoices of both all bills in pdf", or tapping
    the "📎 All Invoices (Combined PDF)" option) — hand back ONE merged PDF
    covering every open invoice, rather than several separate download
    links.
    """
    sorted_invoices = sorted(open_invoices, key=lambda i: i.due_date)
    anchor_invoice = sorted_invoices[0]
    combined_pdf_url = f"/api/firms/{client.id}/invoices/combined-pdf"
    bot_reply_text = f"Sure, here's one combined PDF with all {len(sorted_invoices)} of your open invoices — tap below to download."
    exec_payload = {
        "validation_reason": "Buyer explicitly requested a single combined PDF of all open invoices.",
        "business_name": client.business_name,
        "invoice_pdf_url": combined_pdf_url,
    }
    log = write_audit_log(
        db=db, invoice_id=anchor_invoice.id, incoming_message=message,
        detected_intent="REQUEST_INVOICE", action_taken="RESEND_INVOICE",
        status=AuditStatus.SUCCESS, execution_payload=exec_payload,
    )
    return SimulateMessageResponse(
        invoice_id=anchor_invoice.id, incoming_message=message, detected_intent="REQUEST_INVOICE",
        confidence=1.0, llm_reasoning="Deterministic combined-invoices-PDF interceptor — no LLM call needed.",
        action_taken="RESEND_INVOICE", allowed=True, guardrail_reason=exec_payload["validation_reason"],
        final_amount=None, payment_link_url=None, invoice_pdf_url=combined_pdf_url,
        updated_invoice_status=anchor_invoice.status.value, audit_log_id=log.id,
        bot_reply_text=bot_reply_text,
    )


def _handle_invoice_choice_prompt(
    db: Session, client: Client, message: str, open_invoices: List[Invoice]
) -> SimulateMessageResponse:
    """
    Buyer asked for "an invoice" / "the bill copy" generically while more
    than one invoice is open, without saying "all"/"both" — rather than
    guessing which one they meant (or defaulting to just one), explicitly
    ask: a specific invoice, or one combined PDF with all of them. Each
    invoice gets its own tappable option, plus one option for the combined
    document; tapping either re-enters this same flow with an unambiguous
    message ("invoice copy INV-1007" / "all invoices combined pdf").
    """
    sorted_invoices = sorted(open_invoices, key=lambda i: i.due_date)
    anchor_invoice = sorted_invoices[0]
    prompt_text = (
        f"You have {len(sorted_invoices)} open invoices. Would you like a copy of a "
        f"specific one, or a single combined PDF with all of them?"
    )
    options = [
        ChatOption(
            id=f"OPT_INVPDF_{inv.invoice_number}",
            title=f"📄 {inv.invoice_number} (₹{inv.balance_amount:,.2f})",
            payload=f"invoice copy {inv.invoice_number}",
        )
        for inv in sorted_invoices
    ] + [
        ChatOption(id="OPT_ALL_INVOICES_PDF", title="📎 All Invoices (Combined PDF)", payload="all invoices combined pdf"),
    ]
    exec_payload = {
        "validation_reason": "Buyer asked for an invoice copy without naming one; asked whether they want a specific invoice or a combined document.",
        "prompt_text": prompt_text,
    }
    log = write_audit_log(
        db=db, invoice_id=anchor_invoice.id, incoming_message=message,
        detected_intent="REQUEST_INVOICE", action_taken="INVOICE_CHOICE_PROMPT",
        status=AuditStatus.SUCCESS, execution_payload=exec_payload,
    )
    return SimulateMessageResponse(
        invoice_id=anchor_invoice.id, incoming_message=message, detected_intent="REQUEST_INVOICE",
        confidence=1.0, llm_reasoning="Deterministic invoice-choice-prompt interceptor — no LLM call needed.",
        action_taken="INVOICE_CHOICE_PROMPT", allowed=True, guardrail_reason=exec_payload["validation_reason"],
        final_amount=None, payment_link_url=None, invoice_pdf_url=None,
        updated_invoice_status=anchor_invoice.status.value, audit_log_id=log.id,
        bot_reply_text=prompt_text,
        options=options,
    )


def _create_full_payment_link_response(
    db: Session, client: Client, target_invoice: Invoice, clean_msg: str
) -> SimulateMessageResponse:
    """
    Shared logic for any deterministic full-payment-link path: an explicit
    'PAY <invoice>' command, or a bare 'PAY' resolved to the client's single
    open invoice. Both need identical dispute/zero-balance checks and the
    same Razorpay create+cancel-stale sequence — this avoids maintaining
    that logic in two (or more) separate places.
    """
    if target_invoice.status == InvoiceStatus.DISPUTED:
        exec_payload = {"validation_reason": "Invoice is DISPUTED; payment link blocked."}
        log = write_audit_log(
            db=db, invoice_id=target_invoice.id, incoming_message=clean_msg,
            detected_intent="FULL_PAYMENT", action_taken="NO_ACTION",
            status=AuditStatus.BLOCKED, execution_payload=exec_payload,
        )
        return SimulateMessageResponse(
            invoice_id=target_invoice.id, incoming_message=clean_msg,
            detected_intent="FULL_PAYMENT", confidence=1.0,
            llm_reasoning="Deterministic PAY fast path — no LLM call needed.",
            action_taken="NO_ACTION", allowed=False,
            guardrail_reason=exec_payload["validation_reason"],
            final_amount=None, payment_link_url=None, invoice_pdf_url=None,
            updated_invoice_status=target_invoice.status.value,
            audit_log_id=log.id,
            bot_reply_text="This invoice is currently under dispute — our team needs to resolve that before a payment link can be issued.",
        )

    if target_invoice.balance_amount <= 0:
        exec_payload = {"validation_reason": "Invoice balance already zero."}
        log = write_audit_log(
            db=db, invoice_id=target_invoice.id, incoming_message=clean_msg,
            detected_intent="FULL_PAYMENT", action_taken="NO_ACTION",
            status=AuditStatus.BLOCKED, execution_payload=exec_payload,
        )
        return SimulateMessageResponse(
            invoice_id=target_invoice.id, incoming_message=clean_msg,
            detected_intent="FULL_PAYMENT", confidence=1.0,
            llm_reasoning="Deterministic PAY fast path — no LLM call needed.",
            action_taken="NO_ACTION", allowed=False,
            guardrail_reason=exec_payload["validation_reason"],
            final_amount=None, payment_link_url=None, invoice_pdf_url=None,
            updated_invoice_status=target_invoice.status.value,
            audit_log_id=log.id,
            bot_reply_text=f"Good news — {target_invoice.invoice_number} is already fully paid, nothing owed!",
        )

    try:
        link_res, link_fields = get_or_create_payment_link(db, target_invoice, client, target_invoice.balance_amount)
        exec_payload = {
            "validation_reason": f"Deterministic PAY command for {target_invoice.invoice_number}; full balance.",
            **link_fields,
        }
        audit_status = AuditStatus.SUCCESS
        payment_link_url = link_res["short_url"]
    except (RazorpayClientError, TypeError) as exc:
        exec_payload = {"validation_reason": "Attempted deterministic PAY command.", "error": str(exc)}
        audit_status = AuditStatus.FAILED
        payment_link_url = None

    bot_reply_text = synthesize_bot_reply_text("FULL_PAYMENT", "CREATE_PAYMENT_LINK", audit_status.value, exec_payload)
    log = write_audit_log(
        db=db, invoice_id=target_invoice.id, incoming_message=clean_msg,
        detected_intent="FULL_PAYMENT", action_taken="CREATE_PAYMENT_LINK",
        status=audit_status, execution_payload=exec_payload,
    )
    return SimulateMessageResponse(
        invoice_id=target_invoice.id, incoming_message=clean_msg,
        detected_intent="FULL_PAYMENT", confidence=1.0,
        llm_reasoning="Deterministic PAY fast path — no LLM call needed.",
        action_taken="CREATE_PAYMENT_LINK", allowed=(audit_status == AuditStatus.SUCCESS),
        guardrail_reason=exec_payload["validation_reason"],
        final_amount=target_invoice.balance_amount, payment_link_url=payment_link_url,
        invoice_pdf_url=None, updated_invoice_status=target_invoice.status.value,
        audit_log_id=log.id, bot_reply_text=bot_reply_text,
    )

def _handle_pay_all(db: Session, client: Client, message: str, specified_amount: Optional[float] = None) -> SimulateMessageResponse:
    """
    "PAY ALL" / a natural-language multi-invoice payment creates a single
    payment link covering the client's open invoices.

    specified_amount: if the buyer stated a specific figure ("pay ₹70,000
    against all pending invoices"), that amount is used verbatim (capped at
    the true outstanding total, never exceeded) instead of silently
    substituting the FULL balance — a buyer offering a partial amount here
    must never get a link for more than they actually agreed to pay.
    """
    disputed_invoices = [inv for inv in client.invoices if inv.status == InvoiceStatus.DISPUTED]
    open_invoices = [inv for inv in client.invoices if inv.status not in (InvoiceStatus.PAID, InvoiceStatus.DISPUTED)]
    anchor_invoice = min(open_invoices, key=lambda inv: inv.due_date) if open_invoices else resolve_target_invoice(db, client)

    if not open_invoices:
        execution_payload = {"validation_reason": "PAY ALL requested but client has no open invoices."}
        log = write_audit_log(
            db=db, invoice_id=anchor_invoice.id if anchor_invoice else None,
            incoming_message=message,
            detected_intent="FULL_PAYMENT", action_taken="NO_ACTION",
            status=AuditStatus.BLOCKED,
            execution_payload=execution_payload,
        )
        return SimulateMessageResponse(
            invoice_id=anchor_invoice.id if anchor_invoice else 0,
            incoming_message=message,
            detected_intent="FULL_PAYMENT",
            confidence=1.0,
            llm_reasoning="Deterministic handling for PAY ALL",
            action_taken="NO_ACTION",
            allowed=False,
            guardrail_reason=execution_payload["validation_reason"],
            final_amount=None,
            payment_link_url=None,
            invoice_pdf_url=None,
            updated_invoice_status=anchor_invoice.status.value if anchor_invoice else "PAID",
            audit_log_id=log.id,
            bot_reply_text=f"Hi {client.name}, you have no open invoices to pay right now!",
        )

    # Sum total outstanding across all open invoices
    total_balance = round(sum(inv.balance_amount for inv in open_invoices), 2)

    # A buyer-specified amount is honored verbatim, but never allowed to
    # exceed the real outstanding total (protects against a typo/hallucinated
    # figure creating a link for more than is actually owed).
    amount_to_charge = total_balance
    is_partial = False
    if specified_amount is not None and specified_amount > 0:
        amount_to_charge = round(min(float(specified_amount), total_balance), 2)
        is_partial = amount_to_charge < total_balance

    try:
        link_res, link_fields = get_or_create_payment_link(
            db, anchor_invoice, client, amount_to_charge,
            description=f"{'Partial combined' if is_partial else 'Combined'} payment for {len(open_invoices)} open invoices",
            invoice_numbers=[inv.invoice_number for inv in open_invoices],
        )
    except (RazorpayClientError,TypeError) as exc:
        execution_payload = {
            "validation_reason": f"Attempted combined payment link for {len(open_invoices)} invoices totaling ₹{amount_to_charge:,.2f}.",
            "error": str(exc),
        }
        log = write_audit_log(
            db=db, invoice_id=anchor_invoice.id,
            incoming_message=message,
            detected_intent="FULL_PAYMENT" if not is_partial else "MULTI_INVOICE_PAYMENT",
            action_taken="CREATE_PAYMENT_LINK",
            status=AuditStatus.FAILED,
            execution_payload=execution_payload,
        )
        return SimulateMessageResponse(
            invoice_id=anchor_invoice.id,
            incoming_message=message,
            detected_intent="FULL_PAYMENT" if not is_partial else "MULTI_INVOICE_PAYMENT",
            confidence=1.0,
            llm_reasoning="Deterministic rule for combined balance payment.",
            action_taken="CREATE_PAYMENT_LINK",
            allowed=False,
            guardrail_reason=execution_payload["validation_reason"],
            final_amount=amount_to_charge,
            payment_link_url=None,
            invoice_pdf_url=None,
            updated_invoice_status=anchor_invoice.status.value,
            audit_log_id=log.id,
            bot_reply_text="Sorry, I wasn't able to generate a combined payment link right now — our team will reach out shortly.",
        )
    # get_or_create_payment_link already cancels anchor_invoice's own stale
    # links; also cancel stale links on the OTHER invoices being combined,
    # since a buyer could otherwise still pay one of those individually too.
    for inv in open_invoices:
        if inv.id != anchor_invoice.id:
            cancel_stale_payment_links(db, inv, exclude_link_id=link_res["id"])
    execution_payload = {
        "validation_reason": (
            f"Combined payment link created for {len(open_invoices)} invoices — "
            f"₹{amount_to_charge:,.2f} of ₹{total_balance:,.2f} total outstanding."
            if is_partial else
            f"Combined payment link created for {len(open_invoices)} invoices totaling ₹{total_balance:,.2f}."
        ),
        **link_fields,
        "invoice_numbers": [inv.invoice_number for inv in open_invoices],
        "expires_in_minutes": PAYMENT_LINK_EXPIRY_MINUTES,
    }

    detected_intent_label = "MULTI_INVOICE_PAYMENT" if is_partial else "FULL_PAYMENT"
    log = write_audit_log(
        db=db,
        invoice_id=anchor_invoice.id,
        incoming_message=message,
        detected_intent=detected_intent_label,
        action_taken="CREATE_PAYMENT_LINK",
        status=AuditStatus.SUCCESS,
        execution_payload=execution_payload,
    )

    bot_text = synthesize_bot_reply_text(detected_intent_label, "CREATE_PAYMENT_LINK", "SUCCESS", execution_payload)
    if is_partial:
        bot_text += f" This covers ₹{amount_to_charge:,.2f} of your ₹{total_balance:,.2f} total outstanding."
    if disputed_invoices:
        bot_text += f" Note: {len(disputed_invoices)} disputed invoice(s) are excluded until resolved."

    return SimulateMessageResponse(
        invoice_id=anchor_invoice.id,
        incoming_message=message,
        detected_intent=detected_intent_label,
        confidence=1.0,
        llm_reasoning="Deterministic rule for combined balance payment.",
        action_taken="CREATE_PAYMENT_LINK",
        allowed=True,
        guardrail_reason=execution_payload["validation_reason"],
        final_amount=total_balance,
        payment_link_url=link_res["short_url"],
        invoice_pdf_url=None,
        updated_invoice_status=anchor_invoice.status.value,
        audit_log_id=log.id,
        bot_reply_text=bot_text,
    )

def _normalize_collection_datetime(dt: datetime) -> datetime:
    """
    Calendar-date promises ("tomorrow", "Friday") are stored as naive UTC
    datetimes throughout this app. Use 10:00 IST = 04:30 UTC as the default
    collection follow-up time instead of midnight. Preserve explicit relative
    durations and any datetime that already has a non-midnight clock time.
    """
    if dt.hour == 0 and dt.minute == 0 and dt.second == 0 and dt.microsecond == 0:
        return dt.replace(hour=4, minute=30, second=0, microsecond=0)
    return dt


def _extract_relative_minutes(message: str) -> Optional[int]:
    """
    Extracts a relative duration in minutes from phrases like "remind me in
    10 mins", "after 2 minutes", "in 1 hour", "send it after 2 mins".
    Returns None if no such pattern is present.
    """
    msg = message.strip().lower()
    hour_match = re.search(r'\b(?:in|after)\s+(\d+)\s*(?:hour|hours|hr|hrs)\b', msg)
    if hour_match:
        return int(hour_match.group(1)) * 60
    min_match = re.search(r'\b(?:in|after)\s+(\d+)\s*(?:min|mins|minute|minutes)\b', msg)
    if min_match:
        return int(min_match.group(1))
    return None

def _extract_fast_amount(msg_upper: str) -> float | None:
    """Best-effort deterministic amount parse — used only as a fallback
    when the LLM fails to extract one for an obvious partial-payment message."""
    m = (
        re.search(r'₹\s*([\d,]+(?:\.\d+)?)', msg_upper)
        or re.search(r'\bRS\.?\s*([\d,]+(?:\.\d+)?)', msg_upper)
        or re.search(r'\b(\d{5,})\b', msg_upper)
    )
    if not m:
        return None
    try:
        return float(m.group(1).replace(',', ''))
    except ValueError:
        return None


def _extract_fast_promise_date(message: str) -> Optional[str]:
    """
    Deterministically resolve common future-date phrases used in payment
    promises. This is a safety fallback for transient LLM failures; the LLM
    remains the primary interpreter for richer language.
    """
    msg = message.strip().lower()
    today = datetime.utcnow().date()

    if re.search(r"\b(day after tomorrow)\b", msg):
        return (today + timedelta(days=2)).isoformat()
    if re.search(r"\b(tomorrow|kal)\b", msg):
        return (today + timedelta(days=1)).isoformat()

    rel_match = re.search(r"\bin\s+(\d+)\s+(day|days|week|weeks)\b", msg)
    if rel_match:
        amount = int(rel_match.group(1))
        unit = rel_match.group(2)
        days = amount * (7 if "week" in unit else 1)
        return (today + timedelta(days=days)).isoformat()

    if re.search(r"\b(next\s+week|next week)\b", msg):
        return (today + timedelta(days=7)).isoformat()

    weekdays = {
        "monday": 0, "tuesday": 1, "wednesday": 2,
        "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
    }
    for name, weekday in weekdays.items():
        if re.search(rf"\b(?:next\s+)?{name}\b", msg):
            days_ahead = (weekday - today.weekday()) % 7
            # A bare weekday means a future occurrence, never today.
            if days_ahead == 0:
                days_ahead = 7
            return (today + timedelta(days=days_ahead)).isoformat()

    if re.search(r"\b(month end|end of (?:this )?month)\b", msg):
        if today.month == 12:
            next_month = today.replace(year=today.year + 1, month=1, day=1)
        else:
            next_month = today.replace(month=today.month + 1, day=1)
        return (next_month - timedelta(days=1)).isoformat()

    return None


def pre_process_user_intent(message: str) -> tuple[str | None, str | None, float | None]:
    """Deterministic first-pass classifier for high-confidence buyer intents.

    This function is intentionally conservative: it handles explicit commands
    and obvious financial statements without calling Gemini. Ambiguous natural
    language is left as None so the LLM can interpret it later.
    """
    msg_upper = message.strip().upper()

    inv_match = re.search(r'\b(INV-?\d{4}|\d{4})\b', msg_upper)
    extracted_inv = None
    if inv_match:
        matched_str = inv_match.group(1)
        extracted_inv = (
            matched_str if matched_str.startswith("INV-") else f"INV-{matched_str}"
        )

    has_amount = bool(re.search(r'₹\s*\d+|\bRS\.?\s*\d+|\b\d{5,}\b', msg_upper))
    fast_amount = _extract_fast_amount(msg_upper) if has_amount else None

    # High-confidence exception/safety intents. These must bypass Gemini so a
    # transient model outage cannot turn a buyer claim into a payment action.
    if re.search(r'\b(UTR|TRANSACTION\s*(ID|REFERENCE)?|TXN|PAYMENT\s*REFERENCE|REFERENCE\s*(NO|NUMBER)?)\b', msg_upper):
        if any(x in msg_upper for x in ["ALREADY PAID", "I PAID", "PAID", "PAYMENT MADE", "PAYMENT DONE"]):
            return extracted_inv, "PAYMENT_PROOF", fast_amount

    if any(x in msg_upper for x in [
        "ALREADY PAID", "I ALREADY PAID", "I PAID", "PAYMENT MADE", "PAYMENT DONE",
        "I HAVE PAID", "WE HAVE PAID", "ALREADY SETTLED",
    ]):
        return extracted_inv, "ALREADY_PAID", fast_amount

    if any(x in msg_upper for x in [
        "PAYMENT FAILED", "PAYMENT FAIL", "FAILED PAYMENT", "PAYMENT DIDN'T GO THROUGH",
        "PAYMENT DID NOT GO THROUGH", "COULDN'T PAY", "COULD NOT PAY", "COULDN'T MAKE THE PAYMENT",
    ]):
        return extracted_inv, "PAYMENT_FAILED", fast_amount

    if any(x in msg_upper for x in [
        "PAYMENT PENDING", "PAYMENT IS PENDING", "TRANSACTION PENDING", "STUCK PAYMENT",
        "PAYMENT PROCESSING", "PAYMENT STILL PENDING",
    ]):
        return extracted_inv, "PAYMENT_PENDING", fast_amount

    if any(x in msg_upper for x in [
        "INSTALLMENT", "INSTALLMENTS", "INSTALMENT", "INSTALMENTS", "PAY IN PARTS",
        "PAY IN EMI", "CAN I PAY LATER", "CAN I PAY IN PART", "PAYMENT PLAN",
    ]):
        return extracted_inv, "PAYMENT_PLAN_REQUEST", fast_amount

    if any(x in msg_upper for x in [
        "DISPUTE", "DISPUTED", "WRONG BILL", "WRONG AMOUNT", "INCORRECT AMOUNT",
        "INCORRECT BILL", "BILL IS WRONG", "INVOICE IS WRONG", "DONT AGREE", "DON'T AGREE",
        "DONT ACCEPT", "DON'T ACCEPT", "RETURNED", "DAMAGED GOODS",
    ]):
        return extracted_inv, "DISPUTE", fast_amount

    # Firm-level payment command. Never route this through Gemini.
    if re.search(r'\bPAY\s+ALL\b', msg_upper) or any(x in msg_upper for x in [
        "PAY ALL MY INVOICES", "PAY ALL PENDING INVOICES", "CLEAR ALL MY BILLS",
        "SETTLE ALL MY INVOICES", "PAY EVERYTHING",
    ]):
        return extracted_inv, "MULTI_INVOICE_PAYMENT", None

    # Future payment commitments must beat the generic PAY + amount rule.
    future_signal = bool(re.search(
        r'\b(?:TOMORROW|TONIGHT|NEXT\s+WEEK|NEXT\s+MONTH|'
        r'NEXT\s+(?:MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY)|'
        r'(?:MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY)|'
        r'MONTH\s+END|BY\s+NEXT|LATER|IN\s+\d+\s+(?:DAY|DAYS|WEEK|WEEKS))\b',
        msg_upper
    ))
    future_commitment = bool(re.search(
        r'\b(?:WILL\s+PAY|I\s*\'?LL\s+PAY|CAN\s+PAY|I\s+WILL\s+PAY|DE\s+DUNGA|DE\s+DUNGI|PAY\s+KAR\s+DUNGA|PAY\s+KAR\s+DUNGI)\b',
        msg_upper
    ))

    if future_signal and (fast_amount is not None or future_commitment or re.search(r'\b(PAY|GIVE|CLEAR|SETTLE|DE)\b', msg_upper)):
        return extracted_inv, "PROMISE_TO_PAY", fast_amount

    # Explicit invoice-copy request.
    if any(phrase in msg_upper for phrase in [
        "BILL COPY", "INVOICE COPY", "SEND BILL", "SEND INVOICE", "NEED BILL",
        "NEED INVOICE", "GET BILL", "GET INVOICE", "COPY OF BILL", "COPY OF INVOICE",
        "BILL PLEASE", "INVOICE PLEASE",
    ]):
        return extracted_inv, "REQUEST_INVOICE", None

    # Explicit full-payment language. Keep this after future-signal detection.
    full_signal = any(
        re.search(rf'\b{re.escape(phrase)}\b', msg_upper)
        for phrase in ["FULL", "ALL", "CLEAR", "SETTLE", "EVERYTHING", "COMPLETE"]
    )
    if full_signal and re.search(r'\b(PAY|CLEAR|SETTLE)\b', msg_upper):
        return extracted_inv, "FULL_PAYMENT", None

    # Immediate partial payment: only the explicit PAY verb, never PAYMENT.
    if re.search(r'\bPAY\b', msg_upper) and fast_amount is not None:
        return extracted_inv, "PARTIAL_PAYMENT", fast_amount

    if re.search(r'\bPAY\b', msg_upper):
        return extracted_inv, "FULL_PAYMENT", None

    return extracted_inv, None, None

def _recent_chat_history_for_llm(db: Session, client: Client, limit: int = 6) -> list:
    """
    Same merge-across-invoices logic as /api/firms/{id}/chat-history, trimmed
    to the last `limit` turns — this is what gives the LLM classifier
    conversation memory (see agent_engine._extract_intent_via_llm). Kept
    small deliberately: enough for "wait, not that one" style back-references
    without bloating every single classification call.
    """
    merged = []
    for inv in client.invoices:
        merged.extend(build_invoice_messages(db, inv))
    merged.sort(key=lambda m: m["timestamp"])
    return merged[-limit:]


@app.post("/api/simulate-message", response_model=SimulateMessageResponse)
def simulate_message(req: SimulateMessageRequest, db: Session = Depends(get_db), _auth: None = Depends(require_auth)):
    # 1. Resolve client & target invoice
    client = None
    target_invoice = None

    if req.invoice_id:
        target_invoice = db.query(Invoice).filter(Invoice.id == req.invoice_id).first()
        if not target_invoice:
            raise HTTPException(status_code=404, detail="Invoice not found")
        client = target_invoice.client
    elif req.client_id:
        client = db.query(Client).filter(Client.id == req.client_id).first()
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
    else:
        raise HTTPException(status_code=400, detail="Must provide either invoice_id or client_id")

    # True only when the caller gave ONLY an invoice_id, with no client_id at
    # all — a genuinely narrow, single-invoice request. The chat UI always
    # sends client_id together with invoice_id (the latter is just a "last
    # invoice this conversation touched" convenience hint for continuity,
    # not an instruction to scope every future message to that one invoice
    # forever) — so req.invoice_id being truthy must NOT by itself suppress
    # multi-invoice disambiguation below, or a client with several open
    # invoices would get silently locked onto whichever one happened to be
    # touched last as soon as ANY message set that hint.
    invoice_scope_explicit = req.invoice_id is not None and req.client_id is None

    clean_msg = req.message.strip()
    # Persist every inbound buyer message so the collection agent has a real
    # conversation trail rather than relying only on audit logs.
    record_communication(db, client, target_invoice, "INBOUND", clean_msg, channel="SIMULATOR")
    refresh_case(db, client)

    # 2. Check for deterministic 'PAY ALL' quick command
    if clean_msg.upper() == "PAY ALL":
        return _handle_pay_all(db, client, clean_msg)

    # 2a. Deterministic greeting interceptor — bypasses the LLM entirely.
    GREETING_WORDS = {
        "HI", "HELLO", "HEY", "HII", "HIII", "NAMASTE", "PRANAM",
        "GOOD MORNING", "GOOD EVENING", "GOOD AFTERNOON", "START", "MENU",
    }
    if clean_msg.strip().upper() in GREETING_WORDS:
        greeting_text = build_greeting_reply(client)
        anchor_invoice = target_invoice or resolve_target_invoice(db, client)
        options = [
            ChatOption(id="OPT_STATEMENT", title="📄 Statement", payload="STATEMENT"),
            ChatOption(id="OPT_PAY", title="💳 Pay", payload="PAY"),
            ChatOption(id="OPT_INVOICE", title="📑 Invoice Copy", payload="INVOICE"),
        ]
        exec_payload = {
            "validation_reason": "Buyer sent a greeting; handled deterministically without an LLM call.",
            "business_name": client.business_name,
            "greeting_text": greeting_text,
        }
        log = write_audit_log(
            db=db, invoice_id=anchor_invoice.id if anchor_invoice else None,
            incoming_message=clean_msg,
            detected_intent="GREETING", action_taken="GREETING_RESPONSE",
            status=AuditStatus.SUCCESS, execution_payload=exec_payload,
        )
        return SimulateMessageResponse(
            invoice_id=anchor_invoice.id if anchor_invoice else 0,
            incoming_message=clean_msg,
            detected_intent="GREETING",
            confidence=1.0,
            llm_reasoning="Deterministic greeting interceptor — no LLM call made.",
            action_taken="GREETING_RESPONSE",
            allowed=True,
            guardrail_reason=exec_payload["validation_reason"],
            final_amount=None,
            payment_link_url=None,
            invoice_pdf_url=None,
            updated_invoice_status=anchor_invoice.status.value if anchor_invoice else "PENDING",
            audit_log_id=log.id,
            bot_reply_text=greeting_text,
            options=options,
        )

    # 2a-2. Deterministic conversational-filler interceptor. Real buyers
    # send a lot more than clean payment/invoice commands — thanks, bare
    # "yes"/"no", "bye", "who is this", "help" — none of which carry an
    # actionable intent for the LLM to extract, so they were all landing on
    # UNKNOWN and returning the same "I'm not quite sure I understood that"
    # line regardless of what was actually said. Each category below gets
    # its own short, natural reply instead.
    filler_clean = re.sub(r"[^A-Z\s?]", "", clean_msg.strip().upper()).strip()

    FILLER_REPLIES = {
        # acknowledgement — buyer is closing out / thanking us, nothing more needed
        "ack": {
            "OK", "OKAY", "OKK", "OKIE", "ALRIGHT", "ALRIGHTY", "COOL", "GOT IT",
            "NOTED", "GREAT", "PERFECT", "FINE", "THANKS", "THANK YOU", "TY", "THX",
            "OK THANKS", "OKAY THANKS", "SURE THANKS", "GREAT THANKS", "THANKS A LOT",
        },
        # bare affirmative — buyer wants to do *something* but didn't say what
        "affirmative": {
            "YES", "YEAH", "YEA", "YEP", "YUP", "SURE", "YES PLEASE",
            "OF COURSE", "OFCOURSE", "GO AHEAD",
        },
        # bare negative — declining or not ready right now
        "negative": {"NO", "NOPE", "NAH", "NOT NOW", "NO THANKS", "NOT REALLY"},
        # signing off
        "bye": {"BYE", "GOODBYE", "BYE BYE", "SEE YOU", "TALK LATER", "TTYL"},
        # asking who/what this is
        "identity": {"WHO IS THIS", "WHO ARE YOU", "WHAT IS THIS", "IS THIS A BOT", "ARE YOU A BOT"},
        # explicitly asking what the bot can do
        "help": {"HELP", "MENU", "OPTIONS", "WHAT CAN YOU DO", "?", "WHAT", "HUH"},
    }

    filler_category = next((cat for cat, words in FILLER_REPLIES.items() if filler_clean in words), None)

    if filler_category:
        anchor_invoice = target_invoice or resolve_target_invoice(db, client)
        capabilities = (
            f"tell me how much & when you can pay, ask for a copy of your bill/invoice, "
            f"or let me know if something's wrong with it"
        )
        reply_by_category = {
            "ack": "You're welcome! Let me know if there's anything else I can help with.",
            "affirmative": f"Great — what would you like to do? You can {capabilities}.",
            "negative": f"No problem! I'm here whenever you're ready — {capabilities}.",
            "bye": "Take care! Reach out anytime you're ready to sort out the balance.",
            "identity": (
                f"I'm the automated collections assistant for {client.business_name}. "
                f"I can help you {capabilities}."
            ),
            "help": f"Here's what I can help with — you can {capabilities}.",
        }
        filler_text = reply_by_category[filler_category]
        exec_payload = {
            "validation_reason": f"Buyer sent a conversational filler message ({filler_category}); handled deterministically without an LLM call.",
        }
        log = write_audit_log(
            db=db, invoice_id=anchor_invoice.id if anchor_invoice else None,
            incoming_message=clean_msg,
            detected_intent="CONVERSATIONAL_FILLER", action_taken="NO_ACTION",
            status=AuditStatus.SUCCESS, execution_payload=exec_payload,
        )
        return SimulateMessageResponse(
            invoice_id=anchor_invoice.id if anchor_invoice else 0,
            incoming_message=clean_msg,
            detected_intent="CONVERSATIONAL_FILLER",
            confidence=1.0,
            llm_reasoning=f"Deterministic conversational-filler interceptor ({filler_category}) — no LLM call needed.",
            action_taken="NO_ACTION",
            allowed=True,
            guardrail_reason=exec_payload["validation_reason"],
            final_amount=None,
            payment_link_url=None,
            invoice_pdf_url=None,
            updated_invoice_status=anchor_invoice.status.value if anchor_invoice else "PENDING",
            audit_log_id=log.id,
            bot_reply_text=filler_text,
        )

    # 2b. Deterministic "PAGE N" navigation for a paginated invoice breakdown
    page_match = re.match(r"^PAGE\s+(\d+)$", clean_msg.strip().upper())
    if page_match:
        open_invoices = [inv for inv in client.invoices if inv.status != InvoiceStatus.PAID]
        if open_invoices:
            decision = process_buyer_message(clean_msg, target_invoice or resolve_target_invoice(db, client))
            return _handle_multi_invoice_breakdown(
                db, client, clean_msg, open_invoices, decision, page=int(page_match.group(1))
            )

    # 2c. Deterministic "STATEMENT" command — full itemized breakdown across
    # all open invoices.
    STATEMENT_WORDS = {"STATEMENT", "FULL STATEMENT", "BALANCE", "MY BALANCE", "MY BILLS", "MY INVOICES"}
    if clean_msg.strip().upper() in STATEMENT_WORDS:
        open_invoices = [inv for inv in client.invoices if inv.status != InvoiceStatus.PAID]

        if not open_invoices:
            anchor_invoice = resolve_target_invoice(db, client)
            bot_reply_text = f"Hi {client.name}, good news — you have no outstanding balance with us right now!"
            log = write_audit_log(
                db=db, invoice_id=anchor_invoice.id if anchor_invoice else None,
                incoming_message=clean_msg,
                detected_intent="STATEMENT_REQUEST", action_taken="NO_ACTION",
                status=AuditStatus.SUCCESS,
                execution_payload={"validation_reason": "No open invoices for this client."},
            )
            return SimulateMessageResponse(
                invoice_id=anchor_invoice.id if anchor_invoice else 0,
                incoming_message=clean_msg,
                detected_intent="STATEMENT_REQUEST",
                confidence=1.0,
                llm_reasoning="Deterministic STATEMENT command — no LLM call needed.",
                action_taken="NO_ACTION",
                allowed=True,
                guardrail_reason="No open invoices for this client.",
                final_amount=None,
                payment_link_url=None,
                invoice_pdf_url=None,
                updated_invoice_status=anchor_invoice.status.value if anchor_invoice else "PAID",
                audit_log_id=log.id,
                bot_reply_text=bot_reply_text,
            )

        synthetic_extracted = ExtractedIntent(
            intent=IntentType.REQUEST_INVOICE,
            confidence=1.0,
            reasoning="Deterministic STATEMENT command — no LLM call needed.",
        )
        synthetic_validation = ValidationResult(
            allowed=True,
            action=ActionType.RESEND_INVOICE,
            reason="Buyer requested a full statement via deterministic STATEMENT command.",
        )
        decision = AgentDecision(raw_message=clean_msg, extracted=synthetic_extracted, validation=synthetic_validation)
        return _handle_multi_invoice_breakdown(db, client, clean_msg, open_invoices, decision, page=1)

    # 2c-2. Deterministic generic invoice/bill-copy request. Previously a
    # bare ask like "give me invoice" only got routed correctly when the
    # exact wording matched one of a handful of hardcoded phrases in
    # pre_process_user_intent, OR when the LLM's own extraction of
    # invoice_number/extracted_amount happened to come back exactly null —
    # so near-identical requests could inconsistently return one invoice's
    # PDF directly instead of asking which bill (when more than one is
    # open), and an explicit "...in pdf" for multiple invoices was getting
    # ignored in favour of repeating the itemized picker. This now decides
    # all of that deterministically, without an LLM round-trip.
    msg_upper = clean_msg.strip().upper()
    has_specific_invoice_ref = bool(re.search(r'\b(INV-?\d{4}|\d{4})\b', msg_upper))
    mentions_amount = bool(re.search(r'₹\s*\d+|\bRS\.?\s*\d+|\b\d{5,}\b', msg_upper))
    mentions_pay = bool(re.search(r'\bPAY\b', msg_upper))
    mentions_dispute = any(w in msg_upper for w in ["WRONG", "DISPUTE", "DAMAGED", "INCORRECT", "RETURN"])
    mentions_promise = any(w in msg_upper for w in [
        "REMIND", "TOMORROW", "NEXT WEEK", "MONDAY", "TUESDAY", "WEDNESDAY",
        "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY", "MONTH END", "NEXT MONTH",
    ])
    mentions_invoice_noun = bool(re.search(r'\bINVOICE(S)?\b|\bBILL(S)?\b|\bRECEIPT(S)?\b', msg_upper))

    if (
        mentions_invoice_noun
        and not has_specific_invoice_ref
        and not mentions_amount
        and not mentions_pay
        and not mentions_dispute
        and not mentions_promise
    ):
        if invoice_scope_explicit:
            # Per-invoice chat thread — always resend THAT invoice's PDF
            # directly, regardless of how many other invoices this client has.
            single_inv = target_invoice
            invoice_pdf_url = f"/api/invoices/{single_inv.id}/pdf"
            exec_payload = {
                "validation_reason": "Buyer asked for an invoice copy on a specific invoice's chat thread.",
                "business_name": client.business_name,
                "invoice_pdf_url": invoice_pdf_url,
            }
            log = write_audit_log(
                db=db, invoice_id=single_inv.id, incoming_message=clean_msg,
                detected_intent="REQUEST_INVOICE", action_taken="RESEND_INVOICE",
                status=AuditStatus.SUCCESS, execution_payload=exec_payload,
            )
            return SimulateMessageResponse(
                invoice_id=single_inv.id, incoming_message=clean_msg, detected_intent="REQUEST_INVOICE",
                confidence=1.0, llm_reasoning="Deterministic invoice-request interceptor — no LLM call needed.",
                action_taken="RESEND_INVOICE", allowed=True, guardrail_reason=exec_payload["validation_reason"],
                final_amount=None, payment_link_url=None, invoice_pdf_url=invoice_pdf_url,
                updated_invoice_status=single_inv.status.value, audit_log_id=log.id,
                bot_reply_text="Sure, here's a copy of your invoice — tap below to download the PDF.",
            )

        open_invoices = [inv for inv in client.invoices if inv.status != InvoiceStatus.PAID]
        wants_all = any(w in msg_upper for w in ["BOTH", "ALL"])
        wants_pdf = "PDF" in msg_upper

        if not open_invoices:
            anchor_invoice = resolve_target_invoice(db, client)
            bot_reply_text = f"Hi {client.name}, good news — you have no outstanding balance with us right now!"
            log = write_audit_log(
                db=db, invoice_id=anchor_invoice.id if anchor_invoice else None,
                incoming_message=clean_msg,
                detected_intent="REQUEST_INVOICE", action_taken="NO_ACTION",
                status=AuditStatus.SUCCESS,
                execution_payload={"validation_reason": "No open invoices for this client."},
            )
            return SimulateMessageResponse(
                invoice_id=anchor_invoice.id if anchor_invoice else 0,
                incoming_message=clean_msg, detected_intent="REQUEST_INVOICE",
                confidence=1.0, llm_reasoning="Deterministic invoice-request interceptor — no LLM call needed.",
                action_taken="NO_ACTION", allowed=True,
                guardrail_reason="No open invoices for this client.",
                final_amount=None, payment_link_url=None, invoice_pdf_url=None,
                updated_invoice_status=anchor_invoice.status.value if anchor_invoice else "PAID",
                audit_log_id=log.id, bot_reply_text=bot_reply_text,
            )

        if len(open_invoices) > 1 and wants_all and wants_pdf:
            return _handle_all_invoices_pdf(db, client, clean_msg, open_invoices)

        if len(open_invoices) > 1:
            return _handle_invoice_choice_prompt(db, client, clean_msg, open_invoices)

        single_inv = open_invoices[0]
        invoice_pdf_url = f"/api/invoices/{single_inv.id}/pdf"
        exec_payload = {
            "validation_reason": "Buyer asked for an invoice copy; single open invoice resolved deterministically.",
            "business_name": client.business_name,
            "invoice_pdf_url": invoice_pdf_url,
        }
        log = write_audit_log(
            db=db, invoice_id=single_inv.id, incoming_message=clean_msg,
            detected_intent="REQUEST_INVOICE", action_taken="RESEND_INVOICE",
            status=AuditStatus.SUCCESS, execution_payload=exec_payload,
        )
        return SimulateMessageResponse(
            invoice_id=single_inv.id, incoming_message=clean_msg, detected_intent="REQUEST_INVOICE",
            confidence=1.0, llm_reasoning="Deterministic invoice-request interceptor — no LLM call needed.",
            action_taken="RESEND_INVOICE", allowed=True, guardrail_reason=exec_payload["validation_reason"],
            final_amount=None, payment_link_url=None, invoice_pdf_url=invoice_pdf_url,
            updated_invoice_status=single_inv.status.value, audit_log_id=log.id,
            bot_reply_text="Sure, here's a copy of your invoice — tap below to download the PDF.",
        )

    # 2d. Deterministic bare-invoice-number interceptor — e.g. tapping an
    # "INV-1006" ChatOption button, or typing just the invoice number with
    # no accompanying verb. There's genuinely no actionable intent in a bare
    # invoice number alone, so instead of letting the LLM guess (and
    # correctly return UNKNOWN, since there IS no intent to extract), show a
    # focused mini-menu scoped to that one invoice.
    bare_inv_match = re.match(r"^(INV-?\d{4})$", clean_msg.strip().upper())
    if bare_inv_match:
        raw = bare_inv_match.group(1)
        inv_number = raw if raw.startswith("INV-") else f"INV-{raw[3:]}"
        selected_inv = (
            db.query(Invoice)
            .filter(Invoice.client_id == client.id, Invoice.invoice_number == inv_number)
            .first()
        )
        if selected_inv:
            bot_reply_text = (
                f"Selected {selected_inv.invoice_number} — ₹{selected_inv.balance_amount:,.2f} due "
                f"{selected_inv.due_date.strftime('%d %b %Y')}.\n\n"
                f"Reply PAY to get a payment link for this invoice, INVOICE for a copy, "
                f"or tell me how much & when you can pay."
            )
            options = [
                ChatOption(id=f"OPT_PAY_{selected_inv.invoice_number}", title="💳 Pay Full Amount", payload=f"pay {selected_inv.invoice_number}"),
                ChatOption(id=f"OPT_PARTIAL_{selected_inv.invoice_number}", title="💰 Pay Partial Amount", payload="I can pay ₹"),
                ChatOption(id=f"OPT_PDF_{selected_inv.invoice_number}", title="📄 Get Copy", payload=f"invoice copy {selected_inv.invoice_number}"),
            ]
            exec_payload = {"validation_reason": "Buyer selected a specific invoice via bare invoice number.", "invoice_number": inv_number}
            log = write_audit_log(
                db=db, invoice_id=selected_inv.id,
                incoming_message=clean_msg,
                detected_intent="INVOICE_SELECTED", action_taken="SHOW_INVOICE_MENU",
                status=AuditStatus.SUCCESS, execution_payload=exec_payload,
            )
            return SimulateMessageResponse(
                invoice_id=selected_inv.id,
                incoming_message=clean_msg,
                detected_intent="INVOICE_SELECTED",
                confidence=1.0,
                llm_reasoning="Deterministic bare-invoice-number interceptor — no LLM call needed.",
                action_taken="SHOW_INVOICE_MENU",
                allowed=True,
                guardrail_reason=exec_payload["validation_reason"],
                final_amount=None,
                payment_link_url=None,
                invoice_pdf_url=None,
                updated_invoice_status=selected_inv.status.value,
                audit_log_id=log.id,
                bot_reply_text=bot_reply_text,
                options=options,
            )
    # 2e. Deterministic relative-reminder interceptor. Checked BEFORE the
    # PAY-keyword fast path in step 3 — otherwise a message like "send me a
    # payment link after 2 mins" gets its "after 2 mins" silently discarded,
    # because pre_process_user_intent's substring check on "PAY" matches
    # inside the word "PAYMENT" and forces an immediate CREATE_PAYMENT_LINK
    # whenever the LLM's confidence dips below the override threshold.
    # Also fixes "remind me in N mins" being LLM-dependent (confirmed flaky).
    relative_minutes = _extract_relative_minutes(clean_msg)
    if relative_minutes is not None and relative_minutes > 0:
        anchor_invoice = target_invoice or resolve_target_invoice(db, client)
        reminder_dt = datetime.utcnow() + timedelta(minutes=relative_minutes)
        exec_payload = {
            "validation_reason": f"Buyer asked for a reminder/delayed action in {relative_minutes} minute(s); handled deterministically without an LLM call.",
            "reminder_scheduled_for": reminder_dt.isoformat(),
            "scheduler": "COLLECTION_ACTION",
        }
        log = write_audit_log(
            db=db, invoice_id=anchor_invoice.id if anchor_invoice else None,
            incoming_message=clean_msg,
            detected_intent="PROMISE_TO_PAY", action_taken="SCHEDULE_REMINDER",
            status=AuditStatus.SUCCESS, execution_payload=exec_payload,
        )
        if anchor_invoice:
            create_promise(db, client, anchor_invoice, anchor_invoice.balance_amount, reminder_dt, clean_msg)
            schedule_action(db, client, "FOLLOW_UP_PROMISE", reminder_dt, anchor_invoice,
                            reason="Buyer requested a future payment/reminder",
                            dedupe_key=f"promise-followup:{anchor_invoice.id}:{int(reminder_dt.timestamp())}")
        db.commit()
        bot_reply_text = f"No problem, I'll follow up with you in {relative_minutes} minute(s)."
        return SimulateMessageResponse(
            invoice_id=anchor_invoice.id if anchor_invoice else 0,
            incoming_message=clean_msg,
            detected_intent="PROMISE_TO_PAY",
            confidence=1.0,
            llm_reasoning="Deterministic relative-reminder interceptor — no LLM call needed.",
            action_taken="SCHEDULE_REMINDER",
            allowed=True,
            guardrail_reason=exec_payload["validation_reason"],
            final_amount=None,
            payment_link_url=None,
            invoice_pdf_url=None,
            updated_invoice_status=anchor_invoice.status.value if anchor_invoice else "PENDING",
            audit_log_id=log.id,
            bot_reply_text=bot_reply_text,
        )
    # 3. Fast pre-processing for explicit invoice & key phrases
    fast_inv_num, fast_intent, fast_amount = pre_process_user_intent(clean_msg)
    fallback_invoice = target_invoice or resolve_target_invoice(db, client)

    if fast_inv_num:
        specific_inv = (
            db.query(Invoice)
            .filter(Invoice.client_id == client.id, Invoice.invoice_number == fast_inv_num)
            .first()
        )
        if specific_inv:
            target_invoice = specific_inv

    # 3a. Deterministic "PAY <INVOICE>" fast path
    if fast_inv_num and fast_intent == "FULL_PAYMENT" and target_invoice:
        return _create_full_payment_link_response(db, client, target_invoice, clean_msg)


    # 3b. Deterministic bare "PAY" fast path — e.g. tapping the greeting's
    # "💳 Pay" button. Rather than assuming full payment, this now asks
    # Full vs Partial explicitly — same pattern as the bare-invoice-number
    # interceptor (step 2d) — since a buyer tapping generic "PAY" hasn't
    # actually committed to paying the whole balance.
    if fast_intent == "FULL_PAYMENT" and not fast_inv_num:
        open_invoices = [inv for inv in client.invoices if inv.status != InvoiceStatus.PAID]

        if not open_invoices:
            anchor_invoice = resolve_target_invoice(db, client)
            log = write_audit_log(
                db=db, invoice_id=anchor_invoice.id if anchor_invoice else None,
                incoming_message=clean_msg,
                detected_intent="FULL_PAYMENT", action_taken="NO_ACTION",
                status=AuditStatus.BLOCKED,
                execution_payload={"validation_reason": "No open invoices for this client."},
            )
            return SimulateMessageResponse(
                invoice_id=anchor_invoice.id if anchor_invoice else 0,
                incoming_message=clean_msg, detected_intent="FULL_PAYMENT",
                confidence=1.0, llm_reasoning="Deterministic bare-PAY fast path — no LLM call needed.",
                action_taken="NO_ACTION", allowed=False,
                guardrail_reason="No open invoices for this client.",
                final_amount=None, payment_link_url=None, invoice_pdf_url=None,
                updated_invoice_status=anchor_invoice.status.value if anchor_invoice else "PAID",
                audit_log_id=log.id,
                bot_reply_text=f"Hi {client.name}, good news — you have no outstanding balance with us right now!",
            )

        if len(open_invoices) == 1:
            inv = open_invoices[0]
            bot_reply_text = (
                f"You have ₹{inv.balance_amount:,.2f} outstanding on {inv.invoice_number}. "
                f"Would you like to pay the full amount, or a partial amount?"
            )
            options = [
                ChatOption(id=f"OPT_PAYFULL_{inv.invoice_number}", title="💳 Pay Full Amount", payload=f"pay {inv.invoice_number}"),
                ChatOption(id=f"OPT_PARTIAL_{inv.invoice_number}", title="💰 Pay Partial Amount", payload="I can pay ₹"),
            ]
            exec_payload = {"validation_reason": "Buyer tapped/typed generic PAY; asked to choose full vs partial before generating any link."}
            log = write_audit_log(
                db=db, invoice_id=inv.id, incoming_message=clean_msg,
                detected_intent="FULL_PAYMENT", action_taken="ASK_PAYMENT_TYPE",
                status=AuditStatus.SUCCESS, execution_payload=exec_payload,
            )
            return SimulateMessageResponse(
                invoice_id=inv.id, incoming_message=clean_msg, detected_intent="FULL_PAYMENT",
                confidence=1.0, llm_reasoning="Deterministic bare-PAY fast path — no LLM call needed.",
                action_taken="ASK_PAYMENT_TYPE", allowed=True,
                guardrail_reason=exec_payload["validation_reason"],
                final_amount=None, payment_link_url=None, invoice_pdf_url=None,
                updated_invoice_status=inv.status.value, audit_log_id=log.id,
                bot_reply_text=bot_reply_text, options=options,
            )

        # More than one open invoice — ask which one first (unchanged from before)
        synthetic_extracted = ExtractedIntent(
            intent=IntentType.FULL_PAYMENT, confidence=1.0,
            reasoning="Deterministic bare-PAY command — no LLM call needed.",
        )
        synthetic_validation = ValidationResult(
            allowed=True, action=ActionType.CREATE_PAYMENT_LINK,
            reason="Buyer requested to pay via deterministic PAY command; multiple invoices open.",
        )
        decision = AgentDecision(raw_message=clean_msg, extracted=synthetic_extracted, validation=synthetic_validation)
        return _handle_multi_invoice_breakdown(db, client, clean_msg, open_invoices, decision, page=1)
    # 4. PRIMARY ROUTER: deterministic high-confidence intents first.
    # These paths intentionally DO NOT call Gemini. The LLM is reserved for
    # genuinely ambiguous/natural-language messages. This keeps common
    # collection actions fast, reduces API usage, and prevents a transient
    # Gemini outage from changing a safe future promise into a payment action.
    deterministic_intents = {
        "FULL_PAYMENT", "PARTIAL_PAYMENT", "PROMISE_TO_PAY",
        "ALREADY_PAID", "PAYMENT_PENDING", "PAYMENT_FAILED",
        "PAYMENT_PROOF", "PAYMENT_PLAN_REQUEST", "DISPUTE",
        "MULTI_INVOICE_PAYMENT", "REQUEST_INVOICE",
    }

    if fast_intent in deterministic_intents:
        open_invoices = [inv for inv in client.invoices if inv.status != InvoiceStatus.PAID]

        # Firm-level PAY ALL / multi-invoice requests never need Gemini.
        if fast_intent == "MULTI_INVOICE_PAYMENT":
            return _handle_pay_all(db, client, clean_msg)

        # If the buyer did not identify an invoice and several invoices are
        # open, amount alone is NOT enough to choose one. Ask explicitly.
        if (
            len(open_invoices) > 1
            and not invoice_scope_explicit
            and not fast_inv_num
            and fast_intent in {
                "FULL_PAYMENT", "PARTIAL_PAYMENT", "PROMISE_TO_PAY",
                "REQUEST_INVOICE", "ALREADY_PAID", "PAYMENT_PENDING",
                "PAYMENT_FAILED", "PAYMENT_PROOF", "DISPUTE",
            }
        ):
            synthetic_extracted = ExtractedIntent(
                intent=IntentType[fast_intent],
                extracted_amount=fast_amount,
                confidence=1.0,
                reasoning="Deterministic high-confidence intent; invoice scope is ambiguous across multiple open invoices.",
            )
            if fast_intent == "PROMISE_TO_PAY":
                promise_date = _extract_fast_promise_date(clean_msg)
                if promise_date:
                    synthetic_extracted.promise_date_iso = promise_date
            synthetic_validation = ValidationResult(
                allowed=True,
                action=ActionType.NO_ACTION,
                reason="Multiple open invoices and buyer did not specify which invoice; ask before taking a financial action.",
            )
            decision = AgentDecision(
                raw_message=clean_msg,
                extracted=synthetic_extracted,
                validation=synthetic_validation,
            )
            return _handle_multi_invoice_breakdown(db, client, clean_msg, open_invoices, decision, page=1)

        active_inv = target_invoice or fallback_invoice

        # No invoice exists for an invoice-scoped action: do not guess.
        if fast_intent in {
            "FULL_PAYMENT", "PARTIAL_PAYMENT", "PROMISE_TO_PAY",
            "ALREADY_PAID", "PAYMENT_PENDING", "PAYMENT_FAILED",
            "PAYMENT_PROOF", "PAYMENT_PLAN_REQUEST", "DISPUTE",
        } and active_inv is None:
            synthetic_extracted = ExtractedIntent(
                intent=IntentType[fast_intent], extracted_amount=fast_amount,
                confidence=1.0,
                reasoning="Deterministic intent detected, but no invoice is available to safely scope the action.",
            )
            synthetic_validation = ValidationResult(
                allowed=False, action=ActionType.NO_ACTION,
                reason="No invoice could be resolved safely; no financial action taken.",
            )
            decision = AgentDecision(raw_message=clean_msg, extracted=synthetic_extracted, validation=synthetic_validation)
        else:
            extracted = ExtractedIntent(
                intent=IntentType[fast_intent],
                extracted_amount=fast_amount,
                invoice_number=fast_inv_num,
                confidence=1.0,
                reasoning="Deterministic high-confidence intent — no LLM call needed.",
            )

            if fast_intent == "PROMISE_TO_PAY":
                promise_date = _extract_fast_promise_date(clean_msg)
                if promise_date:
                    extracted.promise_date_iso = promise_date
            elif fast_intent == "PAYMENT_PROOF":
                ref_match = re.search(r"\b(?:UTR|REF(?:ERENCE)?|TXN|TRANSACTION(?:\s+ID)?)\s*[:#-]?\s*([A-Z0-9_-]{5,})\b", clean_msg, re.I)
                if ref_match:
                    extracted.payment_reference = ref_match.group(1)

            if fast_intent == "FULL_PAYMENT":
                validation = ValidationResult(
                    allowed=True, action=ActionType.CREATE_PAYMENT_LINK,
                    final_amount=active_inv.balance_amount,
                    reason=f"Deterministic PAY command for {active_inv.invoice_number}; full current balance.",
                )
            else:
                validation = validate_intent_action(extracted, active_inv)

            decision = AgentDecision(raw_message=clean_msg, extracted=extracted, validation=validation)
    else:
        # 5. LLM fallback — only genuinely ambiguous/unstructured language
        # reaches Gemini. Preserve recent multi-invoice chat context here.
        recent_history = _recent_chat_history_for_llm(db, client)
        decision = process_buyer_message(clean_msg, target_invoice or fallback_invoice, history=recent_history)

    extracted = decision.extracted
    val = decision.validation

    # Normalize date-only promise reminders before writing the audit entry so
    # the persisted schedule and the actual CollectionAction use the same
    # timestamp.
    if val.allowed and val.action == ActionType.SCHEDULE_REMINDER and val.reminder_date:
        val.reminder_date = _normalize_collection_datetime(val.reminder_date)

    # Firm-level actions must be resolved against the whole account, not one
    # arbitrary invoice. This handles natural-language variants of PAY ALL.
    # extracted.extracted_amount is passed through deliberately — a buyer
    # saying "pay ₹70,000 against all pending invoices" means ₹70,000, NOT
    # the full outstanding balance; _handle_pay_all only falls back to the
    # full total when no specific amount was actually stated.
    if extracted.intent == IntentType.MULTI_INVOICE_PAYMENT:
        return _handle_pay_all(db, client, clean_msg, specified_amount=extracted.extracted_amount)

    if extracted.intent == IntentType.CLARIFICATION_REQUIRED:
        open_invoices = [inv for inv in client.invoices if inv.status != InvoiceStatus.PAID]
        if len(open_invoices) > 1:
            decision.validation.action = ActionType.NO_ACTION
            decision.validation.allowed = True
            decision.validation.reason = "Message is ambiguous across multiple open invoices; ask the buyer to select one."
            return _handle_multi_invoice_breakdown(db, client, clean_msg, open_invoices, decision)

    # 6. Specific invoice override from LLM extraction (if fast-regex didn't already catch it)
    if extracted.invoice_number and not fast_inv_num:
        specific_inv = (
            db.query(Invoice)
            .filter(Invoice.client_id == client.id, Invoice.invoice_number == extracted.invoice_number)
            .first()
        )
        if specific_inv:
            target_invoice = specific_inv
            decision = process_buyer_message(clean_msg, target_invoice, history=recent_history)
            extracted = decision.extracted
            val = decision.validation

    # 7. Multi-Invoice Disambiguation
    open_invoices = [inv for inv in client.invoices if inv.status != InvoiceStatus.PAID]
    if (
        not invoice_scope_explicit
        and not extracted.invoice_number
        and not fast_inv_num
        and len(open_invoices) > 1
        and extracted.intent in (
            IntentType.REQUEST_INVOICE,
            IntentType.PARTIAL_PAYMENT,
            IntentType.FULL_PAYMENT,
            IntentType.PROMISE_TO_PAY,
        )
    ):
        return _handle_multi_invoice_breakdown(db, client, clean_msg, open_invoices, decision)

    if not target_invoice:
        target_invoice = fallback_invoice

    # 8. Execute validated action & prepare response
    payment_link_url = None
    invoice_pdf_url = None
    audit_status = None  # set explicitly per-branch below; overrides the generic val.allowed calculation at the bottom

    if val.allowed and val.action == ActionType.CREATE_PAYMENT_LINK:
        try:
            link_res, link_fields = get_or_create_payment_link(db, target_invoice, client, val.final_amount)
            payment_link_url = link_res["short_url"]
            exec_payload = {"validation_reason": val.reason, **link_fields}
            audit_status = AuditStatus.SUCCESS
        except (RazorpayClientError,TypeError) as exc:
            exec_payload = {"validation_reason": val.reason, "error": str(exc)}
            audit_status = AuditStatus.FAILED
    elif val.action == ActionType.GREETING_RESPONSE:
        exec_payload = {
            "validation_reason": val.reason,
            "business_name": client.business_name,
            "greeting_text": build_greeting_reply(client),
        }
    elif val.allowed and val.action == ActionType.RESEND_INVOICE:
        invoice_pdf_url = f"/api/invoices/{target_invoice.id}/pdf"
        exec_payload = {
            "validation_reason": val.reason,
            "business_name": client.business_name,
            "simulated_action": f"Invoice copy for {target_invoice.invoice_number} resent to {client.phone_number}",
            "invoice_pdf_url": invoice_pdf_url,
        }
    elif val.allowed and val.action == ActionType.SCHEDULE_REMINDER:
        exec_payload = {
            "validation_reason": val.reason,
            "reminder_scheduled_for": val.reminder_date.isoformat() if val.reminder_date else None,
            "scheduler": "COLLECTION_ACTION",
        }
    elif val.allowed and val.action == ActionType.FLAG_DISPUTE:
        target_invoice.status = InvoiceStatus.DISPUTED
        case = get_or_create_case(db, client)
        case.status = CollectionCaseStatus.DISPUTED
        case.escalation_reason = extracted.dispute_reason or "Buyer disputed the invoice"
        exec_payload = {"dispute_reason": extracted.dispute_reason, "validation_reason": val.reason, "escalated": True}
    elif val.allowed and val.action == ActionType.VERIFY_PAYMENT:
        exec_payload = {
            "validation_reason": val.reason,
            "payment_reference": extracted.payment_reference,
            "verification_required": True,
        }
        # Never mutate the ledger from a buyer claim alone. Put it in the
        # wholesaler's exception queue instead.
        case = get_or_create_case(db, client)
        case.status = CollectionCaseStatus.ESCALATED
        case.escalation_reason = "Buyer claims a payment was already made; provider verification required."
    elif extracted.intent == IntentType.PAYMENT_FAILED and target_invoice and target_invoice.status != InvoiceStatus.DISPUTED and target_invoice.balance_amount > 0:
        try:
            link_res, link_fields = get_or_create_payment_link(db, target_invoice, client, target_invoice.balance_amount)
            payment_link_url = link_res["short_url"]
            val.action = ActionType.CREATE_PAYMENT_LINK
            exec_payload = {"validation_reason": "Previous payment failed; issued/reused a fresh payment link for the current balance.", **link_fields}
            audit_status = AuditStatus.SUCCESS
        except (RazorpayClientError, TypeError) as exc:
            exec_payload = {"validation_reason": "Buyer reported a failed payment; no ledger mutation performed.", "error": str(exc)}
            audit_status = AuditStatus.FAILED
    elif val.allowed and val.action == ActionType.ESCALATE:
        exec_payload = {"validation_reason": val.reason, "escalated": True}
        case = get_or_create_case(db, client)
        case.status = CollectionCaseStatus.ESCALATED
        case.escalation_reason = extracted.reasoning or "Buyer requested a payment plan / additional time."
    elif val.allowed and val.action == ActionType.ASK_CLARIFICATION:
        clarification_text = (
            f"I want to make sure I get this right — could you tell me the exact amount you can "
            f"pay now, and when you'll pay the rest of the ₹{target_invoice.balance_amount:,.2f} "
            f"balance on {target_invoice.invoice_number}?"
            if target_invoice else
            "I want to make sure I get this right — could you tell me the exact amount and date you can pay?"
        )
        exec_payload = {"validation_reason": val.reason, "clarification_text": clarification_text}
    else:
        exec_payload = {"validation_reason": val.reason, "business_name": client.business_name, "llm_reasoning": extracted.reasoning}

    status_enum = audit_status if audit_status is not None else (AuditStatus.SUCCESS if val.allowed else AuditStatus.BLOCKED)
    log = write_audit_log(
        db=db,
        invoice_id=target_invoice.id if target_invoice else None,
        incoming_message=clean_msg,
        detected_intent=extracted.intent.value,
        action_taken=val.action.value,
        status=status_enum,
        execution_payload=exec_payload,
    )

    if val.allowed and val.action == ActionType.SCHEDULE_REMINDER and val.reminder_date and target_invoice:
        # Date-only promises should follow up at a sensible collection time
        # (10:00 IST) rather than midnight. Relative reminders keep their exact
        # requested delay.
        reminder_dt = _normalize_collection_datetime(val.reminder_date)
        val.reminder_date = reminder_dt
        amount = extracted.extracted_amount or target_invoice.balance_amount
        create_promise(db, client, target_invoice, amount, reminder_dt, clean_msg)
        schedule_action(db, client, "FOLLOW_UP_PROMISE", reminder_dt, target_invoice,
                        reason="Buyer promised to pay",
                        dedupe_key=f"promise-followup:{target_invoice.id}:{int(reminder_dt.timestamp())}")
    elif extracted.intent == IntentType.PARTIAL_PAYMENT and extracted.remaining_promise_amount and extracted.promise_date_iso and target_invoice:
        try:
            future_dt = datetime.strptime(extracted.promise_date_iso, "%Y-%m-%d")
            create_promise(db, client, target_invoice, extracted.remaining_promise_amount, future_dt, clean_msg)
            schedule_action(db, client, "FOLLOW_UP_PROMISE", future_dt, target_invoice,
                            reason="Buyer committed to pay the remaining balance later",
                            dedupe_key=f"promise-followup:{target_invoice.id}:{int(future_dt.timestamp())}")
        except ValueError:
            pass
    elif extracted.intent == IntentType.PAYMENT_PLAN_REQUEST and target_invoice:
        case = get_or_create_case(db, client)
        case.status = CollectionCaseStatus.ESCALATED
    reconcile_promises(db, client)
    refresh_case(db, client)
    bot_reply_text = synthesize_bot_reply_text(extracted.intent.value, val.action.value, status_enum.value, exec_payload)
    record_communication(db, client, target_invoice, "OUTBOUND", bot_reply_text, channel="SIMULATOR")
    db.commit()

    return SimulateMessageResponse(
        invoice_id=target_invoice.id if target_invoice else 0,
        incoming_message=clean_msg,
        detected_intent=extracted.intent.value,
        confidence=extracted.confidence,
        llm_reasoning=extracted.reasoning,
        action_taken=val.action.value,
        allowed=val.allowed,
        guardrail_reason=val.reason,
        final_amount=val.final_amount,
        payment_link_url=payment_link_url,
        invoice_pdf_url=invoice_pdf_url,
        updated_invoice_status=target_invoice.status.value if target_invoice else None,
        audit_log_id=log.id,
        bot_reply_text=bot_reply_text,
    )
# --------------------------------------------------------------------------
# RAZORPAY WEBHOOK — kept for when ngrok/production webhook IS configured
# --------------------------------------------------------------------------

@app.post("/webhook/razorpay")
async def razorpay_webhook(request: Request, db: Session = Depends(get_db)):
    raw_body = await request.body()
    body_str = raw_body.decode("utf-8")
    signature = request.headers.get("X-Razorpay-Signature", "")

    logger.info("Webhook received. Signature present: %s", bool(signature))

    try:
        is_valid = verify_webhook_signature(body_str, signature)
    except RazorpayClientError as exc:
        write_audit_log(
            db=db, invoice_id=None,
            incoming_message="Webhook received (signature verification errored)",
            detected_intent="WEBHOOK_RECONCILIATION", action_taken="VERIFY_SIGNATURE",
            status=AuditStatus.FAILED, execution_payload={"error": str(exc)},
        )
        raise HTTPException(status_code=500, detail=str(exc))

    if not is_valid:
        write_audit_log(
            db=db, invoice_id=None,
            incoming_message="Webhook received (INVALID signature)",
            detected_intent="WEBHOOK_RECONCILIATION", action_taken="VERIFY_SIGNATURE",
            status=AuditStatus.FAILED,
            execution_payload={"body_preview": body_str[:200]},
        )
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    try:
        data = json.loads(body_str)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Malformed JSON body")

    event = data.get("event", "")

    if event not in ("payment_link.paid", "payment.captured"):
        return JSONResponse({"status": "ignored", "event": event}, status_code=200)

    payload = data.get("payload", {})
    payment_entity = payload.get("payment", {}).get("entity", {})
    payment_link_entity = payload.get("payment_link", {}).get("entity", {})

    razorpay_payment_id = payment_entity.get("id")
    razorpay_payment_link_id = payment_link_entity.get("id") or payment_entity.get("invoice_id")
    amount_paise = payment_entity.get("amount")

    if not razorpay_payment_id or amount_paise is None:
        raise HTTPException(status_code=400, detail="Webhook payload missing payment id/amount")

    notes = payment_entity.get("notes", {}) or payment_link_entity.get("notes", {}) or {}
    invoice_number = notes.get("invoice_number")
    invoice_numbers_raw = notes.get("invoice_numbers", "")
    invoice_numbers = [x.strip() for x in str(invoice_numbers_raw).split(",") if x.strip()]
    if not invoice_number and invoice_numbers:
        invoice_number = invoice_numbers[0]
    # If Razorpay omitted notes from the webhook, fetch the authoritative
    # payment-link metadata before deciding the payment is unmatched.
    if not invoice_number and razorpay_payment_link_id:
        try:
            link_data = fetch_payment_link(razorpay_payment_link_id)
            link_notes = link_data.get("notes", {}) or {}
            invoice_number = link_notes.get("invoice_number")
            invoice_numbers = [x.strip() for x in str(link_notes.get("invoice_numbers", "")).split(",") if x.strip()]
        except RazorpayClientError:
            pass
    invoice = db.query(Invoice).filter(Invoice.invoice_number == invoice_number).first() if invoice_number else None

    if not invoice:
        write_audit_log(
            db=db, invoice_id=None,
            incoming_message=f"Webhook event: {event}",
            detected_intent="WEBHOOK_RECONCILIATION", action_taken="RECONCILE_PAYMENT",
            status=AuditStatus.FAILED,
            execution_payload={"reason": "unmatched invoice_number", "razorpay_payment_id": razorpay_payment_id},
        )
        return JSONResponse({"status": "unmatched_invoice"}, status_code=200)

    amount_paid_rupees = paise_to_rupees(amount_paise)
    allocation_invoices = [invoice]
    if invoice_numbers:
        allocation_invoices = (
            db.query(Invoice)
            .filter(Invoice.client_id == invoice.client_id, Invoice.invoice_number.in_(invoice_numbers))
            .all()
        ) or [invoice]
    result = apply_captured_payment(
        db, invoice, razorpay_payment_id, razorpay_payment_link_id, amount_paid_rupees,
        allocation_invoices=allocation_invoices,
    )
    applied = result["applied"]

    write_audit_log(
        db=db, invoice_id=invoice.id,
        incoming_message=f"Webhook event: {event}",
        detected_intent="PAYMENT_RECEIVED", action_taken="RECONCILE_PAYMENT",
        status=AuditStatus.SUCCESS if applied else AuditStatus.BLOCKED,
        execution_payload={
            "razorpay_payment_id": razorpay_payment_id,
            "amount_paid": amount_paid_rupees,
            "already_processed": not applied,
            "new_paid_amount": invoice.paid_amount,
            "new_status": invoice.status.value,
            "invoice_number": invoice.invoice_number,  # NEW (Feature 5)
            "invoice_pdf_url": result["invoice_pdf_url"],  # NEW (Feature 5)
        },
    )
    return JSONResponse({"status": "reconciled" if applied else "already_processed"}, status_code=200)


# --------------------------------------------------------------------------
# MANUAL SYNC — still available for on-demand testing
# --------------------------------------------------------------------------

@app.post("/api/sync-payment-link/{invoice_id}")
def sync_payment_link(invoice_id: int, db: Session = Depends(get_db), _auth: None = Depends(require_auth)):
    invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")

    link_logs = (
        db.query(AuditLog)
        .filter(
            AuditLog.invoice_id == invoice_id,
            AuditLog.action_taken == "CREATE_PAYMENT_LINK",
            AuditLog.status == AuditStatus.SUCCESS,
        )
        .order_by(AuditLog.timestamp.desc())
        .all()
    )
    if not link_logs:
        raise HTTPException(status_code=404, detail="No payment link found for this invoice")

    synced = []
    for log in link_logs:
        exec_payload = json.loads(log.execution_payload) if log.execution_payload else {}
        link_id = exec_payload.get("razorpay_payment_link_id")
        if not link_id:
            continue
        try:
            link_data = fetch_payment_link(link_id)
        except RazorpayClientError:
            continue
        if link_data.get("status") != "paid":
            continue
        for p in link_data.get("payments", []):
            if p.get("status") != "captured":
                continue
            amount_paid_rupees = paise_to_rupees(p["amount"])
            result = apply_captured_payment(db, invoice, p["payment_id"], link_id, amount_paid_rupees)
            if result["applied"]:
                synced.append({
                    "payment_id": p["payment_id"],
                    "amount": amount_paid_rupees,
                    "invoice_pdf_url": result["invoice_pdf_url"],  # NEW (Feature 5)
                })
                # NEW (Feature 5): manual sync is a reconciliation path too —
                # give it the same PAYMENT_RECEIVED audit log the webhook and
                # poller write, so a manually-synced clearance also produces
                # a chat bubble + receipt download, not just a silent JSON
                # response that never reaches the WhatsApp simulator.
                write_audit_log(
                    db=db, invoice_id=invoice.id,
                    incoming_message="Manual sync detected a captured payment",
                    detected_intent="PAYMENT_RECEIVED", action_taken="RECONCILE_PAYMENT",
                    status=AuditStatus.SUCCESS,
                    execution_payload={
                        "razorpay_payment_id": p["payment_id"],
                        "payment_link_id": link_id,
                        "amount_paid": amount_paid_rupees,
                        "new_paid_amount": invoice.paid_amount,
                        "new_status": invoice.status.value,
                        "invoice_number": invoice.invoice_number,
                        "invoice_pdf_url": result["invoice_pdf_url"],
                    },
                )

    return {
        "synced_payments": synced,
        "invoice_status": invoice.status.value,
        "paid_amount": invoice.paid_amount,
        "balance_amount": invoice.balance_amount,
    }
@app.post("/api/invoices/{invoice_id}/check-now")
def check_payment_now(invoice_id: int, db: Session = Depends(get_db), _auth: None = Depends(require_auth)):
    """
    Manual, on-demand version of what the poller does automatically —
    lets the frontend force an immediate Razorpay check instead of waiting
    up to POLL_INTERVAL_SECONDS, then returns the latest state so the chat
    UI can just re-render like it would after any other message.
    """
    invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
    result = sync_payment_link(invoice_id, db)
    return {
        "invoice_id": invoice_id,
        "invoice_number": invoice.invoice_number,
        **result,
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "LedgerRecover AI"}


class OnboardInvoiceRequest(BaseModel):
    # If client_id is set, the invoice is attached to that existing Firm and
    # client_name/business_name/phone_number are ignored. Otherwise a new
    # Firm is created from those fields (or an existing one reused if the
    # phone number already matches, preserved for backward compatibility).
    client_id: Optional[int] = None
    client_name: Optional[str] = None
    business_name: Optional[str] = None
    phone_number: Optional[str] = None
    total_amount: float
    upfront_paid: Optional[float] = 0.0
    # NEW (Feature 3): explicit due_date (YYYY-MM-DD) takes priority over
    # due_days when both are present. due_days kept for backward compat
    # with any existing callers that only send the relative form.
    due_date: Optional[str] = None
    due_days: Optional[int] = 30


@app.post("/api/onboard-client")
def onboard_client_and_invoice(payload: OnboardInvoiceRequest, db: Session = Depends(get_db), _auth: None = Depends(require_auth)):
    """
    Wholesaler-side onboarding: creates (or reuses) a client, creates an
    invoice with an optional upfront down-payment, and dispatches the
    initial outbound WhatsApp message that kicks off the buyer's chat thread.
    """
    if payload.total_amount <= 0:
        raise HTTPException(status_code=400, detail="Total amount must be positive")

    client = None
    if payload.client_id is not None:
        client = db.query(Client).filter(Client.id == payload.client_id).first()
        if not client:
            raise HTTPException(status_code=404, detail=f"Firm id {payload.client_id} not found")
    else:
        if not (payload.client_name and payload.business_name and payload.phone_number):
            raise HTTPException(
                status_code=400,
                detail="client_name, business_name and phone_number are required when creating a new firm",
            )
        client = db.query(Client).filter(Client.phone_number == payload.phone_number).first()
        if not client:
            client = Client(
                name=payload.client_name,
                business_name=payload.business_name,
                phone_number=payload.phone_number,
            )
            db.add(client)
            db.commit()
            db.refresh(client)

    upfront = round(payload.upfront_paid or 0.0, 2)
    upfront = min(upfront, payload.total_amount)  # never let upfront exceed the bill itself

    count = db.query(Invoice).count() + 1
    inv_number = f"INV-{1000 + count}"
    now = datetime.utcnow()

    # NEW (Feature 3): parse explicit due_date if the frontend sent one,
    # falling back to the relative due_days otherwise.
    if payload.due_date:
        try:
            due_dt = datetime.strptime(payload.due_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="due_date must be in YYYY-MM-DD format")
    else:
        due_dt = now + timedelta(days=payload.due_days or 30)

    invoice = Invoice(
        invoice_number=inv_number,
        client_id=client.id,
        total_amount=payload.total_amount,
        paid_amount=upfront,
        balance_amount=payload.total_amount,  # placeholder; recompute_invoice_ledger sets the real value below
        status=InvoiceStatus.PENDING,
        due_date=due_dt,
        created_at=now,
    )
    recompute_invoice_ledger(invoice)  # single source of truth for balance/status derivation
    db.add(invoice)
    db.commit()
    db.refresh(invoice)

    if upfront > 0:
        db.add(PaymentRecord(
            invoice_id=invoice.id,
            razorpay_payment_id=f"counter_cash_{invoice.id}_{uuid.uuid4().hex[:6]}",
            razorpay_payment_link_id="COUNTER_PAYMENT",
            amount_paid=upfront,
            paid_at=now,
        ))
        db.commit()

    init_message = (
        f"Hi {client.name} ({client.business_name}), this is an automated message from your supplier. "
        f"Invoice {inv_number} for ₹{payload.total_amount:,.2f} has been generated. "
        + (f"We've recorded your upfront payment of ₹{upfront:,.2f}. " if upfront > 0 else "")
        + f"Remaining balance: ₹{invoice.balance_amount:,.2f}, due by {invoice.due_date.strftime('%d %b %Y')}. "
        f"Reply here anytime to arrange payment, request your bill copy, or flag any issue."
    )

    write_audit_log(
        db=db,
        invoice_id=invoice.id,
        incoming_message="SYSTEM: Client & Invoice Onboarded",
        detected_intent="OUTBOUND_COLLECTION_INIT",
        action_taken="DISPATCH_WHATSAPP_BOT",
        status=AuditStatus.SUCCESS,
        execution_payload={
            "recipient_phone": client.phone_number,
            "dispatched_text": init_message,
            "invoice_number": inv_number,
            "balance_remaining": invoice.balance_amount,
        },
    )

    return {
        "status": "success",
        "client_id": client.id,
        "invoice_id": invoice.id,
        "invoice_number": inv_number,
        "balance_amount": invoice.balance_amount,
        "paid_amount": invoice.paid_amount,
        "invoice_status": invoice.status.value,
    }


@app.get("/api/chat-history/{invoice_id}")
def chat_history(invoice_id: int, db: Session = Depends(get_db), _auth: None = Depends(require_auth)):
    """
    Reconstructs the full buyer<->bot conversation thread for an invoice from
    the audit trail, so switching invoices (or reloading the page) shows the
    real persisted history rather than a JS-memory-only chat.
    """
    invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")

    messages = build_invoice_messages(db, invoice)
    return {"invoice_id": invoice_id, "messages": messages}


# --------------------------------------------------------------------------
# FIRM-CENTRIC ENDPOINTS (Features 1 & 2)
# --------------------------------------------------------------------------

def _firm_status(invoices: List[Invoice]) -> str:
    """Worst-case rollup status for a firm across all its invoices."""
    statuses = {inv.status for inv in invoices}
    if InvoiceStatus.DISPUTED in statuses:
        return InvoiceStatus.DISPUTED.value
    if InvoiceStatus.PENDING in statuses:
        return InvoiceStatus.PENDING.value
    if InvoiceStatus.PARTIALLY_PAID in statuses:
        return InvoiceStatus.PARTIALLY_PAID.value
    return InvoiceStatus.PAID.value


def _compute_firm_stats(db: Session, client: Client) -> dict:
    """
    Deterministic payment-history stats for one firm, computed straight from
    the ledger — the ONLY numbers that ever reach the LLM narrative (see
    agent_engine.generate_firm_insight_narrative). risk_level is a plain
    rule-based classification here, not an LLM judgment call — kept
    explainable and consistent, same discipline as the rest of this app.
    Shared by the per-firm Insights panel and the Collections Copilot.
    """
    invoices = client.invoices
    open_invoices = [inv for inv in invoices if inv.status != InvoiceStatus.PAID]
    paid_invoices = [inv for inv in invoices if inv.status == InvoiceStatus.PAID]
    disputed_invoices = [inv for inv in invoices if inv.status == InvoiceStatus.DISPUTED]

    now = datetime.utcnow()
    total_outstanding = round(sum(inv.balance_amount for inv in open_invoices), 2)
    overdue_invoices = [inv for inv in open_invoices if inv.due_date < now]
    max_days_overdue = max([(now - inv.due_date).days for inv in overdue_invoices], default=0)

    # On-time rate across historically PAID invoices: compare the LAST
    # payment's paid_at against that invoice's due_date.
    on_time_count = 0
    late_count = 0
    late_day_totals = []
    for inv in paid_invoices:
        if not inv.payments:
            continue
        last_payment = max(inv.payments, key=lambda p: p.paid_at)
        if last_payment.paid_at.date() <= inv.due_date.date():
            on_time_count += 1
        else:
            late_count += 1
            late_day_totals.append((last_payment.paid_at.date() - inv.due_date.date()).days)

    total_scored = on_time_count + late_count
    on_time_rate = round((on_time_count / total_scored) * 100, 1) if total_scored else None
    avg_days_late = round(sum(late_day_totals) / len(late_day_totals), 1) if late_day_totals else None

    invoice_ids = [inv.id for inv in invoices]
    dispute_log_count = 0
    broken_promises = 0
    last_log = None
    if invoice_ids:
        dispute_log_count = (
            db.query(AuditLog)
            .filter(AuditLog.invoice_id.in_(invoice_ids), AuditLog.detected_intent == "DISPUTE")
            .count()
        )
        # Broken-promise proxy: a PROMISE_TO_PAY was logged against an
        # invoice that is STILL open and overdue right now.
        promised_invoice_ids = {
            row[0] for row in db.query(AuditLog.invoice_id)
            .filter(AuditLog.invoice_id.in_(invoice_ids), AuditLog.detected_intent == "PROMISE_TO_PAY")
            .all()
        }
        broken_promises = sum(1 for inv in overdue_invoices if inv.id in promised_invoice_ids)
        last_log = (
            db.query(AuditLog)
            .filter(AuditLog.invoice_id.in_(invoice_ids))
            .order_by(AuditLog.timestamp.desc())
            .first()
        )

    if disputed_invoices or max_days_overdue > 14 or broken_promises >= 2:
        risk_level = "HIGH"
    elif overdue_invoices or broken_promises >= 1 or (on_time_rate is not None and on_time_rate < 50):
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "client_id": client.id,
        "business_name": client.business_name,
        "contact_name": client.name,
        "phone_number": client.phone_number,
        "total_outstanding": total_outstanding,
        "open_invoice_count": len(open_invoices),
        "overdue_invoice_count": len(overdue_invoices),
        "max_days_overdue": max_days_overdue,
        "disputed_invoice_count": len(disputed_invoices),
        "paid_invoice_count_historical": len(paid_invoices),
        "on_time_payment_rate_pct": on_time_rate,
        "avg_days_late_when_late": avg_days_late,
        "broken_promises": broken_promises,
        "dispute_count": dispute_log_count,
        "last_contacted": last_log.timestamp.isoformat() if last_log else None,
        "risk_level": risk_level,
    }


def _compute_firm_analysis(db: Session, client: Client) -> dict:
    """
    Full time-series analysis for one firm — powers the "📊 Analysis Report"
    modal (as opposed to _compute_firm_stats, which powers the quick "🧠 AI
    Insights" blurb). Builds on the same deterministic stats, plus
    chronological trend data for charting: running balance over time,
    monthly collections, invoice status breakdown, and a days-late trend
    across paid invoices (to see whether this firm is improving or
    worsening, not just where they stand today). Every number here is
    computed directly from the ledger — the LLM narrative is layered on top
    of the same stats dict _compute_firm_stats already produces, never
    touching the chart data itself.
    """
    invoices = sorted(client.invoices, key=lambda inv: inv.created_at)

    # Running balance over time: each invoice created adds its total_amount,
    # each payment subtracts what was paid — walked in strict chronological
    # order across ALL of this firm's invoices together.
    events = []
    for inv in invoices:
        events.append((inv.created_at, inv.total_amount, "invoice_created", inv.invoice_number))
        for p in inv.payments:
            events.append((p.paid_at, -p.amount_paid, "payment", inv.invoice_number))
    events.sort(key=lambda e: e[0])
    running = 0.0
    balance_history = []
    for dt, delta, kind, inv_num in events:
        running = round(running + delta, 2)
        balance_history.append({
            "date": dt.isoformat(), "balance": running, "event": kind, "invoice_number": inv_num,
        })

    # Monthly collections: sum of payments per calendar month.
    monthly = {}
    for inv in invoices:
        for p in inv.payments:
            key = p.paid_at.strftime("%Y-%m")
            monthly[key] = round(monthly.get(key, 0.0) + p.amount_paid, 2)
    monthly_collections = [{"month": k, "amount": v} for k, v in sorted(monthly.items())]

    # Current status breakdown.
    status_breakdown = {"PENDING": 0, "PARTIALLY_PAID": 0, "PAID": 0, "DISPUTED": 0}
    for inv in invoices:
        status_breakdown[inv.status.value] += 1

    # Days-late trend across historically PAID invoices, in due-date order —
    # negative = paid early, 0 = on time, positive = days late. Charting
    # this in order is what actually shows a trend, not just a single rate.
    days_to_pay_trend = []
    for inv in invoices:
        if inv.status == InvoiceStatus.PAID and inv.payments:
            last_payment = max(inv.payments, key=lambda p: p.paid_at)
            days_late = (last_payment.paid_at.date() - inv.due_date.date()).days
            days_to_pay_trend.append({
                "invoice_number": inv.invoice_number,
                "due_date": inv.due_date.isoformat(),
                "days_late": days_late,
            })

    stats = _compute_firm_stats(db, client)
    narrative = generate_firm_insight_narrative(stats)

    return {
        **stats,
        "summary": narrative.summary,
        "recommended_action": narrative.recommended_action,
        "balance_history": balance_history,
        "monthly_collections": monthly_collections,
        "status_breakdown": status_breakdown,
        "days_to_pay_trend": days_to_pay_trend,
        "invoices": [_serialize_invoice(inv, db) for inv in invoices],
    }


def send_manual_reminder(db: Session, invoice: Invoice, is_overdue: bool) -> str:
    """
    Writes a wholesaler-triggered reminder straight into a buyer's chat
    thread — used by the Collections Copilot's SEND_REMINDERS action. Same
    rendering mechanism as the automatic overdue poller (see
    OVERDUE_AUTO_REMINDER), but under its own detected_intent since this one
    is human-initiated and NOT deduped (a wholesaler explicitly asking to
    remind again should actually send again, unlike the automatic poller
    which must never repeat itself).
    """
    client = invoice.client
    if is_overdue:
        days = max((datetime.utcnow() - invoice.due_date).days, 1)
        text = (
            f"⚠️ Hi {client.name}, following up — Invoice {invoice.invoice_number} "
            f"(₹{invoice.balance_amount:,.2f}) was due on {invoice.due_date.strftime('%d %b %Y')} "
            f"and is now {days} day(s) overdue. Please arrange payment at the earliest, or let us "
            f"know if there's an issue with this bill."
        )
    else:
        text = (
            f"👋 Hi {client.name}, just a friendly reminder — Invoice {invoice.invoice_number} "
            f"(₹{invoice.balance_amount:,.2f}) is due on {invoice.due_date.strftime('%d %b %Y')}. "
            f"Let us know if you'd like to arrange payment ahead of time!"
        )
    write_audit_log(
        db=db, invoice_id=invoice.id,
        incoming_message="WHOLESALER: manual reminder sent via Collections Copilot",
        detected_intent="MANUAL_COPILOT_REMINDER", action_taken="SEND_MANUAL_REMINDER",
        status=AuditStatus.SUCCESS,
        execution_payload={"reminder_text": text, "is_overdue": is_overdue},
    )
    record_communication(db, client, invoice, "OUTBOUND", text, channel="SIMULATOR")
    return text


def _serialize_invoice(inv: Invoice, db: Session) -> dict:
    last_log = (
        db.query(AuditLog)
        .filter(AuditLog.invoice_id == inv.id)
        .order_by(AuditLog.timestamp.desc())
        .first()
    )
    is_overdue = inv.status != InvoiceStatus.PAID and inv.due_date < datetime.utcnow()
    return {
        "id": inv.id,
        "invoice_number": inv.invoice_number,
        "client_name": inv.client.name,
        "business_name": inv.client.business_name,
        "phone_number": inv.client.phone_number,
        "total_amount": inv.total_amount,
        "paid_amount": inv.paid_amount,
        "balance_amount": inv.balance_amount,
        "status": inv.status.value,
        "due_date": inv.due_date.isoformat(),
        "is_overdue": is_overdue,
        "days_overdue": max((datetime.utcnow() - inv.due_date).days, 0) if is_overdue else 0,
        "last_contacted": last_log.timestamp.isoformat() if last_log else None,
    }


@app.get("/api/firms")
def list_firms(view: str = "active", db: Session = Depends(get_db), _auth: None = Depends(require_auth)):
    """
    Firm-level rollup of the ledger: one row per Client, with total
    outstanding balance / invoice count / worst-case status across
    that firm's invoices MATCHING THE REQUESTED VIEW. This is what the
    grouped 'Firm View' renders.

    view="active" (default) includes only invoices with balance_amount > 0.
    view="settled" includes only PAID invoices, kept for audit/GST history
    without cluttering the active view. A firm with zero invoices matching
    the view is omitted entirely from that response.

    When view="settled", the response also includes a top-level
    total_settled figure: total_amount summed across every settled invoice
    in the system (not just the firms returned), so the frontend can show
    "total historical revenue collected" even though each firm's own
    total_outstanding is always 0 in this view (by definition — settled
    invoices have no balance left).
    """
    if view not in ("active", "settled"):
        raise HTTPException(status_code=400, detail="view must be 'active' or 'settled'")

    clients = db.query(Client).all()
    result = []
    total_settled = 0.0

    for client in clients:
        if view == "active":
            invoices = [inv for inv in client.invoices if inv.balance_amount > 0]
        else:
            invoices = [inv for inv in client.invoices if inv.status == InvoiceStatus.PAID]
            total_settled += sum(inv.total_amount for inv in invoices)

        if not invoices:
            continue

        last_log = (
            db.query(AuditLog)
            .join(Invoice, AuditLog.invoice_id == Invoice.id)
            .filter(Invoice.client_id == client.id)
            .order_by(AuditLog.timestamp.desc())
            .first()
        )
        result.append({
            "client_id": client.id,
            "name": client.name,
            "business_name": client.business_name,
            "phone_number": client.phone_number,
            "total_outstanding": round(sum(inv.balance_amount for inv in invoices), 2),
            "invoice_count": len(invoices),
            "status": _firm_status(invoices),
            "is_overdue": any(
                inv.status != InvoiceStatus.PAID and inv.due_date < datetime.utcnow()
                for inv in invoices
            ),
            "last_contacted": last_log.timestamp.isoformat() if last_log else None,
        })
    result.sort(key=lambda f: f["total_outstanding"], reverse=True)

    response = {"firms": result, "view": view}
    if view == "settled":
        response["total_settled"] = round(total_settled, 2)
    return response


@app.get("/api/firms/{client_id}/invoices")
def firm_invoices(client_id: int, db: Session = Depends(get_db), _auth: None = Depends(require_auth)):
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Firm not found")
    invoices = (
        db.query(Invoice)
        .filter(Invoice.client_id == client_id)
        .order_by(Invoice.due_date.asc())
        .all()
    )
    return {
        "client_id": client_id,
        "business_name": client.business_name,
        "invoices": [_serialize_invoice(inv, db) for inv in invoices],
    }


@app.get("/api/firms/{client_id}/account-summary")
def firm_account_summary(client_id: int, db: Session = Depends(get_db), _auth: None = Depends(require_auth)):
    """
    Deterministic (non-LLM) consolidated summary of a firm's outstanding
    invoices — used to render the interactive Account Summary bubble when
    a Firm is first selected in the WhatsApp simulator.
    """
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Firm not found")

    unpaid = [inv for inv in client.invoices if inv.status != InvoiceStatus.PAID]
    unpaid.sort(key=lambda inv: inv.due_date)
    total_outstanding = round(sum(inv.balance_amount for inv in unpaid), 2)

    return {
        "client_id": client_id,
        "name": client.name,
        "business_name": client.business_name,
        "phone_number": client.phone_number,
        "total_outstanding": total_outstanding,
        "unpaid_invoices": [
            {
                "invoice_id": inv.id,
                "invoice_number": inv.invoice_number,
                "balance_amount": inv.balance_amount,
                "due_date": inv.due_date.isoformat(),
                "status": inv.status.value,
            }
            for inv in unpaid
        ],
    }


@app.get("/api/firms/{client_id}/insights")
def firm_insights(client_id: int, db: Session = Depends(get_db), _auth: None = Depends(require_auth)):
    """
    AI Insights panel for one firm: deterministic payment-history stats
    (see _compute_firm_stats) plus a short LLM-written narrative summary
    and recommended action, generated ONLY from those already-computed
    numbers — never invented independently by the model.
    """
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Firm not found")

    stats = _compute_firm_stats(db, client)
    narrative = generate_firm_insight_narrative(stats)
    return {**stats, "summary": narrative.summary, "recommended_action": narrative.recommended_action}


@app.get("/api/firms/{client_id}/analysis")
def firm_analysis(client_id: int, db: Session = Depends(get_db), _auth: None = Depends(require_auth)):
    """
    Full Analysis Report for one firm: everything in Insights, plus
    chronological trend data for charting (running balance over time,
    monthly collections, status breakdown, days-late trend) and the
    complete invoice history. See _compute_firm_analysis.
    """
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Firm not found")
    return _compute_firm_analysis(db, client)


def _fuzzy_word_hit(words: List[str], targets: List[str], threshold: float = 0.8) -> bool:
    return any(SequenceMatcher(None, w, t).ratio() >= threshold for w in words for t in targets)


def _fuzzy_best_client_match(message: str, clients: List[Client]) -> tuple:
    """Fuzzy-matches a firm name anywhere in free text against real client
    business names/contact names. Used both as a deterministic fallback and
    to actually resolve firm_name_query from the LLM, since matching against
    REAL names beats trusting the LLM to spell a name back correctly."""
    msg_lower = message.lower()
    best, best_score = None, 0.0
    for c in clients:
        score = max(
            SequenceMatcher(None, msg_lower, c.business_name.lower()).ratio(),
            SequenceMatcher(None, msg_lower, c.name.lower()).ratio(),
        )
        if c.business_name.lower() in msg_lower or c.name.lower() in msg_lower:
            score = max(score, 0.9)
        if score > best_score:
            best, best_score = c, score
    return best, best_score


def _deterministic_copilot_fallback(message: str, clients: List[Client]) -> Optional[dict]:
    """
    Typo-tolerant keyword safety net, tried ONLY when the LLM classifier
    itself comes back UNKNOWN (or errors out) — exactly the same "fast
    fallback override" philosophy already used for the buyer-facing agent
    (see pre_process_user_intent in this file). This is what makes the
    Copilot keep working even on typos ("needd a remainder") or if the LLM
    call has a bad day, instead of a single point of failure deciding
    whether the whole feature works.
    """
    words = re.findall(r"[a-z']+", message.lower())
    if not words:
        return None

    def has(*targets):
        return _fuzzy_word_hit(words, list(targets))

    remind_word = has("remind", "reminder", "reminders", "nudge", "followup")
    send_word = has("send", "go", "ahead", "confirm", "please")
    unpaid_word = has("unpaid", "outstanding", "owe", "owes", "owing", "due", "pending", "cleared", "clear", "paid")
    negation = has("havent", "hasnt", "havnt", "didnt", "not", "no")
    who_word = has("who", "which", "what", "any", "list", "show")
    pdf_word = has("pdf", "document", "download", "copy", "receipt")

    if remind_word and send_word:
        return {"intent": "SEND_REMINDERS"}
    if remind_word:
        return {"intent": "LIST_NEEDS_REMINDER"}
    if unpaid_word and (negation or who_word):
        return {"intent": "LIST_UNPAID"}
    if has("total", "portfolio", "overview", "summary", "disputed", "how many"):
        return {"intent": "GENERAL_QUESTION"}

    best_client, best_score = _fuzzy_best_client_match(message, clients)
    if best_client and best_score >= 0.5:
        return {
            "intent": "GENERATE_PDF" if pdf_word else "FIRM_LOOKUP",
            "firm_name_query": best_client.business_name,
        }

    return None


@app.post("/api/copilot/message", response_model=CopilotMessageResponse)
def copilot_message(req: CopilotMessageRequest, db: Session = Depends(get_db), _auth: None = Depends(require_auth)):
    """
    Wholesaler-facing "Collections Copilot" — a portfolio-level chat
    assistant, separate from the buyer-facing simulator. The LLM
    (classify_copilot_message) only classifies intent + scoping parameters;
    every number, firm name, and message it reports back is computed here
    directly from the database, never authored by the model. This mirrors
    the same discipline used throughout the buyer-facing agent.

    IMPORTANT: this endpoint NEVER sends a reminder on the same turn it was
    first requested, no matter how the message was classified. Any request
    that would message a firm always comes back as a proposal + a
    `pending_action` first; the actual send only happens once the wholesaler
    replies with a bare confirmation on the NEXT turn (checked deterministically
    below, before the LLM is even called — a one-word "yes" carries no intent
    the classifier could reliably resolve on its own).
    """
    CONFIRM_WORDS = {
        "YES", "YEAH", "YEA", "YEP", "YUP", "SURE", "GO AHEAD", "DO IT",
        "SEND THEM", "SEND", "CONFIRM", "PLEASE DO", "OK SEND", "SEND IT",
    }
    CANCEL_WORDS = {"NO", "NOPE", "DONT", "CANCEL", "NEVERMIND", "NEVER MIND", "STOP", "SKIP"}
    msg_clean = re.sub(r"[^A-Z\s]", "", req.message.strip().upper()).strip()

    if req.pending_action and req.pending_action.get("type") == "SEND_REMINDERS":
        if msg_clean in CONFIRM_WORDS:
            invoice_ids = req.pending_action.get("invoice_ids", [])
            invoices = db.query(Invoice).filter(Invoice.id.in_(invoice_ids)).all()
            # Re-check freshness: an invoice may have been paid/disputed since
            # the proposal was made a message ago — never send against stale state.
            still_valid = [inv for inv in invoices if inv.status not in (InvoiceStatus.PAID, InvoiceStatus.DISPUTED)]
            if not still_valid:
                return CopilotMessageResponse(
                    reply_text="That batch is no longer valid (those invoices have since been paid or disputed) — ask me again for a fresh list.",
                    intent="SEND_REMINDERS",
                )
            sent_lines = []
            for inv in still_valid:
                is_overdue = inv.due_date < datetime.utcnow()
                send_manual_reminder(db, inv, is_overdue=is_overdue)
                sent_lines.append(f"• {inv.client.business_name} ({inv.invoice_number})")
            reply = f"Sent reminders to {len(still_valid)} firm(s):\n\n" + "\n".join(sent_lines)
            return CopilotMessageResponse(reply_text=reply, intent="SEND_REMINDERS", reminders_sent=len(still_valid))
        if msg_clean in CANCEL_WORDS:
            return CopilotMessageResponse(reply_text="Okay, I won't send anything.", intent="SEND_REMINDERS")
        # Anything else: treat as a brand-new message and fall through to
        # normal classification below — the stale pending_action is simply
        # dropped (the response won't include a new one unless this message
        # itself produces a fresh proposal).

    decision = classify_copilot_message(req.message, history=req.history)
    clients = db.query(Client).all()
    now = datetime.utcnow()

    if decision.intent == CopilotIntent.UNKNOWN:
        fallback = _deterministic_copilot_fallback(req.message, clients)
        if fallback:
            decision = CopilotDecision(
                intent=CopilotIntent(fallback["intent"]),
                due_within_days=None,
                include_overdue=True,
                firm_name_query=fallback.get("firm_name_query"),
                reasoning="Deterministic fuzzy fallback (LLM returned UNKNOWN).",
            )

    if decision.intent == CopilotIntent.LIST_UNPAID:
        unpaid = [c for c in clients if any(inv.status != InvoiceStatus.PAID for inv in c.invoices)]
        if not unpaid:
            return CopilotMessageResponse(
                reply_text="Every firm is fully paid up right now — nothing outstanding. 🎉",
                intent=decision.intent.value,
            )
        ranked = sorted(
            unpaid,
            key=lambda c: -sum(inv.balance_amount for inv in c.invoices if inv.status != InvoiceStatus.PAID),
        )
        lines, firms_payload = [], []
        for c in ranked:
            outstanding = round(sum(inv.balance_amount for inv in c.invoices if inv.status != InvoiceStatus.PAID), 2)
            lines.append(f"• {c.business_name}: ₹{outstanding:,.2f} outstanding")
            firms_payload.append({"client_id": c.id, "business_name": c.business_name, "outstanding": outstanding})
        reply = f"{len(unpaid)} firm(s) haven't cleared their invoices:\n\n" + "\n".join(lines)
        return CopilotMessageResponse(reply_text=reply, intent=decision.intent.value, firms=firms_payload)

    if decision.intent in (CopilotIntent.LIST_NEEDS_REMINDER, CopilotIntent.SEND_REMINDERS):
        # Both intents land here and both ONLY propose — see the docstring
        # above for why an immediate send is never triggered from a single
        # classification, even when the wholesaler's own wording was a
        # direct instruction ("send reminders to everyone overdue").
        days = decision.due_within_days if decision.due_within_days is not None else 3
        cutoff = now + timedelta(days=days)
        matched = []  # list of (client, invoice, is_overdue)
        for c in clients:
            candidate_invoices = [inv for inv in c.invoices if inv.status not in (InvoiceStatus.PAID, InvoiceStatus.DISPUTED)]
            for inv in candidate_invoices:
                is_overdue = inv.due_date < now
                is_due_soon = now <= inv.due_date <= cutoff
                if (is_overdue and decision.include_overdue) or is_due_soon:
                    matched.append((c, inv, is_overdue))
                    break  # one qualifying invoice is enough to include this firm

        if not matched:
            scope = "overdue or due soon" if decision.include_overdue else "due soon"
            return CopilotMessageResponse(
                reply_text=f"No firms are currently {scope} (within {days} day(s)) — you're all caught up.",
                intent=decision.intent.value,
            )

        lines, firms_payload = [], []
        for c, inv, overdue in matched:
            status_label = "overdue" if overdue else f"due within {days} day(s)"
            last_reminder_log = (
                db.query(AuditLog)
                .filter(
                    AuditLog.invoice_id == inv.id,
                    AuditLog.detected_intent.in_(["MANUAL_COPILOT_REMINDER", "OVERDUE_AUTO_REMINDER"]),
                )
                .order_by(AuditLog.timestamp.desc())
                .first()
            )
            cooldown_note = ""
            hours_since = None
            if last_reminder_log:
                hours_since = round((now - last_reminder_log.timestamp).total_seconds() / 3600, 1)
                if hours_since < 24:
                    cooldown_note = f" ⏱ already reminded {hours_since}h ago"
            lines.append(
                f"• {c.business_name} — {status_label} (₹{inv.balance_amount:,.2f} on {inv.invoice_number}){cooldown_note}"
            )
            firms_payload.append({
                "client_id": c.id, "business_name": c.business_name,
                "invoice_number": inv.invoice_number, "overdue": overdue,
                "hours_since_last_reminder": hours_since,
            })
        recently_reminded_count = sum(1 for f in firms_payload if f["hours_since_last_reminder"] is not None and f["hours_since_last_reminder"] < 24)
        cooldown_summary = (
            f" ({recently_reminded_count} of them were already reminded in the last 24h — still included, but flagged above)"
            if recently_reminded_count else ""
        )
        reply = (
            f"{len(matched)} firm(s) could use a reminder{cooldown_summary}:\n\n" + "\n".join(lines) +
            "\n\nReply 'yes' to send reminders to all of them, or 'no' to skip."
        )
        pending_action = {"type": "SEND_REMINDERS", "invoice_ids": [inv.id for _, inv, _ in matched]}
        return CopilotMessageResponse(
            reply_text=reply, intent=decision.intent.value, firms=firms_payload, pending_action=pending_action,
        )

    if decision.intent == CopilotIntent.FIRM_LOOKUP:
        query = (decision.firm_name_query or "").strip()
        if not query:
            return CopilotMessageResponse(reply_text="Which firm would you like to know about?", intent=decision.intent.value)
        best, best_score = _fuzzy_best_client_match(query, clients)
        if not best or best_score < 0.4:
            return CopilotMessageResponse(
                reply_text=f"I couldn't find a firm matching \"{decision.firm_name_query}\".",
                intent=decision.intent.value,
            )
        stats = _compute_firm_stats(db, best)
        narrative = generate_firm_insight_narrative(stats)
        reply = f"{best.business_name} — Risk: {stats['risk_level']}\n\n{narrative.summary}\n\nRecommended: {narrative.recommended_action}"
        return CopilotMessageResponse(reply_text=reply, intent=decision.intent.value, firms=[stats])

    if decision.intent == CopilotIntent.GENERATE_PDF:
        query = (decision.firm_name_query or "").strip()
        if not query:
            return CopilotMessageResponse(reply_text="Which firm's invoice PDF would you like?", intent=decision.intent.value)
        best, best_score = _fuzzy_best_client_match(query, clients)
        if not best or best_score < 0.4:
            return CopilotMessageResponse(
                reply_text=f"I couldn't find a firm matching \"{decision.firm_name_query}\".",
                intent=decision.intent.value,
            )
        open_invoices = sorted([inv for inv in best.invoices if inv.balance_amount > 0], key=lambda i: i.due_date)
        if not open_invoices:
            # No open balance — fall back to their most recent invoice overall so "give me the pdf" still works.
            all_invoices = sorted(best.invoices, key=lambda i: i.due_date, reverse=True)
            if not all_invoices:
                return CopilotMessageResponse(reply_text=f"{best.business_name} has no invoices on file yet.", intent=decision.intent.value)
            pdf_url = f"/api/invoices/{all_invoices[0].id}/pdf"
            reply = f"{best.business_name} has no open balance. Here's their most recent invoice ({all_invoices[0].invoice_number}):"
        elif len(open_invoices) == 1:
            pdf_url = f"/api/invoices/{open_invoices[0].id}/pdf"
            reply = f"Here's {best.business_name}'s invoice ({open_invoices[0].invoice_number}):"
        else:
            pdf_url = f"/api/firms/{best.id}/invoices/combined-pdf"
            reply = f"{best.business_name} has {len(open_invoices)} open invoices — here's a combined PDF with all of them:"
        return CopilotMessageResponse(
            reply_text=reply, intent=decision.intent.value, pdf_url=pdf_url,
            firms=[{"client_id": best.id, "business_name": best.business_name}],
        )

    if decision.intent == CopilotIntent.GENERAL_QUESTION:
        total_outstanding = round(
            sum(inv.balance_amount for c in clients for inv in c.invoices if inv.status != InvoiceStatus.PAID), 2
        )
        disputed_count = sum(1 for c in clients for inv in c.invoices if inv.status == InvoiceStatus.DISPUTED)
        overdue_firm_count = sum(
            1 for c in clients
            if any(inv.status != InvoiceStatus.PAID and inv.due_date < now for inv in c.invoices)
        )
        reply = (
            f"Portfolio snapshot: ₹{total_outstanding:,.2f} total outstanding across {len(clients)} firm(s), "
            f"{overdue_firm_count} firm(s) with an overdue invoice, {disputed_count} disputed invoice(s)."
        )
        return CopilotMessageResponse(reply_text=reply, intent=decision.intent.value)

    return CopilotMessageResponse(
        reply_text=(
            "I can help with: which firms haven't paid, which firms need a reminder, sending "
            "reminders to firms due soon or overdue, looking up a specific firm by name, or "
            "generating an invoice PDF for a firm. Try asking one of those!"
        ),
        intent=decision.intent.value,
    )


@app.get("/api/firms/{client_id}/chat-history")
def firm_chat_history(client_id: int, db: Session = Depends(get_db), _auth: None = Depends(require_auth)):
    """
    Merges the chat threads of every invoice belonging to a firm into one
    chronological conversation, plus tells the frontend which invoice_id
    a new message from this firm should currently be resolved against
    (see resolve_target_invoice) — the same invoice a real message sent
    right now would land on.
    """
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Firm not found")

    invoices = db.query(Invoice).filter(Invoice.client_id == client_id).all()
    merged = []
    for inv in invoices:
        merged.extend(build_invoice_messages(db, inv))
    merged.sort(key=lambda m: m["timestamp"])

    target_invoice = resolve_target_invoice(db, client)

    return {
        "client_id": client_id,
        "target_invoice_id": target_invoice.id if target_invoice else None,
        "messages": merged,
    }


# --------------------------------------------------------------------------
# PDF INVOICE GENERATION (Feature 3)
# --------------------------------------------------------------------------

@app.get("/api/firms/{client_id}/invoices/combined-pdf")
def download_combined_invoices_pdf(client_id: int, db: Session = Depends(get_db), _auth: None = Depends(require_auth)):
    """
    One PDF containing every currently-open (balance > 0) invoice for this
    firm, one full invoice per page after a summary cover page — the
    "combined document" option offered when a buyer with multiple open
    invoices asks for a bill copy without naming a specific one.
    """
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Firm not found")

    open_invoices = sorted(
        [inv for inv in client.invoices if inv.balance_amount > 0],
        key=lambda inv: inv.due_date,
    )
    if not open_invoices:
        raise HTTPException(status_code=400, detail="This firm has no open invoices to combine.")

    pdf_bytes = generate_combined_invoices_pdf_bytes(open_invoices)
    filename = f"{client.business_name.replace(' ', '_')}-combined-invoices.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@app.get("/api/invoices/{invoice_id}/pdf")
def download_invoice_pdf(invoice_id: int, db: Session = Depends(get_db), _auth: None = Depends(require_auth)):
    invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")

    pdf_bytes = generate_invoice_pdf_bytes(invoice)
    filename = f"{invoice.invoice_number}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


# --------------------------------------------------------------------------
# ZERO-BALANCE CLEARANCE RECEIPT PDF (Feature 5)
# --------------------------------------------------------------------------

@app.get("/api/invoices/{invoice_id}/receipt-pdf")
def download_receipt_pdf(invoice_id: int, db: Session = Depends(get_db), _auth: None = Depends(require_auth)):
    invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
    if invoice.status != InvoiceStatus.PAID:
        raise HTTPException(
            status_code=400,
            detail="Receipt is only available once the invoice is fully PAID.",
        )

    pdf_bytes = generate_receipt_pdf_bytes(invoice)
    filename = f"{invoice.invoice_number}-receipt.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )



def extract_invoice_id_fast(user_message: str, active_invoices: list[str]) -> str | None:
    """
    Fast-track deterministic parser to extract invoice IDs from natural language phrases.
    """
    text = user_message.upper().strip()

    # Step 1: Direct Regex Extraction for INV-1006, INV1006, or standalone 1006
    # Captures patterns like "only for 1006", "pay inv-1006", "bill 1006"
    match = re.search(r'(?:INV[-_\s]?)?(\d{4,})', text)
    if match:
        extracted_digits = match.group(1)
        for inv in active_invoices:
            clean_inv_digits = re.sub(r'\D', '', inv)
            if extracted_digits == clean_inv_digits or inv.upper() in text:
                return inv

    # Step 2: Exact substring check across active invoices
    for inv in active_invoices:
        if inv.upper() in text:
            return inv

    # Step 3: Fuzzy Matching Fallback for typos (e.g., "invois 1006", "inv 106")
    for word in text.split():
        for inv in active_invoices:
            if SequenceMatcher(None, word.upper(), inv.upper()).ratio() > 0.8:
                return inv

    return None

import re