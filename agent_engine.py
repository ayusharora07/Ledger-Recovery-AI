import os
import json
import enum
import logging
from datetime import datetime, timedelta
from typing import Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

logger = logging.getLogger("agent_engine")
logging.basicConfig(level=logging.INFO)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY missing. Copy .env.example to .env and fill in your Gemini API key."
    )

client = genai.Client(api_key=GEMINI_API_KEY)
_MODEL_NAME = "gemini-3.6-flash"


class IntentType(str, enum.Enum):
    GREETING = "GREETING"  # NEW (Feature 1)
    PARTIAL_PAYMENT = "PARTIAL_PAYMENT"
    FULL_PAYMENT = "FULL_PAYMENT"
    REQUEST_INVOICE = "REQUEST_INVOICE"
    PROMISE_TO_PAY = "PROMISE_TO_PAY"
    DISPUTE = "DISPUTE"
    ALREADY_PAID = "ALREADY_PAID"
    PAYMENT_PENDING = "PAYMENT_PENDING"
    PAYMENT_FAILED = "PAYMENT_FAILED"
    PAYMENT_PROOF = "PAYMENT_PROOF"
    PAYMENT_PLAN_REQUEST = "PAYMENT_PLAN_REQUEST"
    MULTI_INVOICE_PAYMENT = "MULTI_INVOICE_PAYMENT"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    UNKNOWN = "UNKNOWN"


class ActionType(str, enum.Enum):
    GREETING_RESPONSE = "GREETING_RESPONSE"  # NEW (Feature 1)
    CREATE_PAYMENT_LINK = "CREATE_PAYMENT_LINK"
    RESEND_INVOICE = "RESEND_INVOICE"
    SCHEDULE_REMINDER = "SCHEDULE_REMINDER"
    FLAG_DISPUTE = "FLAG_DISPUTE"
    VERIFY_PAYMENT = "VERIFY_PAYMENT"
    REQUEST_PAYMENT_PROOF = "REQUEST_PAYMENT_PROOF"
    ESCALATE = "ESCALATE"
    ASK_CLARIFICATION = "ASK_CLARIFICATION"
    NO_ACTION = "NO_ACTION"


class CopilotIntent(str, enum.Enum):
    LIST_UNPAID = "LIST_UNPAID"                  # "which firms haven't cleared their invoices"
    LIST_NEEDS_REMINDER = "LIST_NEEDS_REMINDER"  # "which firms need a reminder"
    SEND_REMINDERS = "SEND_REMINDERS"            # "send a reminder to everyone due in a few days"
    FIRM_LOOKUP = "FIRM_LOOKUP"                  # "how is gagan traders doing"
    GENERATE_PDF = "GENERATE_PDF"                # "give me the invoice pdf for gagan traders"
    GENERAL_QUESTION = "GENERAL_QUESTION"        # answerable from a portfolio-level numeric summary
    UNKNOWN = "UNKNOWN"


class CopilotDecision(BaseModel):
    intent: CopilotIntent
    due_within_days: Optional[int] = Field(
        default=None,
        description=(
            "For LIST_NEEDS_REMINDER / SEND_REMINDERS: how many days ahead counts as 'coming due "
            "soon'. Extract from phrasing like 'a few days' (~3), 'this week' (~7), 'next 2 days' (2). "
            "Leave null if the wholesaler didn't specify any soon-due window at all."
        ),
    )
    include_overdue: bool = Field(
        default=True,
        description=(
            "Whether ALREADY-overdue firms should be included in LIST_NEEDS_REMINDER / "
            "SEND_REMINDERS scope. Default true unless the wholesaler explicitly asks only "
            "about upcoming/not-yet-due firms."
        ),
    )
    firm_name_query: Optional[str] = Field(
        default=None,
        description="For FIRM_LOOKUP or GENERATE_PDF: the firm/business name the wholesaler mentioned, as written.",
    )
    reasoning: str


class ExtractedIntent(BaseModel):
    intent: IntentType
    extracted_amount: Optional[float] = Field(
        default=None,
        description="Rupee amount mentioned by buyer, if any. Null if none mentioned.",
    )
    invoice_number: Optional[str] = Field(
        default=None,
        description=(
            "Specific invoice number mentioned by the buyer, if any (e.g. 'INV-1004'). "
            "Null if the buyer refers generically to 'my bill'/'my balance' without naming "
            "a specific invoice, or if multiple invoices are mentioned without singling one out."
        ),
    )
    promise_date_iso: Optional[str] = Field(
        default=None,
        description="ISO date (YYYY-MM-DD) if buyer mentioned a future CALENDAR date/day (e.g. 'tomorrow', 'Friday', 'month end'), else null.",
    )
    promise_relative_minutes: Optional[int] = Field(
        default=None,
        description=(
            "If buyer asked for a reminder/promise using a short RELATIVE duration "
            "(e.g. 'in 10 mins', 'in 2 hours', 'in half an hour'), the number of minutes "
            "from now. Convert hours to minutes (2 hours -> 120). Null if no relative "
            "duration was mentioned — use promise_date_iso instead for calendar dates."
        ),
    )
    dispute_reason: Optional[str] = Field(default=None)
    payment_reference: Optional[str] = Field(default=None, description="Payment/UTR/reference number supplied by buyer, if any.")
    remaining_promise_amount: Optional[float] = Field(default=None, description="Amount buyer says will remain/pay later after a partial payment, if stated.")
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    reasoning: str = Field(default="")


class ValidationResult(BaseModel):
    allowed: bool
    action: ActionType
    final_amount: Optional[float] = None
    reason: str
    reminder_date: Optional[datetime] = None


class AgentDecision(BaseModel):
    raw_message: str
    extracted: ExtractedIntent
    validation: ValidationResult


# --------------------------------------------------------------------------
# SYSTEM PROMPT — restored explicit disambiguation rules + few-shot examples.
# Rule 0 (GREETING) and invoice_number extraction added for Feature 1.
# --------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a strict information-extraction engine for a B2B trade collections system.
You will be given a WhatsApp-style message from a retailer/buyer who owes money on a credit invoice to a wholesaler.

Your ONLY job is to extract structured facts from the message. You must NOT decide whether any payment or action is valid — that is handled by separate deterministic code.

Classify the message into exactly one intent:
- GREETING: the message is ONLY a salutation/opener with no substantive request — "hi", "hello", "hey", "good morning", "namaste", "hi there". If the message combines a greeting with an actual request (e.g. "hi, can I get my bill"), do NOT classify as GREETING — classify it under whichever intent below matches the substantive part of the message instead.
- PARTIAL_PAYMENT: buyer commits to paying SOME amount NOW/TODAY (keywords: "pay", "today", "now", "abhi", "send link"), without explicitly saying it clears the full/entire balance.
- FULL_PAYMENT: buyer commits to clearing the ENTIRE outstanding balance NOW/TODAY. Only use this if the buyer explicitly signals completeness — words like "full", "poora", "puri", "clear all", "everything", "close the account" — NOT just because an amount happens to be mentioned.
- REQUEST_INVOICE: buyer asks for a copy of the bill/invoice/statement (keywords: "bill", "invoice", "copy", "statement", "receipt").
- PROMISE_TO_PAY: buyer promises to pay at a FUTURE date/day, not today (keywords: "tomorrow", "next week", "Friday", "month end", "will pay", "remind me"). This also covers short-term reminder requests like "remind me in 10 mins" / "in 2 hours" — these are still PROMISE_TO_PAY, just with a relative duration instead of a calendar date (see promise_relative_minutes below).
- DISPUTE: buyer disputes the amount, claims goods were damaged/returned, or says the bill is wrong.
- ALREADY_PAID: buyer says they have already paid, transferred, sent money, or settled the bill.
- PAYMENT_PENDING: buyer says a payment is initiated/processing but not yet confirmed.
- PAYMENT_FAILED: buyer says payment/link/transaction failed or was declined.
- PAYMENT_PROOF: buyer provides a UTR, transaction/reference number, screenshot mention, or other proof that payment was made.
- PAYMENT_PLAN_REQUEST: buyer asks for extra time, installments, or a payment plan rather than simply naming one future payment date.
- MULTI_INVOICE_PAYMENT: buyer explicitly asks to pay/clear all pending invoices or the full account balance.
- CLARIFICATION_REQUIRED: message is meaningful but cannot safely be mapped to a single invoice/action without asking a question (for example a bare "kal" when no active promise/context resolves it).
- UNKNOWN: use ONLY if the message truly does not fit any category above after considering all rules below. Do not default to UNKNOWN just because the phrasing is casual or informal.

Disambiguation rules (apply in this order):
0. If the message is nothing more than a greeting/opener with no other request or information -> GREETING. A greeting word followed by a real request ("hey, send me the bill") is NOT a GREETING — evaluate the rest of the message under the rules below instead.
1. If an amount is mentioned AND a "today/now" signal is present AND there is NO explicit "full/all/everything" signal -> PARTIAL_PAYMENT, with extracted_amount set to that number.
2. If a "full/all/everything/clear the account" signal is present (with or without a number) AND a "today/now" signal is present -> FULL_PAYMENT. extracted_amount should be null unless the buyer also states a specific number they believe is the full amount.
3. If a future date/day is mentioned instead of "today/now" -> PROMISE_TO_PAY, and set promise_date_iso, regardless of whether an amount is mentioned.
3b. If a short RELATIVE duration is mentioned instead of a calendar date/day -> PROMISE_TO_PAY.
3c. If the buyer asks for extra time/instalments without a concrete payment date -> PAYMENT_PLAN_REQUEST.
4. If the buyer says payment was already made -> ALREADY_PAID. If they provide a UTR/reference/screenshot/proof -> PAYMENT_PROOF. If they say it is processing/pending -> PAYMENT_PENDING. If they say it failed/declined -> PAYMENT_FAILED.
5. If the buyer explicitly says "pay all", "clear all", "all pending bills", "close my account" or equivalent -> MULTI_INVOICE_PAYMENT.
6. If a message is too ambiguous to safely attach to one invoice/action, prefer CLARIFICATION_REQUIRED rather than guessing.
7. If no "today/now" or future-date signal is present but an amount and "pay" are mentioned, assume PARTIAL_PAYMENT with extracted_amount set to that number.
8. Never invent or guess extracted_amount from context you are not given; if genuinely no amount is stated anywhere, leave it null.

Invoice number extraction (independent of intent):
- Whenever the buyer names a specific invoice number (e.g. "INV-1004", "invoice 1004", "bill number 1004"), set invoice_number to that value, normalized to the "INV-XXXX" form if a bare number is given and clearly refers to this system's numbering.
- If the buyer refers only generically to "my bill", "my balance", "what I owe" without naming a specific invoice, leave invoice_number null — this signals the buyer has NOT disambiguated which invoice they mean (relevant when they have more than one open invoice).
- Extract invoice_number regardless of which intent above the message was classified under.

Numeric shorthand conversion:
- "1 lakh" / "1L" = 100000
- "50k" = 50000
- "2.5 lakh" = 250000
- Numbers may appear in Hindi/Hinglish phrasing (e.g. "ek lakh") — convert these too.

Today's reference date (for resolving "Friday", "tomorrow", "next week", etc.): {today_date}

Examples:
- "hi" -> GREETING, confidence high.
- "hello, good morning" -> GREETING, confidence high.
- "hey, can you send me my bill copy?" -> REQUEST_INVOICE (NOT GREETING — the greeting is just an opener, the request is substantive).
- "I can pay 1 lakh today" -> PARTIAL_PAYMENT, extracted_amount=100000, confidence high (buyer signals "today" + amount, no "full" signal).
- "I will clear the full amount now" -> FULL_PAYMENT, extracted_amount=null, confidence high.
- "I'll pay 50k, that's everything I owe right?" -> FULL_PAYMENT, extracted_amount=50000 (buyer explicitly frames it as the complete balance).
- "Please send me the bill copy" -> REQUEST_INVOICE.
- "Send me a copy of INV-1004" -> REQUEST_INVOICE, invoice_number="INV-1004".
- "I want to clear INV-1002, here's 20000" -> PARTIAL_PAYMENT, extracted_amount=20000, invoice_number="INV-1002".
- "I'll pay next Friday, remind me then" -> PROMISE_TO_PAY, promise_date_iso resolved from reference date.
- "Remind me in 10 mins" -> PROMISE_TO_PAY, promise_relative_minutes=10, promise_date_iso=null.
- "Can you ping me in 2 hours, I'll pay then" -> PROMISE_TO_PAY, promise_relative_minutes=120.
- "This bill amount is wrong, goods were damaged" -> DISPUTE, dispute_reason="damaged goods claim".
- "Maine payment already kar di" -> ALREADY_PAID.
- "Payment processing mein hai" -> PAYMENT_PENDING.
- "Transaction fail ho gaya" -> PAYMENT_FAILED.
- "UTR 1234567890, payment kar di" -> PAYMENT_PROOF, payment_reference="1234567890".
- "15 din do, phir 20k aur next week" -> PAYMENT_PLAN_REQUEST.
- "PAY ALL" -> MULTI_INVOICE_PAYMENT.
- "Aaj 20k bhej raha hu, baaki 60k next week" -> PARTIAL_PAYMENT, extracted_amount=20000, remaining_promise_amount=60000, promise_date_iso resolved to next week.

You must always return a confidence between 0 and 1 that honestly reflects how clearly the message matched a rule above — do not default to 0.0 unless the message is genuinely unintelligible or empty.
"""


def _extract_intent_via_llm(message: str, history: Optional[list] = None) -> ExtractedIntent:
    today_str = datetime.utcnow().strftime("%Y-%m-%d (%A)")
    prompt = _SYSTEM_PROMPT.format(today_date=today_str)

    # NEW: conversation memory. Only the last few turns — just enough for
    # natural back-references ("wait, not that one", "yes the second one",
    # "how much was that again?") without bloating every single-message
    # classification call with the full thread history. Each classification
    # call is still independent/stateless on our side (no server-side
    # session) — we just replay recent turns as prior conversation content.
    contents = []
    if history:
        for turn in history[-6:]:
            role = "user" if turn.get("sender") == "buyer" else "model"
            text = (turn.get("text") or "").strip()
            if text:
                contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
    contents.append(types.Content(role="user", parts=[types.Part(text=message)]))

    try:
        response = client.models.generate_content(
            model=_MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=prompt,
                response_mime_type="application/json",
                response_schema=ExtractedIntent,
                temperature=0.1,
            ),
        )
    except Exception as exc:
        # A real API/network failure (quota, timeout, connectivity). Distinct
        # from a parsing failure below — logged separately so you can tell
        # the two apart when debugging instead of both looking like UNKNOWN.
        logger.warning("Gemini API call failed: %s", exc)
        return ExtractedIntent(
            intent=IntentType.UNKNOWN,
            extracted_amount=None,
            invoice_number=None,
            promise_date_iso=None,
            dispute_reason=None,
            confidence=0.0,
            reasoning=f"LLM API call failed: {exc}",
        )

    # Prefer the SDK's own parsed object (built from the schema server-side)
    # over manually re-parsing response.text — this avoids a second, redundant
    # validation pass that could fail on formatting quirks the SDK already handled.
    parsed_obj = getattr(response, "parsed", None)
    if isinstance(parsed_obj, ExtractedIntent):
        return parsed_obj

    # Fallback: manually parse response.text if .parsed wasn't populated.
    raw_text = (response.text or "").strip()
    if not raw_text:
        logger.warning("Gemini returned an empty response for message: %r", message)
        return ExtractedIntent(
            intent=IntentType.UNKNOWN,
            extracted_amount=None,
            invoice_number=None,
            promise_date_iso=None,
            dispute_reason=None,
            confidence=0.0,
            reasoning="LLM returned an empty response (possibly blocked by safety filters).",
        )

    cleaned = raw_text.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
        return ExtractedIntent(**parsed)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("Failed to parse/validate Gemini JSON output: %s | raw=%r", exc, cleaned[:300])
        return ExtractedIntent(
            intent=IntentType.UNKNOWN,
            extracted_amount=None,
            invoice_number=None,
            promise_date_iso=None,
            dispute_reason=None,
            confidence=0.0,
            reasoning=f"Failed to parse/validate LLM JSON output: {exc}",
        )


MIN_PAYMENT_LINK_AMOUNT = 1.0
MAX_REASONABLE_MULTIPLE = 1.0


def validate_intent_action(extracted: ExtractedIntent, invoice) -> ValidationResult:
    # --------------------------------------------------------------------
    # GREETING bypasses invoice-state checks entirely (Feature 1). A buyer
    # should get greeted regardless of whether this invoice happens to be
    # disputed or already fully paid — those checks below are specifically
    # about whether a PAYMENT/INVOICE/REMINDER action is safe to take, which
    # doesn't apply to a plain greeting.
    # --------------------------------------------------------------------
    if extracted.intent == IntentType.GREETING:
        return ValidationResult(
            allowed=True,
            action=ActionType.GREETING_RESPONSE,
            final_amount=None,
            reason="Buyer sent a greeting; deterministic greeting response will be generated with firm-level context.",
        )

    balance = float(invoice.balance_amount)

    if extracted.intent == IntentType.ALREADY_PAID:
        return ValidationResult(
            allowed=True, action=ActionType.VERIFY_PAYMENT, final_amount=None,
            reason="Buyer claims payment was already made; no ledger change until provider/payment evidence is verified."
        )
    if extracted.intent == IntentType.PAYMENT_PENDING:
        return ValidationResult(
            allowed=True, action=ActionType.VERIFY_PAYMENT, final_amount=None,
            reason="Buyer says payment is pending; verify with payment provider before changing the ledger."
        )
    if extracted.intent == IntentType.PAYMENT_FAILED:
        return ValidationResult(
            allowed=True, action=ActionType.NO_ACTION, final_amount=None,
            reason="Buyer reports a failed payment; do not mark paid. Offer a fresh payment link."
        )
    if extracted.intent == IntentType.PAYMENT_PROOF:
        return ValidationResult(
            allowed=True, action=ActionType.VERIFY_PAYMENT, final_amount=None,
            reason="Buyer supplied payment evidence; verify it before reconciling."
        )
    if extracted.intent == IntentType.PAYMENT_PLAN_REQUEST:
        return ValidationResult(
            allowed=True, action=ActionType.ESCALATE, final_amount=None,
            reason="Payment-plan/credit-extension request requires wholesaler approval."
        )
    if extracted.intent == IntentType.MULTI_INVOICE_PAYMENT:
        return ValidationResult(
            allowed=True, action=ActionType.CREATE_PAYMENT_LINK,
            final_amount=None, reason="Buyer explicitly requested settlement of all outstanding invoices; resolve at firm level."
        )

    if invoice.status.value == "DISPUTED" and extracted.intent != IntentType.DISPUTE:
        return ValidationResult(
            allowed=False,
            action=ActionType.NO_ACTION,
            final_amount=None,
            reason="Invoice is under DISPUTE status. Blocked until human resolution.",
        )

    if balance <= 0:
        return ValidationResult(
            allowed=False,
            action=ActionType.NO_ACTION,
            final_amount=None,
            reason="Invoice balance is already zero or negative.",
        )

    if extracted.intent == IntentType.FULL_PAYMENT:
        return ValidationResult(
            allowed=True,
            action=ActionType.CREATE_PAYMENT_LINK,
            final_amount=round(balance, 2),
            reason="Buyer committed to full payment; link created for exact balance.",
        )

    if extracted.intent == IntentType.PARTIAL_PAYMENT:
        amount = extracted.extracted_amount
        if amount is None or amount <= 0:
            return ValidationResult(
                allowed=False,
                action=ActionType.NO_ACTION,
                final_amount=None,
                reason="PARTIAL_PAYMENT intent detected but no valid amount was extracted.",
            )
        if amount < MIN_PAYMENT_LINK_AMOUNT:
            return ValidationResult(
                allowed=False,
                action=ActionType.NO_ACTION,
                final_amount=None,
                reason=f"Extracted amount ₹{amount} is below minimum payable amount.",
            )
        if amount > balance * MAX_REASONABLE_MULTIPLE:
            return ValidationResult(
                allowed=True,
                action=ActionType.CREATE_PAYMENT_LINK,
                final_amount=round(balance, 2),
                reason=f"Stated amount ₹{amount} exceeds balance ₹{balance}. Capped link to ₹{balance}.",
            )
        return ValidationResult(
            allowed=True,
            action=ActionType.CREATE_PAYMENT_LINK,
            final_amount=round(amount, 2),
            reason=f"Buyer committed to partial payment of ₹{amount}; within balance.",
        )

    if extracted.intent == IntentType.REQUEST_INVOICE:
        return ValidationResult(
            allowed=True,
            action=ActionType.RESEND_INVOICE,
            final_amount=None,
            reason="Buyer requested invoice copy.",
        )

    if extracted.intent == IntentType.PROMISE_TO_PAY:
        reminder_dt = None
        # Relative durations ("in 10 mins") take priority when present — they're
        # more precise than a calendar date and the two are mutually exclusive
        # per the extraction prompt's disambiguation rules.
        if extracted.promise_relative_minutes is not None and extracted.promise_relative_minutes > 0:
            reminder_dt = datetime.utcnow() + timedelta(minutes=extracted.promise_relative_minutes)
        elif extracted.promise_date_iso:
            try:
                reminder_dt = datetime.strptime(extracted.promise_date_iso, "%Y-%m-%d")
            except ValueError:
                reminder_dt = None
        if reminder_dt is None:
            reminder_dt = datetime.utcnow() + timedelta(days=3)
        return ValidationResult(
            allowed=True,
            action=ActionType.SCHEDULE_REMINDER,
            final_amount=None,
            reason=f"Promise to pay recorded. Reminder set for {reminder_dt.date()}.",
            reminder_date=reminder_dt,
        )

    if extracted.intent == IntentType.DISPUTE:
        return ValidationResult(
            allowed=True,
            action=ActionType.FLAG_DISPUTE,
            final_amount=None,
            reason=f"Buyer raised a dispute: {extracted.dispute_reason or 'No reason provided'}.",
        )

    if extracted.intent == IntentType.CLARIFICATION_REQUIRED:
        return ValidationResult(
            allowed=True,
            action=ActionType.ASK_CLARIFICATION,
            final_amount=None,
            reason="Message is meaningful but too ambiguous to safely act on; asking a clarifying question instead of guessing.",
        )

    return ValidationResult(
        allowed=False,
        action=ActionType.NO_ACTION,
        final_amount=None,
        reason="Message intent could not be confidently classified.",
    )


class FirmInsightNarrative(BaseModel):
    summary: str = Field(..., description="3-5 sentence plain-English payment-reliability summary")
    recommended_action: str = Field(..., description="One concrete next step for the wholesaler")


_INSIGHT_PROMPT = """You are a credit-risk analyst summarizing ONE buyer firm's payment history for a wholesaler.
You will be given a JSON object of already-computed stats about this firm.

Write:
- summary: a concise 3-5 sentence plain-English assessment of this firm's payment reliability.
- recommended_action: ONE concrete next step for the wholesaler.

STRICT RULE: do NOT invent, estimate, or restate any number that is not present in the given JSON.
Use the given figures directly and naturally in your sentences. If a field is null or zero, you may
omit mentioning it rather than guessing at a value.
"""


def generate_firm_insight_narrative(stats: dict) -> FirmInsightNarrative:
    """
    Turns deterministic, already-computed payment-history stats for one firm
    (risk_level, on-time rate, broken promises, disputes, etc. — all
    computed in main.py directly from the ledger, never by this function)
    into a short natural-language narrative. Same discipline as the rest of
    this codebase: the LLM explains, it never originates a money figure.
    """
    try:
        response = client.models.generate_content(
            model=_MODEL_NAME,
            contents=json.dumps(stats, default=str),
            config=types.GenerateContentConfig(
                system_instruction=_INSIGHT_PROMPT,
                response_mime_type="application/json",
                response_schema=FirmInsightNarrative,
                temperature=0.2,
            ),
        )
        parsed_obj = getattr(response, "parsed", None)
        if isinstance(parsed_obj, FirmInsightNarrative):
            return parsed_obj
        raw_text = (response.text or "").strip().replace("```json", "").replace("```", "")
        return FirmInsightNarrative(**json.loads(raw_text))
    except Exception as exc:
        logger.warning("Firm insight narrative generation failed, using deterministic fallback: %s", exc)
        return FirmInsightNarrative(
            summary=_fallback_insight_summary(stats),
            recommended_action=_fallback_insight_action(stats),
        )


def _fallback_insight_summary(stats: dict) -> str:
    parts = [
        f"{stats.get('business_name', 'This firm')} has ₹{stats.get('total_outstanding', 0):,.2f} "
        f"outstanding across {stats.get('open_invoice_count', 0)} open invoice(s)."
    ]
    if stats.get("on_time_payment_rate_pct") is not None:
        parts.append(f"They've paid on time {stats['on_time_payment_rate_pct']:.0f}% of the time historically.")
    if stats.get("broken_promises", 0):
        parts.append(f"They have {stats['broken_promises']} unresolved promise-to-pay commitment(s).")
    if stats.get("dispute_count", 0):
        parts.append(f"{stats['dispute_count']} invoice(s) have been disputed.")
    return " ".join(parts)


def _fallback_insight_action(stats: dict) -> str:
    risk = stats.get("risk_level")
    if risk == "HIGH":
        return "Consider a direct call before extending further credit."
    if risk == "MEDIUM":
        return "Send a firm follow-up reminder and monitor closely."
    return "No action needed right now — this firm is in good standing."


_COPILOT_SYSTEM_PROMPT = """You are the intent classifier for a wholesaler-facing "Collections Copilot" — a chat assistant that helps a wholesaler manage their accounts-receivable portfolio (many buyer firms, each with invoices).

Your ONLY job is to classify intent and extract scoping parameters. You must NOT decide which specific firms match, compute any total, or write any message text — all of that is done afterward by deterministic code using real database numbers.

The wholesaler is typing quickly on a phone — expect typos, missing letters, and casual phrasing ("needd", "remainder" instead of "reminder", "wich" instead of "which"). Classify based on intent, not exact spelling; don't let a typo push you to UNKNOWN when the meaning is clear.

Classify the wholesaler's message into exactly one intent:
- LIST_UNPAID: wants to see which firms/buyers still owe money / haven't cleared their invoices (regardless of due date).
- LIST_NEEDS_REMINDER: wants to know which firms *should* get a reminder right now (overdue and/or due soon), without necessarily asking to send anything yet.
- SEND_REMINDERS: wants to actually SEND a reminder message now (e.g. "send a reminder to everyone due this week", "remind all overdue clients", "nudge firms whose due date is coming up").
- FIRM_LOOKUP: asking about ONE specific named firm's status/risk/history (e.g. "how is Singh Fabrics doing", "tell me about Kumar Textiles", "any issues with gagan traders").
- GENERATE_PDF: wants an invoice/receipt/combined PDF document for a named firm (e.g. "give me the invoice pdf for gagan traders", "combined pdf for Singh Fabrics").
- GENERAL_QUESTION: any other question answerable from a portfolio-level numeric summary (e.g. "what's our total outstanding", "how many firms are disputed", "who owes the most").
- UNKNOWN: genuinely doesn't fit any of the above, or is unrelated small talk.

due_within_days extraction (only relevant for LIST_NEEDS_REMINDER / SEND_REMINDERS):
- "a few days" / "coming up soon" -> 3
- "this week" / "next few days" -> 7
- "next N days" -> N
- No timeframe mentioned at all -> leave null (deterministic code will use a sensible default)

Examples:
- "which firms haven't cleared their invoices" -> LIST_UNPAID
- "who do I need to remind" -> LIST_NEEDS_REMINDER
- "which firms needd a remainder?" -> LIST_NEEDS_REMINDER (typo-tolerant)
- "send a reminder to every client whose due date is coming in a few days" -> SEND_REMINDERS, due_within_days=3
- "remind everyone overdue" -> SEND_REMINDERS, include_overdue=true, due_within_days=null
- "how is gagan traders doing" -> FIRM_LOOKUP, firm_name_query="gagan traders"
- "give me the combined pdf for Singh Fabrics" -> GENERATE_PDF, firm_name_query="Singh Fabrics"
- "what's our total outstanding right now" -> GENERAL_QUESTION
"""


def classify_copilot_message(message: str, history: Optional[list] = None) -> CopilotDecision:
    contents = []
    if history:
        for turn in history[-6:]:
            role = "user" if turn.get("sender") == "wholesaler" else "model"
            text = (turn.get("text") or "").strip()
            if text:
                contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
    contents.append(types.Content(role="user", parts=[types.Part(text=message)]))

    try:
        response = client.models.generate_content(
            model=_MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=_COPILOT_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=CopilotDecision,
                temperature=0.1,
            ),
        )
        parsed_obj = getattr(response, "parsed", None)
        if isinstance(parsed_obj, CopilotDecision):
            return parsed_obj
        raw_text = (response.text or "").strip().replace("```json", "").replace("```", "")
        return CopilotDecision(**json.loads(raw_text))
    except Exception as exc:
        logger.warning("Copilot classification failed: %s", exc)
        return CopilotDecision(intent=CopilotIntent.UNKNOWN, reasoning=f"Classification failed: {exc}")


def process_buyer_message(message: str, invoice, history: Optional[list] = None) -> AgentDecision:
    extracted = _extract_intent_via_llm(message, history=history)
    validation = validate_intent_action(extracted, invoice)
    return AgentDecision(raw_message=message, extracted=extracted, validation=validation)