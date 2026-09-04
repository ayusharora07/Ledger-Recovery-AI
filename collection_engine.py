"""Small, database-backed collections orchestration layer for LedgerRecover AI.

This intentionally stays inside the FastAPI app rather than becoming a separate
service. It gives the agent a persistent collection state, promise lifecycle,
next-action queue, and a deterministic priority score.
"""
from __future__ import annotations
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from models import (
    Client, Invoice, AuditLog, InvoiceStatus,
    CollectionCase, CollectionCaseStatus,
    PaymentPromise, PromiseStatus,
    CollectionAction, CollectionActionStatus,
    Communication,
)


def get_or_create_case(db: Session, client: Client) -> CollectionCase:
    case = db.query(CollectionCase).filter(CollectionCase.client_id == client.id).first()
    if not case:
        case = CollectionCase(client_id=client.id)
        db.add(case)
        db.flush()
    return case


def record_communication(db: Session, client: Client, invoice: Invoice | None,
                         direction: str, message: str, channel: str = "SIMULATOR",
                         provider_message_id: str | None = None) -> Communication:
    item = Communication(
        client_id=client.id,
        invoice_id=invoice.id if invoice else None,
        direction=direction,
        channel=channel,
        message=message[:10000],
        provider_message_id=provider_message_id,
    )
    db.add(item)
    case = get_or_create_case(db, client)
    if direction == "INBOUND":
        case.last_customer_response_at = datetime.utcnow()
    else:
        case.last_contact_at = datetime.utcnow()
    db.flush()
    return item


def compute_priority_score(db: Session, client: Client) -> float:
    now = datetime.utcnow()
    score = 0.0
    open_invoices = [i for i in client.invoices if i.status != InvoiceStatus.PAID]
    for inv in open_invoices:
        score += min(float(inv.balance_amount) / 1000.0, 100.0)
        if inv.due_date < now:
            score += min((now - inv.due_date).days * 4.0, 60.0)
        if inv.status == InvoiceStatus.DISPUTED:
            score += 30.0
    broken = db.query(PaymentPromise).filter(
        PaymentPromise.client_id == client.id,
        PaymentPromise.status == PromiseStatus.BROKEN,
    ).count()
    score += broken * 20.0
    recent = db.query(AuditLog).filter(
        AuditLog.invoice_id.in_([i.id for i in open_invoices]) if open_invoices else False,
        AuditLog.detected_intent.in_(["AUTOMATED_REMINDER_FIRED", "OVERDUE_AUTO_REMINDER", "MANUAL_COPILOT_REMINDER"]),
        AuditLog.timestamp >= now - timedelta(days=3),
    ).count() if open_invoices else 0
    # Repeated recent contact should lower priority slightly to avoid spam.
    score -= min(recent * 5.0, 20.0)
    return round(max(score, 0.0), 2)


def refresh_case(db: Session, client: Client) -> CollectionCase:
    case = get_or_create_case(db, client)
    now = datetime.utcnow()
    open_invoices = [i for i in client.invoices if i.status != InvoiceStatus.PAID]
    disputed = [i for i in open_invoices if i.status == InvoiceStatus.DISPUTED]
    if not open_invoices:
        case.status = CollectionCaseStatus.PAID
        case.next_action_at = None
    elif disputed and len(disputed) == len(open_invoices):
        case.status = CollectionCaseStatus.DISPUTED
    elif case.status == CollectionCaseStatus.ESCALATED and case.escalation_reason:
        # Human-review states are sticky until the underlying exception is
        # resolved; routine queue refreshes must not silently downgrade them.
        case.status = CollectionCaseStatus.ESCALATED
    else:
        active_promise = db.query(PaymentPromise).filter(
            PaymentPromise.client_id == client.id,
            PaymentPromise.status == PromiseStatus.ACTIVE,
        ).order_by(PaymentPromise.promised_date.asc()).first()
        if active_promise:
            case.status = CollectionCaseStatus.PROMISED
            case.next_action_at = active_promise.promised_date
        elif any(i.status == InvoiceStatus.PARTIALLY_PAID for i in open_invoices):
            case.status = CollectionCaseStatus.PARTIALLY_PAID
        elif any(i.due_date < now for i in open_invoices):
            case.status = CollectionCaseStatus.OVERDUE
        elif any(i.due_date <= now + timedelta(days=3) for i in open_invoices):
            case.status = CollectionCaseStatus.DUE
        else:
            case.status = CollectionCaseStatus.NOT_DUE
    case.priority_score = compute_priority_score(db, client)
    db.flush()
    return case


def create_promise(db: Session, client: Client, invoice: Invoice | None,
                   promised_amount: float | None, promised_date: datetime,
                   source_message: str) -> PaymentPromise:
    case = get_or_create_case(db, client)
    # New promise supersedes older still-active promises for the same invoice/case.
    q = db.query(PaymentPromise).filter(
        PaymentPromise.client_id == client.id,
        PaymentPromise.status == PromiseStatus.ACTIVE,
    )
    if invoice:
        q = q.filter(PaymentPromise.invoice_id == invoice.id)
    for old in q.all():
        old.status = PromiseStatus.CANCELLED
    promise = PaymentPromise(
        client_id=client.id,
        invoice_id=invoice.id if invoice else None,
        case_id=case.id,
        promised_amount=round(float(promised_amount), 2) if promised_amount is not None else None,
        promised_date=promised_date,
        status=PromiseStatus.ACTIVE,
        source_message=source_message[:5000],
    )
    db.add(promise)
    case.status = CollectionCaseStatus.PROMISED
    case.next_action_at = promised_date
    db.flush()
    return promise


def reconcile_promises(db: Session, client: Client) -> None:
    now = datetime.utcnow()
    active = db.query(PaymentPromise).filter(
        PaymentPromise.client_id == client.id,
        PaymentPromise.status == PromiseStatus.ACTIVE,
    ).all()
    for promise in active:
        invoice = promise.invoice
        if not invoice:
            continue
        if invoice.status == InvoiceStatus.PAID:
            promise.status = PromiseStatus.FULFILLED
            promise.fulfilled_at = now
        elif promise.promised_date < now:
            # If a promised amount exists, compare payments against the amount
            # promised since the promise was created. Otherwise any clearance
            # after the promise is treated as fulfillment.
            paid_since = sum(
                p.amount_paid for p in invoice.payments
                if p.paid_at >= promise.created_at
            )
            if promise.promised_amount is None or paid_since + 0.01 >= promise.promised_amount:
                promise.status = PromiseStatus.FULFILLED
                promise.fulfilled_at = now
            else:
                promise.status = PromiseStatus.BROKEN
                promise.broken_at = now
    db.flush()


def schedule_action(db: Session, client: Client, action_type: str,
                    scheduled_at: datetime, invoice: Invoice | None = None,
                    reason: str | None = None, dedupe_key: str | None = None) -> CollectionAction | None:
    if dedupe_key:
        existing = db.query(CollectionAction).filter(CollectionAction.dedupe_key == dedupe_key).first()
        if existing and existing.status in (CollectionActionStatus.SCHEDULED, CollectionActionStatus.PROCESSING):
            return existing
    case = get_or_create_case(db, client)
    action = CollectionAction(
        client_id=client.id,
        invoice_id=invoice.id if invoice else None,
        case_id=case.id,
        action_type=action_type,
        status=CollectionActionStatus.SCHEDULED,
        scheduled_at=scheduled_at,
        reason=reason,
        dedupe_key=dedupe_key,
    )
    db.add(action)
    case.next_action_at = scheduled_at
    db.flush()
    return action


def cancel_open_actions_for_invoice(db: Session, invoice: Invoice) -> int:
    rows = db.query(CollectionAction).filter(
        CollectionAction.invoice_id == invoice.id,
        CollectionAction.status == CollectionActionStatus.SCHEDULED,
    ).all()
    for row in rows:
        row.status = CollectionActionStatus.CANCELLED
    return len(rows)


def collection_queue(db: Session, limit: int = 20) -> list[dict]:
    clients = db.query(Client).all()
    now = datetime.utcnow()
    rows = []
    for client in clients:
        reconcile_promises(db, client)
        case = refresh_case(db, client)
        open_invoices = [i for i in client.invoices if i.status != InvoiceStatus.PAID]
        if not open_invoices:
            continue
        target = min(open_invoices, key=lambda i: (i.due_date, -float(i.balance_amount)))
        active_promise = db.query(PaymentPromise).filter(
            PaymentPromise.client_id == client.id,
            PaymentPromise.status == PromiseStatus.ACTIVE,
        ).order_by(PaymentPromise.promised_date.asc()).first()
        broken_count = db.query(PaymentPromise).filter(
            PaymentPromise.client_id == client.id,
            PaymentPromise.status == PromiseStatus.BROKEN,
        ).count()
        rows.append({
            "client_id": client.id,
            "business_name": client.business_name,
            "contact_name": client.name,
            "phone_number": client.phone_number,
            "status": case.status.value,
            "priority_score": case.priority_score,
            "outstanding": round(sum(float(i.balance_amount) for i in open_invoices), 2),
            "open_invoice_count": len(open_invoices),
            "target_invoice": target.invoice_number,
            "target_balance": round(float(target.balance_amount), 2),
            "days_overdue": max((now - target.due_date).days, 0),
            "promise": ({
                "id": active_promise.id,
                "amount": active_promise.promised_amount,
                "date": active_promise.promised_date.isoformat(),
                "status": active_promise.status.value,
            } if active_promise else None),
            "broken_promises": broken_count,
            "next_action_at": case.next_action_at.isoformat() if case.next_action_at else None,
        })
    db.commit()
    rows.sort(key=lambda r: (-r["priority_score"], -r["outstanding"]))
    return rows[:limit]
