"""
LedgerRecover AI - comprehensive demo/test seed data.

Run:
    python seed_data.py

SAFE TO RE-RUN:
    This wipes all application/demo rows and recreates a deterministic dataset.

The dataset is designed to exercise:
- overdue / due-soon / not-due invoices
- partially paid invoices
- paid invoices
- disputed invoices
- multiple open invoices for the same client
- active / fulfilled / broken / cancelled promises
- scheduled / completed / skipped / failed / cancelled collection actions
- inbound/outbound communications
- payment records
- multi-invoice payment allocations
- audit history for common collection intents
- escalated and autonomous-disabled cases

All dates are generated relative to datetime.utcnow(), so the data stays useful
when the demo is run on a different day.
"""

import json
from datetime import datetime, timedelta

from database import SessionLocal, engine, Base
import models

# Payment links seeded into demo chat history must be REAL Razorpay test-mode
# links, never a fabricated URL — a plausible-looking fake string like
# "https://rzp.io/i/demoXYZ" is indistinguishable from a real one to anyone
# reading the transcript, which is worse than showing nothing. If Razorpay
# isn't configured/reachable at seed time, that row is honestly recorded as
# FAILED, exactly the way the live app renders a real API failure.
try:
    from razorpay_client import create_payment_link, RazorpayClientError
    _RAZORPAY_AVAILABLE = True
except Exception as _rzp_import_exc:  # missing/blank .env keys, no network, etc.
    _RAZORPAY_AVAILABLE = False
    _RAZORPAY_IMPORT_ERROR = _rzp_import_exc

    class RazorpayClientError(Exception):
        pass

from models import (
    Client,
    Invoice,
    PaymentRecord,
    PaymentAllocation,
    AuditLog,
    AuditStatus,
    InvoiceStatus,
    CollectionCase,
    CollectionCaseStatus,
    PaymentPromise,
    PromiseStatus,
    CollectionAction,
    CollectionActionStatus,
    Communication,
)


def audit(
    invoice_id,
    message,
    intent,
    action,
    status=AuditStatus.SUCCESS,
    payload=None,
    timestamp=None,
):
    return AuditLog(
        invoice_id=invoice_id,
        incoming_message=message,
        detected_intent=intent,
        action_taken=action,
        status=status,
        execution_payload=json.dumps(payload or {}),
        timestamp=timestamp,
    )


def audit_real_payment_link(
    invoice_id,
    message,
    invoice_number,
    amount_rupees,
    customer_name,
    customer_contact,
    timestamp=None,
):
    """
    Seeds a CREATE_PAYMENT_LINK audit row using an ACTUAL Razorpay test-mode
    link — never a hardcoded fake URL. If Razorpay can't be reached (no
    .env keys yet, offline sandbox, test-mode link quota hit), this is
    recorded as a genuine FAILED row with no short_url, identical to how
    main.py's own PAY fast-path handles a real RazorpayClientError. That
    keeps the demo transcript 100% honest: every link shown either really
    works, or the message honestly says link creation failed.
    """
    if _RAZORPAY_AVAILABLE:
        try:
            link_res = create_payment_link(
                amount_rupees=amount_rupees,
                customer_name=customer_name,
                customer_contact=customer_contact,
                invoice_number=invoice_number,
                description=f"Payment for {invoice_number}",
            )
            payload = {
                "invoice_number": invoice_number,
                "razorpay_payment_link_id": link_res["id"],
                "short_url": link_res["short_url"],
                "amount_paise": link_res.get("amount"),
            }
            return audit(
                invoice_id, message, "FULL_PAYMENT", "CREATE_PAYMENT_LINK",
                status=AuditStatus.SUCCESS, payload=payload, timestamp=timestamp,
            )
        except RazorpayClientError as exc:
            print(f"  ! Razorpay call failed while seeding {invoice_number}'s payment link "
                  f"({exc}) — seeding as FAILED, not a fake link.")
        except Exception as exc:
            print(f"  ! Unexpected error creating real payment link for {invoice_number} "
                  f"({exc}) — seeding as FAILED, not a fake link.")
    else:
        print(f"  ! Razorpay not configured (check RAZORPAY_KEY_ID/SECRET in .env: "
              f"{_RAZORPAY_IMPORT_ERROR}) — seeding {invoice_number}'s payment link as FAILED, "
              f"not a fake link.")

    return audit(
        invoice_id, message, "FULL_PAYMENT", "CREATE_PAYMENT_LINK",
        status=AuditStatus.FAILED,
        payload={
            "validation_reason": "Attempted deterministic PAY command.",
            "error": "Razorpay unavailable at seed time — no link was fabricated.",
        },
        timestamp=timestamp,
    )


def communication(
    client_id,
    invoice_id,
    direction,
    message,
    channel="SIMULATOR",
    provider_message_id=None,
    created_at=None,
):
    return Communication(
        client_id=client_id,
        invoice_id=invoice_id,
        direction=direction,
        channel=channel,
        message=message,
        provider_message_id=provider_message_id,
        created_at=created_at or datetime.utcnow(),
    )


def seed():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        # ------------------------------------------------------------------
        # 1. Clean every table that belongs to the current application.
        # ------------------------------------------------------------------
        # Delete children first because several tables have foreign keys.
        for model in (
            PaymentAllocation,
            Communication,
            CollectionAction,
            PaymentPromise,
            CollectionCase,
            AuditLog,
            PaymentRecord,
            Invoice,
            Client,
        ):
            db.query(model).delete(synchronize_session=False)

        db.commit()

        now = datetime.utcnow()

        # ------------------------------------------------------------------
        # 2. Clients
        # ------------------------------------------------------------------
        clients_data = [
            ("Rakesh Kumar", "9999900001", "Kumar Textiles"),
            ("Sunita Sharma", "9999900002", "Sharma Garments"),
            ("Vikram Singh", "9999900003", "Singh Fabrics Co."),
            ("Priya Agarwal", "9999900004", "Agarwal Wholesale Suits"),
            ("Manoj Verma", "9999900005", "Verma Retail Traders"),
            ("Gagan Mehta", "9999900006", "Gagan Electronics"),
            ("Amit Jain", "9999900007", "Jain Distributors"),
            ("Neha Kapoor", "9999900008", "Kapoor Home Supplies"),
        ]

        clients = {}
        for name, phone, business in clients_data:
            client = Client(
                name=name,
                phone_number=phone,
                business_name=business,
            )
            db.add(client)
            clients[business] = client

        db.commit()

        # ------------------------------------------------------------------
        # 3. Invoices
        # ------------------------------------------------------------------
        # Kumar Textiles:
        #   - heavily overdue, large balance
        #   - second overdue invoice -> useful for PAY ALL / ambiguity tests
        #
        # Sharma Garments:
        #   - partially paid
        #
        # Singh Fabrics:
        #   - due soon
        #
        # Agarwal Wholesale Suits:
        #   - disputed
        #
        # Verma Retail Traders:
        #   - fully paid
        #
        # Gagan Electronics:
        #   - multiple invoices, one partial and one pending
        #
        # Jain Distributors:
        #   - not due yet
        #
        # Kapoor Home Supplies:
        #   - overdue and used for escalation / promise scenarios
        def make_invoice(
            number,
            client,
            total,
            paid,
            status,
            due_days,
            age_days,
        ):
            return Invoice(
                invoice_number=number,
                client_id=client.id,
                total_amount=float(total),
                paid_amount=float(paid),
                balance_amount=round(float(total) - float(paid), 2),
                status=status,
                due_date=now + timedelta(days=due_days),
                created_at=now - timedelta(days=age_days),
            )

        inv = {}

        inv["INV-1001"] = make_invoice(
            "INV-1001", clients["Kumar Textiles"],
            150000, 0, InvoiceStatus.PENDING, -12, 45
        )
        inv["INV-1002"] = make_invoice(
            "INV-1002", clients["Kumar Textiles"],
            75000, 0, InvoiceStatus.PENDING, -4, 30
        )

        inv["INV-1003"] = make_invoice(
            "INV-1003", clients["Sharma Garments"],
            80000, 30000, InvoiceStatus.PARTIALLY_PAID, -2, 28
        )

        inv["INV-1004"] = make_invoice(
            "INV-1004", clients["Singh Fabrics Co."],
            220000, 0, InvoiceStatus.PENDING, 2, 10
        )

        inv["INV-1005"] = make_invoice(
            "INV-1005", clients["Agarwal Wholesale Suits"],
            95000, 0, InvoiceStatus.DISPUTED, -8, 40
        )

        inv["INV-1006"] = make_invoice(
            "INV-1006", clients["Verma Retail Traders"],
            60000, 60000, InvoiceStatus.PAID, -15, 45
        )

        inv["INV-1007"] = make_invoice(
            "INV-1007", clients["Gagan Electronics"],
            100000, 25000, InvoiceStatus.PARTIALLY_PAID, -6, 32
        )
        inv["INV-1008"] = make_invoice(
            "INV-1008", clients["Gagan Electronics"],
            45000, 0, InvoiceStatus.PENDING, 5, 12
        )

        inv["INV-1009"] = make_invoice(
            "INV-1009", clients["Jain Distributors"],
            300000, 0, InvoiceStatus.PENDING, 12, 5
        )

        inv["INV-1010"] = make_invoice(
            "INV-1010", clients["Kapoor Home Supplies"],
            125000, 0, InvoiceStatus.PENDING, -20, 55
        )

        inv["INV-1011"] = make_invoice(
            "INV-1011", clients["Kapoor Home Supplies"],
            50000, 0, InvoiceStatus.PENDING, -10, 35
        )

        db.add_all(list(inv.values()))
        db.commit()

        # ------------------------------------------------------------------
        # 4. Payment history
        # ------------------------------------------------------------------
        payment_1003 = PaymentRecord(
            invoice_id=inv["INV-1003"].id,
            razorpay_payment_id="pay_seed_1003_30000",
            razorpay_payment_link_id="plink_seed_1003",
            amount_paid=30000.0,
            paid_at=now - timedelta(days=6),
        )

        payment_1006 = PaymentRecord(
            invoice_id=inv["INV-1006"].id,
            razorpay_payment_id="pay_seed_1006_60000",
            razorpay_payment_link_id="plink_seed_1006",
            amount_paid=60000.0,
            paid_at=now - timedelta(days=14),
        )

        payment_1007 = PaymentRecord(
            invoice_id=inv["INV-1007"].id,
            razorpay_payment_id="pay_seed_1007_25000",
            razorpay_payment_link_id="plink_seed_1007",
            amount_paid=25000.0,
            paid_at=now - timedelta(days=7),
        )

        db.add_all([payment_1003, payment_1006, payment_1007])
        db.commit()

        # ------------------------------------------------------------------
        # 5. Multi-invoice payment allocation
        # ------------------------------------------------------------------
        # A single provider transaction of ₹70,000 is anchored to INV-1001
        # and allocated across two Kumar Textiles invoices. This mirrors the
        # PAY ALL / combined-payment path in the application.
        combined_payment = PaymentRecord(
            invoice_id=inv["INV-1001"].id,
            razorpay_payment_id="pay_seed_multi_70000",
            razorpay_payment_link_id="plink_seed_multi",
            amount_paid=70000.0,
            paid_at=now - timedelta(days=1),
        )
        db.add(combined_payment)
        db.flush()

        # Apply the allocation to the invoice ledger as well.
        inv["INV-1001"].paid_amount = 60000.0
        inv["INV-1001"].balance_amount = 90000.0
        inv["INV-1001"].status = InvoiceStatus.PARTIALLY_PAID

        inv["INV-1002"].paid_amount = 10000.0
        inv["INV-1002"].balance_amount = 65000.0
        inv["INV-1002"].status = InvoiceStatus.PARTIALLY_PAID

        db.add_all([
            PaymentAllocation(
                payment_id=combined_payment.id,
                invoice_id=inv["INV-1001"].id,
                amount_allocated=60000.0,
            ),
            PaymentAllocation(
                payment_id=combined_payment.id,
                invoice_id=inv["INV-1002"].id,
                amount_allocated=10000.0,
            ),
        ])
        db.commit()

        # ------------------------------------------------------------------
        # 6. Collection cases
        # ------------------------------------------------------------------
        cases = {}

        def add_case(
            business,
            status,
            priority,
            next_action_at=None,
            last_contact_at=None,
            last_customer_response_at=None,
            escalation_reason=None,
            autonomous_enabled=True,
        ):
            client = clients[business]
            case = CollectionCase(
                client_id=client.id,
                status=status,
                priority_score=priority,
                next_action_at=next_action_at,
                last_contact_at=last_contact_at,
                last_customer_response_at=last_customer_response_at,
                escalation_reason=escalation_reason,
                autonomous_enabled=autonomous_enabled,
            )
            db.add(case)
            cases[business] = case
            return case

        # These are deliberately varied. refresh_case()/collection_queue()
        # can recalculate them later when you want to test live state changes.
        add_case(
            "Kumar Textiles",
            CollectionCaseStatus.OVERDUE,
            95.0,
            next_action_at=now + timedelta(hours=72),
            last_contact_at=now - timedelta(days=2),
            last_customer_response_at=now - timedelta(days=1),
        )

        add_case(
            "Sharma Garments",
            CollectionCaseStatus.PARTIALLY_PAID,
            42.0,
            next_action_at=now + timedelta(days=2),
            last_contact_at=now - timedelta(days=3),
            last_customer_response_at=now - timedelta(days=2),
        )

        add_case(
            "Singh Fabrics Co.",
            CollectionCaseStatus.DUE,
            25.0,
            next_action_at=now + timedelta(days=1),
        )

        add_case(
            "Agarwal Wholesale Suits",
            CollectionCaseStatus.DISPUTED,
            88.0,
            escalation_reason="Customer disputes invoice amount and reported damaged goods.",
            autonomous_enabled=False,
        )

        add_case(
            "Verma Retail Traders",
            CollectionCaseStatus.PAID,
            0.0,
        )

        add_case(
            "Gagan Electronics",
            CollectionCaseStatus.PARTIALLY_PAID,
            55.0,
            next_action_at=now + timedelta(days=3),
            last_contact_at=now - timedelta(days=1),
        )

        add_case(
            "Jain Distributors",
            CollectionCaseStatus.NOT_DUE,
            10.0,
            next_action_at=now + timedelta(days=7),
        )

        # ESCALATED is intentionally sticky in the collection engine.
        add_case(
            "Kapoor Home Supplies",
            CollectionCaseStatus.ESCALATED,
            120.0,
            next_action_at=None,
            last_contact_at=now - timedelta(days=1),
            last_customer_response_at=now - timedelta(days=4),
            escalation_reason="No payment response after repeated automated collection attempts.",
            autonomous_enabled=True,
        )

        db.commit()

        # ------------------------------------------------------------------
        # 7. Promise lifecycle examples
        # ------------------------------------------------------------------
        # ACTIVE: future promise. This should appear in the dashboard and
        # should NOT be marked broken by reconcile_promises().
        active_promise = PaymentPromise(
            client_id=clients["Gagan Electronics"].id,
            invoice_id=inv["INV-1008"].id,
            case_id=cases["Gagan Electronics"].id,
            promised_amount=45000.0,
            promised_date=now + timedelta(days=2),
            status=PromiseStatus.ACTIVE,
            source_message="I will pay ₹45,000 for INV-1008 day after tomorrow.",
            created_at=now - timedelta(hours=2),
        )

        # FULFILLED: invoice is paid, promise was subsequently fulfilled.
        fulfilled_promise = PaymentPromise(
            client_id=clients["Verma Retail Traders"].id,
            invoice_id=inv["INV-1006"].id,
            case_id=cases["Verma Retail Traders"].id,
            promised_amount=60000.0,
            promised_date=now - timedelta(days=10),
            status=PromiseStatus.FULFILLED,
            source_message="I will pay ₹60,000 for INV-1006 on the promised date.",
            created_at=now - timedelta(days=12),
            fulfilled_at=now - timedelta(days=9),
        )

        # BROKEN: overdue promise with no corresponding payment after creation.
        broken_promise = PaymentPromise(
            client_id=clients["Kapoor Home Supplies"].id,
            invoice_id=inv["INV-1010"].id,
            case_id=cases["Kapoor Home Supplies"].id,
            promised_amount=50000.0,
            promised_date=now - timedelta(days=2),
            status=PromiseStatus.BROKEN,
            source_message="I will pay ₹50,000 for INV-1010 in two days.",
            created_at=now - timedelta(days=5),
            broken_at=now - timedelta(days=1),
        )

        # CANCELLED: demonstrates promise supersession.
        cancelled_promise = PaymentPromise(
            client_id=clients["Kumar Textiles"].id,
            invoice_id=inv["INV-1002"].id,
            case_id=cases["Kumar Textiles"].id,
            promised_amount=20000.0,
            promised_date=now + timedelta(days=1),
            status=PromiseStatus.CANCELLED,
            source_message="I will pay ₹20,000 for INV-1002 tomorrow.",
            created_at=now - timedelta(days=2),
        )

        db.add_all([
            active_promise,
            fulfilled_promise,
            broken_promise,
            cancelled_promise,
        ])
        db.commit()

        # ------------------------------------------------------------------
        # 8. Collection actions
        # ------------------------------------------------------------------
        actions = [
            # Future action: safe to leave in SCHEDULED state.
            CollectionAction(
                client_id=clients["Gagan Electronics"].id,
                invoice_id=inv["INV-1008"].id,
                case_id=cases["Gagan Electronics"].id,
                action_type="FOLLOW_UP_PROMISE",
                status=CollectionActionStatus.SCHEDULED,
                scheduled_at=now + timedelta(days=2),
                reason="Follow up on active payment promise.",
                dedupe_key="seed-followup-promise-gagan-1008",
            ),

            # Future overdue follow-up.
            CollectionAction(
                client_id=clients["Kumar Textiles"].id,
                invoice_id=inv["INV-1001"].id,
                case_id=cases["Kumar Textiles"].id,
                action_type="OVERDUE_FOLLOW_UP",
                status=CollectionActionStatus.SCHEDULED,
                scheduled_at=now + timedelta(hours=72),
                reason="Initial overdue reminder already sent.",
                dedupe_key="seed-overdue-followup-kumar-1001",
            ),

            # Already executed.
            CollectionAction(
                client_id=clients["Sharma Garments"].id,
                invoice_id=inv["INV-1003"].id,
                case_id=cases["Sharma Garments"].id,
                action_type="OVERDUE_FOLLOW_UP",
                status=CollectionActionStatus.COMPLETED,
                scheduled_at=now - timedelta(days=2),
                executed_at=now - timedelta(days=2),
                attempt_count=1,
                reason="Overdue follow-up.",
                result="Reminder sent; customer subsequently made a partial payment.",
                dedupe_key="seed-completed-sharma-1003",
            ),

            # Skipped because the invoice is paid.
            CollectionAction(
                client_id=clients["Verma Retail Traders"].id,
                invoice_id=inv["INV-1006"].id,
                case_id=cases["Verma Retail Traders"].id,
                action_type="FOLLOW_UP_PROMISE",
                status=CollectionActionStatus.SKIPPED,
                scheduled_at=now - timedelta(days=9),
                executed_at=now - timedelta(days=9),
                attempt_count=1,
                reason="Promise follow-up.",
                result="Invoice already paid; automated action skipped.",
                dedupe_key="seed-skipped-paid-verma-1006",
            ),

            # Cancelled because the dispute is not eligible for autonomous
            # collection.
            CollectionAction(
                client_id=clients["Agarwal Wholesale Suits"].id,
                invoice_id=inv["INV-1005"].id,
                case_id=cases["Agarwal Wholesale Suits"].id,
                action_type="OVERDUE_FOLLOW_UP",
                status=CollectionActionStatus.CANCELLED,
                scheduled_at=now - timedelta(days=1),
                executed_at=None,
                attempt_count=0,
                reason="Disputed invoice.",
                result="Cancelled because autonomous collection is disabled.",
                dedupe_key="seed-cancelled-dispute-1005",
            ),

            # Failed action for audit/testing UI.
            CollectionAction(
                client_id=clients["Kapoor Home Supplies"].id,
                invoice_id=inv["INV-1010"].id,
                case_id=cases["Kapoor Home Supplies"].id,
                action_type="BROKEN_PROMISE_FOLLOW_UP",
                status=CollectionActionStatus.FAILED,
                scheduled_at=now - timedelta(hours=6),
                executed_at=now - timedelta(hours=6),
                attempt_count=1,
                reason="Previous payment promise was missed.",
                result="Simulated provider failure during reminder delivery.",
                dedupe_key="seed-failed-broken-promise-1010",
            ),
        ]

        db.add_all(actions)
        db.commit()

        # ------------------------------------------------------------------
        # 9. Communication history
        # ------------------------------------------------------------------
        comms = [
            communication(
                clients["Kumar Textiles"].id,
                inv["INV-1001"].id,
                "OUTBOUND",
                "Reminder: INV-1001 is overdue. Outstanding amount is ₹90,000.",
                created_at=now - timedelta(days=3),
            ),
            communication(
                clients["Kumar Textiles"].id,
                inv["INV-1001"].id,
                "INBOUND",
                "I can pay part of it now and the rest next week.",
                created_at=now - timedelta(days=1),
            ),
            communication(
                clients["Kumar Textiles"].id,
                inv["INV-1002"].id,
                "INBOUND",
                "Please send me the payment link for this invoice.",
                created_at=now - timedelta(hours=20),
            ),

            communication(
                clients["Sharma Garments"].id,
                inv["INV-1003"].id,
                "OUTBOUND",
                "Your invoice is overdue. Please arrange the remaining ₹50,000.",
                created_at=now - timedelta(days=2),
            ),
            communication(
                clients["Sharma Garments"].id,
                inv["INV-1003"].id,
                "INBOUND",
                "I have already paid ₹30,000. I will arrange the rest.",
                created_at=now - timedelta(days=1),
            ),

            communication(
                clients["Agarwal Wholesale Suits"].id,
                inv["INV-1005"].id,
                "INBOUND",
                "This bill amount is wrong; some goods were damaged in transit.",
                created_at=now - timedelta(days=8),
            ),

            communication(
                clients["Gagan Electronics"].id,
                inv["INV-1008"].id,
                "INBOUND",
                "I will pay ₹45,000 for INV-1008 day after tomorrow.",
                created_at=now - timedelta(hours=2),
            ),

            communication(
                clients["Kapoor Home Supplies"].id,
                inv["INV-1010"].id,
                "INBOUND",
                "I promised payment but could not arrange it.",
                created_at=now - timedelta(days=1),
            ),
            communication(
                clients["Kapoor Home Supplies"].id,
                inv["INV-1010"].id,
                "OUTBOUND",
                "Your payment promise has been missed. Please contact the wholesaler.",
                created_at=now - timedelta(hours=12),
            ),

            # Client-level communication without an invoice. Useful for
            # multi-invoice ambiguity / PAY ALL flows.
            communication(
                clients["Gagan Electronics"].id,
                None,
                "INBOUND",
                "I can pay ₹30,000 today. Which invoice should I use?",
                created_at=now - timedelta(hours=5),
            ),
        ]
        db.add_all(comms)
        db.commit()

        # ------------------------------------------------------------------
        # 10. Audit history
        # ------------------------------------------------------------------
        audit_rows = [
            audit(
                inv["INV-1001"].id,
                "SYSTEM: Invoice due date surpassed",
                "OVERDUE_AUTO_REMINDER",
                "SEND_OVERDUE_REMINDER",
                payload={
                    "days_overdue": 12,
                    "overdue_text": (
                        f"⚠️ Hi {clients['Kumar Textiles'].name}, Invoice {inv['INV-1001'].invoice_number} "
                        f"(₹{inv['INV-1001'].balance_amount:,.2f}) was due on "
                        f"{inv['INV-1001'].due_date.strftime('%d %b %Y')} and is now 12 day(s) overdue. "
                        f"Please arrange payment at the earliest, or let us know if there's an issue with this bill."
                    ),
                },
                timestamp=now - timedelta(days=3),
            ),
            # "I can pay part of it now and the rest next week" is a
            # PAYMENT_PLAN_REQUEST live (an installment/extra-time ask, not a
            # concrete date) — which the real engine always routes to
            # ESCALATE, not a made-up "CLARIFICATION_REQUIRED" action.
            audit(
                inv["INV-1001"].id,
                "I can pay part of it now and the rest next week.",
                "PAYMENT_PLAN_REQUEST",
                "ESCALATE",
                payload={"escalated": True},
                timestamp=now - timedelta(days=1),
            ),
            # "Please send me the payment link" for ONE specific invoice is a
            # FULL_PAYMENT request live, not REQUEST_INVOICE — and needs the
            # actual link fields the renderer expects, or it shows a broken
            # "here's your link: ." sentence.
            audit_real_payment_link(
                inv["INV-1002"].id,
                "Please send me the payment link for this invoice.",
                invoice_number="INV-1002",
                amount_rupees=inv["INV-1002"].balance_amount,
                customer_name=clients["Kumar Textiles"].name,
                customer_contact=clients["Kumar Textiles"].phone_number,
                timestamp=now - timedelta(hours=20),
            ),
            # "I have already paid X" is ALREADY_PAID live, which always
            # routes to VERIFY_PAYMENT (never mutates the ledger from a bare
            # claim) — not a "PAYMENT_RECORDED" action that doesn't exist.
            audit(
                inv["INV-1003"].id,
                "I have already paid ₹30,000.",
                "ALREADY_PAID",
                "VERIFY_PAYMENT",
                payload={"payment_reference": None, "verification_required": True},
                timestamp=now - timedelta(days=6),
            ),
            audit(
                inv["INV-1005"].id,
                "This bill amount is wrong; some goods were damaged in transit.",
                "DISPUTE",
                "FLAG_DISPUTE",
                payload={
                    "dispute_reason": "damaged goods claim",
                    "note": "flagged for human review",
                },
                timestamp=now - timedelta(days=8),
            ),
            # An actual captured payment (system/webhook-driven, not typed by
            # the buyer) uses detected_intent="PAYMENT_RECEIVED" — this is
            # the ONLY way it renders as a bot-only confirmation instead of
            # appearing as a fake buyer chat bubble saying "SYSTEM: ...".
            audit(
                inv["INV-1006"].id,
                "SYSTEM: Payment captured",
                "PAYMENT_RECEIVED",
                "PAYMENT_RECEIVED",
                payload={
                    "amount": 60000,
                    "new_status": "PAID",
                    "invoice_number": "INV-1006",
                    "invoice_pdf_url": f"/api/invoices/{inv['INV-1006'].id}/receipt-pdf",
                },
                timestamp=now - timedelta(days=14),
            ),
            audit(
                inv["INV-1007"].id,
                "I paid ₹25,000 already.",
                "ALREADY_PAID",
                "VERIFY_PAYMENT",
                payload={"payment_reference": None, "verification_required": True},
                timestamp=now - timedelta(days=7),
            ),
            audit(
                inv["INV-1008"].id,
                "I will pay ₹45,000 for INV-1008 day after tomorrow.",
                "PROMISE_TO_PAY",
                "SCHEDULE_REMINDER",
                payload={
                    "amount": 45000,
                    "reminder_scheduled_for": (now + timedelta(days=2)).isoformat(),
                },
                timestamp=now - timedelta(hours=2),
            ),
            # A broken promise with no new concrete date is also a
            # PAYMENT_PLAN_REQUEST -> ESCALATE live, same as the INV-1001
            # case above — and it's a successful hand-off, not a bot
            # failure, so status stays SUCCESS.
            audit(
                inv["INV-1010"].id,
                "I promised payment but could not arrange it.",
                "PAYMENT_PLAN_REQUEST",
                "ESCALATE",
                payload={"escalated": True, "note": "Seeded broken-promise follow-up."},
                timestamp=now - timedelta(days=1),
            ),
            # System-triggered escalation (not something the buyer typed) —
            # detected_intent="ESCALATION" is treated as an internal event by
            # build_invoice_messages and renders bot-only, same reasoning as
            # PAYMENT_RECEIVED above.
            audit(
                inv["INV-1010"].id,
                "SYSTEM: Repeated collection attempts without payment.",
                "ESCALATION",
                "ESCALATE",
                payload={
                    "escalated": True,
                    "reason": "No payment response after repeated automated collection attempts.",
                },
                timestamp=now - timedelta(hours=12),
            ),
            # This message pairs with the REAL captured payment seeded above
            # (combined_payment / PaymentAllocation rows) — INV-1001 and
            # INV-1002 already show the ₹70,000 applied (60k/10k split), so
            # this must render as a received-payment confirmation matching
            # that actual ledger state, not a pending payment-link request.
            # detected_intent="PAYMENT_RECEIVED" is treated as an internal
            # event (bot-only, no fake buyer bubble), same as INV-1006 above.
            audit(
                inv["INV-1001"].id,
                "I will pay ₹70,000 against all pending invoices.",
                "PAYMENT_RECEIVED",
                "PAYMENT_RECEIVED",
                payload={
                    "amount": 70000,
                    "new_status": "PARTIALLY_PAID",
                    "invoice_number": "INV-1001 & INV-1002",
                },
                timestamp=now - timedelta(hours=6),
            ),
            audit(
                None,
                "hi",
                "GREETING",
                "GREETING_RESPONSE",
                payload={"fast_path": True, "greeting_text": "Hi! How can I help you today?"},
                timestamp=now - timedelta(hours=6),
            ),
        ]
        db.add_all(audit_rows)
        db.commit()

        # ------------------------------------------------------------------
        # 11. Print a useful summary
        # ------------------------------------------------------------------
        print()
        print("=" * 64)
        print("LedgerRecover AI seed complete")
        print("=" * 64)
        print(f"Clients:             {db.query(Client).count()}")
        print(f"Invoices:            {db.query(Invoice).count()}")
        print(f"Payments:            {db.query(PaymentRecord).count()}")
        print(f"Allocations:         {db.query(PaymentAllocation).count()}")
        print(f"Collection cases:    {db.query(CollectionCase).count()}")
        print(f"Promises:            {db.query(PaymentPromise).count()}")
        print(f"Collection actions:  {db.query(CollectionAction).count()}")
        print(f"Communications:      {db.query(Communication).count()}")
        print(f"Audit logs:          {db.query(AuditLog).count()}")
        print()
        print("Clients / primary scenarios:")
        print("  Kumar Textiles       -> 2 overdue invoices + multi-invoice payment")
        print("  Sharma Garments      -> partially paid + payment history")
        print("  Singh Fabrics Co.    -> due soon")
        print("  Agarwal Wholesale     -> disputed + autonomous disabled")
        print("  Verma Retail Traders -> fully paid + fulfilled promise")
        print("  Gagan Electronics    -> multiple invoices + ACTIVE promise")
        print("  Jain Distributors    -> not due")
        print("  Kapoor Home Supplies -> overdue + BROKEN promise + ESCALATED")
        print()
        print("Key invoices:")
        for number in sorted(inv):
            item = inv[number]
            print(
                f"  {number}: {item.status.value:<15} "
                f"total=₹{item.total_amount:,.0f} "
                f"paid=₹{item.paid_amount:,.0f} "
                f"balance=₹{item.balance_amount:,.0f}"
            )
        print("=" * 64)

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed()