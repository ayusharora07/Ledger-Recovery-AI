from __future__ import annotations
import os
import time
import razorpay
from dotenv import load_dotenv
import hmac
import hashlib

load_dotenv()

RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")
RAZORPAY_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET")

if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
    raise RuntimeError(
        "RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET missing. "
        "Copy .env.example to .env and fill in your Razorpay test keys."
    )

_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))


class RazorpayClientError(Exception):
    """Raised when a Razorpay API call fails or returns an unexpected shape."""
    pass


def rupees_to_paise(amount_rupees: float) -> int:
    return int(round(float(amount_rupees) * 100))


def paise_to_rupees(amount_paise: int) -> float:
    return round(float(amount_paise) / 100, 2)


def _with_retry(fn, *args, max_attempts: int = 3, backoff_seconds: float = 0.6, **kwargs):
    """
    Retries a Razorpay SDK call a couple of times, ONLY for transient
    rate-limit errors (HTTP 429 / "too many requests") — anything else
    (bad request, auth failure, etc.) is raised immediately since retrying
    those would just waste time on a call that will never succeed.
    """
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            if "429" in msg or "too many requests" in msg or "rate limit" in msg:
                time.sleep(backoff_seconds * (attempt + 1))
                continue
            raise
    raise last_exc


# --------------------------------------------------------------------------
# PAYMENT LINKS — used by the AI agent for partial-payment amounts
# --------------------------------------------------------------------------

RAZORPAY_MAX_PAYMENT_LINK_AMOUNT = 500000.0  # Razorpay's default per-transaction cap; raise via Razorpay Support if your business genuinely needs more


def create_payment_link(
    amount_rupees: float,
    customer_name: str,
    customer_contact: str,
    invoice_number: str,
    description: str,
    reference_id: str = None,
    callback_url: str = None,
    expire_by: int = None,  # Unix timestamp; link becomes unpayable after this
    invoice_numbers: list[str] | None = None,
) -> dict:
    if amount_rupees is None or amount_rupees <= 0:
        raise RazorpayClientError(
            f"Refusing to create payment link for non-positive amount: {amount_rupees}"
        )

    if amount_rupees > RAZORPAY_MAX_PAYMENT_LINK_AMOUNT:
        raise RazorpayClientError(
            f"Amount ₹{amount_rupees:,.2f} exceeds Razorpay's default per-transaction limit of "
            f"₹{RAZORPAY_MAX_PAYMENT_LINK_AMOUNT:,.2f}. Ask the buyer to pay in smaller partial "
            f"installments, or contact Razorpay Support to raise this limit for your account."
        )

    amount_paise = rupees_to_paise(amount_rupees)

    payload = {
        "amount": amount_paise,
        "currency": "INR",
        "accept_partial": False,
        "description": description[:255] if description else f"Payment for Invoice {invoice_number}",
        "customer": {"name": customer_name, "contact": customer_contact},
        "notify": {"sms": True, "email": False},
        "notes": {"invoice_number": invoice_number, "source": "LedgerRecoverAI", **({"invoice_numbers": ",".join(invoice_numbers)} if invoice_numbers else {})},
    }

    if expire_by:
        payload["expire_by"] = expire_by
    if reference_id:
        payload["reference_id"] = reference_id
    if callback_url:
        payload["callback_url"] = callback_url
        payload["callback_method"] = "get"

    try:
        response = _with_retry(_client.payment_link.create, payload)
    except Exception as exc:
        raise RazorpayClientError(f"Razorpay payment_link.create failed: {exc}") from exc

    if "id" not in response or "short_url" not in response:
        raise RazorpayClientError(f"Unexpected Razorpay response shape: {response}")

    return response


def fetch_payment_link(payment_link_id: str) -> dict:
    """Authoritative source of a Payment Link's notes/status — used as a
    fallback when a webhook payload doesn't include notes directly."""
    try:
        return _client.payment_link.fetch(payment_link_id)
    except Exception as exc:
        raise RazorpayClientError(f"Razorpay payment_link.fetch failed: {exc}") from exc

def cancel_payment_link(payment_link_id: str) -> dict:
    """Cancels a Razorpay Payment Link so it can no longer be paid. Used to
    invalidate a stale link the instant a new one is issued for the same
    invoice — without this, an old link and a newer one both stay payable,
    letting a buyer pay twice against one invoice."""
    try:
        return _client.payment_link.cancel(payment_link_id)
    except Exception as exc:
        raise RazorpayClientError(f"Razorpay payment_link.cancel failed: {exc}") from exc
# --------------------------------------------------------------------------
# ORDERS — used by the Standard Checkout (full-payment) flow
# --------------------------------------------------------------------------

def create_order(amount_rupees: float, invoice_number: str, receipt: str = None) -> dict:
    """
    Creates a Razorpay Order for the Standard Checkout flow — always the
    FULL current invoice balance at the moment the Pay button is clicked.
    notes.invoice_number lets the webhook/verify-payment path resolve which
    invoice this belongs to, exactly like Payment Links do.
    """
    if amount_rupees is None or amount_rupees <= 0:
        raise RazorpayClientError(
            f"Refusing to create order for non-positive amount: {amount_rupees}"
        )

    payload = {
        "amount": rupees_to_paise(amount_rupees),
        "currency": "INR",
        "notes": {"invoice_number": invoice_number, "source": "LedgerRecoverAI", **({"invoice_numbers": ",".join(invoice_numbers)} if invoice_numbers else {})},
    }
    if receipt:
        payload["receipt"] = receipt[:40]

    try:
        response = _client.order.create(payload)
    except Exception as exc:
        raise RazorpayClientError(f"Razorpay order.create failed: {exc}") from exc

    if "id" not in response:
        raise RazorpayClientError(f"Unexpected Razorpay order response shape: {response}")

    return response


def fetch_order(order_id: str) -> dict:
    """Authoritative source of an Order's notes/status."""
    try:
        return _client.order.fetch(order_id)
    except Exception as exc:
        raise RazorpayClientError(f"Razorpay order.fetch failed: {exc}") from exc


def fetch_payment(payment_id: str) -> dict:
    """
    Authoritative source of truth for a specific payment's captured status
    and EXACT captured amount. Standard Checkout verification must call this
    — never trust the amount the frontend/browser claims was paid.
    """
    try:
        return _client.payment.fetch(payment_id)
    except Exception as exc:
        raise RazorpayClientError(f"Razorpay payment.fetch failed: {exc}") from exc


def verify_payment_signature(order_id: str, payment_id: str, signature: str) -> bool:
    """
    Standard Razorpay Checkout post-payment signature verification
    (HMAC of 'order_id|payment_id' using RAZORPAY_KEY_SECRET). This is
    DIFFERENT from the webhook signature (which uses RAZORPAY_WEBHOOK_SECRET)
    — Checkout's client-side callback is verified with the key secret,
    webhooks are verified with the separate webhook secret.
    """
    try:
        _client.utility.verify_payment_signature({
            "razorpay_order_id": order_id,
            "razorpay_payment_id": payment_id,
            "razorpay_signature": signature,
        })
        return True
    except razorpay.errors.SignatureVerificationError:
        return False
    except Exception:
        return False


# --------------------------------------------------------------------------
# WEBHOOK SIGNATURE — unrelated to the two functions above; separate secret
# --------------------------------------------------------------------------

def verify_webhook_signature(payload_body: str, received_signature: str) -> bool:
    if not RAZORPAY_WEBHOOK_SECRET:
        raise RazorpayClientError(
            "RAZORPAY_WEBHOOK_SECRET is not set in .env — cannot verify webhooks. "
            "Refusing to process webhook for safety."
        )

    try:
        _client.utility.verify_webhook_signature(
            payload_body, received_signature, RAZORPAY_WEBHOOK_SECRET
        )
        return True
    except razorpay.errors.SignatureVerificationError:
        return False
    except Exception:
        return False


def create_order(amount_rupees: float, invoice_number: str, receipt: str = None) -> dict:
    """Creates a Razorpay Order (/v1/orders) for standard checkout."""
    if amount_rupees is None or amount_rupees <= 0:
        raise RazorpayClientError(
            f"Refusing to create order for non-positive amount: {amount_rupees}"
        )

    amount_paise = rupees_to_paise(amount_rupees)

    payload = {
        "amount": amount_paise,
        "currency": "INR",
        "receipt": receipt or f"receipt-{invoice_number}",
        "notes": {
            "invoice_number": invoice_number,
            "source": "LedgerRecoverAI",
            "flow": "standard_checkout_full_payment",
        },
    }

    try:
        order = _client.order.create(payload)
    except Exception as exc:
        raise RazorpayClientError(f"Razorpay order.create failed: {exc}") from exc

    if "id" not in order:
        raise RazorpayClientError(f"Unexpected Razorpay order response shape: {order}")

    return order


def verify_payment_signature(order_id: str, payment_id: str, signature: str) -> bool:
    """Verifies HMAC-SHA256 signature from Razorpay Checkout.js."""
    try:
        _client.utility.verify_payment_signature({
            "razorpay_order_id": order_id,
            "razorpay_payment_id": payment_id,
            "razorpay_signature": signature,
        })
        return True
    except razorpay.errors.SignatureVerificationError:
        return False
    except Exception:
        return False


def fetch_order(order_id: str) -> dict:
    """Fetch an order's live status/payments from Razorpay."""
    try:
        return _client.order.fetch(order_id)
    except Exception as exc:
        raise RazorpayClientError(f"Razorpay order.fetch failed: {exc}") from exc