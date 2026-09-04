# Razorpay Standard Checkout integration

## Backend

Added:
- `POST /api/create-order`
- `POST /api/verify-payment`

`RAZORPAY_KEY_SECRET` is server-only. Signature verification uses HMAC-SHA256 with
`order_id + "|" + payment_id` and constant-time comparison.

The existing Payment Link and webhook functionality is preserved.

## Frontend

`razorpay_checkout_snippet.html` contains the Standard Checkout implementation.
Because the original `templates/index.html` was not uploaded, it has not been
overwritten. Paste/adapt the snippet into the existing dashboard template.

For the real dashboard, set the button's `data-amount-paise` from the invoice
balance and `data-receipt` from the invoice number.

In `main.py`, expose the public key to the template by adding
`"razorpay_key_id": os.getenv("RAZORPAY_KEY_ID")` to the TemplateResponse context,
or use a dedicated context variable. Never expose `RAZORPAY_KEY_SECRET`.

## Run

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open the dashboard and click Pay Now.

The current execution environment could not download packages from PyPI, so
dependency installation was declared/verified but not performed here.
