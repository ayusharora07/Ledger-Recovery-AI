# LedgerRecover AI

**Autonomous B2B Trade-Credit Collections Agent** — built for the Razorpay AI Buildathon.

LedgerRecover AI is a single-tenant dashboard + AI agent that lets a wholesaler who sells on credit (the classic Indian B2B "khata"/trade-credit model) automate the entire collections lifecycle: sending invoices, chasing buyers over a WhatsApp-style chat, understanding free-text replies ("I'll pay 20k today, rest next Friday"), creating Razorpay payment links, reconciling payments, tracking promises-to-pay, escalating unresponsive accounts, and generating invoice/receipt PDFs — with a human wholesaler supervising from one dashboard and an AI "Copilot" to query the whole portfolio in plain English.

---

## 1. The Problem

Small and mid-size wholesalers extend informal credit to retailers/dealers and then spend enormous manual effort chasing payment: remembering who owes what, texting reminders, manually creating payment links, chasing broken promises, and reconciling UPI/bank payments against the right invoice. This is slow, error-prone, and doesn't scale past a handful of accounts.

**LedgerRecover AI turns that manual chase into an autonomous, auditable pipeline** — the AI handles understanding and communication, while every money-moving decision is checked against deterministic, code-enforced guardrails before anything happens to the ledger or a Razorpay payment link.

---

## 2. Core Design Principle: "The LLM Explains, It Never Originates a Number"

This is the single rule the whole backend is built around, and it's worth stating up front because it shapes almost every file in this repo:

- The LLM (Google Gemini, via `agent_engine.py`) is **only ever used to extract structured intent** from a buyer's free-text message (amount mentioned, date mentioned, dispute reason, etc.) or to turn already-computed numbers into a natural-language sentence.
- The LLM **never decides** whether a payment link should be created, never computes a balance, never sets an invoice's status, and never invents a number that isn't already sitting in the database.
- A separate, fully deterministic **validation/guardrail layer** (`validate_intent_action` in `agent_engine.py`, and `recompute_invoice_ledger` in `main.py`) takes the LLM's structured output and decides what, if anything, is actually allowed to happen — capping amounts at the real outstanding balance, blocking any action on a disputed invoice, refusing to reduce a balance except from a verified Razorpay payment, etc.

This means a hallucinated or malformed LLM response can, at worst, cause the agent to ask a clarifying question or take no action — it can never make up money that isn't there.

---

## 3. Feature Overview

| Area | What it does |
|---|---|
| **WhatsApp-style buyer chat simulator** | Free-text messages are classified into an intent (partial payment, full payment, promise-to-pay, dispute, already-paid, payment proof, payment plan request, multi-invoice payment, greeting, etc.) and mapped to a safe action. |
| **Razorpay Payment Links** | Auto-generated for any amount the guardrail layer approves, with expiry, stale-link cancellation (so an old link can't be paid twice), and reuse of an already-open link for the same amount (to conserve Razorpay's test-mode link quota). |
| **Razorpay Standard Checkout** | Order-based full-balance checkout flow (`create_order` / `verify_payment_signature`) as an alternative to Payment Links — see [`INTEGRATION_NOTES.md`](./INTEGRATION_NOTES.md). |
| **Webhook + polling reconciliation** | `/webhook/razorpay` verifies Razorpay's HMAC signature and reconciles captured payments in real time; a background poller (`poll_outstanding_payment_links`) double-checks link status every few seconds as a safety net if a webhook is missed. |
| **Collections engine** | A persistent, per-firm `CollectionCase` state machine (NOT_DUE → DUE → OVERDUE → PROMISED → PARTIALLY_PAID / DISPUTED / ESCALATED → PAID), a deterministic priority score for the "who to chase first" queue, and a promise-to-pay lifecycle (ACTIVE → FULFILLED / BROKEN / CANCELLED). |
| **Autonomous follow-up cadence** | A database-backed action scheduler (`CollectionAction` + `poll_collection_actions`) fires promise-date follow-ups, broken-promise follow-ups, and a 72-hour overdue cadence — with a hard stop that escalates an account to a human after 4 unanswered contact attempts in 7 days, instead of nagging forever. |
| **Collections Copilot** | A second, portfolio-level chat assistant for the *wholesaler* ("which firms haven't paid", "send a reminder to everyone overdue", "give me the PDF for Singh Fabrics"). Intent is classified by the LLM; every number/list/action is computed and executed by deterministic code, with a typo-tolerant keyword fallback if the LLM classifier fails or returns UNKNOWN. |
| **AI risk insights** | A rule-based (non-LLM) risk classifier (LOW/MEDIUM/HIGH, from real overdue days, broken promises, and disputes) feeds a short natural-language narrative generated by Gemini — the LLM is explicitly forbidden from inventing or restating any figure not in the input JSON. |
| **PDF generation** | Dynamically generated invoice PDFs, zero-balance clearance receipts (kept for GST/audit purposes), and combined multi-invoice PDFs — all built with ReportLab, no external service. |
| **Single-tenant auth** | One shared wholesaler password, HMAC-signed expiring session tokens (12h TTL), sent as a Bearer token or a query-param fallback for plain `<a href>` PDF download links. |
| **Full audit trail** | Every inbound message, LLM decision, and system action is written to an `AuditLog` row with the raw payload — the buyer-facing chat thread itself is *reconstructed* from this audit log, not stored separately, so the UI and the audit trail can never drift apart. |

---

## 4. Architecture

```
┌─────────────────────────┐
│   index.html (frontend) │  Tailwind + vanilla JS dashboard
│  - Ledger / firm view    │  - Razorpay Checkout.js + Chart.js
│  - Collection queue      │
│  - Buyer chat simulator  │
│  - Collections Copilot   │
└────────────┬─────────────┘
             │ REST (Bearer/session token)
┌────────────▼─────────────────────────────────────────────┐
│                     main.py  (FastAPI)                    │
│  - auth (login, session tokens)                            │
│  - deterministic ledger recomputation                      │
│  - payment-link lifecycle (create/reuse/cancel/expire)     │
│  - Razorpay webhook + poller reconciliation                │
│  - background collection-action poller                     │
│  - firm analysis / risk scoring / Copilot orchestration     │
│  - PDF endpoints                                            │
└──────┬───────────────┬───────────────┬──────────────┬─────┘
       │               │               │              │
┌──────▼──────┐ ┌──────▼───────┐ ┌─────▼──────┐ ┌─────▼──────┐
│agent_engine │ │collection_   │ │razorpay_   │ │invoice_pdf │
│.py          │ │engine.py     │ │client.py   │ │.py         │
│             │ │              │ │            │ │            │
│Gemini-based │ │Case/Promise/ │ │Payment     │ │ReportLab   │
│intent       │ │Action state  │ │Links,      │ │invoice /   │
│extraction + │ │machine +     │ │Orders,     │ │receipt /   │
│deterministic│ │priority      │ │signature   │ │combined    │
│guardrails   │ │scoring       │ │verification│ │PDFs        │
└─────────────┘ └──────────────┘ └────────────┘ └────────────┘
       │               │               │
       └───────┬───────┴───────┬───────┘
               │                │
        ┌──────▼────────────────▼──────┐
        │   models.py (SQLAlchemy ORM)  │
        │ Client, Invoice, PaymentRecord│
        │ PaymentAllocation, AuditLog,  │
        │ CollectionCase, PaymentPromise│
        │ CollectionAction, Communication│
        └──────────────┬────────────────┘
                        │
                ┌───────▼────────┐
                │  database.py    │  SQLAlchemy engine/session
                │  (SQLite file /  │  (ledger.db by default,
                │   Postgres URL)  │   Postgres via DATABASE_URL)
                └─────────────────┘
```

### Request flow — a buyer message end to end
1. Frontend posts the buyer's free-text message to `/api/simulate-message`.
2. `main.py` resolves which invoice the message applies to (`resolve_target_invoice` — the oldest unpaid invoice for a firm-level conversation) and loads recent chat history for context.
3. `agent_engine.process_buyer_message()`:
   - Calls Gemini with a strict, few-shot system prompt and a Pydantic response schema (`ExtractedIntent`) — the model can *only* return one of a fixed set of intents plus a few typed fields (amount, date, invoice number, dispute reason, etc.).
   - Passes that structured result to `validate_intent_action()`, pure Python, which applies the actual business rules (cap partial payments at the real balance, block disputed invoices, require full-balance for FULL_PAYMENT, etc.) and returns an `ActionType`.
4. `main.py` executes the approved action: creates/reuses a Razorpay payment link, schedules a reminder (`CollectionAction`/timer), flags a dispute, or asks for clarification — and writes everything to `AuditLog`.
5. Payment capture happens later, asynchronously, via the Razorpay webhook (or the background poller as a fallback) — never from the buyer's chat message itself. `apply_captured_payment()` is the single place that ever increments `paid_amount`, and `recompute_invoice_ledger()` is the single place that ever derives `balance_amount`/`status` from it.

### Request flow — the wholesaler's Collections Copilot
Same "LLM classifies, code computes" split: `classify_copilot_message()` returns an intent + scoping parameters (e.g. "remind everyone due within 3 days"); the actual firm list, totals, and reminder-sending are all done by deterministic SQLAlchemy queries in `main.py`, with a fuzzy/keyword fallback (`_deterministic_copilot_fallback`) if the LLM call fails or returns `UNKNOWN`.

---

## 5. Data Model (`models.py`)

- **Client** — a buyer firm (business name, contact name, phone number).
- **Invoice** — one bill. `balance_amount` is a derived, application-only field — it is *never* set directly, only recomputed from `total_amount - paid_amount`.
- **PaymentRecord** / **PaymentAllocation** — a captured Razorpay payment, and how it's split across invoices for a combined "pay all" transaction (so one Razorpay payment never gets double-counted).
- **AuditLog** — append-only record of every inbound message, LLM decision, and system action; also the source of truth the chat UI is reconstructed from.
- **CollectionCase** — one row per client: current lifecycle status, priority score, next scheduled action, autonomy toggle, escalation reason.
- **PaymentPromise** — a buyer's promise to pay a (possibly unspecified) amount by a given date; lifecycle `ACTIVE → FULFILLED/BROKEN/CANCELLED`, reconciled against real payments.
- **CollectionAction** — a scheduled future action (follow-up, broken-promise check, overdue cadence) with a `dedupe_key` so the same follow-up can't be double-scheduled.
- **Communication** — every outbound/inbound message logged for the case timeline (`last_contact_at` / `last_customer_response_at`).

---

## 6. Tech Stack

- **Backend:** FastAPI, SQLAlchemy 2.0, Pydantic
- **Database:** SQLite by default (`ledger.db`), swappable to Postgres via `DATABASE_URL` (`psycopg2-binary` included)
- **AI:** Google Gemini (`google-genai` SDK) for structured intent extraction and narrative generation, with typed Pydantic response schemas throughout
- **Payments:** Razorpay Python SDK — Payment Links, Orders (Standard Checkout), webhooks
- **PDF generation:** ReportLab
- **Frontend:** Single-page dashboard in plain HTML/JS + Tailwind (CDN) + Chart.js, no build step
- **Auth:** HMAC-SHA256-signed session tokens, no external auth provider

---

## 7. Getting Started

### Prerequisites
- Python 3.11+
- A Razorpay **test-mode** account (Key ID + Key Secret + a webhook secret)
- A Google Gemini API key

### Setup
```bash
git clone <this-repo-url>
cd ledgerrecover-ai
pip install -r requirements.txt
```

Create a `.env` file:
```env
GEMINI_API_KEY=your_gemini_api_key
RAZORPAY_KEY_ID=rzp_test_xxxxxxxx
RAZORPAY_KEY_SECRET=your_razorpay_key_secret
RAZORPAY_WEBHOOK_SECRET=your_razorpay_webhook_secret
ADMIN_PASSWORD=choose_a_dashboard_password
SESSION_SECRET=any_long_random_string
# Optional — defaults to sqlite:///./ledger.db
DATABASE_URL=postgresql://user:pass@host:5432/dbname
```

Run:
```bash
uvicorn main:app --reload
```

Open `http://localhost:8000`, log in with `ADMIN_PASSWORD`, and use **+ Onboard Invoice** to create your first test firm/invoice. Use the buyer chat simulator to send yourself test messages ("I'll pay 5000 today", "remind me Friday", "this bill is wrong") and watch the agent's decisions land in the Audit Log.

> For local testing, forward Razorpay webhooks to `/webhook/razorpay` with a tool like `ngrok`, and register the webhook secret in both Razorpay's dashboard and your `.env`.

---

## 8. Key API Endpoints

| Method & Path | Purpose |
|---|---|
| `POST /api/auth/login` | Wholesaler login → session token |
| `POST /api/simulate-message` | Buyer message → AI intent → guardrailed action |
| `POST /api/copilot/message` | Wholesaler portfolio-level chat |
| `GET /api/collections/queue` | Prioritized "who to chase next" list |
| `GET /api/collections/summary` | Portfolio metrics |
| `POST /api/collections/cases/{client_id}/autonomy` | Toggle autonomous follow-ups per firm |
| `GET /api/firms` / `GET /api/firms/{id}/invoices` | Ledger data |
| `GET /api/firms/{id}/insights` / `/analysis` | AI risk narrative / full trend report |
| `POST /api/onboard-client` | Create a firm + first invoice |
| `POST /webhook/razorpay` | Signed payment webhook |
| `POST /api/sync-payment-link/{invoice_id}` | Manual on-demand reconciliation |
| `GET /api/invoices/{id}/pdf` / `/receipt-pdf` | Invoice / receipt PDF |
| `GET /api/firms/{id}/invoices/combined-pdf` | Combined multi-invoice PDF |
| `GET /health` | Liveness check |

---

## 9. Notable Engineering Decisions

- **Deterministic ledger, LLM-free money math** — see Section 2. This is the safety backbone of the whole app.
- **Idempotent payment reconciliation** — every captured payment is keyed by `razorpay_payment_id`; webhook, poller, and manual sync can all fire on the same payment without double-crediting an invoice.
- **Payment link reuse & cancellation** — Razorpay test-mode accounts have a hard cap on links ever created, so an open link for the same amount is reused rather than reissued, and superseded links are actively cancelled so a buyer can't pay a stale link twice.
- **Sticky escalation state** — once a case is `ESCALATED`, routine queue refreshes cannot silently downgrade it back to an automatic state; a human has to resolve the underlying issue.
- **Crash-safe reminders** — promise/overdue follow-ups are persisted as `CollectionAction` rows (not just in-memory timers), and `requeue_pending_reminders()` re-arms anything pending at startup, firing immediately if the scheduled time already passed while the server was down.
- **Typo-tolerant fallbacks** — both the buyer-facing agent and the wholesaler Copilot have a deterministic, fuzzy-matching fallback path for when the LLM call fails or returns `UNKNOWN`, so a single bad model response can't take down the whole feature.

---

## 10. Known Limitations / Work in Progress

- **In-process scheduling:** reminder timers and background pollers run inside the single FastAPI process (`threading.Timer` / `asyncio` tasks) — this is fine for a single-instance deployment but wouldn't survive horizontal scaling without moving to an external job queue.
- **Single-tenant auth:** one shared wholesaler password rather than per-user accounts — intentional for this stage, but the natural next step for a multi-wholesaler product.
- **Standard Checkout integration:** the Orders-based full-payment flow described in [`INTEGRATION_NOTES.md`](./INTEGRATION_NOTES.md) (`/api/create-order`, `/api/verify-payment`) is implemented in `razorpay_client.py`; wiring the corresponding FastAPI routes into `main.py` and the "Pay Now" button in `index.html` is the next integration step.
- **WhatsApp simulator, not live WhatsApp:** all buyer messaging in this build goes through the in-dashboard chat simulator (`channel="SIMULATOR"`) rather than a live WhatsApp Business API — the `Communication`/`AuditLog` model is already channel-aware, so wiring a real provider is additive, not a rewrite.

---

## 11. Repository Structure

```
main.py                 FastAPI app: routes, auth, ledger logic, reconciliation, background pollers
agent_engine.py          Gemini-based intent extraction + deterministic guardrail validation
collection_engine.py     Collection case/promise/action state machine + priority scoring
razorpay_client.py       Razorpay Payment Links, Orders, webhook & signature verification
invoice_pdf.py           ReportLab invoice / receipt / combined PDF generation
models.py                SQLAlchemy ORM models
database.py              Engine/session setup (SQLite by default, Postgres-ready)
index.html               Single-page wholesaler dashboard (Tailwind + vanilla JS)
requirements.txt         Python dependencies
INTEGRATION_NOTES.md     Notes on the Razorpay Standard Checkout integration
```

---

## 12. Credits

Built for the **Razorpay AI Buildathon**. Payments powered by **Razorpay**; intent understanding and narrative generation powered by **Google Gemini**.
