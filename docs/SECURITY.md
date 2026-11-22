# Security & Threat Model

Aegis operates under the assumption that the LLM is **UNTRUSTED**. 

## 1. LLM Trust Boundary
- The LLM is isolated within `AegisAgent`.
- It cannot make HTTP requests.
- It cannot access database schemas or execute raw SQL.
- It cannot access Razorpay API keys.
- Its only output is a strictly typed JSON string mapped to a Pydantic model (`ActionProposal`).

## 2. Policy Enforcement
- The Policy Gate is written in deterministic Python code.
- Configuration (`policy_rules.json`) cannot be mutated by the LLM.
- Financial safety limits (e.g., maximum discount) are hard-enforced.

## 3. Secret Handling
- No secrets are hardcoded in the repository.
- Authentication tokens and API keys are strictly loaded via environment variables at runtime.

## 4. Idempotency & Failure Handling
- Razorpay payments are generated with unique `reference_id`s tied to the Audit Ledger's `action_id` to prevent duplicate billing.
- The velocity/budget/cooldown check and the execution run under a per-customer lock, so concurrent requests cannot both pass the gate (single API process; run one worker).
- The Reconciliation layer resolves unknown states by polling the Razorpay API (`/aegis/reconcile`) or from signed webhooks (`/webhooks/razorpay`, HMAC-SHA256 over the raw body with `RAZORPAY_WEBHOOK_SECRET`).
- A `rzp_live_` key is refused unless `RAZORPAY_ENV=live`; LLM-written link descriptions are clipped.

## 5. Operator Access
- JWT bearer tokens (HS256, `JWT_SECRET_KEY`); refresh tokens can only mint access tokens for active users.
- Spend and state-changing endpoints require role `admin` or `operator`; `viewer` is read-only.
- Logins are throttled (5 failures per username per 5 minutes → HTTP 429); bcrypt's 72-byte limit is enforced on set and treated as a failed login on check.
- Read endpoints (`/customers`, `/aegis/intelligence`, `/aegis/audit`, analytics) are unauthenticated by design for the dashboard. Do not load real customer PII into a publicly reachable deployment without putting these behind auth.

## 6. Audit Ledger Integrity
- `audit_ledger` is append-only: SQLite triggers abort any `UPDATE` or `DELETE`.
- Each event stores `prev_hash` and `event_hash = SHA-256(prev_hash + canonical event JSON)`. `GET /aegis/ledger/verify` recomputes the chain and reports the first event whose content or position was altered (e.g. by someone with raw file access who dropped the triggers).

## 7. Uploads
- CSV uploads are written under a random server-side name in `artifacts/uploads/` (client filenames never reach the filesystem path), capped by `MAX_UPLOAD_MB`, and deleted after ingestion.
