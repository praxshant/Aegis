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
- The Reconciliation layer ensures that unknown states (e.g., network timeout after request) are securely resolved by polling the Razorpay API before retrying.
