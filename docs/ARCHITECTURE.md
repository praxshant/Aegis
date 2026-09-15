# Architecture

Aegis is built around a unidirectional data flow that separates reasoning from execution.

```text
REAL TRANSACTION DATA (Database)
        ↓
FEATURE BUILDER (19-feature contract, strict observation_date)
        ↓
Aegis HistGradientBoostingClassifier (Runtime Inference)
        ↓
INTELLIGENCE ENGINE (Behavioral Signals & Policy Check)
        ↓
CUSTOMER INTELLIGENCE OBJECT (COMPLETE/INCOMPLETE)
        ↓
AEGIS AGENT (Gemini LLM)
        ↓
STRUCTURED ACTION PROPOSAL (Pydantic)
        ↓
POLICY GATE (Deterministic Rules)
        ↓
RAZORPAY ADAPTER (Test API)
        ↓
EXECUTION / AUDIT
        ↓
RECONCILIATION LAYER
```

## 1. Intelligence Engine & Model Layer
Loads the frozen Aegis `HistGradientBoosting_Tuned.pkl` artifact. Constructs a strict 19-feature vector at runtime from raw transactions ensuring no temporal data leakage (`transaction_timestamp <= observation_date`). Generates canonical behavioral features (e.g., recency ratio, declining spend flags) deterministically. If model inference fails, status is `INCOMPLETE` and the system halts financial actions.

## 2. Aegis Agent
A strictly prompted Gemini LLM that reads the Intelligence Object and outputs a JSON `ActionProposal`. It cannot execute anything itself.

## 3. Policy Gate
Evaluates the `ActionProposal` against `config/policy_rules.json`. For example, if `requested_discount_pct` exceeds `MAX_DISCOUNT_PCT`, the request is immediately rejected.

## 4. Execution & Audit
Approved requests pass to the `RazorpayAdapter`, and every state transition is recorded in the immutable `AuditLedger`.
