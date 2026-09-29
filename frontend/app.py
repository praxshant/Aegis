import streamlit as st
import requests
import pandas as pd
import json
import time

st.set_page_config(
    page_title="Aegis | Governed AI Commerce Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000"

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .header-badge {
        display: inline-block;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 8px;
        padding: 4px 12px;
        font-size: 0.75rem;
        color: #4cc9f0;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }

    .intel-card {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .intel-row {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid #1e3a5f;
        font-size: 0.875rem;
    }
    .intel-label { color: #8892a4; }
    .intel-value { color: #e2e8f0; font-weight: 500; }
    .intel-value-green { color: #4ade80; font-weight: 600; }
    .intel-value-yellow { color: #fbbf24; font-weight: 600; }
    .intel-value-red { color: #f87171; font-weight: 600; }

    .stage-box {
        border-radius: 12px;
        padding: 16px;
        height: 100%;
        min-height: 180px;
    }
    .stage-pending  { background: #1e293b; border: 1px solid #334155; }
    .stage-success  { background: #052e16; border: 1px solid #15803d; }
    .stage-denied   { background: #2d0000; border: 1px solid #b91c1c; }
    .stage-failed   { background: #2d0000; border: 1px solid #b91c1c; }
    .stage-inactive { background: #111827; border: 1px dashed #374151; opacity: 0.6; }

    .stage-title { font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em; margin-bottom: 8px; }
    .stage-verdict { font-size: 1.35rem; font-weight: 700; margin: 8px 0; }
    .stage-detail { font-size: 0.8rem; color: #94a3b8; line-height: 1.5; }

    .policy-banner-approved {
        background: linear-gradient(90deg, #052e16, #064e3b);
        border: 2px solid #16a34a;
        border-radius: 10px;
        padding: 16px 24px;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 700;
        color: #4ade80;
        letter-spacing: 0.08em;
        margin: 12px 0;
    }
    .policy-banner-denied {
        background: linear-gradient(90deg, #2d0000, #450a0a);
        border: 2px solid #dc2626;
        border-radius: 10px;
        padding: 16px 24px;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 700;
        color: #f87171;
        letter-spacing: 0.08em;
        margin: 12px 0;
    }

    .audit-row {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 6px 0;
        font-size: 0.82rem;
        border-bottom: 1px solid #1e293b;
    }
    .audit-icon { font-size: 1rem; width: 20px; }
    .audit-event { color: #94a3b8; }

    .razorpay-link {
        display: inline-block;
        background: linear-gradient(135deg, #0ea5e9, #0284c7);
        color: white !important;
        font-weight: 700;
        padding: 10px 24px;
        border-radius: 8px;
        text-decoration: none !important;
        margin-top: 12px;
        font-size: 0.9rem;
        letter-spacing: 0.03em;
    }

    .metric-big { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.75rem; color: #64748b; letter-spacing: 0.08em; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Helper: fetch customers ───────────────────────────────────────────────────
@st.cache_data(ttl=60)
def get_customers():
    try:
        res = requests.get(f"{API_BASE}/customers", timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []

@st.cache_data(ttl=10)
def api_alive():
    try:
        return requests.get(f"{API_BASE}/health", timeout=3).status_code == 200
    except Exception:
        return False

customers = get_customers()

def auth_headers():
    """Bearer header for spend/state-changing endpoints; empty until the operator signs in."""
    tok = st.session_state.get("access_token")
    return {"Authorization": f"Bearer {tok}"} if tok else {}

# ── Header ────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 9])
with col_logo:
    st.markdown("<div style='font-size:3rem; padding-top:8px'>🛡️</div>", unsafe_allow_html=True)
with col_title:
    st.markdown('<div class="header-badge">GOVERNED COMMERCE · ZERO-TRUST AGENT</div>', unsafe_allow_html=True)
    st.markdown("## Aegis: Governed AI Commerce Agent")
    st.markdown("_The LLM recommends. The policy gate decides. Razorpay only executes on deterministic approval._")

st.divider()

if not api_alive():
    st.error("⚠️ API is unreachable. Start with: `uvicorn backend.api.main:app --port 8000`")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🔐 Operator")
if st.session_state.get("access_token"):
    st.sidebar.success(f"Signed in as **{st.session_state.get('operator', 'operator')}**")
    if st.sidebar.button("Sign out", use_container_width=True):
        st.session_state.pop("access_token", None)
        st.session_state.pop("operator", None)
        st.rerun()
else:
    with st.sidebar.form("login_form"):
        _u = st.text_input("Username")
        _p = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", use_container_width=True):
            try:
                r = requests.post(f"{API_BASE}/auth/login", json={"username": _u, "password": _p}, timeout=10)
                if r.status_code == 200:
                    st.session_state["access_token"] = r.json()["access_token"]
                    st.session_state["operator"] = _u
                    st.rerun()
                else:
                    st.error("Invalid credentials")
            except Exception as e:
                st.error(f"Login failed: {e}")

# ── Data Ingestion ──────────────────────────────────────────────────────────
st.sidebar.markdown("---")
with st.sidebar.expander("📥 Data Ingestion", expanded=not customers):
    if not st.session_state.get("access_token"):
        st.caption("🔒 Sign in to ingest a dataset.")
    else:
        up = st.file_uploader("Customer/transaction CSV", type="csv", key="ingest_csv")
        mode = st.selectbox("Activation mode", ["replace", "merge", "new"], index=0,
                            help="replace: archive current data and swap in this run; merge/new: keep existing")
        if st.button("Ingest & activate", use_container_width=True, disabled=up is None):
            try:
                files = {"file": (up.name, up.getvalue(), "text/csv")}
                r = requests.post(f"{API_BASE}/aegis/ingest/csv", files=files,
                                  headers=auth_headers(), timeout=60)
                if r.status_code == 401:
                    st.session_state.pop("access_token", None)
                    st.warning("Session expired — sign in again.")
                    st.stop()
                if r.status_code != 200:
                    st.error(f"Ingest failed: {r.text}")
                    st.stop()
                run_id = r.json()["run_id"]
                prog = st.progress(0, text="Ingesting…")
                status = {}
                for _ in range(120):
                    status = requests.get(f"{API_BASE}/aegis/ingest/status/{run_id}", timeout=10).json()
                    pct = int(status.get("progress_percentage") or 0)
                    prog.progress(min(pct, 100),
                                  text=f"{status.get('status', '')} — {status.get('processed_rows', 0)} rows")
                    state = str(status.get("status", ""))
                    if state == "COMPLETED":
                        break
                    if state.startswith("FAILED"):
                        st.error(f"Ingestion failed: {status}")
                        st.stop()
                    time.sleep(1.0)
                else:
                    st.error("Ingestion timed out.")
                    st.stop()
                act = requests.post(f"{API_BASE}/datasets/activate",
                                    json={"run_id": run_id, "mode": mode},
                                    headers=auth_headers(), timeout=60)
                if act.status_code != 200:
                    st.error(f"Activation failed: {act.text}")
                    st.stop()
                get_customers.clear()
                st.success(f"Ingested {status.get('valid_rows', '?')} valid rows.")
                st.rerun()
            except Exception as e:
                st.error(f"Ingestion error: {e}")

if not customers:
    st.info("No customers loaded yet. Sign in and ingest a dataset from the sidebar to begin.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("## 🎛️ Agent Controls")

def on_customer_change():
    st.session_state.pop("last_intel", None)
    st.session_state.pop("last_result", None)

customer_map = {c["customer_id"]: f"{c['customer_id']} — {c['first_name']} {c['last_name']}" for c in customers}
selected_id = st.sidebar.selectbox("Select Customer", options=list(customer_map.keys()),
                                   format_func=lambda x: customer_map[x], on_change=on_customer_change)

cust_info = next((c for c in customers if c["customer_id"] == selected_id), {})

st.sidebar.markdown("---")
default_prompt = "Assess this customer for a win-back reactivation offer. Be evidence-based and conservative."
st.sidebar.text_area("Agent instruction (editable)", value=default_prompt, key="prompt_box", height=110)
final_prompt = st.session_state.get("prompt_box", default_prompt)

st.sidebar.markdown("---")

# ML Intelligence inspect
if st.sidebar.button("🔬 Inspect ML Intelligence", use_container_width=True):
    with st.spinner("Running Aegis model inference..."):
        try:
            res = requests.get(f"{API_BASE}/aegis/intelligence/{selected_id}", timeout=30)
            st.session_state["last_intel"] = res.json() if res.status_code == 200 else {"error": res.text}
        except Exception as e:
            st.session_state["last_intel"] = {"error": str(e)}

if "last_intel" in st.session_state:
    intel = st.session_state["last_intel"]
    if "error" not in intel:
        s = intel.get("status", "")
        icon = "✅" if s == "COMPLETE" else "🔴"
        st.sidebar.markdown(f"**Status:** {icon} `{s}`")
        st.sidebar.markdown(f"**P(buy 30d):** `{intel.get('purchase_probability_30d', 'N/A')}`")
        st.sidebar.markdown(f"**Band:** `{intel.get('propensity_band', 'N/A')}`")
        st.sidebar.markdown(f"**Segment:** `{intel.get('segment', 'N/A')}`")
        st.sidebar.markdown(f"**Policy Rec:** `{intel.get('recommendation', 'N/A')}`")
        if intel.get("incomplete_reason"):
            st.sidebar.warning(f"⚠️ {intel['incomplete_reason']}")

st.sidebar.markdown("---")
run_clicked = st.sidebar.button("🚀 Run Aegis Agent", type="primary", use_container_width=True,
                                disabled=not st.session_state.get("access_token"))
if not st.session_state.get("access_token"):
    st.sidebar.caption("🔒 Sign in to authorize an agent run.")

# ── Run Agent ─────────────────────────────────────────────────────────────────
if run_clicked:
    st.session_state.pop("last_result", None)
    with st.spinner("Aegis is reasoning over evidence — Gemini → Policy Gate → Razorpay..."):
        try:
            res = requests.post(
                f"{API_BASE}/aegis/reactivate",
                json={"customer_id": selected_id, "prompt": final_prompt},
                headers=auth_headers(),
                timeout=120,
            )
            if res.status_code == 401:
                st.session_state.pop("access_token", None)
                st.warning("Session expired — please sign in again.")
                st.stop()
            data = res.json()
            data["_http_status"] = res.status_code
            st.session_state["last_result"] = data
        except Exception as e:
            st.session_state["last_result"] = {"status": "CONNECTION_ERROR", "_http_status": 0, "error": str(e)}

# ── Customer Intelligence Panel ───────────────────────────────────────────────
if "last_intel" in st.session_state and "error" not in st.session_state["last_intel"]:
    intel = st.session_state["last_intel"]
    fs = intel.get("feature_snapshot", {})

    st.markdown("### 🧠 Customer Intelligence")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown('<div class="metric-label">RECENCY</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-big">{fs.get("recency_days", "—")}</div>', unsafe_allow_html=True)
        st.caption("days since last purchase")
    with c2:
        st.markdown('<div class="metric-label">LIFETIME SPEND</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-big">₹{fs.get("total_net_spend", 0):,.0f}</div>', unsafe_allow_html=True)
        st.caption("cumulative net spend")
    with c3:
        st.markdown('<div class="metric-label">ORDERS</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-big">{fs.get("purchase_order_count", "—")}</div>', unsafe_allow_html=True)
        st.caption("total purchase orders")
    with c4:
        st.markdown('<div class="metric-label">PROPENSITY (ANY ITEM)</div>', unsafe_allow_html=True)
        score = intel.get("purchase_probability_30d", None)
        color_cls = "intel-value-green" if score and score > 0.4 else ("intel-value-yellow" if score and score > 0.2 else "intel-value-red")
        st.markdown(f'<div class="metric-big" style="color: var(--text-color)">{score:.4f}</div>' if score else '<div class="metric-big">—</div>', unsafe_allow_html=True)
        st.caption(f"band: {intel.get('propensity_band', '—')}")
    with c5:
        st.markdown('<div class="metric-label">SEGMENT</div>', unsafe_allow_html=True)
        seg = intel.get("segment", "Unknown")
        st.markdown(f'<div class="metric-big" style="font-size:1.1rem; padding-top:8px">{seg}</div>', unsafe_allow_html=True)
        st.caption(f"policy rec: {intel.get('recommendation', '—')}")
        
    st.markdown("---")
    
    # Next Basket Intelligence
    nb = intel.get("next_basket", {})
    if nb.get("available"):
        st.markdown("### 🛒 Next Basket Intelligence")
        st.markdown(f"<small>Model: `{nb.get('model')} v{nb.get('model_version')}` | Specific Item Signal: `{nb.get('signal')}`</small>", unsafe_allow_html=True)
        
        cols = st.columns([2, 1])
        with cols[0]:
            st.markdown("**Predicted Next Purchase**")
            for i, p in enumerate(nb.get("top_products", [])):
                prob = p.get('probability', 0)
                bar_len = int(prob * 10)
                bar = "█" * bar_len + "▒" * (10 - bar_len)
                st.markdown(f"`{i+1}.` {p.get('product_name')} ` Score: {prob:.2f} ` <span style='color:#00adb5'>{bar}</span>", unsafe_allow_html=True)
                
        with cols[1]:
            st.markdown("**Predicted Departments**")
            for d in nb.get("top_departments", []):
                st.markdown(f"- `{d}`")
            
        with st.expander("Model Provenance"):
            st.code(
                "Algorithm: LightGBM\n"
                "Features: 21 (Leakage Corrected)\n"
                "NDCG@10: 0.524 (holdout)\n"
                "SHA-256: 238a44c50b10d825c2417ae8e278465d9366afa1651bc826cd1b2d81d0e4ea9c",
                language="text"
            )
    else:
        st.markdown("### 🛒 Next Basket Intelligence")
        st.warning(f"⚠️ Next-basket intelligence unavailable: {nb.get('unavailable_reason', 'Not loaded')}")
        
    st.markdown("<br><div style='text-align: center; color: #666;'>⬇ ML INSIGHTS PASSED TO LLM AGENT ⬇</div><br>", unsafe_allow_html=True)

    # Active signals
    sigs = {k: v for k, v in intel.get("behavioral_signals", {}).items() if v}
    if sigs:
        st.markdown("**🚨 Active Signals:** " + " · ".join([f"`{s}`" for s in sigs]))

    st.divider()

# ── Execution Pipeline ────────────────────────────────────────────────────────
if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    http_status = result.get("_http_status", 0)
    status = result.get("status", "UNKNOWN")

    if http_status == 503 or status in ("CONNECTION_ERROR", "AGENT_FAILED"):
        stage = "AGENT_FAILED"
    elif http_status in (422,):
        stage = "INTELLIGENCE_FAILED"
    elif http_status == 404:
        stage = "NOT_FOUND"
    elif status == "DENIED":
        stage = "POLICY_DENIED"
    elif status == "EXECUTED":
        stage = "EXECUTED"
    elif status == "FAILED":
        stage = "EXECUTION_FAILED"
    else:
        stage = "UNKNOWN"

    st.markdown("### ⚙️ Execution Pipeline")

    # ── Policy banner (most important visual element) ─────────────────────
    if stage == "POLICY_DENIED":
        proposal = result.get("proposal", {})
        disc = proposal.get("requested_discount_pct", "?")
        st.markdown(
            f'<div class="policy-banner-denied">'
            f'🚫 POLICY GATE: DENIED — Gemini requested {disc}% discount. Maximum permitted: 25%. '
            f'Razorpay was never called.'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif stage == "EXECUTED":
        proposal = result.get("proposal", {})
        disc = proposal.get("requested_discount_pct", "?")
        st.markdown(
            f'<div class="policy-banner-approved">'
            f'✅ POLICY GATE: APPROVED — {disc}% discount within policy. Payment link created.'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Four pipeline stages ──────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    # Stage 1: AI Proposal
    with col1:
        if stage == "AGENT_FAILED":
            st.error("**🤖 1. AI Proposal**\n\n❌ FAILED")
            err = result.get("detail") or result.get("error", "Gemini unavailable")
            st.caption(f"Reason: {err[:200]}")
        elif stage == "INTELLIGENCE_FAILED":
            st.error("**🤖 1. AI Proposal**\n\n❌ INTELLIGENCE INCOMPLETE")
            st.caption(result.get("detail", "Cannot build propensity features."))
        elif stage in ("POLICY_DENIED", "EXECUTED", "EXECUTION_FAILED"):
            proposal = result.get("proposal", {})
            st.success("**🤖 1. AI Proposal**\n\n✅ GENERATED")
            st.markdown(f"**Intent:** `{proposal.get('intent', '—')}`")
            st.markdown(f"**Discount:** `{proposal.get('requested_discount_pct', '—')}%`")
            st.markdown(f"**Agent Confidence:** `{proposal.get('confidence', '—')}`")
            with st.expander("Reasoning"):
                st.write(proposal.get("reasoning_summary", "—"))
        else:
            st.info("**🤖 1. AI Proposal**\n\n⏸ NOT REACHED")

    # Stage 2: Policy Gate
    with col2:
        if stage == "AGENT_FAILED":
            st.info("**🔒 2. Policy Gate**\n\n⏸ NOT REACHED")
            st.caption("Agent must succeed first.")
        elif stage == "POLICY_DENIED":
            proposal = result.get("proposal", {})
            st.error("**🔒 2. Policy Gate**\n\n🚫 DENIED")
            st.markdown(f"Requested: **{proposal.get('requested_discount_pct', '?')}%**")
            st.markdown(f"Maximum: **25%**")
            
            intel_conf = intel.get('recommendation_confidence', 'Unknown') if 'last_intel' in st.session_state else 'Unknown'
            st.markdown(f"Evidence Confidence: **{intel_conf}**")
            st.caption(result.get("reason", "Policy rule violated"))
        elif stage in ("EXECUTED", "EXECUTION_FAILED"):
            st.success("**🔒 2. Policy Gate**\n\n✅ APPROVED")
            
            intel_conf = intel.get('recommendation_confidence', 'Unknown') if 'last_intel' in st.session_state else 'Unknown'
            st.markdown(f"Evidence Confidence: **{intel_conf}**")
            st.caption("Discount within bounds. Confidence met.")
        else:
            st.info("**🔒 2. Policy Gate**\n\n⏸ NOT REACHED")

    # Stage 3: Razorpay Execution
    with col3:
        if stage in ("AGENT_FAILED", "INTELLIGENCE_FAILED", "NOT_FOUND"):
            st.info("**💸 3. Execution**\n\n⏸ NOT REACHED")
            st.caption("Razorpay is never called without an approved proposal.")
        elif stage == "POLICY_DENIED":
            st.warning("**💸 3. Execution**\n\n🛑 BLOCKED")
            st.caption("**Razorpay was never called.** The LLM cannot authorize payments.")
        elif stage == "EXECUTED":
            rzp = result.get("razorpay", {})
            st.success("**💸 3. Execution**\n\n✅ PAYMENT LINK CREATED")
            amount_inr = rzp.get("amount", 0) / 100
            st.markdown(f"**Amount:** ₹{amount_inr:,.2f} INR")
            short_url = rzp.get("short_url", "")
            if short_url:
                st.markdown(f'<a href="{short_url}" target="_blank" class="razorpay-link">🔗 Open Payment Link</a>', unsafe_allow_html=True)
            st.caption(f"ID: `{rzp.get('id', 'N/A')}`")
        elif stage == "EXECUTION_FAILED":
            st.error("**💸 3. Execution**\n\n❌ RAZORPAY ERROR")
            st.caption(result.get("error", "Unknown error")[:200])
        else:
            st.info("**💸 3. Execution**\n\n⏸ NOT REACHED")

    # Stage 4: Audit
    with col4:
        if stage == "AGENT_FAILED":
            st.info("**🧾 4. Audit**\n\n✅ FAILURE LOGGED")
            st.caption("Even failures are immutably audited. No silent errors.")
        elif stage == "POLICY_DENIED":
            st.success("**🧾 4. Audit**\n\n✅ DENIAL RECORDED")
            st.caption("Full proposal and policy decision recorded in ledger.")
        elif stage == "EXECUTED":
            st.success("**🧾 4. Audit**\n\n✅ RECORDED\n\n⏳ State: CUSTOMER_ACTION_PENDING")
            st.caption("Payment link created. Awaiting customer payment.")
        elif stage == "EXECUTION_FAILED":
            st.error("**🧾 4. Audit**\n\n✅ ERROR RECORDED")
        else:
            st.info("**🧾 4. Audit**\n\n⏸ See ledger below")

    if result.get("action_id") and result["action_id"] != "n/a":
        st.caption(f"🔑 Action ID: `{result['action_id']}`")

st.divider()

# ── Audit Ledger ──────────────────────────────────────────────────────────────
col_a, col_b = st.columns([4, 1])
with col_a:
    st.markdown("### 🧾 Customer Audit Ledger")
with col_b:
    refresh = st.button("🔄 Refresh", use_container_width=True)

if refresh:
    try:
        res = requests.get(f"{API_BASE}/aegis/audit/{selected_id}", timeout=10)
        if res.status_code == 200:
            data = res.json()
            if data:
                df = pd.DataFrame(data)
                
                # Unpack metadata for better presentation
                if "metadata" in df.columns:
                    def parse_meta(m):
                        if isinstance(m, str):
                            try:
                                return json.loads(m)
                            except:
                                return {}
                        return m if isinstance(m, dict) else {}
                        
                    meta_dicts = df["metadata"].apply(parse_meta)
                    df["intent"] = meta_dicts.apply(lambda x: x.get("intent", x.get("proposal", {}).get("intent", "")))
                    df["discount"] = meta_dicts.apply(lambda x: x.get("proposal", {}).get("requested_discount_pct", ""))
                    df["decision"] = meta_dicts.apply(lambda x: x.get("reason", ""))
                    df["policy_rule"] = meta_dicts.apply(lambda x: x.get("policy_rule_id", x.get("proposal", {}).get("policy_rule_id", "")))
                    
                    # Reorder and filter columns for clean display
                    cols_to_show = ["timestamp", "event_type", "status", "intent", "discount", "policy_rule", "decision", "action_id", "actor"]
                    available_cols = [c for c in cols_to_show if c in df.columns]
                    df = df[available_cols]

                st.dataframe(df, use_container_width=True)
            else:
                st.info("No audit events yet for this customer.")
        else:
            st.error("Failed to fetch audit log.")
    except Exception as e:
        st.error(f"Audit fetch failed: {e}")

st.divider()

# ── Reconciliation ────────────────────────────────────────────────────────────
st.markdown("### ⚖️ Reconciliation")
st.caption(
    "Polls Razorpay for all PENDING payment links. "
    "Terminal states (paid / expired / cancelled) move the action to RECONCILED."
)
if st.button("▶ Run Reconciliation", disabled=not st.session_state.get("access_token")):
    try:
        res = requests.post(f"{API_BASE}/aegis/reconcile", headers=auth_headers(), timeout=30)
        if res.status_code == 200:
            data = res.json()
            recs = data.get("reconciled", [])
            if recs:
                st.dataframe(pd.DataFrame(recs), use_container_width=True)
            else:
                st.info("No pending actions to reconcile.")
        else:
            st.error("Reconciliation request failed.")
    except Exception as e:
        st.error(f"Reconciliation error: {e}")
