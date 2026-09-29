import os
import sys
import logging
import json
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import shutil
import pandas as pd
import numpy as np

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, ".env"))

from backend.data.database_manager import DatabaseManager
from backend.intelligence.engine import IntelligenceEngine
from backend.policy.gate import PolicyGate
from backend.razorpay.adapter import get_razorpay_adapter
from backend.audit.ledger import AuditLedger
from backend.reconciliation.reconciler import Reconciler
from backend.agent.tools import AgentTools
from backend.agent.agent import AegisAgent, AegisAgentError
from backend.ingestion.ingestion_service import IngestionService
from backend.auth.auth_manager import auth_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AegisAPI")

app = FastAPI(title="Aegis API", version="1.0.0")

# Global components
_db = None
_ie = None
_policy = None
_rzp = None
_audit = None
_recon = None
_agent = None
_tools = None
_campaigns = None
_ingestion = None

@app.on_event("startup")
async def startup_event():
    global _db, _ie, _policy, _rzp, _audit, _recon, _agent, _tools, _campaigns, _ingestion
    logger.info("Initializing Aegis Components...")
    
    _db = DatabaseManager()
    _policy = PolicyGate()
    _ie = IntelligenceEngine(policy_gate=_policy)
    _rzp = get_razorpay_adapter()
    _audit = AuditLedger(_db)
    _recon = Reconciler(_db, _rzp)
    _tools = AgentTools(_db, _ie)
    _agent = AegisAgent(_tools)
    _ingestion = IngestionService(_db, _ie)
    
    try:
        with open(os.path.join(project_root, "config", "campaigns.json"), "r", encoding="utf-8") as f:
            _campaigns = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load campaigns.json: {e}")
        _campaigns = {}
        
    logger.info("Aegis API ready.")

class ReactivationRequest(BaseModel):
    customer_id: str
    prompt: str = "Assess this customer for a win-back reactivation offer."

class BulkCustomerRequest(BaseModel):
    customer_ids: List[str]

class DatasetImportRequest(BaseModel):
    run_id: str
    mode: str  # "replace" | "merge" | "new"

class LoginRequest(BaseModel):
    username: str
    password: str

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/auth/login")
def login(req: LoginRequest):
    user = auth_manager.authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {
        "access_token": auth_manager.create_access_token({"sub": user["user_id"]}),
        "refresh_token": auth_manager.create_refresh_token({"sub": user["user_id"]}),
        "token_type": "bearer",
    }

@app.get("/auth/me")
def whoami(current_user: dict = Depends(auth_manager.get_current_user)):
    return current_user

@app.get("/customers")
def get_customers(limit: int = 200, status: str = "active"):
    query = """
        SELECT c.customer_id, c.first_name, c.last_name, c.email,
               c.registration_date, c.age, c.gender, c.location,
               c.opted_out, c.propensity_score, c.propensity_band,
               COALESCE(c.status, 'active') as status,
               cs.segment_name
        FROM customers c
        LEFT JOIN (
            SELECT customer_id, segment_name,
                   ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY last_updated DESC) as rn
            FROM customer_segments
        ) cs ON c.customer_id = cs.customer_id AND cs.rn = 1
        WHERE COALESCE(c.status, 'active') = ?
        ORDER BY c.propensity_score DESC NULLS LAST
        LIMIT ?
    """
    df = _db.query_to_dataframe(query, (status, limit))
    df = df.replace({np.nan: None})
    return df.to_dict('records')

@app.post("/customers/archive")
def archive_customers(req: BulkCustomerRequest, current_user: dict = Depends(auth_manager.get_current_user)):
    """Soft-archive customers — preserves all data, audit records, and transactions."""
    if not req.customer_ids:
        raise HTTPException(status_code=400, detail="No customer IDs provided.")
    placeholders = ",".join(["?"]*len(req.customer_ids))
    _db.execute_query(
        f"UPDATE customers SET status = 'archived' WHERE customer_id IN ({placeholders})",
        tuple(req.customer_ids)
    )
    _audit.record_event(
        "bulk", "system", "CUSTOMERS_ARCHIVED", current_user["username"], "SUCCESS",
        {"count": len(req.customer_ids), "customer_ids": req.customer_ids}
    )
    return {"archived": len(req.customer_ids), "customer_ids": req.customer_ids}

@app.post("/customers/delete")
def delete_customers(req: BulkCustomerRequest, current_user: dict = Depends(auth_manager.get_current_user)):
    """Soft-delete customers — marks as deleted but audit ledger and transactions are IMMUTABLE."""
    if not req.customer_ids:
        raise HTTPException(status_code=400, detail="No customer IDs provided.")
    placeholders = ",".join(["?"]*len(req.customer_ids))
    _db.execute_query(
        f"UPDATE customers SET status = 'deleted' WHERE customer_id IN ({placeholders})",
        tuple(req.customer_ids)
    )
    _audit.record_event(
        "bulk", "system", "CUSTOMERS_DELETED", current_user["username"], "SUCCESS",
        {"count": len(req.customer_ids), "customer_ids": req.customer_ids}
    )
    return {"deleted": len(req.customer_ids), "customer_ids": req.customer_ids}

@app.post("/customers/restore")
def restore_customers(req: BulkCustomerRequest, current_user: dict = Depends(auth_manager.get_current_user)):
    """Restore archived/deleted customers back to active."""
    if not req.customer_ids:
        raise HTTPException(status_code=400, detail="No customer IDs provided.")
    placeholders = ",".join(["?"]*len(req.customer_ids))
    _db.execute_query(
        f"UPDATE customers SET status = 'active' WHERE customer_id IN ({placeholders})",
        tuple(req.customer_ids)
    )
    return {"restored": len(req.customer_ids)}

@app.post("/datasets/activate")
def activate_dataset(req: DatasetImportRequest, current_user: dict = Depends(auth_manager.get_current_user)):
    """
    Apply dataset import mode after ingestion completes.
    mode='replace': Archive all existing active customers, then activate new run's customers.
    mode='merge':   Activate new run's customers alongside existing ones.
    mode='new':     Keep existing active customers; new run customers are also activated.
    """
    run_id = req.run_id
    mode = req.mode

    if mode not in ("replace", "merge", "new"):
        raise HTTPException(status_code=400, detail="mode must be 'replace', 'merge', or 'new'")

    run_df = _db.query_to_dataframe("SELECT * FROM ingestion_runs WHERE run_id = ?", (run_id,))
    if run_df.empty:
        raise HTTPException(status_code=404, detail="Run ID not found")

    archived_count = 0
    if mode == "replace":
        # Archive ALL currently active customers that belong to a previous run
        result = _db.query_to_dataframe(
            "SELECT COUNT(*) as c FROM customers WHERE COALESCE(status,'active')='active' AND (ingestion_run_id IS NULL OR ingestion_run_id != ?)",
            (run_id,)
        )
        archived_count = int(result.iloc[0,0])
        _db.execute_query(
            "UPDATE customers SET status='archived' WHERE COALESCE(status,'active')='active' AND (ingestion_run_id IS NULL OR ingestion_run_id != ?)",
            (run_id,)
        )
        _audit.record_event(
            run_id, "system", "DATASET_REPLACED", current_user["username"], "SUCCESS",
            {"run_id": run_id, "archived_count": archived_count, "mode": mode}
        )

    # Activate new run's customers
    _db.execute_query(
        "UPDATE customers SET status='active' WHERE ingestion_run_id = ?",
        (run_id,)
    )
    new_count_df = _db.query_to_dataframe(
        "SELECT COUNT(*) as c FROM customers WHERE ingestion_run_id = ? AND status='active'",
        (run_id,)
    )
    new_count = int(new_count_df.iloc[0,0])

    _audit.record_event(
        run_id, "system", "DATASET_IMPORTED", current_user["username"], "SUCCESS",
        {"run_id": run_id, "mode": mode, "new_customers": new_count, "archived_customers": archived_count}
    )

    return {
        "status": "ok",
        "mode": mode,
        "new_customers_activated": new_count,
        "previous_customers_archived": archived_count
    }

@app.get("/aegis/intelligence/{customer_id}")
def get_customer_intelligence(customer_id: str):
    """
    Diagnostic endpoint: Runs the full Aegis feature engineering + model inference pipeline
    for a given customer and returns their CustomerIntelligence object.
    
    Use this to verify the ML pipeline is working end-to-end before involving the LLM.
    """
    cust_df = _db.get_customer_data(customer_id)
    if cust_df.empty:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    
    segment = None
    if "segment_name" in cust_df.columns:
        segment = cust_df.iloc[0].get("segment_name")
    
    tx_df = _db.query_to_dataframe(
        "SELECT * FROM transactions WHERE customer_id = ?", (customer_id,)
    )
    
    intelligence = _ie.get_intelligence(customer_id, tx_df, segment=segment)
    return intelligence.model_dump()

@app.post("/aegis/reactivate")
def run_reactivation_flow(req: ReactivationRequest, current_user: dict = Depends(auth_manager.get_current_user)):
    """
    End-to-End Reactivation Flow:
    1. Agent Proposes
    2. Policy Validates
    3. Execute if Approved
    4. Audit & Reconcile
    """
    customer_id = req.customer_id

    # actor = the authenticated human who initiated this spend (1B: "who authorized this").
    # Downstream events keep their subsystem actors (Agent/PolicyGate/RazorpayAdapter).
    _audit.record_event("n/a", customer_id, "AGENT_START", current_user["username"], "IN_PROGRESS",
                        {"prompt": req.prompt, "initiated_by": current_user["username"]})
    
    # 1. Compute Intelligence FIRST (used both as agent context and policy ground truth)
    df = _db.get_customer_data(customer_id)
    if df.empty:
        raise HTTPException(status_code=404, detail="Customer not found")

    # Consent gate: never contact an opted-out customer (checked before any LLM/spend work)
    opted_out = df.iloc[0].get("opted_out")
    if pd.notna(opted_out) and int(opted_out) == 1:
        _audit.record_event("n/a", customer_id, "OPT_OUT_BLOCKED", "PolicyGate", "DENIED", {"reason": "CUSTOMER_OPTED_OUT"})
        return {"status": "DENIED", "reason": "CUSTOMER_OPTED_OUT", "proposal": None, "action_id": "n/a"}

    tx_df = _db.query_to_dataframe("SELECT * FROM transactions WHERE customer_id = ?", (customer_id,))
    segment = df.iloc[0].get("segment_name") if "segment_name" in df.columns else None
    intelligence = _ie.get_intelligence(customer_id, tx_df, segment=segment)

    if intelligence.status == "INCOMPLETE":
        _audit.record_event("n/a", customer_id, "INTELLIGENCE_INCOMPLETE", "IntelligenceEngine", "FAILED",
                            {"reason": intelligence.incomplete_reason})
        raise HTTPException(status_code=422, detail=f"Intelligence INCOMPLETE: {intelligence.incomplete_reason}")

    # 2. Agent Proposal — intelligence embedded in prompt (saves API calls)
    try:
        proposal = _agent.propose_action(customer_id, req.prompt,
                                         intelligence_context=intelligence.model_dump())
    except AegisAgentError as e:
        _audit.record_event("n/a", customer_id, "AGENT_FAILED", "Agent", "FAILED", {"reason": str(e)})
        raise HTTPException(status_code=503, detail=f"Agent failed to generate a proposal: {str(e)}")
    except Exception as e:
        _audit.record_event("n/a", customer_id, "AGENT_ERROR", "Agent", "FAILED", {"error": str(e)})
        raise HTTPException(status_code=500, detail="Internal server error during agent execution")
        
    action_id = f"act_{os.urandom(4).hex()}"
    _audit.record_event(action_id, customer_id, "PROPOSAL_GENERATED", "Agent", "SUCCESS", proposal.model_dump())

    # Campaign selection is SERVER-SIDE and deterministic (1C): the ML-grounded policy_rule_id
    # picks the campaign — NOT the LLM's proposal.intent — so the LLM cannot steer the payable amount.
    intel = intelligence.model_dump()
    rule_id = intel.get("policy_rule_id")
    campaign_key = _campaigns.get("rule_to_campaign", {}).get(rule_id)
    campaign = _campaigns.get("campaigns", {}).get(campaign_key) if campaign_key else None

    if campaign is None:
        # No paid campaign maps to this rule (e.g. DEFAULT_01 / STANDARD_ENGAGEMENT) → no spend.
        _audit.record_event(action_id, customer_id, "POLICY_REJECTED", "PolicyGate", "DENIED",
                            {"reason": f"NO_CAMPAIGN_FOR_RULE ({rule_id})"})
        return {"status": "DENIED", "reason": f"NO_CAMPAIGN_FOR_RULE ({rule_id})",
                "proposal": proposal.model_dump(), "action_id": action_id}

    base_amount_inr = campaign["base_amount_inr"]
    currency = campaign.get("currency", "INR")
    discount_pct = proposal.requested_discount_pct
    final_amount_inr = base_amount_inr * (1 - (discount_pct / 100.0))
    amount_paise = round(final_amount_inr * 100)  # round, not int(): int() truncates sub-paise → undercharge

    # 3. Policy Gate — per-campaign caps + eligibility (1C) and velocity/budget/cooldown (1D)
    velocity = _recon.get_customer_velocity(customer_id, _policy.limits.get("ACTION_WINDOW_DAYS", 30))
    auth_result = _policy.authorize_action(proposal.model_dump(), intel,
                                           amount_inr=final_amount_inr, campaign=campaign, velocity=velocity)

    if not auth_result["approved"]:
        _audit.record_event(action_id, customer_id, "POLICY_REJECTED", "PolicyGate", "DENIED", {"reason": auth_result["reason"]})
        return {
            "status": "DENIED",
            "reason": auth_result["reason"],
            "proposal": proposal.model_dump(),
            "action_id": action_id
        }

    _audit.record_event(action_id, customer_id, "POLICY_APPROVED", "PolicyGate", "APPROVED", {"reason": auth_result["reason"]})

    # 4. Execution Request — amount_inr recorded so the 1D budget SUM is real; rule_id is the
    # server-grounded ML rule (matches the campaign that set the amount), not the LLM's.
    _recon.record_execution_request(
        action_id,
        customer_id,
        proposal.action_type,
        proposal.requested_discount_pct,
        rule_id,
        amount_inr=final_amount_inr,
    )
    
    _audit.record_event(action_id, customer_id, "EXECUTION_START", "ExecutionService", "IN_PROGRESS", {"amount_paise": amount_paise, "currency": currency})
    
    try:
        rzp_res = _rzp.create_payment_link(
            amount=amount_paise,
            customer_id=customer_id,
            reference_id=action_id,
            description=f"Aegis Reactivation Offer: {proposal.reasoning_summary}",
            currency=currency
        )
        
        # 6. Reconcile
        _recon.update_execution_status(action_id, "EXECUTED", rzp_res.get("id"))
        _audit.record_event(action_id, customer_id, "EXECUTION_SUCCESS", "RazorpayAdapter", "SUCCESS", {"link": rzp_res.get("short_url")})
        
        return {
            "status": "EXECUTED",
            "proposal": proposal.model_dump(),
            "razorpay": rzp_res,
            "action_id": action_id
        }
        
    except Exception as e:
        _recon.update_execution_status(action_id, "FAILED")
        _audit.record_event(action_id, customer_id, "EXECUTION_FAILED", "RazorpayAdapter", "FAILED", {"error": str(e)})
        return {
            "status": "FAILED",
            "error": str(e),
            "action_id": action_id
        }

@app.get("/aegis/audit/{customer_id}")
def get_customer_audit(customer_id: str):
    events = _audit.get_events_for_customer(customer_id)
    # Normalize to list regardless of what the audit ledger returns
    if isinstance(events, list):
        return events
    if isinstance(events, dict):
        return events.get('events', list(events.values()))
    return []

@app.post("/aegis/reconcile")
def run_reconciliation(current_user: dict = Depends(auth_manager.get_current_user)):
    results = _recon.reconcile_pending_actions()
    for res in results:
        _audit.record_event(res['action_id'], "system", "RECONCILIATION_RUN", "Reconciler", "SUCCESS", res)
    return {"reconciled": results}

@app.post("/aegis/ingest/csv")
async def ingest_csv(background_tasks: BackgroundTasks, file: UploadFile = File(...),
                     current_user: dict = Depends(auth_manager.get_current_user)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="INVALID_FILE: Only CSV files are supported.")
        
    temp_dir = os.path.join(project_root, "artifacts", "uploads")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    run_id = _ingestion.start_ingestion_run(file.filename)
    background_tasks.add_task(_ingestion.process_file_in_background, run_id, temp_path)
    
    return {"status": "accepted", "run_id": run_id}

@app.get("/aegis/ingest/status/{run_id}")
def get_ingestion_status(run_id: str):
    df = _db.query_to_dataframe("SELECT * FROM ingestion_runs WHERE run_id = ?", (run_id,))
    if df.empty:
        raise HTTPException(status_code=404, detail="Run ID not found")
    df = df.replace({np.nan: None})
    return df.iloc[0].to_dict()

@app.get("/aegis/analytics/summary")
def get_analytics_summary():
    stats = {}
    stats["total_customers"] = int(_db.query_to_dataframe("SELECT COUNT(DISTINCT customer_id) FROM customers").iloc[0, 0])
    stats["opted_out"] = int(_db.query_to_dataframe("SELECT COUNT(*) FROM customers WHERE opted_out = 1").iloc[0, 0])
    
    # Avoid errors if table is empty or missing columns
    try:
        segments = _db.query_to_dataframe("SELECT segment_name, COUNT(*) as count FROM customer_segments GROUP BY segment_name")
        stats["segments"] = segments.set_index("segment_name")["count"].to_dict()
    except Exception:
        stats["segments"] = {}
        
    try:
        propensity = _db.query_to_dataframe("SELECT propensity_band, COUNT(*) as count FROM customers WHERE propensity_band IS NOT NULL GROUP BY propensity_band")
        stats["propensity_bands"] = propensity.set_index("propensity_band")["count"].to_dict()
    except Exception:
        stats["propensity_bands"] = {}
        
    try:
        latest_run = _db.query_to_dataframe("SELECT * FROM ingestion_runs ORDER BY started_at DESC LIMIT 1")
        if not latest_run.empty:
            latest_run = latest_run.replace({np.nan: None})
            stats["latest_ingestion"] = latest_run.iloc[0].to_dict()
        else:
            stats["latest_ingestion"] = None
    except Exception:
        stats["latest_ingestion"] = None
        
    return stats

# Serve frontend directory
frontend_dir = os.path.join(project_root, "frontend")
os.makedirs(frontend_dir, exist_ok=True)
app.mount("/ui", StaticFiles(directory=frontend_dir, html=True), name="frontend")

@app.get("/")
def read_root():
    return FileResponse(os.path.join(frontend_dir, "index.html"))
