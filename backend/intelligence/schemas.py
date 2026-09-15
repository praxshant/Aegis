"""
schemas.py — Intelligence layer canonical schemas.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class IntelligenceStatus(str, Enum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"   # propensity could not be generated; no financial action allowed


class EvidenceItem(BaseModel):
    signal: str
    value: float
    threshold: float
    message: str


class NextBasketProduct(BaseModel):
    product_id: int
    product_name: str
    probability: float


class NextBasketIntelligence(BaseModel):
    available: bool
    model: str = "instacart_next_basket"
    model_version: str = "1.1.0"
    status: str  # "COMPLETE" | "UNAVAILABLE" | "ERROR"
    unavailable_reason: Optional[str] = None
    top_products: List[NextBasketProduct] = []
    top_departments: List[str] = []
    top_aisles: List[str] = []
    mean_top_n_probability: Optional[float] = None
    signal: Optional[str] = None   # "likely_to_repurchase_from_usual_basket" | "moderate_repurchase_signal" | "weak_repurchase_signal"


class CustomerIntelligence(BaseModel):
    customer_id: str
    observation_date: str                         # ISO date string, e.g. "2026-08-25"
    status: IntelligenceStatus = IntelligenceStatus.COMPLETE

    # Propensity — None only when status=INCOMPLETE
    segment: Optional[str] = None
    purchase_probability_30d: Optional[float] = None   # raw model output [0,1]
    propensity_band: Optional[str] = None              # 1_LOW / 2_MODERATE / 3_HIGH / 4_VERY_HIGH
    propensity_source: str = "model_inference"         # always "model_inference" in production

    # Policy context
    recommendation: Optional[str] = None
    recommendation_confidence: Optional[str] = None
    policy_rule_id: Optional[str] = None

    # Evidence
    why: List[str] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    evidence_status: str = "NO_ACTIVE_SIGNALS"
    behavioral_signals: Dict[str, bool] = Field(default_factory=dict)

    # Feature snapshot — key Aegis features for agent context (not all 19, just readable ones)
    feature_snapshot: Dict[str, Any] = Field(default_factory=dict)

    # Failure reason — only populated when status=INCOMPLETE
    incomplete_reason: Optional[str] = None

    # Model metadata
    model_type: str = "HistGradientBoostingClassifier"
    model_target: str = "TARGET_next_purchase_30d"

    # Next-Basket Intelligence
    next_basket: Optional[NextBasketIntelligence] = None

    class Config:
        use_enum_values = True
        protected_namespaces = ()
