"""
next_basket_model.py
--------------------
Singleton inference wrapper for the Aegis-trained LightGBM next-basket model.

RULE: This module LOADS an existing artifact. It does NOT train, fit, or modify the model.
RULE: The artifact hash is verified on every load to guarantee provenance.
RULE: If the artifact is missing or corrupt, this module raises loudly — never silently.
"""

import os
import hashlib
import logging
import joblib
import pandas as pd
from typing import Optional, Tuple
from pathlib import Path

from backend.models.next_basket_features import NEXT_BASKET_FEATURES

logger = logging.getLogger("NextBasketModel")

# ── Artifact configuration ────────────────────────────────────────────────────
_DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / "artifacts" / "models" / "next_basket" / "next_basket_model_corrected.pkl"

# SHA-256 of the canonical Aegis artifact (verified against Phase 9 of notebook)
EXPECTED_ARTIFACT_HASH = "238a44c50b10d825c2417ae8e278465d9366afa1651bc826cd1b2d81d0e4ea9c"


class NextBasketModelError(Exception):
    """Raised when the model cannot be loaded or inference fails unrecoverably."""
    pass


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class NextBasketModel:
    """
    Singleton wrapper around the Aegis LightGBM Next Basket model.
    Loaded once at application startup; reused for every inference call.
    """

    _instance: Optional["NextBasketModel"] = None

    def __init__(self, model_path: Optional[Path] = None):
        self._model = None
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self._loaded = False
        self._hash_verified = False

    @classmethod
    def get_instance(cls, model_path: Optional[Path] = None) -> "NextBasketModel":
        if cls._instance is None:
            cls._instance = cls(model_path)
            cls._instance.load()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton — useful for testing."""
        cls._instance = None

    def load(self) -> None:
        """
        Load and verify the model artifact.
        Raises NextBasketModelError if the file is missing, hash mismatches, or load fails.
        """
        path = self._model_path

        if not path.exists():
            raise NextBasketModelError(
                f"Model artifact not found at: {path}\n"
                "Ensure artifacts/models/next_basket/next_basket_model_corrected.pkl is present."
            )

        # Hash verification
        logger.info(f"Verifying model artifact hash: {path}")
        actual_hash = _compute_sha256(path)
        if actual_hash != EXPECTED_ARTIFACT_HASH:
            raise NextBasketModelError(
                f"Model artifact hash mismatch!\n"
                f"  Expected: {EXPECTED_ARTIFACT_HASH}\n"
                f"  Got:      {actual_hash}\n"
                "The artifact may be corrupt or tampered with."
            )
        self._hash_verified = True
        logger.info(f"Artifact hash verified: {actual_hash[:16]}...")

        # Load model
        try:
            payload = joblib.load(path)
            self._model = payload.get("model")
            loaded_features = payload.get("feature_cols")
            model_kind = payload.get("model_kind")
        except Exception as e:
            raise NextBasketModelError(f"Failed to load model from {path}: {e}") from e

        if not self._model:
            raise NextBasketModelError("Loaded payload did not contain a 'model' key.")

        # Verify model type
        if model_kind != "lightgbm":
            raise NextBasketModelError(
                f"Unexpected model kind: {model_kind}. "
                "Expected lightgbm."
            )

        # Verify features
        if loaded_features != NEXT_BASKET_FEATURES:
            raise NextBasketModelError(
                f"Feature name mismatch in model artifact!\n"
                f"  Artifact features: {loaded_features}\n"
                f"  Contract features: {NEXT_BASKET_FEATURES}"
            )

        self._loaded = True
        logger.info(f"Next Basket Model loaded successfully: {model_kind} | {len(NEXT_BASKET_FEATURES)} features")

    def _ensure_loaded(self) -> None:
        if not self._loaded or self._model is None:
            raise NextBasketModelError(
                "Model is not loaded. Call load() or use get_instance()."
            )

    def predict_next_basket(self, candidate_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Run inference on a dataframe of candidates.
        
        Args:
            candidate_frame: DataFrame containing candidate user-product pairs and exactly NEXT_BASKET_FEATURES.
            
        Returns:
            The input dataframe with a new 'score' column containing predictions.
            
        Raises:
            NextBasketModelError: If model inference fails.
        """
        self._ensure_loaded()
        
        # Verify columns
        missing = [c for c in NEXT_BASKET_FEATURES if c not in candidate_frame.columns]
        if missing:
            raise NextBasketModelError(f"Candidate frame missing required features: {missing}")

        try:
            X = candidate_frame[NEXT_BASKET_FEATURES]
            scores = self._model.predict(X, num_iteration=self._model.best_iteration)
            
            result = candidate_frame.copy()
            result['score'] = scores
            return result
        except Exception as e:
            raise NextBasketModelError(f"Model inference failed: {e}") from e

    def metadata(self) -> dict:
        return {
            "model_type": "LightGBM",
            "target": "next_basket_product_probability",
            "feature_count": len(NEXT_BASKET_FEATURES),
            "artifact_hash": EXPECTED_ARTIFACT_HASH,
            "artifact_path": str(self._model_path),
            "loaded": self._loaded,
            "hash_verified": self._hash_verified,
        }


def get_next_basket_model() -> NextBasketModel:
    """Application-level accessor. Returns the loaded singleton."""
    return NextBasketModel.get_instance()
