"""
propensity_model.py
-------------------
Singleton inference wrapper for the Aegis-trained HistGradientBoostingClassifier.

RULE: This module LOADS an existing artifact. It does NOT train, fit, or modify the model.
RULE: The artifact hash is verified on every load to guarantee provenance.
RULE: If the artifact is missing or corrupt, this module raises loudly — never silently.
"""

import os
import hashlib
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Optional
from pathlib import Path

from backend.models.propensity_features import (
    PROPENSITY_FEATURES,
    validate_feature_vector,
    FeatureBuildError,
)

logger = logging.getLogger("PropensityModel")

# ── Artifact configuration ────────────────────────────────────────────────────
_DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / "artifacts" / "models" / "next_purchase_30d" / "model_HistGradientBoosting_Tuned.pkl"

# SHA-256 of the canonical Aegis artifact (verified against source on 2026-08-24)
EXPECTED_ARTIFACT_HASH = "23aa3acebaabc1e84818f8ea727e4693036ecaba03741f1b45b1a3de4d07f7bf"

# Propensity band thresholds — FROZEN from model_audit_report.json
# "thresholds derived empirically from test/validation decile analysis"
_BAND_THRESHOLDS = [
    (0.40, "4_VERY_HIGH"),
    (0.27, "3_HIGH"),
    (0.16, "2_MODERATE"),
    (0.00, "1_LOW"),
]


class PropensityModelError(Exception):
    """Raised when the model cannot be loaded or inference fails unrecoverably."""
    pass


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def assign_propensity_band(score: float) -> str:
    """
    Assign a propensity band from a raw probability score.
    Thresholds are frozen from Aegis model_audit_report.json.
    """
    for threshold, band in _BAND_THRESHOLDS:
        if score >= threshold:
            return band
    return "1_LOW"


class NextPurchasePropensityModel:
    """
    Singleton wrapper around the Aegis HistGradientBoostingClassifier.
    Loaded once at application startup; reused for every inference call.
    """

    _instance: Optional["NextPurchasePropensityModel"] = None

    def __init__(self, model_path: Optional[Path] = None):
        self._model = None
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self._loaded = False
        self._hash_verified = False

    @classmethod
    def get_instance(cls, model_path: Optional[Path] = None) -> "NextPurchasePropensityModel":
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
        Raises PropensityModelError if the file is missing, hash mismatches, or load fails.
        """
        path = self._model_path

        if not path.exists():
            raise PropensityModelError(
                f"Model artifact not found at: {path}\n"
                "Ensure artifacts/models/next_purchase_30d/model_HistGradientBoosting_Tuned.pkl is present."
            )

        # Hash verification
        logger.info(f"Verifying model artifact hash: {path}")
        actual_hash = _compute_sha256(path)
        if actual_hash != EXPECTED_ARTIFACT_HASH:
            raise PropensityModelError(
                f"Model artifact hash mismatch!\n"
                f"  Expected: {EXPECTED_ARTIFACT_HASH}\n"
                f"  Got:      {actual_hash}\n"
                "The artifact may be corrupt or tampered with."
            )
        self._hash_verified = True
        logger.info(f"Artifact hash verified: {actual_hash[:16]}...")

        # Load model
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress sklearn version warnings
                self._model = joblib.load(path)
        except Exception as e:
            raise PropensityModelError(f"Failed to load model from {path}: {e}") from e

        # Verify sklearn model type
        model_type = type(self._model).__name__
        if model_type != "HistGradientBoostingClassifier":
            raise PropensityModelError(
                f"Unexpected model type: {model_type}. "
                "Expected HistGradientBoostingClassifier."
            )

        # Verify feature names if available
        if hasattr(self._model, "feature_names_in_"):
            loaded_features = list(self._model.feature_names_in_)
            if loaded_features != PROPENSITY_FEATURES:
                raise PropensityModelError(
                    f"Feature name mismatch in model artifact!\n"
                    f"  Artifact features: {loaded_features}\n"
                    f"  Contract features: {PROPENSITY_FEATURES}"
                )

        self._loaded = True
        logger.info(f"Model loaded successfully: {model_type} | {len(PROPENSITY_FEATURES)} features")

    def _ensure_loaded(self) -> None:
        if not self._loaded or self._model is None:
            raise PropensityModelError(
                "Model is not loaded. Call load() or use get_instance()."
            )

    def predict_score(self, feature_df: pd.DataFrame) -> float:
        """
        Run inference and return the propensity probability (class=1).

        Args:
            feature_df: Single-row DataFrame with exactly PROPENSITY_FEATURES columns.

        Returns:
            float in [0.0, 1.0] — P(next purchase within 30 days).

        Raises:
            PropensityModelError: If the model is not loaded or inference fails.
            ValueError:           If the feature vector is malformed.
        """
        self._ensure_loaded()
        validate_feature_vector(feature_df)

        try:
            proba = self._model.predict_proba(feature_df)
            score = float(proba[0][1])  # class=1 probability
        except Exception as e:
            raise PropensityModelError(f"Model inference failed: {e}") from e

        if not (0.0 <= score <= 1.0):
            raise PropensityModelError(
                f"Model returned out-of-range probability: {score}. This should never happen."
            )

        return score

    def predict_band(self, feature_df: pd.DataFrame) -> tuple[float, str]:
        """
        Convenience method: run inference and return (score, band).

        Returns:
            (propensity_score: float, propensity_band: str)
        """
        score = self.predict_score(feature_df)
        band = assign_propensity_band(score)
        return score, band

    def metadata(self) -> dict:
        return {
            "model_type": "HistGradientBoostingClassifier",
            "target": "TARGET_next_purchase_30d",
            "feature_count": len(PROPENSITY_FEATURES),
            "artifact_hash": EXPECTED_ARTIFACT_HASH,
            "artifact_path": str(self._model_path),
            "loaded": self._loaded,
            "hash_verified": self._hash_verified,
            "propensity_bands": {
                "4_VERY_HIGH": ">=0.40",
                "3_HIGH": ">=0.27 and <0.40",
                "2_MODERATE": ">=0.16 and <0.27",
                "1_LOW": "<0.16",
            },
        }


def get_propensity_model() -> NextPurchasePropensityModel:
    """Application-level accessor. Returns the loaded singleton."""
    return NextPurchasePropensityModel.get_instance()
