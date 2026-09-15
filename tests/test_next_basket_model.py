import pytest
from pathlib import Path
from backend.models.next_basket_model import NextBasketModel, NextBasketModelError, EXPECTED_ARTIFACT_HASH

def test_model_load_success():
    model = NextBasketModel()
    model.load()
    assert model._loaded is True
    assert model._hash_verified is True
    
def test_model_missing_file():
    model = NextBasketModel(model_path=Path("nonexistent.pkl"))
    with pytest.raises(NextBasketModelError, match="Model artifact not found"):
        model.load()

def test_model_metadata():
    model = NextBasketModel()
    model.load()
    meta = model.metadata()
    assert meta["model_type"] == "LightGBM"
    assert meta["feature_count"] == 21
    assert meta["artifact_hash"] == EXPECTED_ARTIFACT_HASH
