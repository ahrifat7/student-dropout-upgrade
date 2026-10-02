import json

import numpy as np
import pytest

from dropout.config import METADATA_PATH, STAGES, SUMMARY_PATH
from dropout.data import load_dataset
from dropout.registry import load_metadata, load_stage_model, model_columns, predict_table

pytestmark = pytest.mark.skipif(
    not (METADATA_PATH.exists() and SUMMARY_PATH.exists()),
    reason="run `python -m dropout.train` first",
)


@pytest.fixture(scope="module")
def summary():
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("stage", list(STAGES))
def test_saved_model_reproduces_golden_predictions(stage):
    model, _ = load_stage_model(stage)
    features, _ = load_dataset()
    expected = load_metadata()["stages"][stage]["golden_predictions_first5"]
    table = predict_table(model, features.head(5))
    assert table["Predicted outcome"].tolist() == expected
    assert np.allclose(table.filter(like="Probability").sum(axis=1), 1.0)


@pytest.mark.parametrize("stage", list(STAGES))
def test_model_uses_only_stage_columns(stage):
    model, _ = load_stage_model(stage)
    columns = model_columns(model)
    assert load_metadata()["stages"][stage]["columns"] == columns
    if stage == "enrollment":
        assert not any("sem" in c for c in columns)
    if stage == "sem1":
        assert not any("2nd sem" in c for c in columns)


def test_summary_is_internally_consistent(summary):
    assert summary["dataset"]["test_rows"] == 885
    for info in summary["stages"].values():
        assert np.array(info["test"]["confusion"]).sum() == 885
        non_dummy = {k: v["macro_f1"]["mean"] for k, v in info["cv"].items() if k != "dummy"}
        assert info["selected_model"] == max(non_dummy, key=non_dummy.get)
        assert info["test"]["macro_f1"] > info["test_all_models"]["dummy"]["macro_f1"] + 0.2
        low, high = info["test"]["ci95"]["macro_f1"]
        assert low <= info["test"]["macro_f1"] <= high


def test_more_information_does_not_hurt(summary):
    scores = [summary["stages"][s]["test"]["macro_f1"] for s in STAGES]
    assert scores == sorted(scores)


def test_incompatible_library_versions_trigger_safe_retrain(monkeypatch):
    """A pickle from another scikit-learn/LightGBM version must never crash the app."""
    from dropout import registry

    real = registry.library_versions()
    monkeypatch.setattr(registry, "library_versions", lambda: {**real, "scikit-learn": "0.0.1"})
    model, source = registry.load_stage_model("enrollment")
    features, _ = load_dataset()
    expected = load_metadata()["stages"]["enrollment"]["golden_predictions_first5"]
    assert source == "retrained at startup"
    assert predict_table(model, features.head(5))["Predicted outcome"].tolist() == expected
