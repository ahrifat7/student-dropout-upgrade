"""Saving, loading and (if needed) re-creating the deployed stage models."""

from __future__ import annotations

import json
import platform

import joblib
import numpy as np
import pandas as pd

from .config import CLASSES, METADATA_PATH, MODELS_DIR
from .data import load_dataset, stage_columns
from .models import build_estimator


def library_versions() -> dict:
    import lightgbm
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
    }


def _minor(version: str) -> str:
    return ".".join(version.split(".")[:2])


def load_metadata() -> dict:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def model_path(stage: str):
    return MODELS_DIR / f"{stage}.joblib"


def fit_stage_model(stage: str, features: pd.DataFrame, targets: pd.Series, name: str, params: dict):
    columns = stage_columns(features.columns, stage)
    return build_estimator(name, columns, params).fit(features[columns], targets)


def load_stage_model(stage: str):
    """Return (model, source). Falls back to retraining if the artifact is unusable.

    Pickled models are sensitive to library versions. If scikit-learn/LightGBM on
    the host differ from the training environment, we refit from the saved
    hyperparameters (takes a few seconds) instead of crashing.
    """
    metadata = load_metadata()
    saved = metadata["versions"]
    current = library_versions()
    compatible = all(
        _minor(saved[k]) == _minor(current[k]) for k in ("scikit-learn", "lightgbm", "numpy", "pandas")
    )
    if compatible and model_path(stage).exists():
        try:
            return joblib.load(model_path(stage)), "saved artifact"
        except Exception:  # corrupted or incompatible pickle
            pass
    info = metadata["stages"][stage]
    features, targets = load_dataset()
    model = fit_stage_model(stage, features, targets, info["selected_model"], info["best_params"])
    return model, "retrained at startup"


def predict_table(model, features: pd.DataFrame) -> pd.DataFrame:
    """Predicted outcome plus one probability column per class."""
    proba = model.predict_proba(features[model_columns(model)])
    table = pd.DataFrame(proba, columns=[f"Probability: {c}" for c in model.classes_], index=features.index)
    table.insert(0, "Predicted outcome", np.asarray(model.classes_)[proba.argmax(1)])
    return table


def model_columns(model) -> list[str]:
    """Feature columns a fitted model expects (works for Pipelines and LGBMStudent)."""
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return list(model.named_steps["preprocess"].feature_names_in_)


__all__ = ["CLASSES", "fit_stage_model", "library_versions", "load_metadata", "load_stage_model", "model_columns", "model_path", "predict_table"]
