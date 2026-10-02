"""Per-student and global explanations from LightGBM TreeSHAP contributions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def explain_row(model, row: pd.DataFrame, outcome: str, top_k: int = 8) -> pd.DataFrame:
    """Top features pushing one student's score for `outcome` up or down.

    Contributions are on the model's log-odds scale: positive values push towards
    `outcome`, negative values push away from it.
    """
    values, _ = model.contributions(row.iloc[[0]])
    class_index = list(model.classes_).index(outcome)
    contrib = values[0, class_index, :]
    table = pd.DataFrame(
        {
            "feature": model.feature_names_in_,
            "value": [row.iloc[0][c] for c in model.feature_names_in_],
            "contribution": contrib,
        }
    )
    order = np.argsort(-np.abs(table["contribution"].to_numpy()))
    return table.iloc[order].head(top_k).reset_index(drop=True)


def global_importance(model, X: pd.DataFrame, outcome: str = "Dropout") -> pd.DataFrame:
    """Mean |contribution| per feature for one outcome (a SHAP-style importance)."""
    values, _ = model.contributions(X)
    class_index = list(model.classes_).index(outcome)
    importance = np.abs(values[:, class_index, :]).mean(axis=0)
    return (
        pd.DataFrame({"feature": model.feature_names_in_, "mean_abs_contribution": importance})
        .sort_values("mean_abs_contribution", ascending=False)
        .reset_index(drop=True)
    )
