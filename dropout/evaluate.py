"""Evaluation helpers: CV, tuning, bootstrap CIs, slices, calibration, capacity."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

from .config import CLASSES, SEED
from .models import build_estimator

SCORING = {
    "macro_f1": "f1_macro",
    "balanced_accuracy": "balanced_accuracy",
    "accuracy": "accuracy",
}

LGBM_SEARCH_SPACE = {
    "num_leaves": [7, 15, 31],
    "learning_rate": [0.02, 0.03, 0.05, 0.1],
    "n_estimators": [150, 300, 500],
    "min_child_samples": [10, 20, 40],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.9],
    "reg_lambda": [0.0, 1.0, 5.0, 10.0],
    "class_weight": [None, "balanced"],
}


def metric_bundle(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def stratified_folds(folds: int, seed: int = SEED) -> StratifiedKFold:
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)


def cv_scores(estimator, X, y, folds: int) -> dict:
    """Cross-validated scores on the training data only."""
    result = cross_validate(
        estimator, X, y, cv=stratified_folds(folds), scoring=SCORING, n_jobs=1
    )
    return {
        name: {
            "mean": float(np.mean(result[f"test_{name}"])),
            "std": float(np.std(result[f"test_{name}"], ddof=1)),
            "folds": [float(v) for v in result[f"test_{name}"]],
        }
        for name in SCORING
    }


def tune(name: str, X, y, columns, folds: int, n_iter: int) -> dict:
    """Pick hyperparameters with cross-validation on training data. Returns best params."""
    if name == "lightgbm":
        search = RandomizedSearchCV(
            build_estimator("lightgbm", columns),
            LGBM_SEARCH_SPACE,
            n_iter=n_iter,
            scoring="f1_macro",
            cv=stratified_folds(folds),
            random_state=SEED,
            refit=False,
            n_jobs=1,
        )
        search.fit(X[columns], y)
        return dict(search.best_params_)
    if name == "logreg":
        search = GridSearchCV(
            build_estimator("logreg", columns),
            {"classifier__C": [0.01, 0.03, 0.1, 0.3, 1.0]},
            scoring="f1_macro",
            cv=stratified_folds(folds),
            refit=False,
            n_jobs=1,
        )
        search.fit(X[columns], y)
        return {"C": float(search.best_params_["classifier__C"])}
    return {}


def oof_proba(estimator, X, y, folds: int) -> np.ndarray:
    """Out-of-fold class probabilities (each row predicted by a model that never saw it)."""
    return cross_val_predict(
        estimator, X, y, cv=stratified_folds(folds), method="predict_proba", n_jobs=1
    )


def group_cv_macro_f1(estimator, X, y, groups, folds: int = 5) -> dict:
    """Hold out whole macro-indicator groups (cohort proxy) to test cohort leakage."""
    n_groups = int(pd.Series(groups).nunique())
    splitter = GroupKFold(n_splits=min(folds, n_groups))
    scores = []
    for train_idx, test_idx in splitter.split(X, y, groups):
        fitted = estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
        scores.append(
            f1_score(y.iloc[test_idx], fitted.predict(X.iloc[test_idx]), average="macro")
        )
    return {"mean": float(np.mean(scores)), "folds": [float(s) for s in scores], "n_groups": n_groups}


def bootstrap_ci(y_true, y_pred, n_boot: int = 1000, seed: int = SEED) -> dict:
    """95% percentile bootstrap intervals for the headline metrics."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    draws = {"accuracy": [], "balanced_accuracy": [], "macro_f1": []}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        bundle = metric_bundle(y_true[idx], y_pred[idx])
        for key in draws:
            draws[key].append(bundle[key])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()}


def per_class_report(y_true, y_pred, labels=CLASSES) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        label: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i, label in enumerate(labels)
    }


def confusion(y_true, y_pred, labels=CLASSES) -> list[list[int]]:
    return confusion_matrix(y_true, y_pred, labels=labels).tolist()


def capacity_table(y_true, p_dropout, fractions=(0.1, 0.2, 0.3, 0.4)) -> list[dict]:
    """If advisors can only contact the top k% riskiest students, how many dropouts do they reach?"""
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(p_dropout))
    actual = y_true[order] == "Dropout"
    base_rate = float(actual.mean())
    rows = []
    for fraction in fractions:
        k = max(1, math.ceil(fraction * len(order)))
        hit = int(actual[:k].sum())
        rows.append(
            {
                "flagged_share": fraction,
                "students_flagged": k,
                "precision": hit / k,
                "recall": hit / int(actual.sum()),
                "lift": (hit / k) / base_rate,
            }
        )
    return rows


AGE_BINS = [0, 19, 24, 29, 39, 200]
AGE_LABELS = ["<=19", "20-24", "25-29", "30-39", "40+"]
SLICE_COLUMNS = (
    "Course",
    "Gender",
    "Scholarship holder",
    "Debtor",
    "Tuition fees up to date",
    "International",
    "Displaced",
    "Educational special needs",
    "Daytime/evening attendance",
)
MIN_SLICE = 30


def slice_table(features: pd.DataFrame, y_true, y_pred, min_n: int = MIN_SLICE) -> pd.DataFrame:
    """Performance per subgroup (groups smaller than min_n are skipped as too noisy)."""
    frame = features.reset_index(drop=True).copy()
    frame["_true"] = np.asarray(y_true)
    frame["_pred"] = np.asarray(y_pred)
    frame["Age group"] = pd.cut(frame["Age at enrollment"], AGE_BINS, labels=AGE_LABELS).astype(str)
    rows = []
    for column in ("Age group",) + SLICE_COLUMNS:
        if column not in frame.columns:
            continue
        for group, part in frame.groupby(column, observed=True):
            if len(part) < min_n:
                continue
            is_drop = part["_true"] == "Dropout"
            rows.append(
                {
                    "slice": column,
                    "group": str(int(group)) if column != "Age group" and float(group).is_integer() else str(group),
                    "n": len(part),
                    "true_dropout_rate": float(is_drop.mean()),
                    "predicted_dropout_rate": float((part["_pred"] == "Dropout").mean()),
                    "dropout_recall": float((part.loc[is_drop, "_pred"] == "Dropout").mean()) if is_drop.any() else np.nan,
                    "accuracy": float((part["_true"] == part["_pred"]).mean()),
                    "macro_f1": float(f1_score(part["_true"], part["_pred"], average="macro", labels=CLASSES, zero_division=0)),
                }
            )
    return pd.DataFrame(rows)


def calibration(y_true, proba, classes=CLASSES, bins: int = 10) -> dict:
    """Top-label reliability: does '70% confident' mean right about 70% of the time?"""
    proba = np.asarray(proba)
    y_true = np.asarray(y_true)
    pred = np.asarray(classes)[proba.argmax(1)]
    conf = proba.max(1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(1 / len(classes), 1.0, bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (conf > lo) & (conf <= hi) if lo > edges[0] else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        gap = abs(correct[mask].mean() - conf[mask].mean())
        ece += mask.mean() * gap
        rows.append({"confidence": float(conf[mask].mean()), "accuracy": float(correct[mask].mean()), "n": int(mask.sum())})
    onehot = (y_true[:, None] == np.asarray(classes)[None, :]).astype(float)
    return {"ece": float(ece), "brier": float(((proba - onehot) ** 2).sum(1).mean()), "bins": rows}


def worst_mistakes(features: pd.DataFrame, y_true, proba, classes=CLASSES, k: int = 20) -> pd.DataFrame:
    """The k wrong predictions the model was most confident about."""
    proba = np.asarray(proba)
    pred = np.asarray(classes)[proba.argmax(1)]
    wrong = pred != np.asarray(y_true)
    frame = features.reset_index(drop=True).copy()
    frame.insert(0, "true_outcome", np.asarray(y_true))
    frame.insert(1, "predicted_outcome", pred)
    frame.insert(2, "confidence", proba.max(1))
    return frame.loc[wrong].sort_values("confidence", ascending=False).head(k)
