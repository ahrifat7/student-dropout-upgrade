"""Model builders: baselines, the original random forest, and a LightGBM model.

All preprocessing lives inside the estimators, so they can be cross-validated
and deployed without leakage between training and evaluation data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_COLUMNS, SEED

MODEL_NAMES = ("dummy", "logreg", "random_forest", "lightgbm")

DEFAULT_LGBM_PARAMS = {
    "num_leaves": 15,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "class_weight": "balanced",
}


class LGBMStudent(ClassifierMixin, BaseEstimator):
    """LightGBM with native categorical handling and per-feature contributions.

    Coded categorical columns are passed to LightGBM as real categories (not as
    ordered numbers). Because LightGBM works on the original columns, its
    TreeSHAP contributions map directly to the features a user entered.
    """

    def __init__(
        self,
        cat_cols=(),
        num_leaves=15,
        learning_rate=0.05,
        n_estimators=300,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=SEED,
    ):
        self.cat_cols = cat_cols
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.class_weight = class_weight
        self.random_state = random_state

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _as_frame(X) -> pd.DataFrame:
        return X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

    def _cast(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._as_frame(X)[self.feature_names_in_].copy()
        for column in self.feature_names_in_:
            if column in self.cat_cols_:
                X[column] = pd.Categorical(X[column], categories=self.categories_[column])
            else:
                X[column] = X[column].astype(float)
        return X

    # -- sklearn API -------------------------------------------------------
    def fit(self, X, y):
        X = self._as_frame(X)
        self.feature_names_in_ = list(X.columns)
        self.cat_cols_ = [c for c in X.columns if c in set(self.cat_cols)]
        self.categories_ = {c: sorted(pd.unique(X[c].dropna())) for c in self.cat_cols_}
        self.model_ = LGBMClassifier(
            objective="multiclass",
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            subsample_freq=1,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=1,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
        )
        self.model_.fit(self._cast(X), np.asarray(y))
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(self._cast(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def contributions(self, X):
        """TreeSHAP values on the raw (log-odds) scale.

        Returns (values, bias): values has shape (n_rows, n_classes, n_features);
        bias has shape (n_classes,). For each row and class, values.sum + bias
        equals the model's raw score for that class.
        """
        frame = self._cast(X)
        n_features = len(self.feature_names_in_)
        n_classes = len(self.classes_)
        raw = self.model_.predict(frame, pred_contrib=True)
        raw = np.asarray(raw).reshape(len(frame), n_classes, n_features + 1)
        return raw[:, :, :n_features], raw[0, :, n_features]


def build_estimator(name: str, columns, params: dict | None = None):
    """Create an unfitted estimator for the given model name and feature columns."""
    columns = list(columns)
    cats = [c for c in columns if c in CATEGORICAL_COLUMNS]
    nums = [c for c in columns if c not in CATEGORICAL_COLUMNS]
    params = dict(params or {})

    if name == "dummy":
        return DummyClassifier(strategy="most_frequent")

    if name == "lightgbm":
        merged = {**DEFAULT_LGBM_PARAMS, **params}
        return LGBMStudent(cat_cols=tuple(cats), **merged)

    transformers = []
    if cats:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cats,
            )
        )
    scale = name == "logreg"
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    if nums:
        transformers.append(("numeric", Pipeline(numeric_steps), nums))
    preprocess = ColumnTransformer(transformers)

    if name == "logreg":
        classifier = LogisticRegression(
            C=params.get("C", 0.3), max_iter=5000, class_weight="balanced"
        )
    elif name == "random_forest":
        # The configuration used by the original version of this app.
        classifier = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}.")
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])
