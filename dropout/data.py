"""Loading, validating and splitting the UCI student dropout dataset."""

from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    CATEGORICAL_COLUMNS,
    DATA_PATH,
    LABELS_PATH,
    MACRO_COLUMNS,
    SEED,
    STAGES,
    TARGET,
    TEST_SIZE,
)


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Strip stray whitespace from column names (one original header ends in a tab)."""
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    if not frame.columns.is_unique:
        raise ValueError("Column names must be unique after trimming whitespace.")
    return frame


def load_dataset(path=DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Return numeric features and string targets.

    The file is semicolon-separated and may start with a UTF-8 byte-order mark.
    """
    frame = normalize_columns(pd.read_csv(path, sep=";", encoding="utf-8-sig"))
    if TARGET not in frame.columns:
        raise ValueError(f"Dataset must include a '{TARGET}' column.")
    targets = frame[TARGET].astype("string").str.strip()
    valid = targets.notna() & targets.ne("")
    features = frame.loc[valid].drop(columns=TARGET).apply(pd.to_numeric, errors="coerce")
    targets = targets.loc[valid].astype(str)
    if features.empty or features.shape[1] == 0:
        raise ValueError("Dataset must contain labeled rows and at least one feature.")
    return features.reset_index(drop=True), targets.reset_index(drop=True)


def stage_columns(columns, stage: str) -> list[str]:
    """Columns that are known at a given prediction moment.

    enrollment: everything except semester results
    sem1:       enrollment + 1st-semester results
    sem2:       everything
    """
    if stage not in STAGES:
        raise ValueError(f"Unknown stage '{stage}'. Choose from {list(STAGES)}.")
    columns = list(columns)
    sem1 = [c for c in columns if "1st sem" in c]
    sem2 = [c for c in columns if "2nd sem" in c]
    if stage == "enrollment":
        return [c for c in columns if c not in sem1 + sem2]
    if stage == "sem1":
        return [c for c in columns if c not in sem2]
    return columns


def categorical_in(columns) -> list[str]:
    return [c for c in columns if c in CATEGORICAL_COLUMNS]


def split_train_test(features: pd.DataFrame, targets: pd.Series):
    """Stratified, reproducible 80/20 split. The test part stays locked until the end."""
    return train_test_split(
        features, targets, test_size=TEST_SIZE, random_state=SEED, stratify=targets
    )


def cohort_groups(features: pd.DataFrame) -> pd.Series:
    """Group id from the macro-indicator combination (an enrollment-period proxy)."""
    keys = features[list(MACRO_COLUMNS)].round(4).astype(str).agg("|".join, axis=1)
    return keys.astype("category").cat.codes


def prepare_prediction_frame(frame: pd.DataFrame, feature_columns) -> pd.DataFrame:
    """Validate an uploaded frame and return the numeric feature columns in order."""
    frame = normalize_columns(frame)
    missing = [c for c in feature_columns if c not in frame.columns]
    if missing:
        raise ValueError("Missing required feature columns: " + ", ".join(missing))
    features = frame.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    features = features.dropna(how="all")
    if features.empty:
        raise ValueError("The uploaded file must contain at least one populated row.")
    return features


def input_specs(features: pd.DataFrame) -> dict:
    """Observed ranges and options used to build the single-student form."""
    specs = {}
    for column in features.columns:
        series = features[column].dropna()
        if column in CATEGORICAL_COLUMNS:
            options = sorted(series.unique().tolist())
            specs[column] = {
                "kind": "categorical",
                "options": [int(v) if float(v).is_integer() else float(v) for v in options],
                "default": int(series.mode().iloc[0]),
            }
        else:
            is_int = bool((series % 1 == 0).all())
            specs[column] = {
                "kind": "int" if is_int else "float",
                "min": int(series.min()) if is_int else float(series.min()),
                "max": int(series.max()) if is_int else float(series.max()),
                "default": int(series.median()) if is_int else float(series.median()),
            }
    return specs


def load_labels() -> dict:
    """Optional human-readable labels for coded columns (data/data_dictionary.json).

    Format: {"Marital status": {"1": "single", ...}, ...}. We never guess labels:
    if the file is absent the app shows the raw codes.
    """
    if LABELS_PATH.exists():
        return json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return {}
