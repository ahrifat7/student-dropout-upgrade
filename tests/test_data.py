import numpy as np
import pandas as pd
import pytest

from dropout.config import CATEGORICAL_COLUMNS, STAGES
from dropout.data import (
    cohort_groups,
    input_specs,
    load_dataset,
    prepare_prediction_frame,
    split_train_test,
    stage_columns,
)


@pytest.fixture(scope="module")
def data():
    return load_dataset()


def test_dataset_shape_and_labels(data):
    features, targets = data
    assert features.shape == (4424, 36)
    assert "Target" not in features.columns
    assert set(targets.unique()) == {"Dropout", "Enrolled", "Graduate"}
    assert targets.value_counts().to_dict() == {"Graduate": 2209, "Dropout": 1421, "Enrolled": 794}
    assert not features.isna().any().any()


def test_header_whitespace_is_stripped(data):
    features, _ = data
    assert "Daytime/evening attendance" in features.columns
    assert all(c == c.strip() for c in features.columns)


def test_categorical_columns_exist(data):
    features, _ = data
    assert set(CATEGORICAL_COLUMNS) <= set(features.columns)


def test_stages_are_nested_and_leak_free(data):
    features, _ = data
    a = stage_columns(features.columns, "enrollment")
    b = stage_columns(features.columns, "sem1")
    c = stage_columns(features.columns, "sem2")
    assert (len(a), len(b), len(c)) == (24, 30, 36)
    assert set(a) < set(b) < set(c)
    assert not any("sem" in col for col in a)
    assert not any("2nd sem" in col for col in b)


def test_unknown_stage_rejected(data):
    with pytest.raises(ValueError):
        stage_columns(data[0].columns, "week3")


def test_stage_names_cover_all_stage_logic():
    assert set(STAGES) == {"enrollment", "sem1", "sem2"}


def test_split_is_stratified_reproducible_and_disjoint(data):
    features, targets = data
    X_tr, X_te, y_tr, y_te = split_train_test(features, targets)
    X_tr2, X_te2, _, _ = split_train_test(features, targets)
    assert (len(X_tr), len(X_te)) == (3539, 885)
    assert list(X_te.index) == list(X_te2.index)
    assert set(X_tr.index).isdisjoint(X_te.index)
    overall = targets.value_counts(normalize=True)
    for label, share in y_te.value_counts(normalize=True).items():
        assert abs(share - overall[label]) < 0.01


def test_cohort_groups_follow_macro_combinations(data):
    features, _ = data
    assert cohort_groups(features).nunique() == 10


def test_prepare_prediction_frame_reports_missing_columns(data):
    features, _ = data
    with pytest.raises(ValueError, match="Missing required feature columns"):
        prepare_prediction_frame(features.drop(columns=["Age at enrollment"]), list(features.columns))


def test_prepare_prediction_frame_ignores_extra_columns_and_empty_rows(data):
    features, _ = data
    messy = features.head(3).copy()
    messy["notes"] = "ignored"
    messy.loc[len(messy)] = np.nan
    prepared = prepare_prediction_frame(messy, list(features.columns))
    assert list(prepared.columns) == list(features.columns)
    assert len(prepared) == 3


def test_prepare_prediction_frame_rejects_empty_upload(data):
    features, _ = data
    with pytest.raises(ValueError):
        prepare_prediction_frame(pd.DataFrame(columns=list(features.columns)), list(features.columns))


def test_input_specs_defaults_are_valid(data):
    features, _ = data
    specs = input_specs(features)
    assert set(specs) == set(features.columns)
    for spec in specs.values():
        if spec["kind"] == "categorical":
            assert spec["default"] in spec["options"]
        else:
            assert spec["min"] <= spec["default"] <= spec["max"]
