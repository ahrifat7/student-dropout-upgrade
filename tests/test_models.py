import numpy as np
import pytest
from sklearn.base import clone
from sklearn.metrics import f1_score

from dropout.data import load_dataset, split_train_test, stage_columns
from dropout.models import LGBMStudent, build_estimator

FAST = {"n_estimators": 80}


@pytest.fixture(scope="module")
def split():
    features, targets = load_dataset()
    return split_train_test(features, targets)


@pytest.fixture(scope="module")
def fitted(split):
    X_tr, _, y_tr, _ = split
    columns = stage_columns(X_tr.columns, "sem1")
    return build_estimator("lightgbm", columns, FAST).fit(X_tr[columns], y_tr), columns


def test_probabilities_are_valid(fitted, split):
    model, columns = fitted
    _, X_te, _, _ = split
    proba = model.predict_proba(X_te[columns])
    assert proba.shape == (len(X_te), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert list(model.classes_) == ["Dropout", "Enrolled", "Graduate"]


def test_contributions_are_additive(fitted, split):
    """TreeSHAP values plus the bias must reproduce the model's raw scores."""
    model, columns = fitted
    _, X_te, _, _ = split
    sample = X_te[columns].head(10)
    values, bias = model.contributions(sample)
    raw = model.model_.predict(model._cast(sample), raw_score=True)
    assert values.shape == (10, 3, len(columns))
    assert np.allclose(values.sum(axis=2) + bias, raw, atol=1e-8)


def test_unseen_category_and_missing_value_do_not_crash(fitted, split):
    model, columns = fitted
    _, X_te, _, _ = split
    odd = X_te[columns].head(3).copy()
    odd["Course"] = 99999
    odd.loc[odd.index[0], "Admission grade"] = np.nan
    assert model.predict_proba(odd).shape == (3, 3)


def test_estimator_is_clonable(fitted):
    model, _ = fitted
    assert isinstance(clone(model), LGBMStudent)


def test_enrollment_model_cannot_see_semester_columns(split):
    X_tr, _, y_tr, _ = split
    columns = stage_columns(X_tr.columns, "enrollment")
    model = build_estimator("lightgbm", columns, FAST).fit(X_tr[columns], y_tr)
    assert not any("sem" in c for c in model.feature_names_in_)


@pytest.mark.parametrize("name", ["logreg", "random_forest"])
def test_baseline_pipelines_fit_and_predict(split, name):
    X_tr, X_te, y_tr, _ = split
    columns = stage_columns(X_tr.columns, "enrollment")
    model = build_estimator(name, columns).fit(X_tr[columns].head(800), y_tr.head(800))
    assert len(model.predict(X_te[columns].head(5))) == 5


def test_model_clearly_beats_majority_baseline(fitted, split):
    """The original test (balanced accuracy > 0.3) is passed by a majority guesser (0.333)."""
    model, columns = fitted
    _, X_te, _, y_te = split
    macro_f1 = f1_score(y_te, model.predict(X_te[columns]), average="macro")
    dummy = build_estimator("dummy", columns).fit(split[0][columns], split[2])
    dummy_f1 = f1_score(y_te, dummy.predict(X_te[columns]), average="macro")
    assert macro_f1 > dummy_f1 + 0.3
