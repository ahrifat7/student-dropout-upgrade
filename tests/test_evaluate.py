import numpy as np
import pandas as pd

from dropout import evaluate as ev
from dropout.config import CLASSES


def test_metric_bundle_perfect_and_majority():
    y = np.array(["Dropout", "Enrolled", "Graduate", "Graduate"])
    assert ev.metric_bundle(y, y) == {"accuracy": 1.0, "balanced_accuracy": 1.0, "macro_f1": 1.0}
    majority = np.array(["Graduate"] * 4)
    assert ev.metric_bundle(y, majority)["balanced_accuracy"] == 1 / 3


def test_capacity_table_with_perfect_ranking():
    y = np.array(["Dropout"] * 20 + ["Graduate"] * 80)
    p = np.r_[np.linspace(0.99, 0.9, 20), np.linspace(0.1, 0.01, 80)]
    rows = {r["flagged_share"]: r for r in ev.capacity_table(y, p, fractions=(0.1, 0.2, 0.4))}
    assert rows[0.1]["precision"] == 1.0 and rows[0.1]["recall"] == 0.5
    assert rows[0.2]["precision"] == 1.0 and rows[0.2]["recall"] == 1.0
    assert rows[0.4]["precision"] == 0.5
    assert abs(rows[0.2]["lift"] - 5.0) < 1e-9


def test_calibration_is_zero_for_confident_correct_predictions():
    y = np.array(CLASSES * 10)
    proba = np.eye(3)[np.tile([0, 1, 2], 10)]
    result = ev.calibration(y, proba)
    assert result["ece"] == 0.0 and result["brier"] == 0.0


def test_calibration_detects_overconfidence():
    y = np.array(["Graduate"] * 100)
    proba = np.tile([0.0, 0.0, 1.0], (100, 1))
    proba[:50] = [1.0, 0.0, 0.0]  # confidently wrong half of the time
    assert abs(ev.calibration(y, proba)["ece"] - 0.5) < 1e-9


def test_bootstrap_interval_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.choice(CLASSES, 300)
    pred = np.where(rng.random(300) < 0.7, y, rng.choice(CLASSES, 300))
    ci = ev.bootstrap_ci(y, pred, n_boot=200)
    point = ev.metric_bundle(y, pred)
    for key, (low, high) in ci.items():
        assert low <= point[key] <= high


def test_slice_table_skips_tiny_groups():
    n = 120
    features = pd.DataFrame(
        {"Age at enrollment": [18] * 100 + [45] * 20, "Course": [1] * 90 + [2] * 30, "Gender": [0, 1] * 60}
    )
    y = np.array(["Dropout", "Graduate"] * 60)
    table = ev.slice_table(features, y, y, min_n=30)
    assert "40+" not in set(table.loc[table["slice"] == "Age group", "group"])  # only 20 rows
    assert set(table.loc[table["slice"] == "Course", "group"]) == {"1", "2"}
    assert len(features) == n


def test_worst_mistakes_are_wrong_and_sorted_by_confidence():
    features = pd.DataFrame({"x": range(4)})
    y = np.array(["Dropout", "Enrolled", "Graduate", "Graduate"])
    proba = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.6, 0.2, 0.2], [0.05, 0.05, 0.9]])
    worst = ev.worst_mistakes(features, y, proba, k=5)
    assert list(worst["x"]) == [2]
    assert (worst["true_outcome"] != worst["predicted_outcome"]).all()
