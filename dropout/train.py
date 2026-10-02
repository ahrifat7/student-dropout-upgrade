"""Train, evaluate and export the three stage models.

Usage:
    python -m dropout.train                 # full run (a few minutes on one CPU core)
    python -m dropout.train --n-iter 4      # faster, rougher hyperparameter search

Protocol (important for honest results):
  * 80/20 stratified split with a fixed seed. The 20% test set is locked.
  * All tuning and model selection use 5-fold CV on the 80% training data only.
  * Error analysis (slices, calibration, worst mistakes) uses out-of-fold
    predictions on the training data, so the test set stays untouched.
  * The test set is evaluated once per model, after every choice is frozen.
  * The deployed model is refit on all rows with the chosen hyperparameters.
"""

from __future__ import annotations

import argparse
import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import evaluate as ev  # noqa: E402
from .config import (  # noqa: E402
    CLASSES,
    CV_FOLDS,
    FIGURES_DIR,
    METADATA_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    SEED,
    STAGES,
    SUMMARY_PATH,
)
from .data import cohort_groups, load_dataset, split_train_test, stage_columns  # noqa: E402
from .explain import global_importance  # noqa: E402
from .models import MODEL_NAMES, build_estimator  # noqa: E402
from .registry import fit_stage_model, library_versions, model_path, predict_table  # noqa: E402

TUNED = ("logreg", "lightgbm")


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def run(n_iter: int, folds: int) -> dict:
    for folder in (MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    features, targets = load_dataset()
    X_train, X_test, y_train, y_test = split_train_test(features, targets)
    groups = cohort_groups(X_train)
    _log(f"data {features.shape}; train {len(X_train)}, test {len(X_test)}; cohort groups {groups.nunique()}")

    summary = {
        "dataset": {
            "rows": len(features),
            "features": features.shape[1],
            "class_counts": targets.value_counts().to_dict(),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        },
        "settings": {"seed": SEED, "cv_folds": folds, "lightgbm_search_iterations": n_iter},
        "versions": library_versions(),
        "stages": {},
    }
    metadata = {"versions": library_versions(), "stages": {}}
    importance_by_stage, oof_by_stage, test_by_stage = {}, {}, {}

    for stage in STAGES:
        columns = stage_columns(features.columns, stage)
        _log(f"=== stage '{stage}' ({len(columns)} features) ===")
        info = {"n_features": len(columns), "cv": {}, "best_params": {}, "test_all_models": {}}

        for name in MODEL_NAMES:
            params = ev.tune(name, X_train, y_train, columns, folds, n_iter) if name in TUNED else {}
            info["best_params"][name] = params
            info["cv"][name] = ev.cv_scores(build_estimator(name, columns, params), X_train[columns], y_train, folds)
            _log(f"  {name:14s} CV macro-F1 {info['cv'][name]['macro_f1']['mean']:.3f} ± {info['cv'][name]['macro_f1']['std']:.3f}")

        candidates = [n for n in MODEL_NAMES if n != "dummy"]
        selected = max(candidates, key=lambda n: info["cv"][n]["macro_f1"]["mean"])
        params = info["best_params"][selected]
        info["selected_model"] = selected
        _log(f"  selected: {selected}")

        # --- locked test set: evaluated once, after selection ---------------------
        for name in MODEL_NAMES:
            fitted = build_estimator(name, columns, info["best_params"][name]).fit(X_train[columns], y_train)
            info["test_all_models"][name] = ev.metric_bundle(y_test, fitted.predict(X_test[columns]))
            if name == selected:
                proba = fitted.predict_proba(X_test[columns])
                pred = np.asarray(fitted.classes_)[proba.argmax(1)]
                test_by_stage[stage] = {"confusion": ev.confusion(y_test, pred)}
                info["test"] = {
                    **ev.metric_bundle(y_test, pred),
                    "ci95": ev.bootstrap_ci(y_test, pred),
                    "per_class": ev.per_class_report(y_test, pred),
                    "confusion": ev.confusion(y_test, pred),
                    "capacity": ev.capacity_table(y_test, proba[:, list(fitted.classes_).index("Dropout")]),
                }
                if hasattr(fitted, "contributions"):
                    importance_by_stage[stage] = global_importance(fitted, X_train[columns])

        # --- error analysis on out-of-fold training predictions -------------------
        estimator = build_estimator(selected, columns, params)
        oof = ev.oof_proba(estimator, X_train[columns], y_train, folds)
        oof_pred = np.asarray(CLASSES)[oof.argmax(1)]
        oof_by_stage[stage] = oof
        info["oof"] = {
            **ev.metric_bundle(y_train, oof_pred),
            "per_class": ev.per_class_report(y_train, oof_pred),
            "confusion": ev.confusion(y_train, oof_pred),
        }
        info["calibration"] = ev.calibration(y_train, oof)
        info["group_cv"] = ev.group_cv_macro_f1(estimator, X_train[columns], y_train, groups.to_numpy(), folds)
        ev.slice_table(X_train, y_train, oof_pred).to_csv(REPORTS_DIR / f"slices_{stage}.csv", index=False)
        ev.worst_mistakes(X_train[columns], y_train, oof).to_csv(REPORTS_DIR / f"worst_mistakes_{stage}.csv", index=False)
        if stage in importance_by_stage:
            importance_by_stage[stage].to_csv(REPORTS_DIR / f"importance_{stage}.csv", index=False)

        # --- deployed model: refit on all rows with the chosen hyperparameters ----
        final = fit_stage_model(stage, features, targets, selected, params)
        joblib.dump(final, model_path(stage), compress=3)
        golden = predict_table(final, features.head(5))["Predicted outcome"].tolist()
        metadata["stages"][stage] = {
            "selected_model": selected,
            "best_params": params,
            "columns": columns,
            "golden_predictions_first5": golden,
        }
        summary["stages"][stage] = info
        _log(f"  test macro-F1 {info['test']['macro_f1']:.3f} (95% CI {info['test']['ci95']['macro_f1'][0]:.3f}-{info['test']['ci95']['macro_f1'][1]:.3f})")

    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_results_csv(summary)
    _make_figures(summary, importance_by_stage, y_train, oof_by_stage)
    _log("done: models/, reports/ written")
    return summary


def _write_results_csv(summary: dict) -> None:
    import pandas as pd

    rows = []
    for stage, info in summary["stages"].items():
        for name in MODEL_NAMES:
            cv = info["cv"][name]
            rows.append(
                {
                    "stage": stage,
                    "model": name,
                    "selected": name == info["selected_model"],
                    "cv_macro_f1_mean": round(cv["macro_f1"]["mean"], 4),
                    "cv_macro_f1_std": round(cv["macro_f1"]["std"], 4),
                    "cv_balanced_accuracy_mean": round(cv["balanced_accuracy"]["mean"], 4),
                    "cv_accuracy_mean": round(cv["accuracy"]["mean"], 4),
                    "test_macro_f1": round(info["test_all_models"][name]["macro_f1"], 4),
                    "test_accuracy": round(info["test_all_models"][name]["accuracy"], 4),
                }
            )
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "results.csv", index=False)


def _make_figures(summary, importance_by_stage, y_train, oof_by_stage) -> None:
    stages = list(STAGES)
    labels = [STAGES[s] for s in stages]

    # 1) model comparison by stage
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.2
    for i, name in enumerate(MODEL_NAMES):
        means = [summary["stages"][s]["cv"][name]["macro_f1"]["mean"] for s in stages]
        stds = [summary["stages"][s]["cv"][name]["macro_f1"]["std"] for s in stages]
        ax.bar(np.arange(3) + (i - 1.5) * width, means, width, yerr=stds, capsize=3, label=name)
    ax.set_xticks(range(3), labels)
    ax.set_ylabel("Cross-validated macro-F1 (training data)")
    ax.set_title("How much does waiting for more information help?")
    ax.legend(ncol=4, loc="upper left", fontsize=8)
    ax.set_ylim(0, 0.9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "stage_ladder.png", dpi=140)
    plt.close(fig)

    # 2) confusion matrices on the locked test set
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    for ax, stage in zip(axes, stages, strict=True):
        cm = np.array(summary["stages"][stage]["test"]["confusion"])
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
        for r in range(3):
            for c in range(3):
                ax.text(c, r, f"{cm[r, c]}\n{norm[r, c]:.0%}", ha="center", va="center",
                        color="white" if norm[r, c] > 0.5 else "black", fontsize=9)
        ax.set_xticks(range(3), CLASSES, fontsize=8)
        ax.set_yticks(range(3), CLASSES, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_title(f"{STAGES[stage]} ({summary['stages'][stage]['selected_model']})", fontsize=9)
    axes[0].set_ylabel("Actual")
    fig.suptitle("Confusion matrices on the locked test set (row = share of actual class)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=140)
    plt.close(fig)

    # 3) calibration (out-of-fold, training data)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0.3, 1], [0.3, 1], "k--", lw=1, label="perfectly calibrated")
    for stage in stages:
        bins = summary["stages"][stage]["calibration"]["bins"]
        ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], marker="o",
                label=f"{STAGES[stage]} (ECE {summary['stages'][stage]['calibration']['ece']:.3f})")
    ax.set_xlabel("Model confidence")
    ax.set_ylabel("Actual accuracy")
    ax.set_title("Is the model's confidence trustworthy?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration.png", dpi=140)
    plt.close(fig)

    # 4) capacity: flag the riskiest k% of students
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random choice")
    for stage in stages:
        cap = summary["stages"][stage]["test"]["capacity"]
        ax.plot([0] + [c["flagged_share"] for c in cap], [0] + [c["recall"] for c in cap], marker="o", label=STAGES[stage])
    ax.set_xlabel("Share of students flagged (highest P(Dropout) first)")
    ax.set_ylabel("Share of all dropouts reached")
    ax.set_title("Advisor capacity vs dropouts reached (test set)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "capacity.png", dpi=140)
    plt.close(fig)

    # 5) global importance for Dropout
    if importance_by_stage:
        fig, axes = plt.subplots(1, len(importance_by_stage), figsize=(5 * len(importance_by_stage), 4.2))
        axes = np.atleast_1d(axes)
        for ax, (stage, table) in zip(axes, importance_by_stage.items(), strict=True):
            top = table.head(8).iloc[::-1]
            ax.barh(top["feature"], top["mean_abs_contribution"])
            ax.set_title(STAGES[stage], fontsize=9)
            ax.tick_params(axis="y", labelsize=7)
            ax.set_xlabel("mean |SHAP|, Dropout")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "importance.png", dpi=140)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-iter", type=int, default=16, help="LightGBM random-search iterations per stage")
    parser.add_argument("--folds", type=int, default=CV_FOLDS)
    args = parser.parse_args()
    run(args.n_iter, args.folds)


if __name__ == "__main__":
    main()
