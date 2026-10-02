"""Student Support Radar: a decision-support demo for early dropout risk.

Run locally:  streamlit run app.py
Models and reports are created by:  python -m dropout.train
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from dropout.config import CLASSES, FIGURES_DIR, REPORTS_DIR, ROOT, STAGES, SUMMARY_PATH
from dropout.data import (
    input_specs,
    load_dataset,
    load_labels,
    prepare_prediction_frame,
)
from dropout.explain import explain_row
from dropout.registry import load_stage_model, model_columns, predict_table

st.set_page_config(page_title="Student Support Radar", page_icon=":mag:", layout="wide")

STAGE_HELP = {
    "enrollment": "Only information known when the student enrolls. Earliest, and hardest.",
    "sem1": "Adds first-semester results. Available about half a year after enrollment.",
    "sem2": "Adds second-semester results. Strongest signal, but latest: many students have already left.",
}
FORM_GROUPS = {
    "Application": ["Application mode", "Application order", "Course", "Daytime/evening attendance",
                    "Previous qualification", "Previous qualification (grade)", "Admission grade"],
    "Student profile": ["Marital status", "Gender", "Age at enrollment", "Nacionality", "International",
                        "Displaced", "Educational special needs"],
    "Family background": ["Mother's qualification", "Father's qualification",
                          "Mother's occupation", "Father's occupation"],
    "Finances": ["Debtor", "Tuition fees up to date", "Scholarship holder"],
    "National economy at enrollment": ["Unemployment rate", "Inflation rate", "GDP"],
}


# ---------------------------------------------------------------- cached loaders
@st.cache_data(show_spinner=False)
def get_data():
    return load_dataset()


@st.cache_data(show_spinner=False)
def get_summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Loading model...")
def get_model(stage: str):
    return load_stage_model(stage)


def read_text(path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else "_File not found. Run `python -m dropout.train`._"


def pct(value: float) -> str:
    return f"{value:.1%}"


def stage_picker(key: str, default: str = "enrollment") -> str:
    return st.radio(
        "When is the prediction made?",
        list(STAGES),
        index=list(STAGES).index(default),
        format_func=lambda s: STAGES[s],
        horizontal=True,
        key=key,
        help="Each stage only uses information available at that moment.",
    )


# ---------------------------------------------------------------- shell
summary = get_summary()
features, targets = get_data()

st.sidebar.title("Student Support Radar")
page = st.sidebar.radio(
    "Page",
    ["Overview", "Predict a student", "Score a cohort", "Model comparison", "Error analysis", "About and limits"],
)
st.sidebar.caption(f"{len(features):,} students - {features.shape[1]} features - 3 outcomes")
st.sidebar.caption("Data: UCI 'Predict Students' Dropout and Academic Success' (CC BY 4.0).")

st.title("Student Support Radar")
st.caption(
    "Early-warning estimates of Dropout, Enrolled or Graduate. A decision-support demo, "
    "never a verdict on an individual student."
)


# ---------------------------------------------------------------- pages
def page_overview():
    st.subheader("How early can dropout be seen?")
    st.write(
        "Advisors can only help students they notice in time. This project trains one model per "
        "moment (at enrollment, after semester 1, after semester 2) and measures how much "
        "predictive power each extra piece of information adds."
    )
    rows = []
    for stage, info in summary["stages"].items():
        test = info["test"]
        rows.append(
            {
                "Prediction moment": STAGES[stage],
                "Model": info["selected_model"],
                "Macro-F1": f"{test['macro_f1']:.2f}  ({test['ci95']['macro_f1'][0]:.2f}-{test['ci95']['macro_f1'][1]:.2f})",
                "Accuracy": pct(test["accuracy"]),
                "Dropouts found": pct(test["per_class"]["Dropout"]["recall"]),
                "Top 20% flagged reaches": pct(test["capacity"][1]["recall"]) + " of dropouts",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(
        f"Locked test set of {summary['dataset']['test_rows']} students, evaluated once after all choices were made. "
        "Macro-F1 averages the three outcomes equally; brackets show a 95% bootstrap interval."
    )
    left, right = st.columns(2)
    with left:
        st.image(str(FIGURES_DIR / "stage_ladder.png"))
    with right:
        st.image(str(FIGURES_DIR / "capacity.png"))
    st.info(
        "Reading this honestly: the strongest numbers need second-semester results, when many students have "
        "already disengaged. The enrollment-only model is the realistic early warning, and it is much weaker."
    )


def widget_bounds(spec: dict) -> tuple[float, float]:
    """Allow some headroom beyond the observed range, but not below zero where the data has none."""
    span = max(abs(spec["max"] - spec["min"]), 1)
    low = spec["min"] if spec["min"] >= 0 else spec["min"] - span
    return low, spec["max"] + span


def render_form(stage: str, model, specs: dict, labels: dict) -> pd.DataFrame:
    columns = model_columns(model)
    values = {}
    st.caption("Starting values are the most typical student in the dataset. Change them to explore.")
    sem_groups = {
        "First-semester results": [c for c in columns if "1st sem" in c],
        "Second-semester results": [c for c in columns if "2nd sem" in c],
    }
    for title, group in {**FORM_GROUPS, **sem_groups}.items():
        group = [c for c in group if c in columns]
        if not group:
            continue
        with st.expander(title, expanded=title in ("Application", "First-semester results", "Second-semester results")):
            cols = st.columns(2)
            for i, column in enumerate(group):
                spec = specs[column]
                key = f"{stage}-{column}"
                with cols[i % 2]:
                    if spec["kind"] == "categorical":
                        mapping = labels.get(column, {})
                        values[column] = st.selectbox(
                            column,
                            spec["options"],
                            index=spec["options"].index(spec["default"]),
                            format_func=lambda v, m=mapping: f"{v} - {m[str(v)]}" if str(v) in m else str(v),
                            key=key,
                            help="Coded value as in the UCI dataset." if not mapping else None,
                        )
                    elif spec["kind"] == "int":
                        low, high = widget_bounds(spec)
                        values[column] = st.number_input(
                            column, min_value=int(low), max_value=int(high), value=spec["default"], step=1, key=key,
                            help=f"Observed range in the data: {spec['min']} to {spec['max']}.",
                        )
                    else:
                        low, high = widget_bounds(spec)
                        values[column] = st.number_input(
                            column, min_value=float(low), max_value=float(high), value=float(spec["default"]),
                            step=0.1, key=key,
                            help=f"Observed range in the data: {spec['min']:.1f} to {spec['max']:.1f}.",
                        )
    return pd.DataFrame([values], columns=columns).astype(float)


def page_predict():
    st.subheader("Estimate one student")
    stage = stage_picker("predict-stage")
    st.caption(STAGE_HELP[stage])
    model, source = get_model(stage)
    specs, labels = input_specs(features), load_labels()
    row = render_form(stage, model, specs, labels)

    result = predict_table(model, row).iloc[0]
    outcome = result["Predicted outcome"]
    test = summary["stages"][stage]["test"]
    st.divider()
    left, right = st.columns([1, 1.4])
    with left:
        st.metric("Most likely outcome", outcome)
        probs = pd.Series({c: float(result[f"Probability: {c}"]) for c in model.classes_})
        st.bar_chart(probs)
        st.caption(
            f"At this prediction moment the model's overall macro-F1 on unseen students is {test['macro_f1']:.2f}; "
            f"it finds {pct(test['per_class']['Dropout']['recall'])} of actual dropouts. "
            "Probabilities are model scores, not certainties."
        )
    with right:
        if hasattr(model, "contributions"):
            st.markdown(f"**Why: factors pushing towards or away from '{outcome}'**")
            table = explain_row(model, row, outcome, top_k=8).iloc[::-1]
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.barh(
                [f"{f} = {v:g}" for f, v in zip(table["feature"], table["value"], strict=True)],
                table["contribution"],
                color=["#2e7d32" if c > 0 else "#c62828" for c in table["contribution"]],
            )
            ax.axvline(0, color="black", lw=0.8)
            ax.set_xlabel("Contribution (log-odds). Green: towards, red: away")
            ax.tick_params(axis="y", labelsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Per-student explanations are available for the LightGBM models only.")
    st.warning("Decision support only. Use it to offer help earlier, never to penalise or exclude a student.")
    st.caption(f"Model source: {source}.")


def page_cohort():
    st.subheader("Score a cohort")
    stage = stage_picker("cohort-stage")
    model, _ = get_model(stage)
    needed = model_columns(model)
    st.write(
        f"Upload a CSV with these {len(needed)} columns (names unchanged). Other columns, including `Target`, are ignored."
    )
    st.download_button(
        "Download blank CSV template",
        data=pd.DataFrame(columns=needed).to_csv(index=False),
        file_name=f"template_{stage}.csv",
        mime="text/csv",
    )
    st.caption("The app code does not save uploads, but please do not upload real student data to a public demo.")
    uploaded = st.file_uploader("Student records (CSV)", type=["csv"], key=f"upload-{stage}")
    if uploaded is None:
        return
    try:
        records = pd.read_csv(uploaded, sep=None, engine="python", encoding="utf-8-sig")
        if len(records) > 20000:
            raise ValueError("Please upload at most 20,000 rows.")
        prepared = prepare_prediction_frame(records, needed)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(str(error))
        return
    scored = predict_table(model, prepared)
    results = pd.concat([scored, records.loc[prepared.index].reset_index(drop=True).set_index(prepared.index)], axis=1)
    st.caption(f"Scored {len(results):,} records. Scores are model estimates, not certainties.")
    left, right = st.columns([1, 2])
    with left:
        st.bar_chart(results["Predicted outcome"].value_counts().reindex(CLASSES, fill_value=0))
    with right:
        st.dataframe(results.sort_values("Probability: Dropout", ascending=False))
    st.download_button(
        "Download scored cohort", data=results.to_csv(index=False),
        file_name="scored_cohort.csv", mime="text/csv",
    )


def page_comparison():
    st.subheader("Model comparison")
    st.write(
        "Four models, three prediction moments. Hyperparameters and the final model per stage were chosen "
        "with 5-fold cross-validation on the 80% training data only. The 20% test set was used once at the end."
    )
    results = pd.read_csv(REPORTS_DIR / "results.csv")
    results["CV macro-F1"] = results.apply(lambda r: f"{r.cv_macro_f1_mean:.3f} +/- {r.cv_macro_f1_std:.3f}", axis=1)
    view = results.rename(columns={"stage": "Stage", "model": "Model", "selected": "Chosen",
                                   "test_macro_f1": "Test macro-F1", "test_accuracy": "Test accuracy"})
    view["Stage"] = view["Stage"].map(STAGES)
    st.dataframe(view[["Stage", "Model", "Chosen", "CV macro-F1", "Test macro-F1", "Test accuracy"]], hide_index=True)
    st.image(str(FIGURES_DIR / "stage_ladder.png"))
    st.caption(
        "Dummy always predicts the most common outcome (Graduate). Logistic regression is the simple baseline; "
        "random forest is the configuration used by the first version of this app."
    )


def page_errors():
    st.subheader("Where does the model fail?")
    stage = stage_picker("error-stage")
    info = summary["stages"][stage]
    test = info["test"]
    st.markdown("**Per-outcome quality on the locked test set**")
    per_class = pd.DataFrame(test["per_class"]).T.rename_axis("Outcome").reset_index()
    st.dataframe(per_class, hide_index=True)
    st.image(str(FIGURES_DIR / "confusion_matrices.png"))

    st.markdown("**Subgroups (out-of-fold predictions on training data)**")
    slices = pd.read_csv(REPORTS_DIR / f"slices_{stage}.csv")
    chosen = st.selectbox("Subgroup type", slices["slice"].unique())
    st.dataframe(slices[slices["slice"] == chosen].drop(columns="slice").round(3), hide_index=True)
    st.caption("Groups with fewer than 30 students are left out because their scores would be mostly noise.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Calibration**")
        cal = info["calibration"]
        st.write(f"Expected calibration error: {cal['ece']:.3f}; Brier score: {cal['brier']:.3f}.")
        st.image(str(FIGURES_DIR / "calibration.png"))
    with right:
        st.markdown("**If advisors can only contact a few students**")
        cap = pd.DataFrame(test["capacity"])
        cap["flagged_share"] = cap["flagged_share"].map(pct)
        for col in ("precision", "recall"):
            cap[col] = cap[col].map(pct)
        cap["lift"] = cap["lift"].map(lambda v: f"{v:.2f}x")
        st.dataframe(cap, hide_index=True)

    group = info["group_cv"]
    st.markdown(
        f"**Cohort robustness:** holding out whole economic-period groups gives macro-F1 {group['mean']:.3f}, "
        f"versus {info['cv'][info['selected_model']]['macro_f1']['mean']:.3f} with ordinary cross-validation."
    )
    with st.expander("20 most confident mistakes (out-of-fold)"):
        st.dataframe(pd.read_csv(REPORTS_DIR / f"worst_mistakes_{stage}.csv"))
    if (FIGURES_DIR / "importance.png").exists():
        with st.expander("Which features matter most for Dropout?"):
            st.image(str(FIGURES_DIR / "importance.png"))
    st.divider()
    st.markdown(read_text(REPORTS_DIR / "error_analysis.md"))


def page_about():
    st.markdown(read_text(ROOT / "docs" / "MODEL_CARD.md"))


{
    "Overview": page_overview,
    "Predict a student": page_predict,
    "Score a cohort": page_cohort,
    "Model comparison": page_comparison,
    "Error analysis": page_errors,
    "About and limits": page_about,
}[page]()
