"""Streamlit app for the predictive maintenance model.

Run with:  streamlit run app.py
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scripts.predict import read_input, score
from src import config
from src.data import load_dataset

st.set_page_config(page_title="Predictive Maintenance", page_icon="🔧", layout="wide")


@st.cache_resource
def get_bundle():
    return joblib.load(config.MODEL_PATH)


@st.cache_data
def get_data():
    return load_dataset(config.DATA_PATH)


@st.cache_data
def get_metrics():
    with open(config.METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def get_test_predictions():
    return pd.read_csv(config.PREDICTIONS_PATH, parse_dates=[config.DATE])


def decide(scores, threshold):
    scores = scores.copy()
    scores["decision"] = scores["failure_probability"].ge(threshold).map(
        {True: "MAINTENANCE", False: "OK"}
    )
    return scores


def show_device(scores, threshold):
    last = scores.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Latest reading", str(last["date"].date()))
    c2.metric("Failure probability", f"{last['failure_probability']:.3f}")
    c3.metric("Decision", last["decision"])

    chart = scores.set_index("date")[["failure_probability"]].copy()
    chart["threshold"] = threshold
    st.line_chart(chart)

    st.dataframe(scores.tail(15).iloc[::-1], use_container_width=True, hide_index=True)


bundle = get_bundle()
metrics = get_metrics()
saved_threshold = float(bundle["threshold"])

st.title("🔧 Predictive Maintenance")
st.caption("Flags machine-days that look likely to fail, from daily sensor readings.")

threshold = st.sidebar.slider(
    "Alert threshold", 0.01, 0.90, round(saved_threshold, 3), 0.01,
    help="Lower catches more failures but sends more machines for inspection. "
         "The default was tuned on the validation split.",
)
st.sidebar.caption(f"Saved threshold: {saved_threshold:.3f}")
st.sidebar.caption("Decision support only. A person should make the final call.")

tab_score, tab_rank, tab_model = st.tabs(["Score a machine", "Risk ranking", "Model results"])

# ---------------------------------------------------------------- score
with tab_score:
    source = st.radio("Readings from", ["Dataset", "Upload CSV"], horizontal=True)

    if source == "Dataset":
        data = get_data()
        failed = set(data.loc[data[config.TARGET] == 1, config.DEVICE])
        only_failed = st.checkbox("Only machines that failed in the data", value=True)
        devices = sorted(failed) if only_failed else sorted(data[config.DEVICE].unique())
        device = st.selectbox("Machine", devices)
        readings = data[data[config.DEVICE] == device].reset_index(drop=True)
        scores = decide(score(readings, bundle), threshold)
        scores["actual_failure"] = readings[config.TARGET].to_numpy()
        show_device(scores, threshold)
        st.caption(
            "Note: many of these machines were part of the training set, so scores here "
            "look better than they would on new machines. See Risk ranking for held-out ones."
        )
    else:
        st.write(
            "Columns needed: `date` (m/d/YYYY), `device`, `metric1` to `metric9`. "
            "Include each machine's recent history, since deviation features are "
            "computed from the rows you give."
        )
        up = st.file_uploader("CSV file", type="csv")
        if up is not None:
            try:
                readings = read_input(up).sort_values([config.DEVICE, config.DATE])
                readings = readings.reset_index(drop=True)
                scores = decide(score(readings, bundle), threshold)
            except Exception as exc:
                st.error(f"Couldn't read that file: {exc}")
            else:
                latest = scores.groupby("device").tail(1).sort_values(
                    "failure_probability", ascending=False
                )
                st.subheader("Latest reading per machine")
                st.dataframe(latest, use_container_width=True, hide_index=True)
                pick = st.selectbox("Look at one machine", latest["device"])
                show_device(scores[scores["device"] == pick].reset_index(drop=True), threshold)
                st.download_button(
                    "Download all scores", scores.to_csv(index=False), "scores.csv", "text/csv"
                )

# ---------------------------------------------------------------- ranking
with tab_rank:
    st.write(
        "Machines from the held-out test split (never seen in training), ranked by their "
        "highest failure probability."
    )
    preds = get_test_predictions()
    per_device = (
        preds.groupby(config.DEVICE)
        .agg(max_probability=("failure_probability", "max"), failed=(config.TARGET, "max"))
        .sort_values("max_probability", ascending=False)
        .reset_index()
    )
    per_device["flagged"] = per_device["max_probability"] >= threshold

    flagged = per_device[per_device["flagged"]]
    c1, c2, c3 = st.columns(3)
    c1.metric("Machines", len(per_device))
    c2.metric("Flagged", len(flagged))
    c3.metric("Flagged that really failed", int(flagged["failed"].sum()))

    top = st.slider("Show top", 10, 100, 25, 5)
    st.dataframe(per_device.head(top), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- model
with tab_model:
    t = metrics["test_at_threshold"]
    r = metrics["test_ranking"]
    st.write(f"Selected model: **{metrics['selected_model']}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PR-AUC", f"{r['pr_auc']:.3f}", help=f"No-skill baseline: {metrics['baseline_pr_auc']:.4f}")
    c2.metric("ROC-AUC", f"{r['roc_auc']:.3f}")
    c3.metric("Precision", f"{t['precision']:.2f}")
    c4.metric("Recall", f"{t['recall']:.2f}")
    st.caption(
        f"At threshold {t['threshold']:.3f} on the test split: "
        f"{t['tp']} caught, {t['fn']} missed, {t['fp']} false alarms."
    )

    st.subheader("Model comparison")
    comp = pd.read_csv(config.COMPARISON_PATH).drop(columns=["best_params"])
    st.dataframe(comp.round(4), use_container_width=True, hide_index=True)

    figs = config.FIGURE_DIR
    left, right = st.columns(2)
    left.image(str(figs / "pr_curves.png"), caption="Precision-recall")
    right.image(str(figs / "roc_curves.png"), caption="ROC")
    left.image(str(figs / "confusion_matrix.png"), caption="Confusion matrix")
    right.image(str(figs / "feature_importance.png"), caption="Feature importance")
