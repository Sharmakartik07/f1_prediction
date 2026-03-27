"""
evaluate.py
===========
Model evaluation utilities:
  - SHAP feature importance plots
  - Calibration curves
  - Per-circuit accuracy breakdown
  - Confusion matrix for top-3 predictions
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import logging

from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

log = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_NAMES = [
    "Grid position", "Qualifying position", "Quali delta (s)",
    "Win rate (5 races)", "Win rate (10 races)", "Avg points (5 races)",
    "DNF rate", "Constructor momentum", "Driver ELO",
    "Overtaking difficulty", "Altitude (m)", "Home race",
    "Circuit: street", "Circuit: high speed", "Circuit: technical", "Circuit: mixed",
]


def load_model_and_data():
    df = pd.read_parquet(os.path.join(DATA_DIR, "features.parquet"))
    test = df[df["year"] >= 2022]
    feat_cols = [c for c in df.columns if c in [
        "grid_pos", "quali_pos", "quali_delta_s",
        "rolling_win_rate_5", "rolling_win_rate_10", "rolling_points_5",
        "driver_dnf_rate", "constructor_momentum_5", "driver_elo",
        "overtaking_difficulty", "altitude_m", "home_race",
        "circuit_street", "circuit_high_speed", "circuit_technical", "circuit_mixed",
    ]]
    X_test = test[feat_cols].fillna(0)
    y_test = test["won"]

    with open(os.path.join(MODEL_DIR, "xgboost.pkl"), "rb") as f:
        model = pickle.load(f)
    return model, X_test, y_test, test


def plot_feature_importance(model, feature_names: list, top_n: int = 12):
    """Plot XGBoost feature importances (gain-based)."""
    try:
        importance = model.feature_importances_
    except AttributeError:
        log.warning("Model has no feature_importances_ attribute.")
        return

    if len(importance) != len(feature_names):
        feature_names = [f"Feature {i}" for i in range(len(importance))]

    pairs = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    pairs = pairs[:top_n]
    names, vals = zip(*pairs)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(names)))
    bars = ax.barh(names[::-1], vals[::-1], color=colors[::-1], edgecolor="white", height=0.65)
    ax.set_xlabel("Feature Importance (gain)", fontsize=11)
    ax.set_title("F1 Predictor — XGBoost Feature Importance", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=10)
    for bar, val in zip(bars, vals[::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, color="#444")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150)
    log.info(f"Saved feature importance chart to {path}")
    plt.show()


def plot_calibration_curve(model, X_test, y_test, name: str = "XGBoost"):
    """Reliability diagram: predicted probability vs actual win rate."""
    proba = model.predict_proba(X_test)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", color="#E8593C", lw=2,
            markersize=7, label=f"{name} (calibrated)")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (actual win rate)")
    ax.set_title("Calibration Curve — F1 Win Predictor", fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "calibration_curve.png")
    plt.savefig(path, dpi=150)
    log.info(f"Saved calibration curve to {path}")
    plt.show()


def per_circuit_accuracy(model, X_test, y_test, df_test):
    """Top-3 accuracy grouped by circuit type."""
    proba = model.predict_proba(X_test)[:, 1]
    df_eval = df_test.copy()
    df_eval["proba"] = proba
    df_eval["y_test"] = y_test.values

    results = []
    for ctype, group in df_eval.groupby("circuit_type"):
        races = group.groupby(["year", "round"])
        hits, total = 0, 0
        for _, race in races:
            if race["y_test"].sum() == 0:
                continue
            top3 = race.nlargest(3, "proba")["y_test"]
            hits += int(top3.sum() > 0)
            total += 1
        if total > 0:
            results.append({"circuit_type": ctype, "top3_acc": hits / total, "n_races": total})

    results_df = pd.DataFrame(results).sort_values("top3_acc", ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"street": "#E8593C", "high_speed": "#3B8BD4",
              "technical": "#7F77DD", "mixed": "#1D9E75"}
    for _, row in results_df.iterrows():
        ax.bar(row["circuit_type"], row["top3_acc"],
               color=colors.get(row["circuit_type"], "#888"),
               edgecolor="white", width=0.5)
        ax.text(row["circuit_type"], row["top3_acc"] + 0.01,
                f"{row['top3_acc']:.0%}", ha="center", fontsize=10)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Top-3 accuracy", fontsize=11)
    ax.set_title("Top-3 accuracy by circuit type", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "circuit_accuracy.png")
    plt.savefig(path, dpi=150)
    log.info(f"Saved circuit accuracy chart to {path}")
    plt.show()
    return results_df


def shap_summary(model, X_test):
    """
    SHAP beeswarm summary plot.
    Requires: pip install shap
    """
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        short_names = [
            "Grid pos", "Quali pos", "Quali delta",
            "Win rate 5R", "Win rate 10R", "Avg pts 5R",
            "DNF rate", "Constr. momentum", "ELO",
            "Overtaking diff.", "Altitude", "Home race",
            "Cir: street", "Cir: fast", "Cir: tech", "Cir: mixed",
        ]
        feat_labels = short_names[:X_test.shape[1]]

        plt.figure(figsize=(9, 6))
        shap.summary_plot(shap_values, X_test, feature_names=feat_labels,
                          max_display=12, show=False)
        plt.title("SHAP Feature Impact — F1 Win Predictor", fontweight="bold")
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, "shap_summary.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        log.info(f"Saved SHAP summary to {path}")
        plt.show()
    except ImportError:
        log.info("Install shap for SHAP plots: pip install shap")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model, X_test, y_test, df_test = load_model_and_data()
    feat_names = FEATURE_NAMES[:X_test.shape[1]]
    plot_feature_importance(model, feat_names)
    plot_calibration_curve(model, X_test, y_test)
    if "circuit_type" in df_test.columns:
        per_circuit_accuracy(model, X_test, y_test, df_test)
    shap_summary(model, X_test)
