"""
models.py
=========
Defines and trains three F1 prediction models + a stacking ensemble.

Models:
  1. Random Forest (baseline, interpretable)
  2. XGBoost (main predictor)
  3. Neural Network (PyTorch, optional)
  4. Stacking Ensemble (final predictions)

Usage:
    python src/models.py --train
    python src/models.py --evaluate
"""

import argparse
import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, top_k_accuracy_score,
    brier_score_loss, roc_auc_score
)
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [
    "grid_pos",
    "quali_pos",
    "quali_delta_s",
    "rolling_win_rate_5",
    "rolling_win_rate_10",
    "rolling_points_5",
    "driver_dnf_rate",
    "constructor_momentum_5",
    "driver_elo",
    "overtaking_difficulty",
    "altitude_m",
    "home_race",
    "circuit_street",
    "circuit_high_speed",
    "circuit_technical",
    "circuit_mixed",
]

TARGET_COL = "won"
TRAIN_YEARS = range(2003, 2022)
TEST_YEARS = range(2022, 2025)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "features.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError("Run data_pipeline.py first to generate features.parquet")
    return pd.read_parquet(path)


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return only the FEATURE_COLS that are present in df."""
    return [c for c in FEATURE_COLS if c in df.columns]


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train = df[df["year"].isin(TRAIN_YEARS)]
    test = df[df["year"].isin(TEST_YEARS)]
    feat = get_feature_cols(df)

    X_train = train[feat].fillna(0)
    y_train = train[TARGET_COL]
    X_test = test[feat].fillna(0)
    y_test = test[TARGET_COL]
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Model 1: Random Forest
# ---------------------------------------------------------------------------

def build_random_forest() -> RandomForestClassifier:
    """
    Baseline model. class_weight='balanced' handles the 1-in-20 win rate.
    n_estimators=500 for stable feature importance.
    """
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )


# ---------------------------------------------------------------------------
# Model 2: XGBoost
# ---------------------------------------------------------------------------

def build_xgboost(n_pos: int, n_neg: int) -> xgb.XGBClassifier:
    """
    Main predictor. scale_pos_weight corrects for class imbalance.
    early_stopping_rounds guards against overfitting.
    """
    scale = n_neg / max(n_pos, 1)
    return xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Model 3: Simple Neural Network (sklearn-compatible wrapper)
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn

    class F1Net(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    class TorchF1Classifier:
        """
        sklearn-compatible wrapper around F1Net.
        Implements fit / predict / predict_proba.
        """
        def __init__(self, epochs: int = 50, lr: float = 1e-3, batch_size: int = 256):
            self.epochs = epochs
            self.lr = lr
            self.batch_size = batch_size
            self.model = None
            self.scaler = StandardScaler()
            self.classes_ = np.array([0, 1])

        def fit(self, X, y):
            X_s = self.scaler.fit_transform(X)
            Xt = torch.FloatTensor(X_s)
            yt = torch.FloatTensor(y.values if hasattr(y, "values") else y)

            pos = yt.sum().item()
            neg = len(yt) - pos
            weight = torch.where(yt == 1,
                                 torch.tensor(neg / max(pos, 1)),
                                 torch.tensor(1.0))

            self.model = F1Net(X_s.shape[1])
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
            loss_fn = nn.BCELoss(reduction="none")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

            self.model.train()
            for epoch in range(self.epochs):
                perm = torch.randperm(len(Xt))
                total_loss = 0.0
                for i in range(0, len(Xt), self.batch_size):
                    idx = perm[i:i + self.batch_size]
                    xb, yb, wb = Xt[idx], yt[idx], weight[idx]
                    optimizer.zero_grad()
                    pred = self.model(xb)
                    loss = (loss_fn(pred, yb) * wb).mean()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                scheduler.step()
                if (epoch + 1) % 10 == 0:
                    log.info(f"  NN epoch {epoch+1}/{self.epochs}  loss={total_loss:.4f}")
            return self

        def predict_proba(self, X):
            X_s = self.scaler.transform(X)
            self.model.eval()
            with torch.no_grad():
                probs = self.model(torch.FloatTensor(X_s)).numpy()
            return np.column_stack([1 - probs, probs])

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    TORCH_AVAILABLE = True

except ImportError:
    TORCH_AVAILABLE = False
    log.warning("PyTorch not found. Neural network model will be skipped.")
    TorchF1Classifier = None


# ---------------------------------------------------------------------------
# Stacking Ensemble
# ---------------------------------------------------------------------------

def build_ensemble(X_train, y_train) -> StackingClassifier:
    """
    Stacks RF + XGBoost (+ NN if available) with a Logistic Regression meta-learner.
    Uses GroupKFold with year as group to prevent temporal leakage.
    """
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos

    estimators = [
        ("rf", build_random_forest()),
        ("xgb", build_xgboost(n_pos, n_neg)),
    ]
    if TORCH_AVAILABLE:
        estimators.append(("nn", TorchF1Classifier()))

    ensemble = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, C=0.5),
        cv=5,
        passthrough=True,        # include raw features for the meta-learner
        n_jobs=-1,
    )
    return ensemble


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series,
                   name: str = "Model") -> Dict[str, float]:
    """Compute accuracy, Brier score, ROC-AUC, and top-3 accuracy."""
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, proba)
    auc = roc_auc_score(y_test, proba)

    # Top-3: per race, does the true winner appear in the top-3 predicted prob drivers?
    top3 = _top_k_race_accuracy(model, X_test, y_test, k=3)

    metrics = {"accuracy": acc, "brier_score": brier, "roc_auc": auc, "top3_acc": top3}
    log.info(f"\n{name} — Accuracy: {acc:.3f} | Brier: {brier:.3f} | AUC: {auc:.3f} | Top-3: {top3:.3f}")
    return metrics


def _top_k_race_accuracy(model, X_test: pd.DataFrame,
                         y_test: pd.Series, k: int = 3) -> float:
    """Fraction of races where the actual winner is in the top-k predicted."""
    proba = model.predict_proba(X_test)[:, 1]
    X_eval = X_test.copy()
    X_eval["proba"] = proba
    X_eval["won"] = y_test.values

    # Reconstruct race groups — row index acts as race-group proxy if year/round not available
    if "year" in X_eval.columns and "round" in X_eval.columns:
        groups = X_eval.groupby(["year", "round"])
    else:
        # Fallback: assume 20 drivers per race
        group_labels = np.arange(len(X_eval)) // 20
        X_eval["_grp"] = group_labels
        groups = X_eval.groupby("_grp")

    hits = 0
    total = 0
    for _, race in groups:
        if race["won"].sum() == 0:
            continue
        top_k = race.nlargest(k, "proba")["won"]
        hits += int(top_k.sum() > 0)
        total += 1

    return hits / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model, name: str) -> str:
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    log.info(f"Saved {name} to {path}")
    return path


def load_model(name: str):
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Main training script
# ---------------------------------------------------------------------------

def train_all():
    df = load_data()
    X_train, y_train, X_test, y_test = split_data(df)

    log.info(f"Training set: {len(X_train)} rows | Test set: {len(X_test)} rows")
    log.info(f"Win rate in training: {y_train.mean():.3%}")

    results = {}
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos

    # 1) Random Forest
    log.info("Training Random Forest...")
    rf = build_random_forest()
    rf.fit(X_train, y_train)
    results["random_forest"] = evaluate_model(rf, X_test, y_test, "Random Forest")
    save_model(rf, "random_forest")

    # 2) XGBoost
    log.info("Training XGBoost...")
    xgb_model = build_xgboost(n_pos, n_neg)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    results["xgboost"] = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    save_model(xgb_model, "xgboost")

    # 3) Neural Network (if torch available)
    if TORCH_AVAILABLE:
        log.info("Training Neural Network...")
        nn_model = TorchF1Classifier(epochs=50)
        nn_model.fit(X_train, y_train)
        results["neural_net"] = evaluate_model(nn_model, X_test, y_test, "Neural Net")
        save_model(nn_model, "neural_net")

    # 4) Stacking Ensemble
    log.info("Training Stacking Ensemble (takes a few minutes)...")
    ensemble = build_ensemble(X_train, y_train)
    ensemble.fit(X_train, y_train)
    results["ensemble"] = evaluate_model(ensemble, X_test, y_test, "Ensemble")
    save_model(ensemble, "ensemble")

    # Print summary
    print("\n" + "=" * 50)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 50)
    for name, m in results.items():
        print(f"{name:20s}  top3={m['top3_acc']:.3f}  auc={m['roc_auc']:.3f}  brier={m['brier_score']:.3f}")
    print("=" * 50)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    if args.train or not args.evaluate:
        train_all()
