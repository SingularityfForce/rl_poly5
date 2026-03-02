from __future__ import annotations
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from rl_hybrid.models.supervised import TabularWinnerModel, eval_binary
from rl_hybrid.utils.serialization import save_json


def build_tabular_dataset(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    y = (df["winner"] == "UP").astype(int).to_numpy()
    x = df[feature_cols].to_numpy(dtype=float)
    return x, y


def train_supervised(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str], outdir: str) -> dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    model = TabularWinnerModel()
    xtr, ytr = build_tabular_dataset(train_df, feature_cols)
    xva, yva = build_tabular_dataset(val_df, feature_cols)
    xte, yte = build_tabular_dataset(test_df, feature_cols)
    model.fit(xtr, ytr)
    pva = model.predict_proba(xva)
    pte = model.predict_proba(xte)
    m_val = eval_binary(yva, pva)
    m_test = eval_binary(yte, pte)
    joblib.dump(model, out / "tabular_model.joblib")

    prob_true, prob_pred = calibration_curve(yte, pte, n_bins=10)
    plt.figure(figsize=(5, 4))
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], "--")
    plt.title("Calibration")
    plt.savefig(out / "calibration.png", dpi=120, bbox_inches="tight")
    plt.close()
    metrics = {"val": m_val.__dict__, "test": m_test.__dict__}
    save_json(metrics, out / "metrics.json")
    return metrics
