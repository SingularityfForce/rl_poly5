from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss


@dataclass
class SupervisedMetrics:
    logloss: float
    auc: float
    brier: float


class TabularWinnerModel:
    def __init__(self):
        self.model = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.05)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x, y)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)[:, 1]


class GRUWinnerModel(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


def eval_binary(y_true: np.ndarray, p: np.ndarray) -> SupervisedMetrics:
    return SupervisedMetrics(
        logloss=float(log_loss(y_true, p, labels=[0, 1])),
        auc=float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) > 1 else 0.5,
        brier=float(brier_score_loss(y_true, p)),
    )
