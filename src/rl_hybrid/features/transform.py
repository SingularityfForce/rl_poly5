from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


@dataclass
class FeatureTransformer:
    scaler: StandardScaler
    asset_encoder: LabelEncoder
    feature_cols: list[str]

    @classmethod
    def fit(cls, df: pd.DataFrame, feature_cols: list[str]) -> "FeatureTransformer":
        scaler = StandardScaler()
        scaler.fit(df[feature_cols])
        enc = LabelEncoder()
        enc.fit(df["asset"])
        return cls(scaler=scaler, asset_encoder=enc, feature_cols=feature_cols)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.feature_cols] = self.scaler.transform(out[self.feature_cols])
        out["asset_id"] = self.asset_encoder.transform(out["asset"])
        return out

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "FeatureTransformer":
        return joblib.load(path)
