# src/model/trainer.py

import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class ModelTrainer:
    data_path: str = "data/processed/encar_processed.csv"
    model_path: str = "models/price_model.pkl"

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        print(f"📊 학습 데이터 로드: {df.shape}")
        return df

    def train(self):
        df = self.load()

        # -------------------------
        # 1) Feature / Target 분리
        # -------------------------
        numeric_features = ["CarAge", "Mileage"]
        categorical_features = [
            "Manufacturer",
            "Model",
            "Badge",
            "FuelType",
            "Transmission",
            "OfficeCityState",
        ]

        feature_cols = numeric_features + categorical_features

        X = df[feature_cols].copy()
        y = df["Price"].astype(float)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # -------------------------
        # 2) 전처리 파이프라인
        # -------------------------
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", numeric_features),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_features,
                ),
            ]
        )

        # -------------------------
        # 3) 모델 정의 (RandomForest)
        # -------------------------
        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
        )

        model = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", rf),
            ]
        )

        # -------------------------
        # 4) 학습
        # -------------------------
        print("🚀 모델 학습 중...")
        model.fit(X_train, y_train)

        # -------------------------
        # 5) 평가
        # -------------------------
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        rmse = mean_squared_error(y_test, pred, squared=False)
        r2 = r2_score(y_test, pred)

        print("\n📌 평가 결과 (단위: 'Price'가 만원이라고 가정)")
        print(f"  🎯 MAE  : {mae:,.3f} (만 원)")
        print(f"  📉 RMSE : {rmse:,.3f} (만 원)")
        print(f"  📈 R²   : {r2:.4f}")

        # -------------------------
        # 6) 저장
        # -------------------------
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model, self.model_path)
        print(f"\n💾 전체 파이프라인 저장 완료: {self.model_path}")

        return model, {"mae": mae, "rmse": rmse, "r2": r2}
