# src/model/trainer.py

import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


@dataclass
class ModelTrainer:
    data_path: str = "data/processed/encar_processed.csv"
    model_path: str = "models/price_model.pkl"
    use_ensemble: bool = True  # 앙상블 모델 사용 여부
    tune_hyperparameters: bool = False  # 하이퍼파라미터 튜닝 여부

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        print(f"📊 학습 데이터 로드: {df.shape}")
        return df

    def add_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """파생 변수 생성"""
        print("🔧 Feature Engineering 진행 중...")
        
        # 1) 연평균 주행거리
        df['Mileage_per_year'] = df['Mileage'] / (df['CarAge'] + 1)  # 0으로 나누기 방지
        
        # 2) 고급 브랜드 여부
        luxury_brands = ['BMW', 'Mercedes-Benz', '메르세데스-벤츠', 'Audi', '아우디', 
                        'Porsche', '포르쉐', 'Lexus', '렉서스', 'Genesis', '제네시스']
        df['Is_luxury_brand'] = df['Manufacturer'].isin(luxury_brands).astype(int)
        
        # 3) 차량 연식 그룹 (신차, 준신차, 중고, 노후)
        df['Age_group'] = pd.cut(df['CarAge'], 
                                 bins=[-1, 2, 5, 10, 100], 
                                 labels=['신차급', '준신차', '중고', '노후'])
        
        # 4) 주행거리 그룹
        df['Mileage_group'] = pd.cut(df['Mileage'], 
                                      bins=[0, 30000, 80000, 150000, 1000000],
                                      labels=['저주행', '중주행', '고주행', '과다주행'])
        
        print(f"✅ 파생 변수 추가 완료: {df.shape}")
        return df

    def train(self):
        df = self.load()
        df = self.add_feature_engineering(df)

        # -------------------------
        # 1) Feature / Target 분리
        # -------------------------
        numeric_features = ["CarAge", "Mileage", "Mileage_per_year", "Is_luxury_brand"]
        categorical_features = [
            "Manufacturer",
            "Model",
            "Badge",
            "FuelType",
            "Transmission",
            "OfficeCityState",
            "Age_group",
            "Mileage_group"
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
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_features,
                ),
            ]
        )

        # -------------------------
        # 3) 모델 정의
        # -------------------------
        if self.use_ensemble:
            print("🚀 앙상블 모델(RF + XGBoost) 학습 중...")
            
            rf = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42,
            )
            
            xgb = XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            
            # VotingRegressor로 앙상블
            ensemble = VotingRegressor(
                estimators=[('rf', rf), ('xgb', xgb)],
                n_jobs=-1
            )
            
            model = Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", ensemble),
                ]
            )
            
        else:
            print("🚀 RandomForest 모델 학습 중...")
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
        # 4) 하이퍼파라미터 튜닝 (선택적)
        # -------------------------
        if self.tune_hyperparameters and not self.use_ensemble:
            print("🔍 하이퍼파라미터 튜닝 중... (시간이 걸릴 수 있습니다)")
            param_grid = {
                'model__n_estimators': [200, 300, 400],
                'model__max_depth': [15, 20, None],
                'model__min_samples_split': [2, 5, 10],
            }
            
            grid_search = GridSearchCV(
                model, 
                param_grid, 
                cv=3, 
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"✅ 최적 파라미터: {grid_search.best_params_}")
        else:
            model.fit(X_train, y_train)

        # -------------------------
        # 5) 평가
        # -------------------------
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Train 성능
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
        train_r2 = r2_score(y_train, y_pred_train)
        
        # Test 성능
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
        test_r2 = r2_score(y_test, y_pred_test)

        print("\n" + "="*50)
        print(" 모델 평가 결과 (단위: 만원)")
        print("="*50)
        print(f"\n[Train Set]")
        print(f"   MAE  : {train_mae:,.1f} 만원")
        print(f"   RMSE : {train_rmse:,.1f} 만원")
        print(f"   R²   : {train_r2:.4f}")
        
        print(f"\n[Test Set]")
        print(f"   MAE  : {test_mae:,.1f} 만원")
        print(f"   RMSE : {test_rmse:,.1f} 만원")
        print(f"   R²   : {test_r2:.4f}")
        
        # 오버피팅 체크
        if train_r2 - test_r2 > 0.1:
            print(f"\n⚠️  과적합 의심: Train R² ({train_r2:.4f}) >> Test R² ({test_r2:.4f})")
        
        print("="*50)

        # -------------------------
        # 6) 가격대별 성능 분석
        # -------------------------
        self._analyze_by_price_range(y_test, y_pred_test)

        # -------------------------
        # 7) 저장
        # -------------------------
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model, self.model_path)
        print(f"\n💾 모델 저장 완료: {self.model_path}")
        
        # 메타데이터 저장
        metadata = {
            "train_mae": train_mae,
            "train_rmse": train_rmse,
            "train_r2": train_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
            "use_ensemble": self.use_ensemble,
            "n_train": len(X_train),
            "n_test": len(X_test)
        }
        
        metadata_path = self.model_path.replace('.pkl', '_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        print(f"💾 메타데이터 저장: {metadata_path}")

        return model, metadata

    def _analyze_by_price_range(self, y_true, y_pred):
        """가격대별 MAE 분석"""
        print("\n" + "="*50)
        print("💰 가격대별 성능 분석")
        print("="*50)
        
        df_eval = pd.DataFrame({
            'true': y_true,
            'pred': y_pred,
            'error': np.abs(y_true - y_pred)
        })
        
        # 가격대 구간 설정 (만원 단위)
        bins = [0, 500, 1000, 2000, 3000, 10000]
        labels = ['~500만', '500~1000만', '1000~2000만', '2000~3000만', '3000만~']
        
        df_eval['price_range'] = pd.cut(df_eval['true'], bins=bins, labels=labels)
        
        for price_range in labels:
            subset = df_eval[df_eval['price_range'] == price_range]
            if len(subset) > 0:
                mae = subset['error'].mean()
                count = len(subset)
                print(f"  {price_range:15s}: MAE = {mae:>8,.1f} 만원 (n={count:>5,})")


if __name__ == "__main__":
    # 기본 학습
    trainer = ModelTrainer(use_ensemble=True, tune_hyperparameters=False)
    trainer.train()