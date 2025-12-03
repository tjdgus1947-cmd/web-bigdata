# src/model/predictor.py

import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd


class PricePredictor:
    """
    학습된 price_model.pkl을 로드해서
    단일 차량/여러 차량의 가격을 예측하는 클래스
    """

    def __init__(self, model_path: str = "models/price_model.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"모델 파일을 찾을 수 없습니다: {model_path}\n"
                f"→ 먼저 main.py를 실행해서 모델을 학습/저장해 주세요."
            )
        self.model = joblib.load(model_path)

        # Feature Engineering용 컬럼 추가
        self.columns = [
            "Id",
            "Manufacturer",
            "Model",
            "Badge",
            "BadgeDetail",
            "Transmission",
            "FuelType",
            "Year",
            "Mileage",
            "Price",
            "OfficeCityState",
            "CarAge",
            "Mileage_per_year",
            "Is_luxury_brand",
            "Age_group",
            "Mileage_group"
        ]

    def _build_input_row(
        self,
        manufacturer: str,
        model: str,
        badge: str,
        year: int,
        mileage: float,
        fuel_type: str,
        transmission: str,
        region: str,
    ) -> pd.DataFrame:
        """
        사용자가 입력한 정보를 기반으로
        학습 때 사용한 컬럼 구조와 동일한 DataFrame 한 줄 생성
        """

        current_year = datetime.now().year
        car_age = current_year - int(year)

        # Feature Engineering
        mileage_per_year = mileage / (car_age + 1)
        
        luxury_brands = ['BMW', 'Mercedes-Benz', '메르세데스-벤츠', 'Audi', '아우디',
                        'Porsche', '포르쉐', 'Lexus', '렉서스', 'Genesis', '제네시스']
        is_luxury = 1 if manufacturer in luxury_brands else 0
        
        # Age group
        if car_age <= 2:
            age_group = '신차급'
        elif car_age <= 5:
            age_group = '준신차'
        elif car_age <= 10:
            age_group = '중고'
        else:
            age_group = '노후'
        
        # Mileage group
        if mileage <= 30000:
            mileage_group = '저주행'
        elif mileage <= 80000:
            mileage_group = '중주행'
        elif mileage <= 150000:
            mileage_group = '고주행'
        else:
            mileage_group = '과다주행'

        data = {
            "Id": [0],
            "Manufacturer": [manufacturer],
            "Model": [model],
            "Badge": [badge if badge else "미확인"],
            "BadgeDetail": [""],
            "Transmission": [transmission],
            "FuelType": [fuel_type],
            "Year": [int(year)],
            "Mileage": [float(mileage)],
            "Price": [0.0],
            "OfficeCityState": [region],
            "CarAge": [car_age],
            "Mileage_per_year": [mileage_per_year],
            "Is_luxury_brand": [is_luxury],
            "Age_group": [age_group],
            "Mileage_group": [mileage_group]
        }

        df = pd.DataFrame(data, columns=self.columns)
        return df

    def predict_price(
        self,
        manufacturer: str,
        model: str,
        badge: str,
        year: int,
        mileage: float,
        fuel_type: str,
        transmission: str,
        region: str,
    ) -> float:
        """
        단일 차량 가격 예측 (단위: 만원)
        """
        df = self._build_input_row(
            manufacturer=manufacturer,
            model=model,
            badge=badge,
            year=year,
            mileage=mileage,
            fuel_type=fuel_type,
            transmission=transmission,
            region=region,
        )

        pred = self.model.predict(df)[0]
        return float(pred)

    def predict_with_interval(
        self,
        manufacturer: str,
        model: str,
        badge: str,
        year: int,
        mileage: float,
        fuel_type: str,
        transmission: str,
        region: str,
        confidence: float = 0.9
    ) -> dict:
        """
        가격 예측 + 신뢰 구간 반환
        
        Returns:
            dict: {
                'prediction': 예측 가격,
                'lower': 하한 가격,
                'upper': 상한 가격,
                'confidence': 신뢰 수준
            }
        """
        df = self._build_input_row(
            manufacturer=manufacturer,
            model=model,
            badge=badge,
            year=year,
            mileage=mileage,
            fuel_type=fuel_type,
            transmission=transmission,
            region=region,
        )

        # 예측
        pred = self.model.predict(df)[0]

        # 신뢰 구간 추정 (RandomForest/XGBoost의 경우)
        # 개별 트리들의 예측값 분포를 활용
        try:
            if hasattr(self.model, 'named_steps'):
                actual_model = self.model.named_steps['model']
                
                # VotingRegressor인 경우
                if hasattr(actual_model, 'estimators_'):
                    # 각 base estimator의 예측값 수집
                    predictions = []
                    for estimator in actual_model.estimators_:
                        if hasattr(estimator, 'estimators_'):  # RandomForest
                            # 전처리 적용
                            X_transformed = self.model.named_steps['preprocess'].transform(df)
                            tree_preds = [tree.predict(X_transformed)[0] 
                                        for tree in estimator.estimators_]
                            predictions.extend(tree_preds)
                    
                    if predictions:
                        predictions = np.array(predictions)
                        lower_percentile = (1 - confidence) / 2 * 100
                        upper_percentile = (1 + confidence) / 2 * 100
                        
                        lower = np.percentile(predictions, lower_percentile)
                        upper = np.percentile(predictions, upper_percentile)
                    else:
                        # 트리 예측값을 못 구한 경우 단순 추정
                        margin = pred * 0.15  # ±15%
                        lower = pred - margin
                        upper = pred + margin
                else:
                    # 단일 모델인 경우
                    margin = pred * 0.15
                    lower = pred - margin
                    upper = pred + margin
            else:
                margin = pred * 0.15
                lower = pred - margin
                upper = pred + margin

        except Exception as e:
            # 예외 발생 시 단순 추정
            margin = pred * 0.15
            lower = pred - margin
            upper = pred + margin

        return {
            'prediction': float(pred),
            'lower': float(max(0, lower)),  # 음수 방지
            'upper': float(upper),
            'confidence': confidence
        }