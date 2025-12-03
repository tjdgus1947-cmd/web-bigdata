# src/model/predictor.py

import os
from datetime import datetime

import joblib
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

        # 전처리 결과 CSV의 컬럼 구조(12개)에 맞춰서 한 줄짜리 DataFrame을 만들 것이다.
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
            "Price",            # 예측 시에는 dummy 값 사용
            "OfficeCityState",
            "CarAge",
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

        # Price 는 타깃이라 예측 시 의미 없으니 0 넣어둔다 (모델은 이 컬럼 안 씀)
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

        # trainer에서 저장한 Pipeline이 그대로 들어있으니 바로 predict 가능
        pred = self.model.predict(df)[0]
        return float(pred)
