# src/preprocessing/preprocessor.py

import pandas as pd
from datetime import datetime
import os


class Preprocessor:
    def __init__(
        self,
        raw_path: str = "data/raw/encar_premium.csv",
        save_path: str = "data/processed/encar_processed.csv",
    ):
        self.raw_path = raw_path
        self.save_path = save_path

    def run(self) -> pd.DataFrame:
        print("📄 원본 데이터 불러오는 중...")
        # low_memory=False 로 DtypeWarning 줄이기
        df = pd.read_csv(self.raw_path, low_memory=False)
        print(f"로드 완료: {df.shape}")

        # 1) 필요한 컬럼만 사용
        keep_cols = [
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
        ]
        df = df[keep_cols].copy()

        # 2) Price / Mileage / Year 결측 제거
        before = len(df)
        df = df.dropna(subset=["Price", "Mileage", "Year"])
        print(f"❌ 결측(Price/Mileage/Year) 제거: {before} → {len(df)}")

        # 3) Id 기준 중복 제거
        before = len(df)
        df = df.drop_duplicates(subset="Id")
        print(f"🧹 중복 제거: {before} → {len(df)}")

        # 4) 숫자형 변환
        for col in ["Mileage", "Price", "Year"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Year: 202101.0 → 2021
        def _year_to_int(x):
            if pd.isna(x):
                return None
            s = str(int(x))
            return int(s[:4])

        df["Year"] = df["Year"].apply(_year_to_int)

        before = len(df)
        df = df.dropna(subset=["Year", "Price", "Mileage"])
        df["Year"] = df["Year"].astype(int)
        print(f"❌ 연식/가격/주행거리 재검증 후: {before} → {len(df)}")

        # 5) 차량 나이
        current_year = datetime.now().year
        df["CarAge"] = current_year - df["Year"]

        # 6) 문자열 정리 (양쪽 공백 제거)
        str_cols = [
            "Manufacturer",
            "Model",
            "Badge",
            "BadgeDetail",
            "Transmission",
            "FuelType",
            "OfficeCityState",
        ]
        for c in str_cols:
            df[c] = df[c].astype(str).str.strip()

        # 7) 변속기 정규화
        def norm_trans(x: str) -> str:
            x = x.replace(" ", "")
            if "오토" in x or "AT" in x.upper():
                return "오토"
            if "수동" in x or "MT" in x.upper():
                return "수동"
            return "기타"

        df["Transmission_clean"] = df["Transmission"].apply(norm_trans)

        # 8) 연료 정규화
        def norm_fuel(x: str) -> str:
            if "가솔린" in x:
                return "가솔린"
            if "디젤" in x:
                return "디젤"
            if "LPG" in x:
                return "LPG"
            if "하이브리드" in x:
                return "하이브리드"
            if "전기" in x or "EV" in x.upper():
                return "전기"
            return "기타"

        df["FuelType_clean"] = df["FuelType"].apply(norm_fuel)

        # 9) 최종 컬럼 정리
        final_cols = [
            "Id",
            "Manufacturer",
            "Model",
            "Badge",
            "BadgeDetail",
            "Transmission_clean",
            "FuelType_clean",
            "Mileage",
            "Year",
            "Price",
            "CarAge",
            "OfficeCityState",
        ]
        df = df[final_cols].rename(
            columns={
                "Transmission_clean": "Transmission",
                "FuelType_clean": "FuelType",
            }
        )

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        df.to_csv(self.save_path, index=False, encoding="utf-8-sig")
        print(f"💾 전처리 데이터 저장: {self.save_path}")
        print(f"✅ 최종 데이터 Shape: {df.shape}")

        return df
