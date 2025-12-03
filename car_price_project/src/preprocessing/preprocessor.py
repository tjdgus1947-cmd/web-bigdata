# src/preprocessing/preprocessor.py

import pandas as pd
from datetime import datetime
import os


class Preprocessor:
    def __init__(
        self,
        raw_path: str = "data/raw/encar_premium.csv",
        save_path: str = "data/processed/encar_processed.csv",
        remove_outliers: bool = True,
    ):
        self.raw_path = raw_path
        self.save_path = save_path
        self.remove_outliers = remove_outliers

    def run(self) -> pd.DataFrame:
        print("📄 원본 데이터 불러오는 중...")
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

        # 6) 비정상 데이터 필터링
        df = self._filter_invalid_data(df)

        # 7) 이상치 제거 (선택적)
        if self.remove_outliers:
            df = self._remove_outliers(df)

        # 8) 문자열 정리 (양쪽 공백 제거)
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

        # 9) 변속기 정규화
        def norm_trans(x: str) -> str:
            x = x.replace(" ", "")
            if "오토" in x or "AT" in x.upper():
                return "오토"
            if "수동" in x or "MT" in x.upper():
                return "수동"
            return "기타"

        df["Transmission_clean"] = df["Transmission"].apply(norm_trans)

        # 10) 연료 정규화
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

        # 11) 최종 컬럼 정리
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

        # 12) 데이터 품질 리포트
        self._print_quality_report(df)

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        df.to_csv(self.save_path, index=False, encoding="utf-8-sig")
        print(f"💾 전처리 데이터 저장: {self.save_path}")
        print(f"✅ 최종 데이터 Shape: {df.shape}")

        return df

    def _filter_invalid_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """명백히 잘못된 데이터 필터링"""
        before = len(df)
        
        # 1) 가격이 너무 낮거나 높은 경우 (10만원 미만, 10억 이상)
        df = df[(df["Price"] >= 10) & (df["Price"] <= 100000)]
        
        # 2) 주행거리가 음수이거나 비정상적으로 높은 경우 (100만km 이상)
        df = df[(df["Mileage"] >= 0) & (df["Mileage"] <= 1000000)]
        
        # 3) 차량 나이가 음수이거나 50년 이상
        df = df[(df["CarAge"] >= 0) & (df["CarAge"] <= 50)]
        
        # 4) 연식이 미래거나 너무 과거 (1980년 이전)
        current_year = datetime.now().year
        df = df[(df["Year"] >= 1980) & (df["Year"] <= current_year + 1)]
        
        removed = before - len(df)
        if removed > 0:
            print(f"🚫 비정상 데이터 제거: {removed:,}건 ({removed/before*100:.1f}%)")
        
        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """IQR 방식으로 이상치 제거"""
        print("\n🔍 이상치 탐지 및 제거 중...")
        before = len(df)
        
        for col in ['Price', 'Mileage']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            before_col = len(df)
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            removed = before_col - len(df)
            
            if removed > 0:
                print(f"  - {col}: {removed:,}건 제거 "
                      f"(범위: {lower_bound:,.0f} ~ {upper_bound:,.0f})")
        
        total_removed = before - len(df)
        print(f"✅ 총 {total_removed:,}건의 이상치 제거됨 ({total_removed/before*100:.1f}%)")
        
        return df

    def _print_quality_report(self, df: pd.DataFrame):
        """데이터 품질 리포트 출력"""
        print("\n" + "="*50)
        print("📊 데이터 품질 리포트")
        print("="*50)
        
        print(f"\n✅ 최종 레코드 수: {len(df):,}건")
        
        print(f"\n[가격 통계]")
        print(f"  - 평균: {df['Price'].mean():,.0f} 만원")
        print(f"  - 중앙값: {df['Price'].median():,.0f} 만원")
        print(f"  - 최소: {df['Price'].min():,.0f} 만원")
        print(f"  - 최대: {df['Price'].max():,.0f} 만원")
        
        print(f"\n[주행거리 통계]")
        print(f"  - 평균: {df['Mileage'].mean():,.0f} km")
        print(f"  - 중앙값: {df['Mileage'].median():,.0f} km")
        print(f"  - 최소: {df['Mileage'].min():,.0f} km")
        print(f"  - 최대: {df['Mileage'].max():,.0f} km")
        
        print(f"\n[차량 나이 통계]")
        print(f"  - 평균: {df['CarAge'].mean():.1f}년")
        print(f"  - 중앙값: {df['CarAge'].median():.1f}년")
        
        print(f"\n[제조사 분포 (상위 10)]")
        top_manus = df['Manufacturer'].value_counts().head(10)
        for manu, count in top_manus.items():
            print(f"  - {manu}: {count:,}대 ({count/len(df)*100:.1f}%)")
        
        print(f"\n[연료 타입 분포]")
        for fuel, count in df['FuelType'].value_counts().items():
            print(f"  - {fuel}: {count:,}대 ({count/len(df)*100:.1f}%)")
        
        print("="*50 + "\n")


if __name__ == "__main__":
    prep = Preprocessor(remove_outliers=True)
    prep.run()