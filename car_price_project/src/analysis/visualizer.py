# src/analysis/visualizer.py

import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import platform

if platform.system() == "Windows":
    matplotlib.rc("font", family="Malgun Gothic")  # Windows 기본 폰트
else:
    matplotlib.rc("font", family="AppleGothic")    # Mac
matplotlib.rcParams['axes.unicode_minus'] = False

class Visualizer:
    def __init__(self,
                 data_path: str = "data/processed/encar_processed.csv",
                 save_dir: str = "visualizations"):
        self.data_path = data_path
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        print(f"📄 시각화용 데이터 로드: {df.shape}")
        return df

    # -----------------------------
    # 1) 가격 분포 히스토그램
    # -----------------------------
    def plot_price_hist(self, df: pd.DataFrame):
        plt.figure(figsize=(8, 5))
        # Price 단위: 만원 → 백만원 단위로 변환(보기 좋게)
        price_million = df["Price"] / 100  # 1,870만원 → 18.7
        plt.hist(price_million, bins=50)
        plt.xlabel("가격 (백만원)")
        plt.ylabel("차량 대수")
        plt.title("중고차 가격 분포 (엔카진단 차량)")
        plt.tight_layout()

        path = os.path.join(self.save_dir, "price_hist.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  ✅ 저장: {path}")

    # -----------------------------
    # 2) 제조사별 평균 가격 (상위 15개)
    # -----------------------------
    def plot_price_by_manufacturer(self, df: pd.DataFrame):
        # 제조사별 개수 기준 상위 15개만
        manu_counts = df["Manufacturer"].value_counts()
        top_manus = manu_counts.head(15).index

        sub = df[df["Manufacturer"].isin(top_manus)].copy()
        grp = (
            sub.groupby("Manufacturer")["Price"]
            .mean()
            .sort_values(ascending=False)
        )
        price_million = grp / 100  # 백만원 단위

        plt.figure(figsize=(10, 6))
        plt.bar(price_million.index, price_million.values)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("평균 가격 (백만원)")
        plt.title("제조사별 평균 가격 (상위 15 제조사)")
        plt.tight_layout()

        path = os.path.join(self.save_dir, "price_by_manufacturer_top15.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  ✅ 저장: {path}")

    # -----------------------------
    # 3) 연료 타입별 평균 가격
    # -----------------------------
    def plot_price_by_fuel(self, df: pd.DataFrame):
        grp = (
            df.groupby("FuelType")["Price"]
            .mean()
            .sort_values(ascending=False)
        )
        price_million = grp / 100

        plt.figure(figsize=(8, 5))
        plt.bar(price_million.index, price_million.values)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("평균 가격 (백만원)")
        plt.title("연료 타입별 평균 가격")
        plt.tight_layout()

        path = os.path.join(self.save_dir, "price_by_fueltype.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  ✅ 저장: {path}")

    # -----------------------------
    # 4) 차량 나이 vs 평균 가격 (감가곡선)
    # -----------------------------
    def plot_price_vs_car_age(self, df: pd.DataFrame):
        # 이상한 음수, 0살 등은 제거 (안 맞는 값 조금 정리)
        sub = df[(df["CarAge"] >= 0) & (df["CarAge"] <= 20)].copy()
        grp = (
            sub.groupby("CarAge")["Price"]
            .mean()
            .sort_index()
        )
        price_million = grp / 100

        plt.figure(figsize=(8, 5))
        plt.plot(price_million.index, price_million.values, marker="o")
        plt.xlabel("차량 나이 (년)")
        plt.ylabel("평균 가격 (백만원)")
        plt.title("차량 나이에 따른 평균 중고차 가격 (감가 곡선)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        path = os.path.join(self.save_dir, "price_vs_car_age.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  ✅ 저장: {path}")

    # -----------------------------
    # 5) 주행거리 vs 가격 산점도
    # -----------------------------
    def plot_price_vs_mileage(self, df: pd.DataFrame):
        # 너무 많으면 샘플링 (최대 5000개 정도)
        sub = df.copy()
        if len(sub) > 5000:
            sub = sub.sample(5000, random_state=42)

        price_million = sub["Price"] / 100
        mileage_10k = sub["Mileage"] / 10000  # 만 km 단위

        plt.figure(figsize=(8, 5))
        plt.scatter(mileage_10k, price_million, alpha=0.3)
        plt.xlabel("주행거리 (만 km)")
        plt.ylabel("가격 (백만원)")
        plt.title("주행거리 vs 가격 산점도 (샘플링)")
        plt.tight_layout()

        path = os.path.join(self.save_dir, "price_vs_mileage_scatter.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  ✅ 저장: {path}")

    # -----------------------------
    # 전체 실행
    # -----------------------------
    def run(self):
        df = self.load_data()
        print("🖼 시각화 생성 중...")

        self.plot_price_hist(df)
        self.plot_price_by_manufacturer(df)
        self.plot_price_by_fuel(df)
        self.plot_price_vs_car_age(df)
        self.plot_price_vs_mileage(df)

        print("✅ 시각화 완료! 'visualizations' 폴더를 확인하세요.")


if __name__ == "__main__":
    viz = Visualizer()
    viz.run()
