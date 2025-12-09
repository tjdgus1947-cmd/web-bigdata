# src/analysis/correlation_analyzer.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import platform

if platform.system() == "Windows":
    matplotlib.rc("font", family="Malgun Gothic")
else:
    matplotlib.rc("font", family="AppleGothic")
matplotlib.rcParams['axes.unicode_minus'] = False


class CorrelationAnalyzer:
    def __init__(self,
                 data_path: str = "data/processed/encar_processed.csv",
                 save_dir: str = "visualizations/correlation"):
        self.data_path = data_path
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        print(f"📊 데이터 로드: {df.shape}")
        return df

    def calculate_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """수치형 변수들의 상관계수 계산"""
        print("\n" + "="*60)
        print("📈 상관관계 분석")
        print("="*60)
        
        # 수치형 변수만 선택
        numeric_cols = ['Price', 'Year', 'Mileage', 'CarAge']
        corr_df = df[numeric_cols].corr()
        
        print("\n📊 상관계수 매트릭스:")
        print(corr_df.to_string())
        
        # 가격과의 상관관계만 추출
        print("\n🎯 가격(Price)과의 상관계수:")
        price_corr = corr_df['Price'].sort_values(ascending=False)
        for var, corr_val in price_corr.items():
            if var != 'Price':
                direction = "양의" if corr_val > 0 else "음의"
                strength = self._get_correlation_strength(abs(corr_val))
                print(f"  - {var:12s}: {corr_val:>7.3f}  ({direction} 상관관계, {strength})")
        
        return corr_df

    def _get_correlation_strength(self, corr_value: float) -> str:
        """상관계수 강도 해석"""
        if corr_value >= 0.7:
            return "매우 강함"
        elif corr_value >= 0.5:
            return "강함"
        elif corr_value >= 0.3:
            return "중간"
        elif corr_value >= 0.1:
            return "약함"
        else:
            return "매우 약함"

    def plot_correlation_heatmap(self, corr_df: pd.DataFrame):
        """상관계수 히트맵"""
        print("\n📈 상관계수 히트맵 생성 중...")
        
        plt.figure(figsize=(10, 8))
        
        # 히트맵 생성
        mask = np.triu(np.ones_like(corr_df, dtype=bool))  # 상삼각 마스크
        sns.heatmap(corr_df, 
                    annot=True,  # 숫자 표시
                    fmt='.3f',   # 소수점 3자리
                    cmap='coolwarm',  # 색상 맵
                    center=0,    # 0을 중심으로
                    vmin=-1, vmax=1,
                    square=True,
                    linewidths=1,
                    cbar_kws={"shrink": 0.8},
                    mask=mask)  # 상삼각만 표시
        
        plt.title('변수 간 상관관계 히트맵', fontsize=16, pad=20)
        plt.tight_layout()
        
        path = os.path.join(self.save_dir, "correlation_heatmap.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 저장: {path}")

    def plot_correlation_bar(self, df: pd.DataFrame):
        """가격과의 상관계수 막대그래프"""
        print("\n📈 상관계수 막대그래프 생성 중...")
        
        # 수치형 변수만 선택
        numeric_cols = ['Year', 'Mileage', 'CarAge']
        corr_with_price = df[numeric_cols + ['Price']].corr()['Price'].drop('Price')
        corr_with_price = corr_with_price.sort_values()
        
        # 한글 이름 매핑
        name_map = {
            'Year': '연식',
            'Mileage': '주행거리',
            'CarAge': '차량 나이'
        }
        corr_with_price.index = [name_map.get(x, x) for x in corr_with_price.index]
        
        # 플롯
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#3498db' if x < 0 else '#e74c3c' for x in corr_with_price.values]
        bars = ax.barh(corr_with_price.index, corr_with_price.values, color=colors)
        
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel('상관계수', fontsize=12)
        ax.set_title('가격과의 상관관계', fontsize=14, pad=15)
        ax.grid(True, alpha=0.3, axis='x')
        
        # 값 표시
        for i, (idx, val) in enumerate(corr_with_price.items()):
            ax.text(val + (0.03 if val > 0 else -0.03), i, 
                   f'{val:.3f}', 
                   va='center', 
                   ha='left' if val > 0 else 'right',
                   fontsize=11,
                   fontweight='bold')
        
        plt.tight_layout()
        
        path = os.path.join(self.save_dir, "correlation_with_price.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 저장: {path}")

    def plot_scatter_with_correlation(self, df: pd.DataFrame):
        """주요 변수들과 가격의 산점도 + 상관계수"""
        print("\n📈 산점도 with 상관계수 생성 중...")
        
        variables = [
            ('Year', '연식'),
            ('Mileage', '주행거리'),
            ('CarAge', '차량 나이')
        ]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (var, var_name) in enumerate(variables):
            ax = axes[idx]
            
            # 샘플링 (너무 많으면)
            plot_df = df.sample(min(3000, len(df)), random_state=42)
            
            # 산점도
            ax.scatter(plot_df[var], plot_df['Price'], alpha=0.3, s=10)
            
            # 상관계수 계산
            corr = df[[var, 'Price']].corr().iloc[0, 1]
            
            # 추세선
            z = np.polyfit(plot_df[var], plot_df['Price'], 1)
            p = np.poly1d(z)
            ax.plot(plot_df[var].sort_values(), 
                   p(plot_df[var].sort_values()), 
                   "r--", linewidth=2, alpha=0.8)
            
            ax.set_xlabel(var_name, fontsize=11)
            ax.set_ylabel('가격 (만원)', fontsize=11)
            ax.set_title(f'{var_name} vs 가격\n(상관계수: {corr:.3f})', 
                        fontsize=12, pad=10)
            ax.grid(True, alpha=0.3)
            
            # 상관계수 텍스트 박스
            strength = self._get_correlation_strength(abs(corr))
            direction = "양의" if corr > 0 else "음의"
            textstr = f'{direction} 상관관계\n{strength}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        path = os.path.join(self.save_dir, "scatter_with_correlation.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 저장: {path}")

    def generate_correlation_table(self, corr_df: pd.DataFrame):
        """PDF용 상관관계 표 생성"""
        print("\n📋 상관관계 표 생성 중...")
        
        # 가격과의 상관관계만
        price_corr = corr_df['Price'][['Price', 'Year', 'Mileage']]
        
        # 표 형식으로 출력
        print("\n" + "="*50)
        print("📊 변수 간 상관계수 (PDF용)")
        print("="*50)
        print("\n변수         가격      연식    주행거리")
        print("-" * 50)
        
        rows = ['가격', '연식', '주행거리']
        vars = ['Price', 'Year', 'Mileage']
        
        for i, row_name in enumerate(rows):
            values = []
            for var in vars:
                val = corr_df.loc[vars[i], var]
                values.append(f"{val:7.2f}")
            print(f"{row_name:8s}  {'  '.join(values)}")
        
        print("="*50)
        
        # CSV로도 저장
        export_df = corr_df.loc[vars, vars]
        export_df.index = rows
        export_df.columns = rows
        
        csv_path = os.path.join(self.save_dir, "correlation_table.csv")
        export_df.to_csv(csv_path, encoding='utf-8-sig')
        print(f"\n💾 CSV 저장: {csv_path}")

    def analyze_categorical_correlation(self, df: pd.DataFrame):
        """범주형 변수와 가격의 관계 분석"""
        print("\n📊 범주형 변수 분석...")
        
        categorical_vars = ['FuelType', 'Transmission', 'Manufacturer']
        
        for var in categorical_vars:
            if var in df.columns:
                print(f"\n▶ {var}별 평균 가격:")
                avg_price = df.groupby(var)['Price'].agg(['mean', 'count']).sort_values('mean', ascending=False)
                print(avg_price.head(10).to_string())

    def run(self):
        """전체 상관관계 분석 실행"""
        print("\n" + "="*60)
        print("🔍 상관관계 분석 시작")
        print("="*60)
        
        # 데이터 로드
        df = self.load_data()
        
        # 상관계수 계산
        corr_df = self.calculate_correlation(df)
        
        # 시각화
        self.plot_correlation_heatmap(corr_df)
        self.plot_correlation_bar(df)
        self.plot_scatter_with_correlation(df)
        
        # 표 생성
        self.generate_correlation_table(corr_df)
        
        # 범주형 변수 분석
        self.analyze_categorical_correlation(df)
        
        print("\n" + "="*60)
        print("✅ 상관관계 분석 완료!")
        print(f"📁 저장 위치: {self.save_dir}")
        print("="*60)


if __name__ == "__main__":
    analyzer = CorrelationAnalyzer()
    analyzer.run()