# src/analysis/model_evaluator.py

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import platform
import joblib
from sklearn.metrics import mean_absolute_error, r2_score

if platform.system() == "Windows":
    matplotlib.rc("font", family="Malgun Gothic")
else:
    matplotlib.rc("font", family="AppleGothic")
matplotlib.rcParams['axes.unicode_minus'] = False


class ModelEvaluator:
    def __init__(self,
                 model_path: str = "models/price_model.pkl",
                 data_path: str = "data/processed/encar_processed.csv",
                 save_dir: str = "visualizations/model_evaluation"):
        self.model_path = model_path
        self.data_path = data_path
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def load_data_and_predict(self):
        """데이터 로드 및 예측"""
        print("📊 데이터 로드 및 예측 중...")
        
        # 모델 로드
        model = joblib.load(self.model_path)
        
        # 데이터 로드
        df = pd.read_csv(self.data_path)
        
        # Feature Engineering (trainer.py와 동일하게)
        df['Mileage_per_year'] = df['Mileage'] / (df['CarAge'] + 1)
        luxury_brands = ['BMW', 'Mercedes-Benz', '메르세데스-벤츠', 'Audi', '아우디',
                        'Porsche', '포르쉐', 'Lexus', '렉서스', 'Genesis', '제네시스']
        df['Is_luxury_brand'] = df['Manufacturer'].isin(luxury_brands).astype(int)
        df['Age_group'] = pd.cut(df['CarAge'], bins=[-1, 2, 5, 10, 100],
                                 labels=['신차급', '준신차', '중고', '노후'])
        df['Mileage_group'] = pd.cut(df['Mileage'],
                                      bins=[0, 30000, 80000, 150000, 1000000],
                                      labels=['저주행', '중주행', '고주행', '과다주행'])
        
        # Feature 준비
        feature_cols = ["CarAge", "Mileage", "Mileage_per_year", "Is_luxury_brand",
                       "Manufacturer", "Model", "Badge", "FuelType",
                       "Transmission", "OfficeCityState", "Age_group", "Mileage_group"]
        
        X = df[feature_cols]
        y_true = df['Price']
        y_pred = model.predict(X)
        
        print(f"✅ 예측 완료: {len(df):,}건")
        return df, y_true, y_pred, model

    def plot_residuals(self, y_true, y_pred):
        """잔차 플롯 - 모델의 편향 확인"""
        print("📈 잔차 플롯 생성 중...")
        
        residuals = y_pred - y_true
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1) 예측값 vs 잔차
        axes[0].scatter(y_pred, residuals, alpha=0.3, s=10)
        axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('예측 가격 (만원)')
        axes[0].set_ylabel('잔차 (예측값 - 실제값)')
        axes[0].set_title('잔차 플롯 (Residual Plot)')
        axes[0].grid(True, alpha=0.3)
        
        # 2) 잔차 히스토그램
        axes[1].hist(residuals, bins=50, edgecolor='black')
        axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('잔차 (만원)')
        axes[1].set_ylabel('빈도')
        axes[1].set_title('잔차 분포')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(self.save_dir, "residual_plot.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 저장: {path}")

    def plot_prediction_vs_actual(self, y_true, y_pred):
        """예측값 vs 실제값 산점도"""
        print("📈 예측값 vs 실제값 플롯 생성 중...")
        
        plt.figure(figsize=(8, 8))
        
        # 샘플링 (너무 많으면)
        if len(y_true) > 5000:
            indices = np.random.choice(len(y_true), 5000, replace=False)
            y_true_sample = y_true.iloc[indices]
            y_pred_sample = y_pred[indices]
        else:
            y_true_sample = y_true
            y_pred_sample = y_pred
        
        plt.scatter(y_true_sample, y_pred_sample, alpha=0.3, s=10)
        
        # 완벽한 예측선 (y=x)
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='완벽한 예측')
        
        plt.xlabel('실제 가격 (만원)')
        plt.ylabel('예측 가격 (만원)')
        plt.title('예측값 vs 실제값')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # R² 표시
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}',
                transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=12, verticalalignment='top')
        
        plt.tight_layout()
        path = os.path.join(self.save_dir, "prediction_vs_actual.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 저장: {path}")

    def plot_feature_importance(self, model, feature_names=None):
        """변수 중요도 시각화"""
        print("📈 변수 중요도 플롯 생성 중...")
        
        try:
            # Pipeline에서 실제 모델 추출
            if hasattr(model, 'named_steps'):
                actual_model = model.named_steps['model']
                
                # VotingRegressor인 경우 첫 번째 모델 사용
                if hasattr(actual_model, 'estimators_'):
                    actual_model = actual_model.estimators_[0]
            else:
                actual_model = model
            
            # Feature importance 추출
            if hasattr(actual_model, 'feature_importances_'):
                importances = actual_model.feature_importances_
                
                # 전처리 후 feature 이름 가져오기
                if feature_names is None:
                    preprocessor = model.named_steps['preprocess']
                    feature_names = preprocessor.get_feature_names_out()
                
                # 상위 20개만 표시
                indices = np.argsort(importances)[-20:]
                
                plt.figure(figsize=(10, 8))
                plt.barh(range(len(indices)), importances[indices])
                plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
                plt.xlabel('중요도')
                plt.title('변수 중요도 (Feature Importance) - Top 20')
                plt.tight_layout()
                
                path = os.path.join(self.save_dir, "feature_importance.png")
                plt.savefig(path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  ✅ 저장: {path}")
            else:
                print("  ⚠️  이 모델은 feature_importances_를 지원하지 않습니다.")
                
        except Exception as e:
            print(f"  ⚠️  변수 중요도 추출 실패: {str(e)}")

    def analyze_price_range_accuracy(self, df, y_true, y_pred):
        """가격대별 정확도 분석"""
        print("📈 가격대별 정확도 분석 중...")
        
        # 가격대 구간 설정
        bins = [0, 500, 1000, 2000, 3000, 10000]
        labels = ['~500만', '500~1000만', '1000~2000만', '2000~3000만', '3000만~']
        
        df_eval = pd.DataFrame({
            'true': y_true,
            'pred': y_pred,
            'error': np.abs(y_true - y_pred)
        })
        
        df_eval['price_range'] = pd.cut(df_eval['true'], bins=bins, labels=labels)
        
        # 가격대별 통계
        stats = []
        for price_range in labels:
            subset = df_eval[df_eval['price_range'] == price_range]
            if len(subset) > 0:
                stats.append({
                    'price_range': price_range,
                    'count': len(subset),
                    'mae': subset['error'].mean(),
                    'r2': r2_score(subset['true'], subset['pred'])
                })
        
        stats_df = pd.DataFrame(stats)
        
        # 시각화
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # MAE by price range
        axes[0].bar(stats_df['price_range'], stats_df['mae'])
        axes[0].set_xlabel('가격대')
        axes[0].set_ylabel('MAE (만원)')
        axes[0].set_title('가격대별 평균 절대 오차 (MAE)')
        axes[0].tick_params(axis='x', rotation=30)
        axes[0].grid(True, alpha=0.3)
        
        # R² by price range
        axes[1].bar(stats_df['price_range'], stats_df['r2'])
        axes[1].set_xlabel('가격대')
        axes[1].set_ylabel('R² Score')
        axes[1].set_title('가격대별 R² Score')
        axes[1].tick_params(axis='x', rotation=30)
        axes[1].axhline(y=0.8, color='r', linestyle='--', linewidth=1, alpha=0.5)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(self.save_dir, "price_range_accuracy.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 저장: {path}")
        
        return stats_df

    def analyze_by_manufacturer(self, df, y_true, y_pred):
        """제조사별 성능 분석"""
        print("📈 제조사별 성능 분석 중...")
        
        df_eval = df.copy()
        df_eval['true'] = y_true
        df_eval['pred'] = y_pred
        df_eval['error'] = np.abs(y_true - y_pred)
        
        # 제조사별 통계 (상위 15개)
        manu_counts = df_eval['Manufacturer'].value_counts()
        top_manus = manu_counts.head(15).index
        
        stats = []
        for manu in top_manus:
            subset = df_eval[df_eval['Manufacturer'] == manu]
            stats.append({
                'manufacturer': manu,
                'count': len(subset),
                'mae': subset['error'].mean(),
                'r2': r2_score(subset['true'], subset['pred'])
            })
        
        stats_df = pd.DataFrame(stats).sort_values('mae')
        
        # 시각화
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # MAE by manufacturer
        axes[0].barh(stats_df['manufacturer'], stats_df['mae'])
        axes[0].set_xlabel('MAE (만원)')
        axes[0].set_title('제조사별 평균 절대 오차 (MAE)')
        axes[0].grid(True, alpha=0.3)
        
        # R² by manufacturer
        axes[1].barh(stats_df['manufacturer'], stats_df['r2'])
        axes[1].set_xlabel('R² Score')
        axes[1].set_title('제조사별 R² Score')
        axes[1].axvline(x=0.8, color='r', linestyle='--', linewidth=1, alpha=0.5)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(self.save_dir, "manufacturer_accuracy.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 저장: {path}")
        
        return stats_df

    def generate_report(self):
        """전체 평가 리포트 생성"""
        print("\n" + "="*60)
        print("📊 모델 평가 리포트 생성 시작")
        print("="*60 + "\n")
        
        # 데이터 로드 및 예측
        df, y_true, y_pred, model = self.load_data_and_predict()
        
        # 전체 성능 지표
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f"\n📌 전체 성능")
        print(f"  - MAE : {mae:,.1f} 만원")
        print(f"  - R²  : {r2:.4f}")
        print(f"  - 데이터 수: {len(df):,}건\n")
        
        # 각종 플롯 생성
        self.plot_residuals(y_true, y_pred)
        self.plot_prediction_vs_actual(y_true, y_pred)
        self.plot_feature_importance(model)
        
        # 가격대별 분석
        print()
        price_stats = self.analyze_price_range_accuracy(df, y_true, y_pred)
        print("\n📌 가격대별 MAE:")
        print(price_stats[['price_range', 'mae', 'r2']].to_string(index=False))
        
        # 제조사별 분석
        print()
        manu_stats = self.analyze_by_manufacturer(df, y_true, y_pred)
        print("\n📌 제조사별 성능 (Top 5):")
        print(manu_stats.head()[['manufacturer', 'mae', 'r2']].to_string(index=False))
        
        print("\n" + "="*60)
        print(f"✅ 평가 리포트 생성 완료!")
        print(f"📁 저장 위치: {self.save_dir}")
        print("="*60 + "\n")


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.generate_report()