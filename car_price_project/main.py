from src.api.encar_api import EncarAPICrawler
from src.preprocessing.preprocessor import Preprocessor
from src.model.trainer import ModelTrainer
from src.analysis.visualizer import Visualizer
from src.analysis.model_evaluator import ModelEvaluator
import os

def main():
    raw_path = "data/raw/encar_premium.csv"

    print("=" * 60)
    print("🚗 중고차 가격 예측 시스템 - 전체 파이프라인 실행")
    print("=" * 60)

    # ==== 1) 데이터 수집 ====
    print("\n==== 1) 엔카 데이터 수집 ====")
    if os.path.exists(raw_path):
        print("📁 기존 수집 파일 발견 → 재수집 스킵")
        print(f"   (재수집하려면 {raw_path} 파일을 삭제하세요)")
    else:
        crawler = EncarAPICrawler()
        df = crawler.crawl()
        if df is not None and not df.empty:
            crawler.save(df)
        else:
            print("❌ 데이터 수집 실패")
            return

    # ==== 2) 전처리 ====
    print("\n==== 2) 데이터 전처리 ====")
    prep = Preprocessor(remove_outliers=True)  # 이상치 제거 활성화
    df = prep.run()

    # ==== 3) 모델 학습 ====
    print("\n==== 3) 모델 학습 ====")
    trainer = ModelTrainer(
        use_ensemble=True,           # 앙상블 모델 사용
        tune_hyperparameters=False   # 하이퍼파라미터 튜닝 (시간이 오래 걸림)
    )
    trainer.train()

    # ==== 4) 기본 시각화 생성 ====
    print("\n==== 4) 기본 시각화 생성 ====")
    viz = Visualizer()
    viz.run()

    # ==== 5) 모델 평가 리포트 생성 ====
    print("\n==== 5) 모델 평가 리포트 생성 ====")
    evaluator = ModelEvaluator()
    evaluator.generate_report()

    # ==== 6) 상관관계 분석 ====
    print("\n==== 6) 상관관계 분석 ====")
    from src.analysis.correlation_analyzer import CorrelationAnalyzer
    corr_analyzer = CorrelationAnalyzer()
    corr_analyzer.run()

    # ==== 완료 ====
    print("\n" + "=" * 60)
    print("✅ 전체 파이프라인 실행 완료!")
    print("=" * 60)
    print("\n📌 다음 단계:")
    print("  1. 예측하기: python predict_mycar.py")
    print("  2. 웹 앱 실행: streamlit run app.py")
    print("  3. 시각화 확인: visualizations/ 폴더")
    print("  4. 상관관계 분석: visualizations/correlation/ 폴더")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()