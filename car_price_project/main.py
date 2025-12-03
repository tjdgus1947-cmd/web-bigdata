from src.api.encar_api import EncarAPICrawler
from src.preprocessing.preprocessor import Preprocessor
from src.model.trainer import ModelTrainer
from src.analysis.visualizer import Visualizer
import os

def main():
    raw_path = "data/raw/encar_premium.csv"

    print("📡 엔카진단 전체 데이터 수집 시작...")
    if os.path.exists(raw_path):
        print("📁 기존 수집 파일 발견 → 재수집 스킵")
    else:
        crawler = EncarAPICrawler(page_size=200)
        df = crawler.crawl()
        if not df.empty:
            crawler.save(df)

    print("\n==== 2) 전처리 시작 ====")
    prep = Preprocessor()
    df = prep.run()

    print("\n==== 3) 모델 학습 ====")
    trainer = ModelTrainer()
    trainer.train()

    print("\n==== 4) 시각화 생성 ====")
    viz = Visualizer()
    viz.run()

if __name__ == "__main__":
    main()
