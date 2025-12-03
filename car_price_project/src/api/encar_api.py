import requests
import pandas as pd
import os

class EncarAPICrawler:
    def __init__(self):
        self.url = (
            "https://api.encar.com/search/car/list/premium"
            "?count=true"
            "&q=(And.Hidden.N._.CarType.Y._.(Or.ServiceMark.EncarDiagnosisP0._.ServiceMark.EncarDiagnosisP1._.ServiceMark.EncarDiagnosisP2.))"
            "&sr=%7CModifiedDate%7C0%7C12"
        )

    def crawl(self):
        print("📡 엔카진단 전체 데이터 요청 중...")

        r = requests.get(self.url, timeout=25)

        if r.status_code != 200:
            print(f"❌ HTTP ERROR: {r.status_code}")
            return None
        
        data = r.json()
        results = data.get("SearchResults", [])

        print(f"📦 총 {len(results):,}개 차량 수집됨")

        df = pd.DataFrame(results)
        return df

    def save(self, df, path="data/raw/encar_premium.csv"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"💾 저장 완료: {path} (rows={len(df)})")


if __name__ == "__main__":
    crawler = EncarAPICrawler()
    df = crawler.crawl()
    if df is not None:
        crawler.save(df)
