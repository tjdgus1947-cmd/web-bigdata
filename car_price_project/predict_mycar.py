# predict_mycar.py
from src.model.predictor import PricePredictor


def safe_input_int(msg: str, default: int | None = None) -> int:
    val = input(msg).strip()
    if val == "":
        if default is not None:
            return default
        raise ValueError("값을 입력해야 합니다.")
    try:
        return int(val)
    except ValueError:
        if default is not None:
            print(f"⚠ 숫자로 변환 실패 → 기본값 {default} 사용")
            return default
        raise


def safe_input_float(msg: str, default: float | None = None) -> float:
    val = input(msg).strip()
    if val == "":
        if default is not None:
            return default
        raise ValueError("값을 입력해야 합니다.")
    try:
        return float(val)
    except ValueError:
        if default is not None:
            print(f"⚠ 숫자로 변환 실패 → 기본값 {default} 사용")
            return default
        raise


def main():
    print("======================================")
    print("   🚗 내 차 중고차 예상 가격 계산기")
    print("   (엔카 진단 차량 기반 모델)")
    print("======================================\n")

    # 1) 사용자 입력 받기
    manufacturer = input("제조사 (예: 현대, 기아, BMW 등): ").strip() or "현대"
    model = input("모델명 (예: 아반떼 (CN7), 쏘렌토 4세대): ").strip() or "아반떼 (CN7)"
    badge = input("트림/배지 (예: 1.6 인스퍼레이션, 디젤 2.0 프레스티지) [엔터로 생략 가능]: ").strip()

    year = safe_input_int("연식 (예: 2021): ", default=2021)
    mileage = safe_input_float("주행거리 (km, 예: 88410): ", default=50000.0)

    fuel_type = input("연료 (가솔린/디젤/LPG/전기/하이브리드 등): ").strip() or "가솔린"
    transmission = input("변속기 (자동/수동 등, 예: 오토, 자동): ").strip() or "오토"
    region = input("등록 지역 (예: 서울, 경기, 부산 등): ").strip() or "경기"

    # 2) 예측기 로드 & 예측
    predictor = PricePredictor(model_path="models/price_model.pkl")
    price_m = predictor.predict_price(
        manufacturer=manufacturer,
        model=model,
        badge=badge,
        year=year,
        mileage=mileage,
        fuel_type=fuel_type,
        transmission=transmission,
        region=region,
    )

    # 3) 결과 출력
    price_krw = int(price_m * 10000)

    print("\n======================================")
    print("          💰 예측 결과")
    print("======================================")
    print(f"차량: {manufacturer} {model} ({badge or '트림 미입력'})")
    print(f"연식: {year}년 / 주행거리: {int(mileage):,} km")
    print(f"연료: {fuel_type} / 변속기: {transmission} / 지역: {region}")
    print("--------------------------------------")
    print(f"▶ 예상 중고차 가격: 약 {price_m:,.1f} 만원")
    print(f"   (≈ {price_krw:,} 원)")
    print("※ 실제 거래가는 시세, 사고이력, 옵션 등에 따라 달라질 수 있습니다.")
    print("======================================\n")


if __name__ == "__main__":
    main()
