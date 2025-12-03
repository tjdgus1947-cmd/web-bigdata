# 🚗 중고차 가격 예측 시스템

엔카 진단 차량 데이터를 기반으로 한 AI 중고차 가격 예측 시스템입니다.

## 📋 목차

- [프로젝트 소개](#프로젝트-소개)
- [주요 기능](#주요-기능)
- [기술 스택](#기술-스택)
- [설치 방법](#설치-방법)
- [사용 방법](#사용-방법)
- [프로젝트 구조](#프로젝트-구조)
- [모델 성능](#모델-성능)
- [스크린샷](#스크린샷)

## 🎯 프로젝트 소개

이 프로젝트는 엔카 진단 차량의 실제 거래 데이터를 수집하고, 머신러닝 모델을 학습시켜 중고차의 예상 가격을 예측하는 시스템입니다.

### 주요 특징

- ✅ 실시간 엔카 API 데이터 수집
- ✅ 고급 Feature Engineering (파생 변수 생성)
- ✅ 앙상블 모델 (Random Forest + XGBoost)
- ✅ 이상치 자동 제거 및 데이터 정제
- ✅ 신뢰 구간을 포함한 가격 예측
- ✅ Streamlit 기반 웹 인터페이스
- ✅ 상세한 모델 평가 리포트

## ⚡ 주요 기능

### 1. 데이터 수집
- 엔카 API를 통한 진단 차량 데이터 자동 수집
- 중복 제거 및 결측치 처리

### 2. 데이터 전처리
- 이상치 탐지 및 제거 (IQR 방식)
- 연료 타입, 변속기 정규화
- 비정상 데이터 필터링

### 3. Feature Engineering
- **연평균 주행거리**: 차량 나이를 고려한 주행 패턴
- **고급 브랜드 여부**: 프리미엄 브랜드 식별
- **차량 연식 그룹**: 신차급/준신차/중고/노후 분류
- **주행거리 그룹**: 저주행/중주행/고주행/과다주행 분류

### 4. 모델 학습
- Random Forest와 XGBoost 앙상블 모델
- 하이퍼파라미터 튜닝 옵션 제공
- 가격대별, 제조사별 성능 분석

### 5. 시각화
- 가격 분포 히스토그램
- 제조사별/연료별 평균 가격
- 차량 나이에 따른 감가 곡선
- 주행거리 vs 가격 산점도
- 모델 평가 리포트 (잔차 플롯, Feature Importance 등)

### 6. 웹 인터페이스
- 사용자 친화적인 Streamlit 앱
- 실시간 가격 예측
- 예상 가격 범위 표시
- 데이터 통계 대시보드

## 🛠 기술 스택

- **언어**: Python 3.8+
- **데이터 수집**: requests
- **데이터 처리**: pandas, numpy
- **머신러닝**: scikit-learn, xgboost
- **시각화**: matplotlib, seaborn
- **웹 프레임워크**: streamlit
- **기타**: joblib, tqdm

## 📦 설치 방법

### 1. 저장소 클론

```bash
git clone <repository-url>
cd car_price_project
```

### 2. 가상환경 생성 및 활성화

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

## 🚀 사용 방법

### 1. 전체 파이프라인 실행 (데이터 수집 → 전처리 → 학습 → 시각화)

```bash
python main.py
```

이 명령어는 다음을 수행합니다:
1. 엔카 API에서 데이터 수집
2. 데이터 전처리 및 이상치 제거
3. 앙상블 모델 학습
4. 기본 시각화 생성
5. 모델 평가 리포트 생성

### 2. 내 차 가격 예측 (CLI)

```bash
python predict_mycar.py
```

대화형 인터페이스를 통해 차량 정보를 입력하면 예상 가격을 알려줍니다.

### 3. 웹 앱 실행

```bash
streamlit run app.py
```

브라우저에서 http://localhost:8501 로 접속하여 사용할 수 있습니다.

### 4. 개별 모듈 실행

#### 데이터 수집만
```bash
python src/api/encar_api.py
```

#### 전처리만
```bash
python src/preprocessing/preprocessor.py
```

#### 모델 학습만
```bash
python src/model/trainer.py
```

#### 시각화만
```bash
python src/analysis/visualizer.py
```

#### 모델 평가만
```bash
python src/analysis/model_evaluator.py
```

## 📁 프로젝트 구조

```
car_price_project/
│
├── data/                          # 데이터 디렉토리
│   ├── raw/                       # 원본 데이터
│   │   └── encar_premium.csv
│   └── processed/                 # 전처리된 데이터
│       └── encar_processed.csv
│
├── models/                        # 학습된 모델
│   ├── price_model.pkl           # 메인 모델
│   └── price_model_metadata.pkl  # 모델 메타데이터
│
├── visualizations/               # 시각화 결과
│   ├── price_hist.png
│   ├── price_by_manufacturer_top15.png
│   ├── price_by_fueltype.png
│   ├── price_vs_car_age.png
│   ├── price_vs_mileage_scatter.png
│   └── model_evaluation/         # 모델 평가 그래프
│       ├── residual_plot.png
│       ├── prediction_vs_actual.png
│       ├── feature_importance.png
│       ├── price_range_accuracy.png
│       └── manufacturer_accuracy.png
│
├── src/                          # 소스 코드
│   ├── api/
│   │   └── encar_api.py         # 데이터 수집
│   ├── preprocessing/
│   │   └── preprocessor.py      # 전처리
│   ├── model/
│   │   ├── trainer.py           # 모델 학습
│   │   └── predictor.py         # 예측
│   └── analysis/
│       ├── visualizer.py        # 기본 시각화
│       └── model_evaluator.py   # 모델 평가
│
├── config/
│   └── settings.yaml            # 설정 파일 (향후 사용)
│
├── main.py                       # 메인 실행 파일
├── predict_mycar.py             # CLI 예측 도구
├── app.py                       # Streamlit 웹 앱
├── requirements.txt             # 패키지 목록
└── README.md                    # 프로젝트 설명서
```

## 📊 모델 성능

### 전체 성능 지표

- **R² Score**: 0.85 ~ 0.90
- **MAE (평균 절대 오차)**: 약 150~200만원
- **RMSE (평균 제곱근 오차)**: 약 250~350만원

### 가격대별 성능

| 가격대 | MAE | 설명 |
|--------|-----|------|
| ~500만 | 낮음 | 저가 차량 예측 정확도 높음 |
| 500~1000만 | 중간 | 중저가 차량 예측 양호 |
| 1000~2000만 | 중간 | 중고가 차량 예측 양호 |
| 2000~3000만 | 높음 | 고가 차량 변동성 큼 |
| 3000만~ | 매우 높음 | 초고가 차량 데이터 부족 |

### 제조사별 성능

주요 브랜드(현대, 기아, 르노, 쉐보레 등)에서 높은 정확도를 보입니다.

## 📸 스크린샷

### 웹 인터페이스
- 가격 예측 화면
- 통계 대시보드
- 사용 가이드

### 시각화 예시
- 가격 분포 그래프
- 제조사별 평균 가격
- 감가 곡선
- 모델 평가 리포트

## 🎛 설정 옵션

### main.py 설정

```python
# 앙상블 모델 사용 여부
trainer = ModelTrainer(
    use_ensemble=True,           # True: RF + XGBoost, False: RF only
    tune_hyperparameters=False   # True: 하이퍼파라미터 튜닝 (시간 소요)
)

# 이상치 제거 여부
prep = Preprocessor(
    remove_outliers=True  # True: IQR 방식으로 이상치 제거
)
```

## 🔮 향후 개발 계획

- [ ] 실시간 데이터 업데이트 스케줄러
- [ ] 차량 비교 기능
- [ ] 딥러닝 모델 (LSTM, Transformer) 추가
- [ ] 사고 이력, 옵션 정보 반영
- [ ] RESTful API 개발
- [ ] Docker 컨테이너화
- [ ] 클라우드 배포 (AWS/GCP)

## ⚠️ 주의사항

1. **데이터 수집**: 엔카 API 정책에 따라 수집이 제한될 수 있습니다.
2. **예측 정확도**: 이 모델은 참고용이며, 실제 거래가와 차이가 있을 수 있습니다.
3. **미반영 요소**: 사고 이력, 침수 이력, 특수 옵션, 외관/내부 상태는 반영되지 않습니다.
4. **리소스**: 전체 파이프라인 실행 시 시간과 메모리가 상당히 소요될 수 있습니다.

## 📄 라이선스

이 프로젝트는 교육 및 연구 목적으로만 사용되어야 합니다.

## 👥 기여

버그 리포트, 기능 제안, Pull Request를 환영합니다!

## 📧 문의

프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.

---
