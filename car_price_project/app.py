# app.py
import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
from src.model.predictor import PricePredictor

# 페이지 설정
st.set_page_config(
    page_title="중고차 가격 예측기",
    page_icon="🚗",
    layout="wide"
)

# 스타일 설정
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        padding: 1rem 0;
    }
    .result-box {
        background-color: #f0f8ff;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        border-left: 5px solid #1f77b4;
    }
    .price-result {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ff4b4b;
        text-align: center;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# 헤더
st.markdown('<p class="main-header">🚗 중고차 가격 예측기</p>', unsafe_allow_html=True)
st.markdown("**엔카 진단 차량 데이터 기반 AI 예측 시스템**")
st.markdown("---")

# 사이드바 - 모델 정보
with st.sidebar:
    st.header("📊 모델 정보")
    
    # 모델 메타데이터 로드
    metadata_path = "models/price_model_metadata.pkl"
    if os.path.exists(metadata_path):
        metadata = joblib.load(metadata_path)
        
        st.metric("Test R² Score", f"{metadata['test_r2']:.4f}")
        st.metric("Test MAE", f"{metadata['test_mae']:,.0f} 만원")
        st.metric("Test RMSE", f"{metadata['test_rmse']:,.0f} 만원")
        
        if metadata.get('use_ensemble'):
            st.info("🔥 앙상블 모델 (RF + XGBoost)")
        else:
            st.info("🌲 Random Forest 모델")
        
        st.caption(f"학습 데이터: {metadata['n_train']:,}건")
        st.caption(f"테스트 데이터: {metadata['n_test']:,}건")
    else:
        st.warning("모델 메타데이터를 찾을 수 없습니다.")
    
    st.markdown("---")
    st.markdown("### 💡 사용 팁")
    st.markdown("""
    - 정확한 차량 정보를 입력하세요
    - 트림/배지는 선택 입력입니다
    - 예측 가격은 참고용입니다
    """)

# 메인 컨텐츠
tab1, tab2, tab3 = st.tabs(["🔮 가격 예측", "📈 통계 정보", "ℹ️ 사용 가이드"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("차량 기본 정보")
        
        # 제조사 선택 (데이터에서 추출)
        data_path = "data/processed/encar_processed.csv"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            manufacturers = sorted(df['Manufacturer'].unique().tolist())
            manufacturer = st.selectbox("제조사 *", manufacturers, index=0)
            
            # 선택한 제조사의 모델 목록
            models = sorted(df[df['Manufacturer'] == manufacturer]['Model'].unique().tolist())
            model = st.selectbox("모델명 *", models, index=0 if models else 0)
        else:
            manufacturer = st.text_input("제조사 *", value="현대")
            model = st.text_input("모델명 *", value="아반떼 (CN7)")
        
        badge = st.text_input("트림/배지 (선택)", 
                             placeholder="예: 1.6 인스퍼레이션",
                             help="입력하지 않아도 됩니다")
        
        year = st.number_input("연식 *", 
                              min_value=1980, 
                              max_value=datetime.now().year,
                              value=2021)
        
        mileage = st.number_input("주행거리 (km) *", 
                                 min_value=0.0, 
                                 max_value=500000.0,
                                 value=50000.0,
                                 step=1000.0,
                                 help="정확한 주행거리를 입력하세요")
    
    with col2:
        st.subheader("차량 추가 정보")
        
        fuel_type = st.selectbox("연료 타입 *", 
                                ["가솔린", "디젤", "LPG", "하이브리드", "전기"],
                                index=0)
        
        transmission = st.selectbox("변속기 *",
                                   ["오토", "수동", "기타"],
                                   index=0)
        
        # 지역 선택
        if os.path.exists(data_path):
            regions = sorted(df['OfficeCityState'].unique().tolist())
            region = st.selectbox("등록 지역 *", regions, 
                                 index=regions.index("경기") if "경기" in regions else 0)
        else:
            region = st.text_input("등록 지역 *", value="경기")
        
        st.markdown("---")
        st.caption("* 필수 입력 항목")
    
    # 예측 버튼
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_btn = st.button("💰 가격 예측하기", use_container_width=True, type="primary")
    
    # 예측 실행
    if predict_btn:
        try:
            with st.spinner("🔄 AI 모델이 가격을 예측하고 있습니다..."):
                predictor = PricePredictor(model_path="models/price_model.pkl")
                price_m = predictor.predict_price(
                    manufacturer=manufacturer,
                    model=model,
                    badge=badge if badge else "",
                    year=int(year),
                    mileage=float(mileage),
                    fuel_type=fuel_type,
                    transmission=transmission,
                    region=region,
                )
                
                price_krw = int(price_m * 10000)
                
                # 결과 표시
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.success("✅ 예측 완료!")
                
                st.markdown(f"""
                ### 입력 정보
                - **차량**: {manufacturer} {model} {f'({badge})' if badge else ''}
                - **연식**: {int(year)}년 / **주행거리**: {int(mileage):,} km
                - **연료**: {fuel_type} / **변속기**: {transmission}
                - **지역**: {region}
                """)
                
                st.markdown('<p class="price-result">💰 예상 가격: {0:,.0f} 만원</p>'.format(price_m), 
                           unsafe_allow_html=True)
                st.markdown(f'<p style="text-align: center; font-size: 1.2rem; color: #666;">≈ {price_krw:,} 원</p>', 
                           unsafe_allow_html=True)
                
                st.warning("⚠️ 실제 거래가는 시세, 사고이력, 옵션, 외관/내부 상태 등에 따라 달라질 수 있습니다.")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # 가격 범위 추정 (±10%)
                st.markdown("---")
                st.subheader("📊 예상 가격 범위")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("최소 예상가", f"{price_m * 0.9:,.0f} 만원", 
                             delta=f"-{price_m * 0.1:,.0f}", delta_color="inverse")
                with col2:
                    st.metric("예측 가격", f"{price_m:,.0f} 만원")
                with col3:
                    st.metric("최대 예상가", f"{price_m * 1.1:,.0f} 만원",
                             delta=f"+{price_m * 0.1:,.0f}")
                
        except FileNotFoundError:
            st.error("❌ 모델 파일을 찾을 수 없습니다. main.py를 먼저 실행하여 모델을 학습해주세요.")
        except Exception as e:
            st.error(f"❌ 예측 중 오류 발생: {str(e)}")

with tab2:
    st.subheader("📈 데이터 통계 정보")
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("전체 차량 수", f"{len(df):,}대")
        with col2:
            st.metric("평균 가격", f"{df['Price'].mean():,.0f}만원")
        with col3:
            st.metric("평균 주행거리", f"{df['Mileage'].mean():,.0f}km")
        with col4:
            st.metric("평균 차량 나이", f"{df['CarAge'].mean():.1f}년")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏭 제조사 분포 (Top 10)")
            manu_counts = df['Manufacturer'].value_counts().head(10)
            st.bar_chart(manu_counts)
        
        with col2:
            st.markdown("#### ⛽ 연료 타입 분포")
            fuel_counts = df['FuelType'].value_counts()
            st.bar_chart(fuel_counts)
        
        st.markdown("---")
        st.markdown("#### 💰 가격 분포")
        st.line_chart(df['Price'].value_counts().sort_index())
        
    else:
        st.warning("데이터 파일을 찾을 수 없습니다. 먼저 데이터 수집 및 전처리를 실행해주세요.")

with tab3:
    st.subheader("ℹ️ 사용 가이드")
    
    st.markdown("""
    ### 📝 사용 방법
    
    1. **차량 정보 입력**
       - '가격 예측' 탭에서 차량의 기본 정보를 입력합니다
       - 제조사, 모델, 연식, 주행거리 등은 필수 입력 사항입니다
       - 트림/배지는 선택 사항이지만, 입력하면 더 정확한 예측이 가능합니다
    
    2. **가격 예측**
       - '가격 예측하기' 버튼을 클릭합니다
       - AI 모델이 입력된 정보를 분석하여 예상 가격을 계산합니다
    
    3. **결과 확인**
       - 예측된 가격과 가격 범위를 확인합니다
       - 실제 거래가는 차량 상태, 옵션 등에 따라 달라질 수 있습니다
    
    ---
    
    ### 🎯 정확도 향상을 위한 팁
    
    - **정확한 주행거리**: 현재 계기판에 표시된 정확한 주행거리를 입력하세요
    - **상세한 트림 정보**: 가능하면 트림/배지 정보를 입력하세요 (예: 1.6 터보 프레스티지)
    - **지역 정보**: 차량이 등록된 지역을 정확히 선택하세요
    
    ---
    
    ### ⚠️ 주의사항
    
    - 이 시스템은 엔카 진단 차량 데이터를 기반으로 학습되었습니다
    - 예측 가격은 **참고용**이며, 실제 거래가와 다를 수 있습니다
    - 사고 이력, 침수 이력, 특별한 옵션, 외관/내부 상태 등은 반영되지 않습니다
    - 실제 거래 시에는 전문가의 검수를 받으시길 권장합니다
    
    ---
    
    ### 📞 문의사항
    
    시스템 관련 문의사항이나 개선 제안이 있으시면 연락 주세요!
    """)

# 푸터
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🚗 중고차 가격 예측 시스템 v1.0</p>
    <p>Powered by AI & Encar Data</p>
</div>
""", unsafe_allow_html=True)