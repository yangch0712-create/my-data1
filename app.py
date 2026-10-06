import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="서울 100년 기온 변화 분석",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded" # 사이드바 기본 열림 설정
)

# 2. 사이드바 메뉴 구성
st.sidebar.title("📌 메뉴")
page = st.sidebar.radio(
    "이동할 페이지를 선택하세요:",
    ["1. 서울 100년 기온 변화 (메인)", "2. 선형회귀 모델 분석 및 비교"]
)

# 3. 데이터 로드 함수
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/seoul.csv"
    try:
        df = pd.read_csv(url, encoding='cp949')
    except Exception:
        df = pd.read_csv(url, encoding='utf-8')
    
    df.columns = df.columns.str.strip()
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['연도'] = df['날짜'].dt.year
    
    station_col = [c for c in df.columns if '지점' in c][0]
    avg_col = [c for c in df.columns if '평균' in c][0]
    min_col = [c for c in df.columns if '최저' in c][0]
    max_col = [c for c in df.columns if '최고' in c][0]
    
    df[avg_col] = pd.to_numeric(df[avg_col], errors='coerce')
    df[min_col] = pd.to_numeric(df[min_col], errors='coerce')
    df[max_col] = pd.to_numeric(df[max_col], errors='coerce')
    
    df_clean = df[['날짜', '연도', station_col, avg_col, min_col, max_col]].copy()
    df_clean.columns = ['날짜', '연도', '지점', '평균기온(℃)', '최저기온(℃)', '최고기온(℃)']
    
    yearly_df = df_clean.groupby('연도').agg(
        연평균기온=('평균기온(℃)', 'mean'),
        연평균최저기온=('최저기온(℃)', 'mean'),
        연평균최고기온=('최고기온(℃)', 'mean'),
        관측일수=('평균기온(℃)', 'count')
    ).reset_index()
    
    yearly_df = yearly_df[yearly_df['관측일수'] >= 300].copy()
    yearly_df['10년이동평균'] = yearly_df['연평균기온'].rolling(window=10, min_periods=1).mean()
    
    return df_clean, yearly_df

# PAGE 1: 메인 기온 데이터 분석
if page == "1. 서울 100년 기온 변화 (메인)":
    st.title("🌡️ 서울 지난 100년간 연평균 기온 변화")
    st.write("기상청 서울 관측 데이터(`seoul.csv`)를 바탕으로 한 기온 변화 추이 분석 앱입니다.")
    st.info("👈 좌측 사이드바에서 페이지를 변경하여 **선형회귀 모델 분석** 결과도 확인하실 수 있습니다.")
    
    try:
        raw_df, yearly_df = load_data()
        
        # KPI 지표
        min_year = int(yearly_df['연도'].min())
        max_year = int(yearly_df['연도'].max())
        first_avg = yearly_df.iloc[0]['연평균기온']
        last_avg = yearly_df.iloc[-1]['연평균기온']
        temp_diff = last_avg - first_avg
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("분석 기간", f"{min_year}년 ~ {max_year}년")
        col2.metric("관측 초기 연평균", f"{first_avg:.1f} ℃")
        col3.metric("최근 연평균", f"{last_avg:.1f} ℃")
        col4.metric("기온 변화량", f"{temp_diff:+.1f} ℃", delta=f"{temp_diff:.1f} ℃")
        
        st.divider()
        
        # 메인 시각화
        st.subheader("📈 연도별 평균 기온 추이 및 10년 이동평균선")
        fig = px.line(
            yearly_df, 
            x='연도', 
            y='연평균기온', 
            title=f'서울 연평균 기온 변화 ({min_year} - {max_year})',
            labels={'연도': '연도', '연평균기온': '연평균 기온 (℃)'},
            markers=True
        )
        fig.add_scatter(
            x=yearly_df['연도'], 
            y=yearly_df['10년이동평균'], 
            mode='lines', 
            name='10년 이동평균선',
            line=dict(color='orange', width=3, dash='dash')
        )
        fig.update_traces(hovertemplate='<b>%{x}년</b><br>연평균 기온: %{y:.2f}℃')
        fig.update_layout(hovermode="x unified", xaxis=dict(showgrid=True), yaxis=dict(showgrid=True, title="기온 (℃)"))
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # 요약 통계
        st.subheader("📋 원본 데이터 요약 통계 (Summary Statistics)")
        summary_df = raw_df[['지점', '평균기온(℃)', '최저기온(℃)', '최고기온(℃)']].describe(include='all')
        rename_dict = {
            'count': '개수(일수)', 'unique': '고유값 수', 'top': '최다 관측 지점', 'freq': '최다 관측 지점 일수',
            'mean': '평균', 'std': '표준편차', 'min': '최소', '25%': '25% (1분위)', '50%': '중앙값 (50%)', '75%': '75% (3분위)', 'max': '최대'
        }
        summary_df.index = [rename_dict.get(idx, idx) for idx in summary_df.index]
        st.dataframe(summary_df, use_container_width=True)
        
        # 상세 연도별 데이터
        with st.expander("📊 연도별 집계 데이터 보기"):
            st.dataframe(
                yearly_df[['연도', '연평균기온', '연평균최저기온', '연평균최고기온', '관측일수']].style.format({
                    '연평균기온': '{:.2f} ℃', '연평균최저기온': '{:.2f} ℃', '연평균최고기온': '{:.2f} ℃', '관측일수': '{:,.0f} 일'
                }), 
                use_container_width=True
            )
    except Exception as e:
        st.error(f"오류 발생: {e}")

# PAGE 2: 머신러닝 분석
else:
    st.title("🤖 연평균 기온 선형회귀 모델 분석 및 비교")
    st.write("서울의 연평균 기온 데이터를 활용하여 **전체 데이터 모델**, **최근 50년 학습 모델(1956~2005)**, **최근 100년 학습 모델(1906~2005)**을 구축하고 최근 20년(2006~2025) 테스트 데이터에 대한 예측 성능을 비교합니다.")

    try:
        raw_df, yearly_df = load_data()
        
        # 모델 1: 전체 데이터
        X_full = yearly_df[['연도']]
        y_full = yearly_df['연평균기온']
        model_full = LinearRegression().fit(X_full, y_full)
        pred_full = model_full.predict(X_full)
        
        mae_full = mean_absolute_error(y_full, pred_full)
        mse_full = mean_squared_error(y_full, pred_full)
        r2_full = r2_score(y_full, pred_full)
        slope_full = model_full.coef_[0]
        
        # 데이터 분할 (테스트: 2006 ~ 2025 공통)
        test_df = yearly_df[(yearly_df['연도'] >= 2006) & (yearly_df['연도'] <= 2025)]
        X_test = test_df[['연도']]
        y_test = test_df['연평균기온']
        
        # 모델 2: 최근 50년 학습 (1956 ~ 2005)
        train_50 = yearly_df[(yearly_df['연도'] >= 1956) & (yearly_df['연도'] <= 2005)]
        X_train_50 = train_50[['연도']]
        y_train_50 = train_50['연평균기온']
        model_50 = LinearRegression().fit(X_train_50, y_train_50)
        pred_test_50 = model_50.predict(X_test)
        
        mae_50 = mean_absolute_error(y_test, pred_test_50)
        mse_50 = mean_squared_error(y_test, pred_test_50)
        r2_50 = r2_score(y_test, pred_test_50)
        slope_50 = model_50.coef_[0]
        
        # 모델 3: 최근 100년 학습 (1906 ~ 2005)
        train_100 = yearly_df[(yearly_df['연도'] >= 1906) & (yearly_df['연도'] <= 2005)]
        X_train_100 = train_100[['연도']]
        y_train_100 = train_100['연평균기온']
        model_100 = LinearRegression().fit(X_train_100, y_train_100)
        pred_test_100 = model_100.predict(X_test)
        
        mae_100 = mean_absolute_error(y_test, pred_test_100)
        mse_100 = mean_squared_error(y_test, pred_test_100)
        r2_100 = r2_score(y_test, pred_test_100)
        slope_100 = model_100.coef_[0]
        
        # 1. 전체 데이터 모델 평가
        st.subheader("1. 전체 데이터(전체 기간) 선형회귀 모델 평가")
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        f_col1.metric("기울기 (10년당 상승폭)", f"{slope_full * 10:+.3f} ℃")
        f_col2.metric("MAE (평균 절대 오차)", f"{mae_full:.3f} ℃")
        f_col3.metric("MSE (평균 제곱 오차)", f"{mse_full:.3f} ℃")
        f_col4.metric("R² (결정계수)", f"{r2_full:.3f}")
        
        st.divider()
        
        # 2. 모델 비교
        st.subheader("2. 학습 기간별 모델 비교 (공통 테스트 데이터: 2006년~2025년)")
        metrics_data = {
            '비교 항목': ['학습 기간 (Train Years)', '테스트 기간 (Test Years)', '기울기 (10년당 기온 변화량)', 'MAE (평균 절대 오차)', 'MSE (평균 제곱 오차)', 'R² (결정계수)'],
            '최근 50년 학습 모델': ['1956년 ~ 2005년', '2006년 ~ 2025년', f"{slope_50 * 10:+.3f} ℃", f"{mae_50:.3f} ℃", f"{mse_50:.3f} ℃", f"{r2_50:.3f}"],
            '최근 100년 학습 모델': ['1906년 ~ 2005년', '2006년 ~ 2025년', f"{slope_100 * 10:+.3f} ℃", f"{mae_100:.3f} ℃", f"{mse_100:.3f} ℃", f"{r2_100:.3f}"]
        }
        st.table(pd.DataFrame(metrics_data))
        
        # 3. 회귀선 시각화
        st.subheader("📈 학습 기간별 회귀선 및 테스트 데이터 시각화")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yearly_df['연도'], y=yearly_df['연평균기온'], mode='markers', name='과거 관측 데이터', marker=dict(color='gray', opacity=0.4, size=6)))
        fig.add_trace(go.Scatter(x=test_df['연도'], y=test_df['연평균기온'], mode='markers+lines', name='테스트 데이터 (2006~2025)', marker=dict(color='red', size=8), line=dict(color='red', width=2)))
        
        years_plot = np.arange(1906, 2026).reshape(-1, 1)
        fig.add_trace(go.Scatter(x=years_plot.flatten(), y=model_50.predict(years_plot), mode='lines', name=f'50년 학습 회귀선 (기울기: +{slope_50*10:.2f}℃/10년)', line=dict(color='blue', width=3, dash='dash')))
        fig.add_trace(go.Scatter(x=years_plot.flatten(), y=model_100.predict(years_plot), mode='lines', name=f'100년 학습 회귀선 (기울기: +{slope_100*10:.2f}℃/10년)', line=dict(color='green', width=3)))
        
        fig.update_layout(title="서울 연평균 기온 회귀선 및 2006~2025년 예측 성능 비교", xaxis_title="연도", yaxis_title="연평균 기온 (℃)", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        # 4. 인사이트
        st.subheader("💡 분석 결과 및 인사이트")
        st.markdown(f"""
        - **회귀선 기울기**: 최근 50년 학습 모델(10년당 {slope_50*10:+.2f}℃)이 100년 학습 모델(10년당 {slope_100*10:+.2f}℃)보다 기울기가 가파릅니다.
        - **예측 성능**: 최근 온난화 가속 경향을 반영한 **50년 학습 모델**이 오차(MAE: {mae_50:.3f}℃)가 더 적어 예측 성능이 우수합니다.
        """)
    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
