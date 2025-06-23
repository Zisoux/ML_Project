import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from PIL import Image
import base64
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np

st.set_page_config(layout="wide")
st.markdown("""
    <style>
    .block-container {
        max-width: 1500px;
        margin-left: auto;
        margin-right: auto;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    .desc-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 30px;
        text-align: center;
    }

    .section-header {
        font-size: 1.6rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #3366cc;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🕵️ 은평구 건강지표 분석 대시보드")

# -------------------------------
# 1. 소개 페이지
# -------------------------------
with open("은평구_건강분석_웹페이지.html", "r", encoding="utf-8") as f:
    intro_html = f.read()
st.components.v1.html(intro_html, height=500, scrolling=True)

st.markdown("---")

# -------------------------------
# 2. LSTM 예측 결과
# -------------------------------
def predict_lstm_timeseries(series):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))
    X, y = [], []
    for i in range(len(scaled) - 3):
        X.append(scaled[i:i+3])
        y.append(scaled[i+3])
    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(32, activation='relu', input_shape=(3, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, verbose=0)

    pred_scaled = model.predict(scaled[-3:].reshape((1, 3, 1)))
    return scaler.inverse_transform(pred_scaled)[0][0]

st.markdown("<div class='section-header'>📈 진관동 비만율 LSTM 예측 - 반복학습으로 인해 초기화가 이루어지는 점 참고해주세요!</div>", unsafe_allow_html=True)
df_lstm = pd.read_csv("은평구_이상치분석결과.csv", encoding="cp949")
obesity = df_lstm[df_lstm['항목명'] == '비만율']
series = df_lstm.sort_values(by='기준일자')['진관동']
predicted = predict_lstm_timeseries(series)
st.metric("예측된 진관동 다음 해 비만율", f"{predicted:.2f}%")

st.markdown("---")

# -------------------------------
# 3. 지도 시각화 (비만율 변화율)
# -------------------------------
st.markdown("<div class='section-header'>📌 비만 증가율 지도 (Folium) - 마커를 클릭해보세요!</div>", unsafe_allow_html=True)
with open("은평구_비만율_지도.html", "r", encoding="utf-8") as f:
    map_html = f.read()
st.components.v1.html(map_html, height=600, scrolling=False)

st.markdown("---")

# -------------------------------
# 4. 클러스터링 지도
# -------------------------------
st.markdown("<div class='section-header'>📍 클러스터링 지도 시각화 - 마커를 클릭해보세요!</div>", unsafe_allow_html=True)
cluster_df = pd.read_csv("은평구_건강지표_클러스터링.csv")
coords = {
    "진관동": [37.6344, 126.9184], "신사제2동": [37.6026, 126.9129],
    "불광제1동": [37.6101, 126.9313], "불광제2동": [37.6098, 126.9272],
    "응암제1동": [37.5999, 126.9187], "구산동": [37.6134, 126.9093],
    "녹번동": [37.6005, 126.9356], "역촌동": [37.6066, 126.9222],
    "신사제1동": [37.5982, 126.9178]
}
cluster_colors = {0: 'green', 1: 'blue', 2: 'red'}

m = folium.Map(location=[37.615, 126.92], zoom_start=12)
marker_cluster = MarkerCluster().add_to(m)

for _, row in cluster_df.iterrows():
    dong = row['행정동']
    if dong in coords:
        folium.Marker(
            location=coords[dong],
            popup=folium.Popup(
                f"""
                <div style='font-size:14px; width:280px;'>
                <b style='font-size:16px'>{dong}</b><br>
                클러스터: <b>{row['클러스터']}</b><br>
                비만율 평균: {row['비만율']:.1f}%<br>
                고혈압 신규 이용률: {row['고혈압신규의료이용률']:.1f}%<br>
                당뇨병 신규 이용률: {row['당뇨병신규의료이용률']:.1f}%
                </div>
                """,
                max_width=300
            ),
            icon=folium.Icon(color=cluster_colors.get(row['클러스터'], 'gray'))
        ).add_to(marker_cluster)

st_data = st_folium(m, width=1000, height=600)

st.markdown("---")

# -------------------------------
# 5. 클러스터링 테이블 (강조 포함)
# -------------------------------
st.markdown("<div class='section-header'>📊 클러스터링 데이터 테이블</div>", unsafe_allow_html=True)
st.dataframe(cluster_df.style.highlight_max(axis=0, color="lightyellow"))

st.markdown("""
### 🧠 클러스터 해석 (예시)
- 🟢 <b>Cluster 0:</b> 상대적으로 건강지표가 양호한 지역<br>
- 🔵 <b>Cluster 1:</b> 평균 수준의 건강 상태<br>
- 🔴 <b>Cluster 2:</b> 건강지표가 열악한 지역으로 정책적 개입이 필요한 곳
""", unsafe_allow_html=True)

st.markdown("---")

# -------------------------------
# 6. 분석 이미지 시각화
# -------------------------------
col1, col2 = st.columns(2)
with col1:
    st.markdown("### 🧪 이상치 탐지 결과 시각화")
    st.image(Image.open("이상치탐지결과.png"), use_container_width=True)
with col2:
    st.markdown("### 🧪 클러스터링 결과 시각화")
    st.image(Image.open("클러스터링결과.png"), use_container_width=True)

st.markdown("---")

# -------------------------------
# 7. CSV 다운로드
# -------------------------------
st.markdown("<div class='section-header'>📥 데이터 다운로드</div>", unsafe_allow_html=True)
st.download_button(
    label="📄 이상치 분석 원본 CSV 다운로드",
    data=open("은평구_이상치분석결과.csv", "rb").read(),
    file_name="은평구_이상치분석결과.csv",
    mime="text/csv"
)
st.download_button(
    label="📄 클러스터링 결과 CSV 다운로드",
    data=open("은평구_건강지표_클러스터링.csv", "rb").read(),
    file_name="은평구_건강지표_클러스터링.csv",
    mime="text/csv"
)
