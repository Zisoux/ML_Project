import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# 페이지 설정
st.set_page_config(layout="wide")
st.title("은평구 건강지표 클러스터링 분석")

# CSS로 중앙 정렬
st.markdown("""
    <style>
    .block-container {
        max-width: 1300px;
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
    </style>
""", unsafe_allow_html=True)

# 설명 상단 박스
st.markdown("""
<div class="desc-box">
    <h4>📊 분석 목적</h4>
    <p>은평구의 각 행정동을 대상으로 비만율, 고혈압, 당뇨병 신규 의료 이용률을 분석하고, K-Means 기반 클러스터링을 수행하여 유사한 건강 수준을 가진 지역을 그룹화했습니다.</p>
</div>
""", unsafe_allow_html=True)

# 데이터 불러오기
cluster_df = pd.read_csv("은평구_건강지표_클러스터링.csv")

# 좌표 정의
coords = {
    "진관동": [37.6344, 126.9184],
    "신사제2동": [37.6026, 126.9129],
    "불광제1동": [37.6101, 126.9313],
    "불광제2동": [37.6098, 126.9272],
    "응암제1동": [37.5999, 126.9187],
    "구산동": [37.6134, 126.9093],
    "녹번동": [37.6005, 126.9356],
    "역촌동": [37.6066, 126.9222],
    "신사제1동": [37.5982, 126.9178]
}

# 클러스터 색상 지정
cluster_colors = {0: 'green', 1: 'blue', 2: 'red'}

# 지도 만들기
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

# 중앙 정렬 시작
st.markdown('<div class="centered">', unsafe_allow_html=True)

# 지도 시각화
st.markdown("## 📍 클러스터링 지도 시각화")
st_folium(m, width=1000, height=600)

# 결과 테이블
st.markdown("## 📋 행정동별 클러스터링 결과")
st.dataframe(cluster_df.style.highlight_max(axis=0), use_container_width=True)

# 클러스터 요약 설명
st.markdown("""
### 🧠 클러스터 해석 (예시)
- 🟢 <b>Cluster 0:</b> 상대적으로 건강지표가 양호한 지역
- 🔵 <b>Cluster 1:</b> 평균 수준의 건강 상태
- 🔴 <b>Cluster 2:</b> 건강지표가 열악한 지역으로 정책적 개입이 필요한 곳
""", unsafe_allow_html=True)

# CSV 다운로드 버튼
csv = cluster_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="📥 클러스터링 결과 CSV 다운로드",
    data=csv,
    file_name="은평구_건강지표_클러스터링.csv",
    mime="text/csv"
)

# 중앙 정렬 끝
st.markdown('</div>', unsafe_allow_html=True)
