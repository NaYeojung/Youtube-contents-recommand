import streamlit as st

st.set_page_config(
    page_title="YouTube 성장 어시스턴트",
    page_icon="📈",
    layout="wide"
)

# =========================
# Hero Section
# =========================
hero_left, hero_cen, hero_right = st.columns([0.6, 0.1, 0.3])

with hero_left:
    st.markdown(
        "### 유튜브 제목, 썸네일, 업로드 패턴까지\n데이터로 조회수를 예측합니다."
    )

    st.markdown(
        """
        우리는 단순한 '제목 추천기' 가 아닙니다.  
        **채널의 시청자 반응, 업로드 타이밍, 키워드 구조, 썸네일 스타일**을 분석하고  
        실제 조회수에 영향을 주는 요소만 뽑아 제안합니다.
        """.strip()
    )

    # 태그 칩 3개
    tag_cols = st.columns(3)
    with tag_cols[0]:
        st.markdown(
            """
            <div style="
                background-color:#EEF2FF;
                color:#4F46E5;
                font-weight:500;
                border-radius:999px;
                padding:6px 12px;
                border:1px solid #C7D2FE;
                font-size:0.8rem;
                text-align:center;
            ">
            📊 조회수 예측 모델
            </div>
            """,
            unsafe_allow_html=True
        )
    with tag_cols[1]:
        st.markdown(
            """
            <div style="
                background-color:#F0FDFA;
                color:#0F766E;
                font-weight:500;
                border-radius:999px;
                padding:6px 12px;
                border:1px solid #99F6E4;
                font-size:0.8rem;
                text-align:center;
            ">
            🧠 제목 템플릿 추천
            </div>
            """,
            unsafe_allow_html=True
        )
    with tag_cols[2]:
        st.markdown(
            """
            <div style="
                background-color:#FFF7ED;
                color:#C2410C;
                font-weight:500;
                border-radius:999px;
                padding:6px 12px;
                border:1px solid #FED7AA;
                font-size:0.8rem;
                text-align:center;
            ">
            🎯 채널 맞춤 키워드
            </div>
            """,
            unsafe_allow_html=True
        )

with hero_right:
    with st.container(border=True):
        st.caption("오늘의 목표")
        st.markdown("**영상 올리기 전에, 진짜로 클릭할 제목인지 확인하세요.**")
        st.write(
            "제목과 썸네일은 클릭률을, 클릭률은 조회수를 좌우합니다. "
            "이 도구는 '감'이 아니라 **데이터**로 그 답을 줍니다."
        )

st.markdown("---")


# =========================
# Feature Cards Section
# =========================
st.subheader("무엇을 도와줄까요?")

feat_col1, feat_col2 = st.columns(2)

with feat_col1:
    with st.container(border=True):
        st.markdown("**기능 01 · 영상 단위 분석**")
        st.markdown("#### 제목 / 썸네일 기반 조회수 예측")
        st.write(
            "- 영상의 제목, 썸네일 특징, 키워드 톤을 분석해\n"
            "- 예상 조회수와 클릭 잠재력을 점수화하고\n"
            "- 강조 단어, 감정 훅, 숫자/이모지 사용 등 개선 포인트를 제안합니다."
        )

        st.markdown(
            """
            <div style="
                display:inline-block;
                background-color:#4338CA;
                color:#fff;
                font-size:0.9rem;
                font-weight:600;
                padding:8px 14px;
                border-radius:8px;
                border:1px solid #4338CA;
                margin-top:0.5rem;
            ">
                ▶ 영상별 조회수 예측
            </div>
            """,
            unsafe_allow_html=True
        )
        st.caption("제목과 썸네일 이미지를 넣으면 결과를 볼 수 있어요.")
        st.markdown(
            '<p style="color:#4338CA; font-size:0.8rem; font-weight:500;">'
            '좌측 사이드바에서 "조회수 예측" 페이지로 이동하세요.'
            "</p>",
            unsafe_allow_html=True,
        )

with feat_col2:
    with st.container(border=True):
        st.markdown("**기능 02 · 채널 단위 분석**")
        st.markdown("#### 채널 맞춤 분석 & 추천 제목 생성")
        st.write(
            "- 채널 업로드 빈도, 시간대, 반응(좋아요율/댓글율)을 분석하고\n"
            "- 잘 터진 영상의 패턴을 학습한 뒤\n"
            "- 그 채널만을 위한 신규 영상용 추천 제목과 키워드를 생성합니다."
        )

        st.markdown(
            """
            <div style="
                display:inline-block;
                background-color:#065F46;
                color:#fff;
                font-size:0.9rem;
                font-weight:600;
                padding:8px 14px;
                border-radius:8px;
                border:1px solid #065F46;
                margin-top:0.5rem;
            ">
                ▶ 채널 분석 & 추천 제목
            </div>
            """,
            unsafe_allow_html=True
        )
        st.caption("채널 ID만 입력하면 자동 분석돼요.")
        st.markdown(
            '<p style="color:#065F46; font-size:0.8rem; font-weight:500;">'
            '좌측 사이드바에서 "추천 제목 대시보드" 페이지로 이동하세요.'
            "</p>",
            unsafe_allow_html=True,
        )

st.markdown("---")


# =========================
# Why this matters / 신뢰 섹션
# =========================
st.subheader("왜 이 도구인가?")

why1, why2, why3 = st.columns(3)

with why1:
    with st.container(border=True):
        st.markdown("**📈 실제 성능 기반**")
        st.write(
            "모델은 채널의 실제 업로드 기록과 반응(조회수, 좋아요, 댓글)을 "
            "기반으로 학습된 지표를 사용합니다."
        )

with why2:
    with st.container(border=True):
        st.markdown("**🧠 제목 구조 분석**")
        st.write(
            "그냥 '멋진 말'을 추천하는 게 아닙니다. "
            "상위 퍼포먼스 제목에서 반복적으로 쓰인 **슬롯(주제·상황·감정 훅)** "
            "패턴을 추출해 조합합니다."
        )

with why3:
    with st.container(border=True):
        st.markdown("**🎯 크리에이터 친화**")
        st.write(
            "완성 문장만 던져주고 끝나지 않습니다.\n"
            "대체 제목 후보 / 슬롯별 강력 키워드 / 템플릿 트렌드까지 함께 제시하므로\n"
            "다양하게 활용할 수 있습니다."
        )

st.markdown("---")


# =========================
# Final CTA
# =========================
st.markdown(
    """
    <div style="text-align:center; font-size:0.9rem; line-height:1.5; color:#6B7280;">
        크리에이터의 시간을 아끼고,<br/>
        제목과 썸네일로 끌 수 있는 잠재 조회수를 극대화하기 위해 만들어졌습니다.
        <br/><br/>
        이제 사이드바에서 원하는 기능을 선택해 시작하세요 👇
    </div>
    """,
    unsafe_allow_html=True
)
