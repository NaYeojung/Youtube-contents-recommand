import streamlit as st

st.set_page_config(
    page_title="YouTube 성장 어시스턴트",
    page_icon="📈",
    layout="wide"
)

# =========================
# Hero Section
# =========================
hero_left, hero_right = st.columns([0.6, 0.4])

with hero_left:
    st.markdown(
        """
<div style="font-size:2rem; font-weight:700; line-height:1.3; color:#111;">
    유튜브 제목, 썸네일, 업로드 패턴까지<br/>
    데이터로 조회수를 예측합니다.
</div>

<div style="
    font-size:1rem;
    line-height:1.5;
    color:#444;
    margin-top:1rem;
">
    우리는 단순한 '제목 추천기'가 아닙니다.<br/>
    <b>채널의 시청자 반응, 업로드 타이밍, 키워드 구조, 썸네일 스타일</b>을 분석하고,<br/>
    <span style="color:#4F46E5; font-weight:600;">
        실제 조회수에 영향을 주는 요소
    </span>만 뽑아 제안합니다.
</div>

<div style="
    margin-top:1.5rem;
    display:flex;
    flex-wrap:wrap;
    gap:0.5rem;
    font-size:0.8rem;
    color:#555;
">
    <div style="
        background-color:#EEF2FF;
        color:#4F46E5;
        font-weight:500;
        border-radius:999px;
        padding:6px 12px;
        border:1px solid #C7D2FE;
        white-space:nowrap;
    ">
        📊 조회수 예측 모델
    </div>
    <div style="
        background-color:#F0FDFA;
        color:#0F766E;
        font-weight:500;
        border-radius:999px;
        padding:6px 12px;
        border:1px solid #99F6E4;
        white-space:nowrap;
    ">
        🧠 제목 템플릿 추천
    </div>
    <div style="
        background-color:#FFF7ED;
        color:#C2410C;
        font-weight:500;
        border-radius:999px;
        padding:6px 12px;
        border:1px solid #FED7AA;
        white-space:nowrap;
    ">
        🎯 채널 맞춤 키워드
    </div>
</div>
        """,
        unsafe_allow_html=True
    )

with hero_right:
    st.markdown(
        """
<div style="
    border:1px solid #E5E7EB;
    border-radius:12px;
    padding:16px 20px;
    background:radial-gradient(circle at 20% 20%, #EEF2FF 0%, #FFFFFF 60%);
    box-shadow:0 20px 40px rgba(0,0,0,0.04);
">
    <div style="font-size:0.8rem; font-weight:600; color:#4F46E5; margin-bottom:0.5rem;">
        오늘의 목표
    </div>
    <div style="font-size:1rem; font-weight:600; color:#111; line-height:1.4;">
        영상 올리기 전에,<br/>
        진짜로 클릭할 제목인지 확인하세요.
    </div>
    <div style="font-size:0.8rem; color:#444; line-height:1.5; margin-top:0.75rem;">
        제목과 썸네일은 클릭률을, 클릭률은 조회수를 좌우합니다.<br/>
        이 도구는 "감"이 아니라 <b>데이터</b>로 그 답을 줍니다.
    </div>
</div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)

# =========================
# Feature Cards Section
# =========================
feat1, feat2 = st.columns(2)

with feat1:
    card_html_1 = """
        <div style="
            border:1px solid #E5E7EB;
            border-radius:12px;
            padding:20px 24px;
            background-color:#FFFFFF;
            box-shadow:0 16px 32px rgba(0,0,0,0.03);
            height:100%;
        ">
        <div style="font-size:0.8rem; font-weight:600; color:#4338CA;">
            기능 01
        </div>

        <div style="
            font-size:1.1rem;
            font-weight:600;
            color:#111;
            margin-top:0.4rem;
            line-height:1.4;
        ">
            제목 / 썸네일 기반<br/>
            조회수 예측 &amp; 개선 인사이트
        </div>

        <div style="
            font-size:0.9rem;
            color:#444;
            margin-top:0.75rem;
            line-height:1.5;
        ">
            • 영상의 제목, 썸네일 특징, 키워드 톤을 분석해<br/>
            • 예상 조회수와 클릭 잠재력을 점수화하고<br/>
            • 개선 포인트(강조 단어, 감정 훅, 숫자/이모지 사용 등)를 제안합니다.
        </div>

        <div style="margin-top:1rem;">
            <div style="
                display:inline-block;
                background-color:#4338CA;
                color:#fff;
                font-size:0.9rem;
                font-weight:600;
                padding:8px 14px;
                border-radius:8px;
                border:1px solid #4338CA;
            ">
                ▶ 영상별 조회수 예측 바로가기
            </div>
            <div style="
                font-size:0.7rem;
                color:#6B7280;
                margin-top:0.5rem;
            ">
                (제목과 썸네일 이미지를 넣으면 결과를 볼 수 있어요)
            </div>
        </div>
    </div>"""
    st.markdown(card_html_1, unsafe_allow_html=True)

    st.markdown(
        """<div style="font-size:0.8rem; color:#4338CA; font-weight:500; margin-top:0.5rem;">
            👉 좌측 사이드바에서 "조회수 예측" 페이지로 이동하세요.
        </div>""",
        unsafe_allow_html=True
    )

with feat2:
    card_html_2 = """<div style="
        border:1px solid #E5E7EB;
        border-radius:12px;
        padding:20px 24px;
        background-color:#FFFFFF;
        box-shadow:0 16px 32px rgba(0,0,0,0.03);
        height:100%;
    ">
        <div style="font-size:0.8rem; font-weight:600; color:#065F46;">
            기능 02
        </div>

        <div style="
            font-size:1.1rem;
            font-weight:600;
            color:#111;
            margin-top:0.4rem;
            line-height:1.4;
        ">
            채널 맞춤 분석 +<br/>
            조회수형 추천 제목 생성
        </div>

        <div style="
            font-size:0.9rem;
            color:#444;
            margin-top:0.75rem;
            line-height:1.5;
        ">
            • 채널의 업로드 빈도, 시간대, 반응(좋아요율/댓글율)을 분석하고<br/>
            • 잘 터진 영상의 패턴을 추출해서<br/>
            • 그 채널만을 위한 신규 영상용 추천 제목과 키워드를 생성합니다.
        </div>

        <div style="margin-top:1rem;">
            <div style="
                display:inline-block;
                background-color:#065F46;
                color:#fff;
                font-size:0.9rem;
                font-weight:600;
                padding:8px 14px;
                border-radius:8px;
                border:1px solid #065F46;
            ">
                ▶ 채널 분석 &amp; 추천 제목 받기
            </div>
            <div style="
                font-size:0.7rem;
                color:#6B7280;
                margin-top:0.5rem;
            ">
                (채널 ID만 입력하면 자동 분석돼요)
            </div>
        </div>
    </div>"""
    st.markdown(card_html_2, unsafe_allow_html=True)

    st.markdown(
        """<div style="font-size:0.8rem; color:#065F46; font-weight:500; margin-top:0.5rem;">
            👉 좌측 사이드바에서 "추천 제목 대시보드" 페이지로 이동하세요.
        </div>""",
        unsafe_allow_html=True
    )

st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)

# =========================
# Credibility / Value Props
# =========================
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
<div style="
    background-color:#F9FAFB;
    border:1px solid #E5E7EB;
    border-radius:10px;
    padding:16px;
    font-size:0.8rem;
    line-height:1.5;
    height:100%;
">
    <div style="
        font-weight:600;
        color:#111;
        font-size:0.8rem;
        margin-bottom:0.4rem;
    ">
        📈 실제 성능 기반
    </div>
    <div style="color:#444;">
        모델은 채널의 실제 업로드 기록과 반응(조회수, 좋아요, 댓글)을 기반으로 학습된 지표를 사용합니다.
    </div>
</div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        """
<div style="
    background-color:#F9FAFB;
    border:1px solid #E5E7EB;
    border-radius:10px;
    padding:16px;
    font-size:0.8rem;
    line-height:1.5;
    height:100%;
">
    <div style="
        font-weight:600;
        color:#111;
        font-size:0.8rem;
        margin-bottom:0.4rem;
    ">
        🧠 제목 구조 분석
    </div>
    <div style="color:#444;">
        '멋있어 보이는 문장'이 아니라, 상위 퍼포먼스 제목에서 반복적으로 쓰인
        <b>슬롯(주제·상황·감정 훅)</b>을 추출해 조합합니다.
    </div>
</div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        """
<div style="
    background-color:#F9FAFB;
    border:1px solid #E5E7EB;
    border-radius:10px;
    padding:16px;
    font-size:0.8rem;
    line-height:1.5;
    height:100%;
">
    <div style="
        font-weight:600;
        color:#111;
        font-size:0.8rem;
        margin-bottom:0.4rem;
    ">
        🎯 크리에이터 친화
    </div>
    <div style="color:#444;">
        '최종 문장'만 던져주지 않습니다.<br/>
        대체 제목 후보 / 슬롯별 강력 키워드 / 썸네일 훅까지 같이 드립니다.<br/>
        바로 썸네일 텍스트로 복사해서 써도 될 정도로.
    </div>
</div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)

st.markdown(
    """
<div style="
    text-align:center;
    font-size:0.8rem;
    color:#6B7280;
    line-height:1.5;
    padding-bottom:3rem;
">
    이 도구는 크리에이터의 시간을 아끼고,<br/>
    제목과 썸네일로 끌 수 있는 잠재 조회수를 극대화하기 위해 만들어졌습니다.
    <br/><br/>
    사이드바에서 원하는 기능을 선택해 시작하세요 👇
</div>
    """,
    unsafe_allow_html=True
)
