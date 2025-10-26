# -*- coding: utf-8 -*-
import os, sys, tempfile, json, io, ast
import numpy as np
import pandas as pd
import streamlit as st
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Matplotlib 한글 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# =========================================
# 경로/모듈 로드
# =========================================
WEB_DIR = r"C:\web"  # 필요 시 수정
if WEB_DIR not in sys.path:
    sys.path.append(WEB_DIR)

try:
    from feature import process_user_input
except Exception as e:
    st.set_page_config(page_title="TubeBoost", layout="wide", page_icon="🎬")
    st.error(f"feature.py import 실패: {e}")
    st.stop()

st.set_page_config(page_title="조회수 예측", layout="wide", page_icon="🎬")

# =========================================
# 유틸 함수 / 헬퍼
# =========================================
FEATURE_LABELS = {
    "title_length": "제목 길이",
    "word_count": "제목 단어 수",
    "emoji_count": "제목 이모지 수",
    "special_char_count": "특수문자 수",
    "person_count": "썸네일 사람 수",
    "object_count": "썸네일 객체 수",
    "has_text": "썸네일 텍스트 포함 여부",
    "brightness": "썸네일 밝기",
    "contrast": "썸네일 대비",
    "has_question_mark": "제목 ? 포함 여부",
    "has_exclamation": "제목 ! 포함 여부",
    "duration": "영상 길이",
}
def get_label(k: str) -> str:
    return FEATURE_LABELS.get(k, k)

def _safe_mean(series_like):
    try:
        return float(pd.to_numeric(series_like, errors="coerce").mean())
    except Exception:
        return np.nan

def _predict_views_by_regressor(model, df, cols):
    pred = model.predict(df[cols].values)
    # 로그 스케일 회귀 모델 가정 → expm1 시도
    try:
        return int(np.expm1(pred[0]))
    except Exception:
        try:
            return int(float(pred[0]))
        except Exception:
            return None

def load_csv_compat(path: str) -> pd.DataFrame:
    """인코딩/engine 차이로 인한 로드 실패를 줄이기 위한 안전한 CSV 로더"""
    encodings = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin1"]
    engines = ["c", "python"]
    last_err = None
    for enc in encodings:
        for eng in engines:
            try:
                with open(path, "r", encoding=enc, errors="ignore") as f:
                    return pd.read_csv(f, engine=eng)
            except Exception as e:
                last_err = e
                continue
    raise last_err if last_err else FileNotFoundError(path)

def parse_object_labels(obj_details):
    """object_details(dict/list/str)에서 라벨별 등장 빈도를 시리즈로 반환"""
    try:
        v = obj_details
        if isinstance(v, str):
            v = json.loads(v)
    except Exception:
        try:
            v = ast.literal_eval(obj_details)
        except Exception:
            v = {}
    if isinstance(v, dict):
        objs = v.get("objects", [])
    elif isinstance(v, list):
        objs = v
    else:
        objs = []
    labels = [o.get("label", "unknown") for o in objs if isinstance(o, dict)]
    if not labels:
        return pd.Series(dtype=int)
    return pd.Series(labels).value_counts().sort_values(ascending=False)

def extract_text_samples(text_details, topk=5):
    """text_details(list/str)에서 높은 확률 텍스트 샘플만 뽑아서 문자열 리스트로"""
    try:
        v = text_details
        if isinstance(v, str):
            v = json.loads(v)
    except Exception:
        try:
            v = ast.literal_eval(text_details)
        except Exception:
            v = []
    rows = []
    for t in v if isinstance(v, list) else []:
        txt = t.get("text", "")
        prob = float(t.get("probability", 0.0))
        rows.append((prob, txt))
    rows.sort(key=lambda x: x[0], reverse=True)
    return [f"{txt} (p={prob:.2f})" for prob, txt in rows[:topk] if txt]

def chip(text: str):
    # 제목에서 추출한 주요 명사 chip
    return (
        "<span style='display:inline-block;"
        "padding:4px 10px;margin:4px;border-radius:999px;"
        "background:#F1F5F9;border:1px solid #E2E8F0;"
        "font-size:12px'>"
        f"{text}</span>"
    )

def color_chip(name: str):
    # 추출된 대표 색상 시각용 칩
    bg = name
    return (
        "<span style='display:inline-block;"
        "padding:6px 12px;margin:4px;border-radius:10px;"
        "border:1px solid #e5e7eb;"
        f"background:{bg};color:#111827;font-weight:600'>{name}</span>"
    )

def get_peer_subset(md: pd.DataFrame, my_subs: int, pct: float, min_rows: int = 100) -> pd.DataFrame:
    """
    md에서 내 구독자수 ±pct% 범위의 영상들만 추림.
    min_rows 미만이면 범위를 점차 늘려서(최대 ±50%) 충분한 표본 확보.
    """
    if "subscriber_count" not in md.columns:
        return pd.DataFrame()
    subs = pd.to_numeric(md["subscriber_count"], errors="coerce")
    md2 = md.copy()
    md2["subscriber_count"] = subs

    p = max(0.01, pct / 100.0)
    my = float(my_subs)
    lo, hi = my * (1 - p), my * (1 + p)

    peer = pd.DataFrame()
    for _ in range(10):  # 최대 10번 (대략 ±50%까지)
        peer = md2[(md2["subscriber_count"] >= lo) & (md2["subscriber_count"] <= hi)]
        if len(peer) >= min_rows or p >= 0.5:
            break
        p *= 1.25
        lo, hi = my * (1 - p), my * (1 + p)
    return peer

def plot_sensitivity_barh(delta_df: pd.DataFrame, topn: int = 5, title: str = ""):
    """
    delta_df: columns => ['지표','증감(예측)']
    topn개 지표만 가로 막대로 시각화.
    """
    if delta_df is None or delta_df.empty or "증감(예측)" not in delta_df.columns:
        return

    plot_df = (
        delta_df.dropna(subset=["증감(예측)"])
                .sort_values("증감(예측)", ascending=False)
                .head(topn)
    )
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ["#80D2FF" if v > 0 else "#E57373" for v in plot_df["증감(예측)"]]

    ax.barh(plot_df["지표"], plot_df["증감(예측)"], color=colors)
    ax.invert_yaxis()
    ax.axvline(0, color="gray", lw=1)
    ax.set_title(title)

    for i, v in enumerate(plot_df["증감(예측)"]):
        ax.text(
            v + (0.02 * max(plot_df["증감(예측)"].max(), 1)),
            i,
            f"{int(v):+}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=8,
        )

    st.pyplot(fig)

def render_change_row(label, cur_val, tgt_val, gain=None):
    """
    지표별로 '지금 vs 추천' 비교 카드 한 줄 렌더
    """
    diff=float(cur_val)-float(tgt_val)
    arrow = "➡️"
    gain_txt = ""
    if gain is not None:
        if gain > 0:
            gain_txt = f"<span style='color:#16a34a;font-weight:600;'>+{gain:,}↑</span>"
        elif gain < 0:
            gain_txt = f"<span style='color:#dc2626;font-weight:600;'>{gain:,}↓</span>"

    if abs(diff) < 1e-6:
        base_phrase = "가 평균과 유사한 수준입니다."
    else:
        if diff > 0:
            if "길이" in label:
                base_phrase = "가 긴 편입니다."
            elif "수" in label:
                base_phrase = "가 많은 편입니다."
            elif "밝기" in label or "대비" in label:
                base_phrase = "가 높은 편입니다."
            else:
                base_phrase = "가 높은 편입니다."
        else:
            if "길이" in label:
                base_phrase = "가 짧은 편입니다."
            elif "수" in label:
                base_phrase = "가 적은 편입니다."
            elif "밝기" in label or "대비" in label:
                base_phrase = "가 낮은 편입니다."
            else:
                base_phrase = "가 낮은 편입니다."

    st.markdown(
        f"""
        <div style="
            border:1px solid #e5e7eb;
            border-radius:10px;
            padding:8px 12px;
            margin-bottom:6px;
            background:#f9fafb;
            font-size:18px;
            line-height:2;
        ">
            <div style="font-weight:600; color:#111827;">{label}
            <span style="font-weigth:300;">{base_phrase}</span>
            </div>
            <div style="color:#374151;">
                <span style="color:#4b5563;"> 현재 </span>
                <strong style="color:#111827;"> {cur_val:.1f} </strong>
                {arrow}
                <span style="color:#4b5563;"> 추천 </span>
                <strong style="color:#111827;"> {tgt_val:.1f} </strong>
                {("&nbsp; &nbsp; &nbsp;" + gain_txt) if gain_txt else ""}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================
# 사이드바 (피어 비교 설정)
# =========================================
with st.sidebar:
    st.subheader("비교 기준 설정")
    peer_pct = st.slider(
        "구독자 범위(±%)",
        min_value=1,
        max_value=50,
        value=15,
        step=1,
        help="내 구독자 수의 ±범위로 유사 구독자 채널을 선택합니다.",
    )
    peer_min_rows = st.number_input(
        "최소 표본 수",
        min_value=20,
        value=100,
        step=10,
        help="범위 내 표본이 부족하면 자동으로 범위를 넓힙니다.",
    )

    st.divider()
    st.subheader("모델/데이터 경로")
    saved_models_dir = st.text_input(
        "saved_models 경로",
        value=os.path.join(WEB_DIR, "saved_models"),
    )
    model_datas_path = st.text_input(
        "model_datas.csv 경로",
        value=os.path.join(WEB_DIR, "model_datas.csv"),
    )

# =========================================
# 헤더 / 히어로 섹션
# =========================================
hero_left, hero_cen, hero_right = st.columns([0.6, 0.1, 0.3])
with hero_left:
    st.markdown("### 🎬 영상 조회수 예측")
    st.markdown(
        "제목과 썸네일을 기반으로 이 영상이 어느 정도의 조회수를 낼 수 있을지 "
        "**사전에 진단**합니다."
    )
    st.caption(
        "제목 톤 · 키워드 훅 · 이모지/특수문자 사용 · 썸네일 구도와 가시성 등을 바탕으로 "
        "클릭 잠재력을 추정하고 개선 포인트를 제안해요."
    )

with hero_right:
    with st.container(border=True):
        st.caption("이 페이지에서 할 수 있는 것")
        st.markdown(
            "- 지금 작성한 제목이 잘 클릭되는 구조인지 확인\n"
            "- 썸네일/카피 조합의 강점·약점 체크\n"
            "- 조회수 잠재력(예측값) 확인"
        )

st.markdown("---")

# =========================================
# 1. 사용자 입력 폼
# =========================================
st.subheader("1. 분석할 영상 정보 입력")

with st.container(border=True):
    st.caption("제목 / 썸네일 / 메타 정보를 입력하세요. 실행하면 분석이 시작됩니다.")

    with st.form("predict_form"):
        c1, c2 = st.columns([3, 2])
        with c1:
            title = st.text_input(
                "제목",
                value="인생여행지였던 뉴욕에 다시 혼자 입장! 근데 왜 이렇게 변함...? (찐 뉴요커가 추천해준 빵집, 맛집, 카페 다 뿌시긴함ㅣ스킴스 엉뽕 대난리)",
            )
            thumbnail_url = st.text_input(
                "썸네일 URL",
                value="https://i.ytimg.com/vi/D-jOG2ybV1s/hq720.jpg",
            )
            thumbnail_file = st.file_uploader(
                "또는 이미지 업로드",
                type=["png", "jpg", "jpeg", "webp"],
            )
        with c2:
            duration = st.text_input("영상 길이 (MM:SS)", value="36:40")
            subscriber = st.number_input(
                "구독자 수",
                min_value=0,
                value=230000,
                step=1000,
            )
            total_videos = st.number_input(
                "누적 업로드 수",
                min_value=0,
                value=600,
                step=1,
            )
            category = st.text_input("카테고리", value="General")
        run = st.form_submit_button("조회수 예측 실행 🔍")

# =========================================
# 2. 실제 분석 실행 및 결과 출력
# =========================================
if run:
    # -------------------------------------
    # 썸네일 입력 정리 (URL 또는 업로드 파일)
    # -------------------------------------
    tmp_path = None
    if thumbnail_url.strip():
        thumb_input = thumbnail_url.strip()
    elif thumbnail_file is not None:
        suffix = os.path.splitext(thumbnail_file.name)[1].lower() or ".jpg"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        with open(tmp_path, "wb") as f:
            f.write(thumbnail_file.read())
        thumb_input = tmp_path
    else:
        st.error("썸네일 URL을 입력하거나 파일을 업로드해주세요.")
        st.stop()

    try:
        # -------------------------------------
        # 2-1. feature.py로 피처 추출 & 클러스터 예측
        # -------------------------------------
        df, cluster = process_user_input(
            title,
            thumb_input,
            duration,
            subscriber,
            total_videos,
            category,
        )
        row = df.iloc[0].to_dict()

        # -------------------------------------
        # 2-2. 클러스터별 회귀 모델 불러와 조회수 예측
        # -------------------------------------
        predicted_views = None
        feat_cols = [
            'duration','subscriber_count','brightness','contrast',
            'title_length','word_count','emoji_count','special_char_count',
            'is_clickbait','has_question_mark','has_exclamation',
            'pub_year','pub_month','pub_weekday',
            'color_red','color_blue','color_green','color_yellow',
            'color_purple','color_brown','color_grey','color_white','color_pink',
            'person_count','object_count','has_text',
            'person_left','person_middle','person_right',
            'person_small','person_medium','person_large',
            'text_left','text_middle','text_right',
            'text_small','text_medium','text_large'
        ]
        for c in feat_cols:
            if c not in df.columns:
                df[c] = 0

        import joblib
        reg, cluster_mean, cluster_top10_mean = None, None, None
        model_path = os.path.join(saved_models_dir, f"model_cluster_{cluster}.pkl")
        if os.path.exists(model_path):
            try:
                reg = joblib.load(model_path)
                predicted_views = _predict_views_by_regressor(reg, df, feat_cols)
            except Exception as e:
                st.warning(f"회귀 모델 로딩/예측 실패: {e}")

        # -------------------------------------
        # 2-3. 클러스터 통계 (평균 / 상위10%)
        # -------------------------------------
        md = None
        if os.path.exists(model_datas_path):
            try:
                md = load_csv_compat(model_datas_path)
                md["cluster"] = pd.to_numeric(md.get("cluster"), errors="coerce")
                filt = md[md["cluster"] == cluster]
                if len(filt):
                    cluster_mean = int(_safe_mean(filt.get("view_count")))
                    top10 = filt.nlargest(max(1, int(len(filt)*0.1)), "view_count")
                    cluster_top10_mean = int(_safe_mean(top10.get("view_count")))
            except Exception as e:
                st.warning(f"model_datas.csv 읽기 실패: {e}")

        # -------------------------------------
        # 2-4. 유사 구독자군(peer) 통계
        # -------------------------------------
        peer_mean_views = None
        peer_top10_mean_views = None
        peer_avg_core = None
        peer_avg_dict = {}

        if md is not None and len(md):
            peer_df = get_peer_subset(md, int(row.get("subscriber_count", 0)), peer_pct, peer_min_rows)
            if len(peer_df):
                # 조회수 평균
                if "view_count" in peer_df.columns:
                    peer_mean_views = int(_safe_mean(peer_df["view_count"]))
                    top10_peer = peer_df.nlargest(
                        max(1, int(len(peer_df) * 0.1)),
                        "view_count"
                    )
                    peer_top10_mean_views = int(_safe_mean(top10_peer["view_count"]))

                # 핵심 지표 평균 (길이/밝기 등)
                core_cols = [
                    "duration","title_length","word_count","emoji_count","special_char_count",
                    "has_question_mark","has_exclamation","brightness","contrast",
                    "person_count","object_count","has_text"
                ]
                peer_avg_core = {}
                for c in core_cols:
                    if c in peer_df.columns:
                        peer_avg_core[c] = round(_safe_mean(peer_df[c]), 3)

                # 레이아웃 분포 평균(사람, 텍스트 위치 등)
                dist_cols = [
                    "text_left","text_middle","text_right",
                    "text_small","text_medium","text_large",
                    "person_left","person_middle","person_right",
                    "person_small","person_medium","person_large",
                ]
                for c in dist_cols:
                    if c in peer_df.columns:
                        peer_avg_dict[c] = float(_safe_mean(peer_df[c]))
                    else:
                        peer_avg_dict[c] = 0.0

        # -------------------------------------
        # 2-5. 향상 시뮬레이션 (피어 평균으로 맞췄을 때)
        # -------------------------------------
        current_pred_views, improved_pred_views = None, None
        lift_abs, lift_pct = None, None
        top_change_rows, drivers_text = pd.DataFrame(), ""

        if reg is not None and peer_avg_core:
            # 내부적으로 예측해보는 헬퍼
            def simulate(modified):
                new_df = df.copy()
                for k, v in modified.items():
                    new_df.at[0, k] = v
                return _predict_views_by_regressor(reg, new_df, feat_cols)

            sims = []
            target_keys = [
                'title_length','emoji_count','special_char_count','person_count','has_text',
                'brightness','contrast','word_count','object_count'
            ]
            for k in target_keys:
                if k not in peer_avg_core:
                    continue
                cur_val = float(df.iloc[0].get(k, 0))
                peer_val = float(peer_avg_core[k])
                # 정수형으로 보는 지표는 반올림
                if k in ['emoji_count','special_char_count','person_count','has_text',
                         'word_count','object_count','title_length']:
                    peer_val = int(round(peer_val))

                pred_peer = simulate({k: peer_val})
                diff_val = None
                if predicted_views and pred_peer:
                    diff_val = int(pred_peer) - int(predicted_views)

                sims.append({
                    "지표":k,
                    "현재값":cur_val,
                    "평균값":peer_val,
                    "평균으로 맞출 때 예측":pred_peer,
                    "증감(예측)":diff_val
                })

            delta_views_df = pd.DataFrame(sims)

            # 여러 개선을 동시에 적용
            improving_changes = {
                r["지표"]: r["평균값"]
                for _, r in delta_views_df.iterrows()
                if r["증감(예측)"] and r["증감(예측)"] > 0
            }
            pred_combo = simulate(improving_changes) if improving_changes else None

            current_pred_views = int(predicted_views) if predicted_views else None
            improved_pred_views = int(pred_combo) if pred_combo else None

            if current_pred_views and improved_pred_views:
                lift_abs = improved_pred_views - current_pred_views
                lift_pct = (
                    (lift_abs / current_pred_views) * 100
                    if current_pred_views > 0 else None
                )

            top_change_rows = (
                delta_views_df
                .dropna(subset=["증감(예측)"])
                .sort_values("증감(예측)", ascending=False)
                .head(3)
            )
            drivers_text = ", ".join(
                get_label(v) for v in top_change_rows["지표"].tolist()
            ) if not top_change_rows.empty else ""

        # -------------------------------------
        # 2-6. 상단 요약 섹션
        # -------------------------------------
        st.success(f"분석이 완료됐습니다. 클러스터 {cluster}가 적용됩니다.")
        st.markdown(" ")

        topA, topB = st.columns([3, 2])
        with topA:
            with st.container(border=True):
                st.markdown("#### 📌 인사이트 요약")
                lines = []

                if current_pred_views:
                    lines.append(
                        f"- 현재 구성으로 예상되는 조회수: 약 {current_pred_views:,}회"
                    )
                if improved_pred_views and lift_abs and lift_pct:
                    lines.append(
                        f"- 핵심 지표를 평균 수준으로 조정 시: 약 {improved_pred_views:,}회 예상"
                    )
                    lines.append(
                        f"  (추정 +{lift_abs:,} / {lift_pct:.1f}% 향상)"
                    )
                if drivers_text:
                    lines.append(
                        f"- 특히 **{drivers_text}** 지표가 조회수에 큰 영향을 주는 신호"
                    )

                # if cluster_mean:
                #     lines.append(
                #         f"- 이 클러스터 평균 조회수: 약 {cluster_mean:,}회"
                #     )
                # if cluster_top10_mean:
                #     lines.append(
                #         f"- 상위 10% 평균 조회수: 약 {cluster_top10_mean:,}회"
                #     )

                # if peer_mean_views:
                #     lines.append(
                #         f"- 유사 구독자 채널 평균 조회수: 약 {peer_mean_views:,}회"
                #     )
                # if peer_top10_mean_views:
                #     lines.append(
                #         f"- 유사 구독자 상위 10% 평균: 약 {peer_top10_mean_views:,}회"
                #     )

                if lines:
                    st.write("\n".join(lines))
                else:
                    st.write("요약 정보를 계산할 수 없습니다.")

        with topB:
            with st.container(border=True):
                st.markdown("#### 썸네일 미리보기")
                try:
                    st.image(thumb_input, caption="", use_container_width=True)
                except Exception:
                    st.caption("썸네일 이미지를 표시할 수 없습니다.")

        st.markdown("---")

        # -------------------------------------
        # 3. 제목 분석
        # -------------------------------------
        st.subheader("2. 제목 분석")

        with st.container(border=True):
            tcol1, tcol2, tcol3 = st.columns(3)
            tcol1.metric("제목 길이", int(row.get("title_length", 0)))
            tcol2.metric("단어 수", int(row.get("word_count", 0)))
            tcol3.metric("특수문자 수", int(row.get("special_char_count", 0)))

            tcol1.metric("이모지 수", int(row.get("emoji_count", 0)))
            tcol2.metric("? 포함", "Yes" if int(row.get("has_question_mark", 0)) else "No")
            tcol3.metric("! 포함", "Yes" if int(row.get("has_exclamation", 0)) else "No")

            nouns = [
                row.get("top_noun_1",""),
                row.get("top_noun_2",""),
                row.get("top_noun_3",""),
            ]
            nouns = [n for n in nouns if str(n).strip()]
            if nouns:
                st.markdown("**상위 명사(Top Nouns)**")
                st.markdown("".join(chip(n) for n in nouns), unsafe_allow_html=True)

        st.markdown("---")

        # -------------------------------------
        # 4. 썸네일 분석
        # -------------------------------------
        st.subheader("3. 썸네일 분석")

        with st.container(border=True):
            scol1, scol2, scol3 = st.columns(3)
            scol1.metric("밝기", f"{float(row.get('brightness',0)):.1f}")
            scol2.metric("대비",  f"{float(row.get('contrast',0)):.1f}")
            scol3.metric("텍스트 존재", "Yes" if int(row.get("has_text", 0)) else "No")

            scol1.metric("사람 수", int(row.get("person_count", 0)))
            scol2.metric("객체 수", int(row.get("object_count", 0)))
            try:
                h, w = row.get("thumbnail_size", (None,None))
                scol3.metric("이미지 크기", f"{w}×{h}" if h and w else "—")
            except Exception:
                pass

            st.markdown("**주요 색상**")
            dom_colors = []
            if isinstance(row.get("dominant_colors"), str):
                dom_colors = [
                    c.strip()
                    for c in row["dominant_colors"].split(",")
                    if c.strip()
                ]
            if dom_colors:
                st.markdown(
                    "".join(color_chip(c) for c in dom_colors),
                    unsafe_allow_html=True
                )

            # --- 세부 썸네일 분석 (객체/텍스트 레이아웃) ---
            st.markdown(" ")
            with st.expander("🔍 썸네일 세부 분석"):
                # 객체 라벨 빈도
                lbl_counts = parse_object_labels(row.get("object_details", {}))

                # OCR 텍스트 디테일 가공
                text_details_raw = row.get("text_details", [])
                if isinstance(text_details_raw, str):
                    try:
                        text_details_raw = ast.literal_eval(text_details_raw)
                    except Exception:
                        text_details_raw = []

                text_tokens = []
                for item in text_details_raw:
                    txt = str(item.get("text", "")).strip()
                    if txt:
                        text_tokens.extend(txt.split())
                text_counts = pd.Series(Counter(text_tokens)).sort_values(ascending=False).head(10)

                c1_, c2_ = st.columns([1, 2])

                if not lbl_counts.empty:
                    c1_.markdown("**객체 라벨 분포**")
                    c1_.dataframe(
                        lbl_counts.rename("개수").to_frame(),
                        use_container_width=True
                    )
                else:
                    c1_.info("객체 인식 결과가 없습니다.")

                # 객체 박스 위치 시각화 (상위 5개)
                # 복붙한 로직 사용
                def clamp(v, lo, hi):
                    return max(lo, min(hi, v))

                def safe_parse_details(raw_val):
                    if isinstance(raw_val, (dict, list)):
                        return raw_val
                    if raw_val is None:
                        return None
                    if isinstance(raw_val, str):
                        raw_val = raw_val.strip()
                        if not raw_val:
                            return None
                        try:
                            return json.loads(raw_val)
                        except json.JSONDecodeError:
                            pass
                        try:
                            return ast.literal_eval(raw_val)
                        except Exception:
                            pass
                    return raw_val

                object_details_raw = safe_parse_details(row.get("object_details", {}))
                if object_details_raw is None:
                    object_details_raw = {}
                if isinstance(object_details_raw, dict):
                    objects_list = object_details_raw.get("objects", [])
                elif isinstance(object_details_raw, list):
                    objects_list = object_details_raw
                else:
                    objects_list = []

                all_boxes = []
                for obj in objects_list:
                    if not isinstance(obj, dict):
                        continue
                    try:
                        x = float(obj.get("x", 0))
                        y = float(obj.get("y", 0))
                        w_ = float(obj.get("width", 0) or obj.get("w", 0))
                        h_ = float(obj.get("height", 0) or obj.get("h", 0))
                    except Exception:
                        continue
                    if w_ > 0 and h_ > 0:
                        all_boxes.append((x, y, w_, h_))

                if len(all_boxes) == 0:
                    c2_.caption("객체 박스를 표시할 수 없습니다.")
                else:
                    max_x = max(x+w for (x,y,w,h) in all_boxes)
                    max_y = max(y+h for (x,y,w,h) in all_boxes)
                    if max_x <= 0: max_x = 1.0
                    if max_y <= 0: max_y = 1.0

                    CANVAS_W = 1280.0
                    CANVAS_H = 720.0

                    def scale_box_any(x, y, w_, h_):
                        sx = CANVAS_W / max_x
                        sy = CANVAS_H / max_y
                        return {
                            "x": x * sx,
                            "y": y * sy,
                            "w": w_ * sx,
                            "h": h_ * sy,
                        }

                    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=100)
                    ax.set_xlim([0, CANVAS_W])
                    ax.set_ylim([0, CANVAS_H])
                    ax.invert_yaxis()
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    used_label_positions = []

                    def get_non_overlapping_position(x, y, used_positions, min_dist=20, max_tries=20):
                        cand_x, cand_y = x, y
                        tries = 0
                        while tries < max_tries:
                            conflict = False
                            for (ux, uy) in used_positions:
                                dx = abs(cand_x - ux)
                                dy = abs(cand_y - uy)
                                if dx < min_dist and dy < min_dist:
                                    conflict = True
                                    break
                            if not conflict:
                                break
                            cand_y += 15
                            tries += 1
                        cand_x = clamp(cand_x, 0, CANVAS_W - 5)
                        cand_y = clamp(cand_y, 0, CANVAS_H - 5)
                        return cand_x, cand_y

                    sized_objects = []
                    for obj in objects_list:
                        if not isinstance(obj, dict):
                            continue
                        try:
                            ox = float(obj.get("x", 0))
                            oy = float(obj.get("y", 0))
                            ow = float(obj.get("width", 0) or obj.get("w", 0))
                            oh = float(obj.get("height", 0) or obj.get("h", 0))
                            label_val = str(obj.get("label", "obj"))
                        except Exception:
                            continue
                        if ow <= 0 or oh <= 0:
                            continue
                        area = ow * oh
                        sized_objects.append({
                            "x": ox, "y": oy, "w": ow, "h": oh,
                            "label": label_val,
                            "area": area
                        })
                    sized_objects = sorted(
                        sized_objects,
                        key=lambda d: d["area"],
                        reverse=True
                    )[:5]

                    for box in sized_objects:
                        scaled_o = scale_box_any(box["x"], box["y"], box["w"], box["h"])
                        label_txt = box["label"]
                        lw = 2.5 if label_txt.lower() == "person" else 1.5
                        ls = "--" if label_txt.lower() == "person" else "-"

                        rect_o = plt.Rectangle(
                            (scaled_o["x"], scaled_o["y"]),
                            scaled_o["w"],
                            scaled_o["h"],
                            fill=False,
                            linewidth=lw,
                            linestyle=ls
                        )
                        ax.add_patch(rect_o)

                        raw_lx = scaled_o["x"] + scaled_o["w"] + 5
                        raw_ly = scaled_o["y"] - 5
                        raw_lx = clamp(raw_lx, 0, CANVAS_W - 5)
                        raw_ly = clamp(raw_ly, 0, CANVAS_H - 5)

                        final_lx, final_ly = get_non_overlapping_position(
                            raw_lx,
                            raw_ly,
                            used_label_positions,
                            min_dist=20,
                            max_tries=20
                        )
                        used_label_positions.append((final_lx, final_ly))

                        ax.text(
                            final_lx,
                            final_ly,
                            f"[OBJ] {label_txt}",
                            fontsize=8,
                            va="top",
                            ha="left"
                        )

                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=100)
                    buf.seek(0)
                    c2_.image(buf, caption="객체 배치 맵 (정규화된 좌표)", use_container_width=True)
                    plt.close(fig)

                # OCR 텍스트 단어 + 신뢰도 테이블
                st.markdown(" ")
                st.markdown("**OCR 텍스트 단어 빈도 / 신뢰도**")

                # text_details_raw 다시 파싱해서 단어별 신뢰도 계산
                def safe_parse_details(raw_val):
                    if isinstance(raw_val, (dict, list)):
                        return raw_val
                    if raw_val is None:
                        return None
                    if isinstance(raw_val, str):
                        raw_val = raw_val.strip()
                        if not raw_val:
                            return None
                        try:
                            return json.loads(raw_val)
                        except json.JSONDecodeError:
                            pass
                        try:
                            return ast.literal_eval(raw_val)
                        except Exception:
                            pass
                    return raw_val

                text_details_list = safe_parse_details(row.get("text_details", []))
                word_conf_map = {}
                if isinstance(text_details_list, list):
                    for item in text_details_list:
                        text_val = str(item.get("text", "")).strip()
                        conf = float(item.get("probability", 0.0))
                        if not text_val:
                            continue
                        for w in text_val.split():
                            if w not in word_conf_map:
                                word_conf_map[w] = {"count": 0, "conf_sum": 0.0}
                            word_conf_map[w]["count"] += 1
                            word_conf_map[w]["conf_sum"] += conf

                if word_conf_map:
                    conf_df = pd.DataFrame([
                        {
                            "단어": w,
                            "빈도": v["count"],
                            "신뢰도(%)": round((v["conf_sum"] / v["count"]) * 100, 1)
                        }
                        for w, v in word_conf_map.items()
                    ])
                    conf_df = conf_df.sort_values(
                        ["빈도", "신뢰도(%)"],
                        ascending=[False, False]
                    ).reset_index(drop=True)
                    st.dataframe(conf_df, use_container_width=True)
                else:
                    st.info("OCR 텍스트 인식 결과가 없습니다.")

                # 텍스트/사람 배치 비교 (내 영상 vs 피어 평균)
                st.markdown(" ")
                st.markdown("**텍스트 / 인물 배치 특성 (내 영상 vs 평균)**")

                if "peer_avg_dict" not in locals():
                    peer_avg_dict = {}

                bar_width = 0.35
                my_color = "#80D2FF"
                peer_color = "#5C6AFF"

                # 텍스트 위치 분포
                c_txt_pos, c_txt_size = st.columns(2)
                with c_txt_pos:
                    text_pos_keys = ["text_left","text_middle","text_right"]
                    x_idx = range(len(text_pos_keys))
                    my_vals = [int(row.get(k, 0)) for k in text_pos_keys]
                    peer_vals = [peer_avg_dict.get(k, 0.0) for k in text_pos_keys]

                    fig, ax = plt.subplots()
                    ax.bar([i - bar_width/2 for i in x_idx], my_vals, bar_width,
                           label="내 영상", color=my_color)
                    ax.bar([i + bar_width/2 for i in x_idx], peer_vals, bar_width,
                           label="평균", color=peer_color)
                    ax.set_xticks(list(x_idx))
                    ax.set_xticklabels(["왼쪽","중앙","오른쪽"])
                    ax.set_ylabel("텍스트 박스 수")
                    ax.set_title("텍스트 위치 분포")
                    ax.legend()
                    st.pyplot(fig)

                with c_txt_size:
                    text_size_keys = ["text_small","text_medium","text_large"]
                    x_idx = range(len(text_size_keys))
                    my_vals = [int(row.get(k, 0)) for k in text_size_keys]
                    peer_vals = [peer_avg_dict.get(k, 0.0) for k in text_size_keys]

                    fig, ax = plt.subplots()
                    ax.bar([i - bar_width/2 for i in x_idx], my_vals, bar_width,
                           label="내 영상", color=my_color)
                    ax.bar([i + bar_width/2 for i in x_idx], peer_vals, bar_width,
                           label="평균", color=peer_color)
                    ax.set_xticks(list(x_idx))
                    ax.set_xticklabels(["작음","중간","큼"])
                    ax.set_ylabel("텍스트 박스 수")
                    ax.set_title("텍스트 크기 분포")
                    ax.legend()
                    st.pyplot(fig)

                # 사람 위치/크기 분포
                c_p_pos, c_p_size = st.columns(2)
                with c_p_pos:
                    person_pos_keys = ["person_left","person_middle","person_right"]
                    x_idx = range(len(person_pos_keys))
                    my_vals = [int(row.get(k, 0)) for k in person_pos_keys]
                    peer_vals = [peer_avg_dict.get(k, 0.0) for k in person_pos_keys]

                    fig, ax = plt.subplots()
                    ax.bar([i - bar_width/2 for i in x_idx], my_vals, bar_width,
                           label="내 영상", color=my_color)
                    ax.bar([i + bar_width/2 for i in x_idx], peer_vals, bar_width,
                           label="평균", color=peer_color)
                    ax.set_xticks(list(x_idx))
                    ax.set_xticklabels(["왼쪽","중앙","오른쪽"])
                    ax.set_ylabel("사람 수")
                    ax.set_title("사람 위치 분포")
                    ax.legend()
                    st.pyplot(fig)

                with c_p_size:
                    person_size_keys = ["person_small","person_medium","person_large"]
                    x_idx = range(len(person_size_keys))
                    my_vals = [int(row.get(k, 0)) for k in person_size_keys]
                    peer_vals = [peer_avg_dict.get(k, 0.0) for k in person_size_keys]

                    fig, ax = plt.subplots()
                    ax.bar([i - bar_width/2 for i in x_idx], my_vals, bar_width,
                           label="내 영상", color=my_color)
                    ax.bar([i + bar_width/2 for i in x_idx], peer_vals, bar_width,
                           label="평균", color=peer_color)
                    ax.set_xticks(list(x_idx))
                    ax.set_xticklabels(["작음","중간","큼"])
                    ax.set_ylabel("사람 수")
                    ax.set_title("사람 크기 분포")
                    ax.legend()
                    st.pyplot(fig)

        st.markdown("---")

        # -------------------------------------
        # 5. A/B 개선 시뮬레이션 (핵심 지표 조정)
        # -------------------------------------
        if 'reg' in locals() and reg is not None and peer_avg_core:
            st.subheader("4. 개선 시뮬레이션 (A/B 가이드)")

            with st.container(border=True):
                st.caption(
                    "유사 구독자 채널 평균 수준으로 일부 지표(제목 길이, 밝기 등)를 맞췄을 때 "
                    "예상 조회수 변화를 추정합니다."
                )

                # 시뮬 결과 요약 KPI
                kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
                kpi_col1.metric(
                    label="현재 예상 조회수",
                    value=(f"{current_pred_views:,}" if current_pred_views is not None else "—")
                )
                kpi_col2.metric(
                    label="개선 후 예상 조회수",
                    value=(f"{improved_pred_views:,}" if improved_pred_views is not None else "—"),
                    delta=(f"+{lift_abs:,}" if (lift_abs is not None and lift_abs > 0) else None)
                )
                kpi_col3.metric(
                    label="향상률(%)",
                    value=(f"{lift_pct:.1f}%" if lift_pct is not None else "—"),
                    delta=(f"+{lift_pct:.1f}%" if (lift_pct is not None and lift_pct > 0) else None)
                )

                st.markdown("### 수정하면 좋은 지표 Top 3")
                if not top_change_rows.empty:
                    for _, r in top_change_rows.iterrows():
                        render_change_row(
                            label   = get_label(r["지표"]),
                            cur_val = r["현재값"],
                            tgt_val = r["평균값"],
                            gain    = r["증감(예측)"]
                        )
                else:
                    st.info("개선 효과를 추정할 수 있는 지표가 충분하지 않습니다.")

                # 상세 표
                st.markdown("")
                st.markdown("#### 세부 비교표")
                sims_k = []
                # sims 재구성 (위에서 만든 delta_views_df 기반으로)
                if 'delta_views_df' in locals():
                    for _, row_sim in delta_views_df.iterrows():
                        sims_k.append({
                            "지표": get_label(row_sim["지표"]),
                            "현재값": row_sim["현재값"],
                            "평균값": row_sim["평균값"],
                            "평균으로 맞출 때 예측": row_sim["평균으로 맞출 때 예측"],
                            "증감(예측)": row_sim["증감(예측)"]
                        })
                if sims_k:
                    st.dataframe(pd.DataFrame(sims_k), use_container_width=True)
                else:
                    st.caption("세부 비교 데이터를 표시할 수 없습니다.")

    except Exception as e:
        st.exception(e)

    finally:
        if 'tmp_path' in locals() and tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
