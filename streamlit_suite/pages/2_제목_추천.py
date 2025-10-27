import os, json, numpy as np, pandas as pd, streamlit as st, altair as alt
import ast
import re

# =========================
# 기본 설정
# =========================
st.set_page_config(page_title="TubeBoost", layout="wide", page_icon="📺")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# =========================
# 공용 유틸
# =========================
def load_csv(fname):
    path = os.path.join(DATA_DIR, fname)
    return pd.read_csv(path) if os.path.exists(path) else None

def load_json(fname):
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def first_non_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

def parse_count_series(series):
    """
    문자열 형태의 수치(조회수/좋아요/댓글 등)를 float로 변환
    예: "12,345", "1.2만", "3.4천", "982", "1,234회"
    """
    if series is None:
        return pd.Series(dtype="float64")

    def _parse_one(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        s = s.replace(",", "").replace("회", "").strip()

        if s.endswith("만"):
            try:
                return float(s[:-1]) * 10000
            except:
                return np.nan

        if s.endswith("천"):
            try:
                return float(s[:-1]) * 1000
            except:
                return np.nan

        try:
            return float(s)
        except:
            return np.nan

    return series.apply(_parse_one)

def load_channel_df():
    """
    channel(@KoreanCryingGuy)_videos_metadata.csv 를 불러와
    분석용 파생 컬럼 생성
    """
    ch_path = os.path.join(DATA_DIR, "channel(@KoreanCryingGuy)_videos_metadata.csv")
    if not os.path.exists(ch_path):
        return None

    df_ch = pd.read_csv(ch_path)

    # 업로드 시각 -> datetime
    if "published_date" in df_ch.columns:
        df_ch["_published_dt"] = pd.to_datetime(df_ch["published_date"], errors="coerce")
    else:
        df_ch["_published_dt"] = pd.NaT

    # 숫자형 지표 파싱
    df_ch["_views"]    = parse_count_series(df_ch["view_count"]    if "view_count"    in df_ch.columns else None)
    df_ch["_likes"]    = parse_count_series(df_ch["like_count"]    if "like_count"    in df_ch.columns else None)
    df_ch["_comments"] = parse_count_series(df_ch["comment_count"] if "comment_count" in df_ch.columns else None)

    # 반응 비율
    df_ch["_like_rate"] = np.where(
        (df_ch["_views"].notna()) & (df_ch["_views"] > 0),
        df_ch["_likes"] / df_ch["_views"],
        np.nan
    )
    df_ch["_comment_rate"] = np.where(
        (df_ch["_views"].notna()) & (df_ch["_views"] > 0),
        df_ch["_comments"] / df_ch["_views"],
        np.nan
    )

    # 업로드 요일 / 시간
    df_ch["_dow"] = df_ch["_published_dt"].dt.dayofweek  # 월=0 ... 일=6
    df_ch["_hour"] = df_ch["_published_dt"].dt.hour      # 업로드 시각 (없으면 NaN)

    return df_ch

def build_channel_summary(df_ch: pd.DataFrame):
    """
    채널 KPI 요약 계산
    반환 summary:
      total_videos : 총 업로드 수
      recent_30_cnt : 최근 30일 업로드 수
      avg_views_all : 전체 평균 조회수
      median_views_all : 전체 중앙값 조회수
      avg_views_recent5 : 최근 5개 영상 평균 조회수
      avg_like_rate : 평균 좋아요율
      top_hits : [{title, views}] (상위 3개)
    """
    summary = {}

    # 총 업로드 수
    summary["total_videos"] = len(df_ch)

    # 최근 30일 업로드 수
    if df_ch["_published_dt"].notna().any():
        now = pd.Timestamp.now(tz=None)
        recent_mask = df_ch["_published_dt"] >= (now - pd.Timedelta(days=30))
        summary["recent_30_cnt"] = int(recent_mask.sum())
    else:
        summary["recent_30_cnt"] = None

    # 조회수 통계
    if df_ch["_views"].notna().any():
        summary["avg_views_all"] = int(df_ch["_views"].mean())
        summary["median_views_all"] = int(df_ch["_views"].median())
    else:
        summary["avg_views_all"] = None
        summary["median_views_all"] = None

    # 최근 5개 평균 조회수
    if df_ch["_published_dt"].notna().any():
        recent_5 = df_ch.sort_values("_published_dt", ascending=False).head(5)
    else:
        recent_5 = df_ch.tail(5)
    if recent_5["_views"].notna().any():
        summary["avg_views_recent5"] = int(recent_5["_views"].mean())
    else:
        summary["avg_views_recent5"] = None

    # 좋아요율 평균
    if df_ch["_like_rate"].notna().any():
        summary["avg_like_rate"] = float(df_ch["_like_rate"].mean())
    else:
        summary["avg_like_rate"] = None

    # 히트 영상 TOP 3 (조회수 높은 순)
    top_hits_rows = df_ch.sort_values("_views", ascending=False).head(3)
    top_list = []
    for _, r in top_hits_rows.iterrows():
        vid_title = r.get("title", "(제목 없음)")
        vid_views = r.get("_views", np.nan)
        if pd.notna(vid_views):
            top_list.append({
                "title": str(vid_title),
                "views": int(vid_views),
            })
    summary["top_hits"] = top_list

    # 키 보장
    for k in ["avg_like_rate", "median_views_all", "avg_views_recent5"]:
        summary.setdefault(k, None)

    return summary

# =========================
# 데이터 로드
# =========================
reco      = first_non_none(load_csv("reco_results.csv"),           load_csv("reco_results_demo.csv"))
tpl_stats = first_non_none(load_csv("template_keyword_stats.csv"), load_csv("template_keyword_stats_demo.csv"))
kw_stats  = first_non_none(load_csv("keyword_stats.csv"),          load_csv("keyword_stats_demo.csv"))
feat_imp  = first_non_none(load_csv("feature_importances.csv"),    load_csv("feature_importances_demo.csv"))
p10_style = first_non_none(load_json("p10_style.json"),            load_json("p10_style_demo.json"))
candidates_df = first_non_none(load_csv("title_recommend_candidates.csv"), None)

if reco is None:
    st.error("reco_results.csv가 필요합니다.")
    st.stop()

# =========================
# 세션 상태
# =========================
if "analysis_mode" not in st.session_state:
    st.session_state["analysis_mode"] = False

# =========================
# 사이드바 필터
# =========================
with st.sidebar:
    st.subheader("필터")
    tpl_choices = ["(전체)"] + sorted(reco["template"].dropna().unique().tolist())
    tpl_sel = st.selectbox("템플릿", tpl_choices, help="특정 템플릿만 보고 싶을 때 선택하세요.")

# 필터 적용
df = reco.copy()
if tpl_sel != "(전체)":
    df = df[df["template"] == tpl_sel]
df["predicted_views"] = pd.to_numeric(df["predicted_views"], errors="coerce")
df = df.dropna(subset=["predicted_views"])

# =========================
# 헤더/Hero 섹션
# =========================
hero_left, hero_cen, hero_right = st.columns([0.6, 0.1, 0.3])
with hero_left:
    st.markdown("### 📺 채널 기반 추천 콘텐츠")
    st.markdown(
        "채널의 업로드 패턴·조회수·반응(좋아요/댓글)을 읽고, "
        "**해당 채널에서 잘 터지는 제목 스타일**을 뽑아냅니다."
        " 내 채널 영상 뿐만 아니라 유사한 채널들의 인기 동영상을 통해 **트렌드를 반영**합니다."
    )
    st.caption(
        "업로드 타이밍, 시청자 반응도, 터진 영상의 공통 구조를 기반으로 "
        "‘조회수형 제목 템플릿’을 자동 생성합니다."
    )

with hero_right:
    with st.container(border=True):
        st.caption("이 페이지에서 할 수 있는 것")
        st.markdown(
            "- 채널 업로드 성향/성과 지표 확인\n"
            "- 기간 필터로 특정 구간만 분석\n"
            "- 그 채널 전용 추천 제목 & 슬롯 키워드 확보"
        )

st.markdown("---")

# =========================
# 1. 채널 ID 입력
# =========================
st.subheader("1. 채널 선택 및 분석 시작")
with st.container(border=True):
    c1, c3, c2 = st.columns([2, 0.03, 1])

    with c1:
        with st.form("predict_form"):
            title_input = st.text_input("채널 ID", value="@KoreanCryingGuy")
            run = st.form_submit_button("추천 제목 생성 🔍")

            if run:
                st.session_state["analysis_mode"] = True

    with c2:
        st.caption("채널 ID만 입력하면 아래 전체 분석이 한 번에 생성됩니다.")

# =========================
# 2. 채널 분석 섹션
# =========================
if st.session_state["analysis_mode"]:
    ch_df = load_channel_df()

    if ch_df is not None and len(ch_df) > 0:
        # 2-1. 업로드 타임라인 인덱스 기반 슬라이더
        timeline_df = ch_df.dropna(subset=["_published_dt"]).copy()
        timeline_df = timeline_df.sort_values("_published_dt").reset_index(drop=True)
        timeline_df["_t_idx"] = timeline_df.index

        if len(timeline_df) > 0:
            min_idx = int(timeline_df["_t_idx"].min())
            max_idx = int(timeline_df["_t_idx"].max())
        else:
            min_idx = 0
            max_idx = 0

        st.markdown(" ")
        st.subheader("2. 채널 분석")

        # 상단: 기간 선택 / 설명
        with st.container(border=True):
            top_left, top_cen, top_right = st.columns([0.6, 0.03, 0.37])

            with top_left:
                if len(timeline_df) > 0:
                    default_start = min_idx
                    default_end   = max_idx

                    idx_range = st.slider(
                        "분석할 업로드 구간",
                        min_value=min_idx,
                        max_value=max_idx,
                        value=(default_start, default_end),
                        step=1,
                        help="왼쪽→오른쪽 슬라이더로 분석할 업로드 범위를 지정하세요.",
                    )
                    start_idx, end_idx = idx_range

                    start_dt = timeline_df.loc[
                        timeline_df["_t_idx"] == start_idx, "_published_dt"
                    ].min()
                    end_dt   = timeline_df.loc[
                        timeline_df["_t_idx"] == end_idx, "_published_dt"
                    ].max()
                    end_dt_inclusive = end_dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

                    mask_in_range = (
                        (ch_df["_published_dt"].notna()) &
                        (ch_df["_published_dt"] >= start_dt) &
                        (ch_df["_published_dt"] <= end_dt_inclusive)
                    )
                    ch_sub = ch_df[mask_in_range].copy()

                    if len(ch_sub) == 0:
                        st.warning("이 구간에 해당하는 영상이 없어요. 전체 채널로 대체합니다.")
                        ch_sub = ch_df.copy()
                        st.caption(f"(전체 {len(ch_sub)}개 영상 분석)")
                    else:
                        st.caption(
                            f"{start_dt.date()} ~ {end_dt.date()} / {len(ch_sub)}개 영상"
                        )
                else:
                    st.info("날짜 정보가 없어 전체 영상을 분석합니다.")
                    ch_sub = ch_df.copy()
                    st.caption(f"(전체 {len(ch_sub)}개 영상 분석)")

            with top_right:
                st.caption(
                    "📌 이 구간만 따로 분석하면, 예를 들어 최근 1~2달만의 분위기가 "
                    "예전이랑 얼마나 달라졌는지 확인할 수 있어요."
                )

        # 2-2. 채널 KPI 카드
        summary = build_channel_summary(ch_sub)

        st.markdown(" ")

        kpi_cols = st.columns(3)

        def fmt_views(v):
            if v is None:
                return "-"
            return f"{int(v):,}회"

        # 업로드 현황
        with kpi_cols[0]:
            with st.container(border=True):
                st.markdown("**업로드 현황**")
                st.markdown(f"### 총 {summary['total_videos']}개 업로드")
                recent30 = summary.get("recent_30_cnt")
                if recent30 is not None:
                    st.caption(f"최근 30일 업로드: {recent30}개")
                else:
                    st.caption("최근 30일 업로드: -")

        # 조회수 퍼포먼스
        with kpi_cols[1]:
            with st.container(border=True):
                st.markdown("**조회수 평균**")
                st.markdown(f"### {fmt_views(summary.get('avg_views_all'))}")
                st.caption(
                    f"최근 5개 평균: {fmt_views(summary.get('avg_views_recent5'))}"
                )

        # 시청자 반응도
        with kpi_cols[2]:
            with st.container(border=True):
                st.markdown("**시청자 반응도**")
                avg_like_rate = summary.get("avg_like_rate")
                if avg_like_rate is not None and not np.isnan(avg_like_rate):
                    like_pct = avg_like_rate * 100.0
                    st.markdown(f"### {like_pct:.2f}%")
                else:
                    st.markdown("### 평균 좋아요율 -")
                st.caption("좋아요율 = 좋아요수 / 조회수")

        # 2-3. 상위 퍼포먼스 영상 TOP 5
        st.markdown(" ")
        with st.container(border=True):
            st.markdown("#### 🔥 상위 퍼포먼스 영상 TOP 5")

            top5_df = (
                ch_sub.sort_values("_views", ascending=False)
                .loc[:, ["title", "_views", "_likes", "_comments", "_published_dt"]]
                .head(5)
                .rename(columns={
                    "title": "제목",
                    "_views": "조회수",
                    "_likes": "좋아요",
                    "_comments": "댓글",
                    "_published_dt": "업로드일"
                })
            )

            # 보기 좋은 문자열 포맷
            top5_df["조회수"] = top5_df["조회수"].map(
                lambda x: f"{int(x):,}" if pd.notna(x) else "-"
            )
            if "좋아요" in top5_df.columns:
                top5_df["좋아요"] = top5_df["좋아요"].map(
                    lambda x: f"{int(x):,}" if pd.notna(x) else "-"
                )
            if "댓글" in top5_df.columns:
                top5_df["댓글"] = top5_df["댓글"].map(
                    lambda x: f"{int(x):,}" if pd.notna(x) else "-"
                )

            st.dataframe(top5_df, use_container_width=True)

        # 2-4. 조회수 추이 / 반응 분포
        st.markdown(" ")
        with st.container(border=True):
            st.markdown("#### 📈 조회수 추이 & 반응도 상관")

            # 조회수 추이
            st.markdown("**최근 조회수 추이**")
            time_df = ch_sub.dropna(subset=["_published_dt", "_views"]).copy()
            if len(time_df) > 0:
                time_df = time_df.sort_values("_published_dt")
                chart_views_over_time = (
                    alt.Chart(time_df)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("_published_dt:T", title=""),
                        y=alt.Y("_views:Q", title=""),
                        tooltip=["title", "_published_dt", "_views"]
                    )
                    .properties(height=200)
                )
                st.altair_chart(chart_views_over_time, use_container_width=True)
            else:
                st.caption("업로드일/조회수 정보가 부족해 추이를 표시할 수 없습니다.")

            st.markdown("---")

            # 반응 vs 조회수 산점도
            corr_cols = st.columns(2)

            with corr_cols[0]:
                st.markdown("**조회수 ↔ 좋아요수**")
                like_df = ch_sub.dropna(subset=["_views", "_likes"]).copy()
                if len(like_df) > 0:
                    chart_like = (
                        alt.Chart(like_df)
                        .mark_circle(size=60)
                        .encode(
                            x=alt.X("_views:Q", title="조회수"),
                            y=alt.Y("_likes:Q", title="좋아요수"),
                            tooltip=["title", "_views", "_likes"]
                        )
                        .properties(height=220)
                    )
                    st.altair_chart(chart_like, use_container_width=True)
                else:
                    st.caption("좋아요 데이터가 충분하지 않습니다.")

            with corr_cols[1]:
                st.markdown("**조회수 ↔ 댓글수**")
                cmt_df = ch_sub.dropna(subset=["_views", "_comments"]).copy()
                if len(cmt_df) > 0:
                    chart_cmt = (
                        alt.Chart(cmt_df)
                        .mark_circle(size=60)
                        .encode(
                            x=alt.X("_views:Q", title="조회수"),
                            y=alt.Y("_comments:Q", title="댓글수"),
                            tooltip=["title", "_views", "_comments"]
                        )
                        .properties(height=220)
                    )
                    st.altair_chart(chart_cmt, use_container_width=True)
                else:
                    st.caption("댓글 데이터가 충분하지 않습니다.")

        # 2-5. 업로드 타이밍 분석
        st.markdown(" ")
        with st.container(border=True):
            st.markdown("#### 🕒 업로드 타이밍 성과")

            timing_cols = st.columns(2)

            # 요일별 평균 조회수
            with timing_cols[0]:
                st.markdown("**요일별 평균 조회수**")
                dow_df = ch_sub.dropna(subset=["_dow", "_views"]).copy()
                if len(dow_df) > 0:
                    dow_label = ["월","화","수","목","금","토","일"]

                    def map_dow(d):
                        if pd.isna(d):
                            return "?"
                        try:
                            idx = int(d)
                        except Exception:
                            return "?"
                        if 0 <= idx <= 6:
                            return dow_label[idx]
                        return "?"

                    dow_df["_dow_label"] = dow_df["_dow"].apply(map_dow)
                    dow_mean = (
                        dow_df.groupby("_dow_label", as_index=False)["_views"].mean()
                              .rename(columns={"_views": "avg_views"})
                    )

                    chart_dow = (
                        alt.Chart(dow_mean)
                        .mark_bar()
                        .encode(
                            x=alt.X("_dow_label:N", title="요일"),
                            y=alt.Y("avg_views:Q", title="평균 조회수"),
                            tooltip=["_dow_label", "avg_views"]
                        )
                        .properties(height=200)
                    )
                    st.altair_chart(chart_dow, use_container_width=True)
                else:
                    st.caption("요일 정보가 부족합니다.")

            # 시간대별 평균 조회수
            with timing_cols[1]:
                st.markdown("**업로드 시간대별 평균 조회수**")
                st.caption("시간 정보가 부족합니다.")

        st.markdown("---")

        # =========================
        # 3. 추천 제목 섹션
        # =========================
        st.subheader("3. 추천 제목 (채널 맞춤)")

        # candidates_df → 템플릿별 후보제목, 슬롯별 키워드
        template_info = {}

        if candidates_df is not None and len(candidates_df) > 0:
            for tpl_name, g in candidates_df.groupby("template"):
                cand_titles = (
                    g["title_candidate"]
                    .dropna()
                    .drop_duplicates()
                    .tolist()
                )

                first_row = g.iloc[0]

                def safe_parse(v):
                    try:
                        return ast.literal_eval(v) if isinstance(v, str) else v
                    except Exception:
                        return {}

                slot_mapping = safe_parse(first_row.get("slot_mapping", "{}"))
                slot_scores  = safe_parse(first_row.get("slot_scores", "{}"))

                # 슬롯별 상위 키워드 후보 구성
                slot_top = {}
                for col in g.columns:
                    if col.endswith("_top_keywords"):
                        slot_name = col.replace("_top_keywords", "")
                        kw_list = safe_parse(first_row.get(col, []))
                        sc_list = safe_parse(first_row.get(f"{slot_name}_top_scores", []))

                        if isinstance(kw_list, (list, tuple)):
                            items = []
                            for i_kw, kw in enumerate(kw_list):
                                if kw is None or (isinstance(kw, float) and np.isnan(kw)):
                                    continue
                                score_val = None
                                if isinstance(sc_list, (list, tuple)) and i_kw < len(sc_list):
                                    score_val = sc_list[i_kw]
                                items.append({"keyword": kw, "score": score_val})
                            if items:
                                slot_top[slot_name] = items

                template_info[tpl_name] = {
                    "candidates": cand_titles,
                    "slot_mapping": slot_mapping,
                    "slot_scores": slot_scores,
                    "slot_top": slot_top,
                }

        # 추천 제목들 (예측 조회수 높은 순)
        df_view = df.sort_values(["predicted_views"], ascending=False).reset_index(drop=True)

        for _, row_item in df_view.iterrows():
            rec_title = row_item.get("recommended_title", "")
            tpl_name  = row_item.get("template", "-")

            with st.container(border=True):
                st.markdown(f"##### {rec_title}")
                st.write(f"템플릿: {tpl_name}")

                with st.expander("자세히 보기"):
                    tinfo = template_info.get(tpl_name, {})
                    cand_list = tinfo.get("candidates", []) if isinstance(tinfo, dict) else []
                    slot_top  = tinfo.get("slot_top", {})   if isinstance(tinfo, dict) else {}

                    left_col, right_col = st.columns([0.45, 0.55])

                    # (A) 같은 템플릿 후보 제목들
                    with left_col:
                        st.markdown("**같은 템플릿 후보 제목들**")

                        if len(cand_list) == 0:
                            st.caption("후보 없음")
                        else:
                            safe_tpl_key = re.sub(r"[^a-zA-Z0-9_-]", "_", tpl_name)
                            session_key = f"show_count_{safe_tpl_key}"

                            if session_key not in st.session_state:
                                st.session_state[session_key] = 10

                            show_n = st.session_state[session_key]
                            if show_n > len(cand_list):
                                show_n = len(cand_list)

                            visible_candidates = cand_list[:show_n]

                            for i_c, cand_title in enumerate(visible_candidates, start=1):
                                is_main = (cand_title == rec_title)

                                # 카드 스타일
                                base_style = (
                                    "border:1px solid #D1D5DB;"
                                    "border-radius:6px;"
                                    "padding:10px 12px;"
                                    "margin-bottom:8px;"
                                    "font-size:0.9rem;"
                                    "line-height:1.4;"
                                    "background-color:#FFFFFF;"
                                    "color:#111;"
                                    "white-space:normal;"
                                    "word-break:break-word;"
                                )
                                if is_main:
                                    base_style += "border:1px solid #A78BFA;background-color:#F5F3FF;"

                                label_html = (
                                    '<div style="font-size:0.7rem; color:#6B21A8; font-weight:500; margin-top:4px;">(현재 추천 제목)</div>'
                                    if is_main else ""
                                )

                                st.markdown(
                                    f"""
                                    <div style="{base_style}">
                                        <div style="font-weight:500;">{i_c}. {cand_title}</div>
                                        {label_html}
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                            # 더 보기 버튼
                            if show_n < len(cand_list):
                                if st.button("더 보기", key=f"morebtn_{safe_tpl_key}"):
                                    st.session_state[session_key] = min(show_n + 10, len(cand_list))
                                    st.rerun()

                    # (B) 슬롯별 상위 키워드 후보
                    with right_col:
                        st.markdown("**슬롯별 상위 키워드 후보**")

                        if len(slot_top) == 0:
                            st.caption("슬롯 후보 키워드 정보 없음")
                        else:
                            for slot_name, items in slot_top.items():
                                slot_df = pd.DataFrame(items)
                                slot_df["score"] = pd.to_numeric(slot_df["score"], errors="coerce").fillna(0.0)

                                max_score = slot_df["score"].max()
                                if max_score > 0:
                                    slot_df["rel_score"] = (slot_df["score"] / max_score) * 100.0
                                else:
                                    slot_df["rel_score"] = 0.0

                                slot_df = slot_df.sort_values("rel_score", ascending=False).head(5)

                                # 슬롯 블록 헤더
                                st.markdown(
                                    f"""
                                    <div style="
                                        font-weight:600;
                                        font-size:0.95rem;
                                        margin-top:1rem;
                                        margin-bottom:0.5rem;
                                        color:#222;
                                    ">
                                        {slot_name} 슬롯
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                                # 각 키워드 막대형 게이지
                                for _, rslot in slot_df.iterrows():
                                    kw_text   = str(rslot.get("keyword", ""))
                                    rel_score = float(rslot.get("rel_score", 0.0))
                                    score_label = f"{rel_score:.0f}"

                                    st.markdown(
                                        f"""
                                        <div style="
                                            border:1px solid #E5E7EB;
                                            border-radius:6px;
                                            padding:8px 10px;
                                            margin-bottom:6px;
                                            background-color:#FAFAFA;
                                            font-size:0.85rem;
                                            line-height:1.4;
                                        ">
                                            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                                                <div style="font-weight:500; color:#111;">{kw_text}</div>
                                                <div style="font-size:0.8rem; color:#555;">적합도 {score_label}</div>
                                            </div>
                                            <div style="
                                                width:100%;
                                                background-color:#EEE;
                                                border-radius:4px;
                                                height:6px;
                                                overflow:hidden;
                                            ">
                                                <div style="
                                                    width:{rel_score}%;
                                                    background-color:#4F46E5;
                                                    height:6px;
                                                    border-radius:4px;
                                                "></div>
                                            </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )

        st.markdown("---")

        # =========================
        # 4. 템플릿/키워드 트렌드
        # =========================
        st.subheader("4. 템플릿 / 키워드 트렌드 패턴")

        # -------------------------------------------------
        # (B) 실제 업로드에서 쓰였던 템플릿 분석
        #     title_template.json 기반
        # -------------------------------------------------
        st.markdown("#### 📚 실제 업로드에 많이 쓰인 템플릿")
        st.caption(
            "과거 업로드 영상의 제목에서 반복적으로 등장한 문장 구조(템플릿)를 뽑았습니다. "
            "이 채널이 '자주 쓰는 톤'과 '먹히는 포맷'이 뭔지 볼 수 있어요."
        )

        # title_template.json 불러오기
        template_json_path = os.path.join(DATA_DIR, "title_template.json")
        template_df = None
        if os.path.exists(template_json_path):
            try:
                with open(template_json_path, "r", encoding="utf-8") as f:
                    raw_list = json.load(f)  # [{original_title, template, keywords, ...}, ...]
                template_df = pd.DataFrame(raw_list)
            except Exception as e:
                st.warning(f"title_template.json 로드 실패: {e}")
                template_df = None
        else:
            st.info("title_template.json 파일을 찾을 수 없습니다.")
            template_df = None

        if template_df is not None and len(template_df) > 0:
            # 최소 필요한 컬럼 정리
            work_df = template_df.copy()
            if "original_title" not in work_df.columns:
                work_df["original_title"] = "(제목 없음)"
            if "template" not in work_df.columns:
                work_df["template"] = "(unknown)"
            if "keywords" not in work_df.columns:
                work_df["keywords"] = [[] for _ in range(len(work_df))]

            # 템플릿별 그룹화
            grp = (
                work_df.groupby("template", as_index=False)
                    .agg({
                            "original_title": list,   # 이 템플릿으로 된 영상들 제목 리스트
                            "keywords": list         # 이 템플릿의 키워드들 모음(리스트들의 리스트일 확률 높음)
                        })
            )
            grp["count"] = grp["original_title"].apply(lambda lst: len(lst))

            # 많이 쓴 템플릿 순으로 정렬
            grp = grp.sort_values("count", ascending=False).reset_index(drop=True)

            # ---- 캐러셀 상태 관리 ----
            # 한 "슬라이드"에서 보여줄 카드 수
            CARDS_PER_PAGE = 3

            # 전체 템플릿을 CARDS_PER_PAGE씩 묶으면 몇 페이지?
            total_templates = len(grp)
            total_pages = int(np.ceil(total_templates / CARDS_PER_PAGE))  # 예: 8개 템플릿이면 3페이지(3,3,2)

            # 세션 스테이트 초기화
            if "tpl_carousel_page" not in st.session_state:
                st.session_state["tpl_carousel_page"] = 0  # 0-based index (0페이지 = Top1~3)

            
            # ---- 현재 페이지에 해당하는 3개 템플릿만 뽑기 ----
            start_idx = st.session_state["tpl_carousel_page"] * CARDS_PER_PAGE
            end_idx   = start_idx + CARDS_PER_PAGE
            page_slice = grp.iloc[start_idx:end_idx].reset_index(drop=True)

            # 3등분 레이아웃
            card_cols = st.columns(3)

            for c_i in range(3):
                col = card_cols[c_i]

                if c_i >= len(page_slice):
                    # 이 페이지에 실제 카드 개수가 2개만 있다면, 마지막 칸은 비워둔다
                    continue

                row_tpl = page_slice.iloc[c_i]

                tpl_str = row_tpl["template"]
                tpl_cnt = int(row_tpl["count"])

                # keywords 필드가 리스트들의 리스트 형태일 수 있으므로 평탄화
                kw_all = []
                for kw_list in row_tpl["keywords"]:
                    if isinstance(kw_list, list):
                        for k in kw_list:
                            if isinstance(k, (str, int, float)):
                                kw_all.append(str(k))
                    elif isinstance(kw_list, str):
                        kw_all.append(kw_list)

                # 가장 많이 등장한 키워드 Top5
                if len(kw_all) > 0:
                    kw_freq_s = (
                        pd.Series(kw_all)
                        .value_counts()
                        .head(5)
                    )
                    kw_top5 = kw_freq_s.index.tolist()
                else:
                    kw_top5 = []

                # 키워드 칩 HTML
                kw_html = ""
                for kw in kw_top5:
                    kw_html += f"""
                    <span style="
                        display:inline-block;
                        background-color:#F9FAFB;
                        border:1px solid #E5E7EB;
                        border-radius:999px;
                        padding:4px 10px;
                        font-size:12px;
                        line-height:1.2;
                        color:#374151;
                        margin:2px 4px 2px 0;
                        white-space:nowrap;
                    ">{kw}</span>
                    """

                col.markdown(
                    f"""
                    <div style="
                        border:1px solid #E5E7EB;
                        border-radius:12px;
                        padding:16px 16px 12px;
                        background:#FFFFFF;
                        box-shadow:0 12px 24px rgba(0,0,0,0.03);
                        height:100%;
                        display:flex;
                        flex-direction:column;
                        justify-content:space-between;
                    ">
                    <div style="font-size:0.8rem;font-weight:600;color:#6B21A8;">
                        TOP {start_idx + c_i + 1} 템플릿
                    </div>
                    <div style="
                        font-size:1rem;
                        font-weight:600;
                        color:#111;
                        line-height:1.5;
                        word-break:break-word;
                        margin-top:4px;
                    ">
                        {tpl_str}
                    </div>

                    <div style="
                        font-size:0.9rem;
                        color:#4B5563;
                        margin-top:8px;
                        line-height:1.4;
                    ">
                        총 <b style="color:#111;">{tpl_cnt}개</b> 영상에서 사용
                    </div>

                    <div style="margin-top:10px;font-size:0.8rem;color:#6B7280;">
                        반복적으로 쓰인 키워드
                    </div>
                    <div style="margin-top:4px;display:flex;flex-wrap:wrap;">
                        {kw_html if kw_html else "<span style='font-size:0.8rem;color:#9CA3AF;'>-</span>"}
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                # 상단: 네비게이션 (이전 / 현재페이지 / 다음)
                
            st.markdown("")
            nav_left, nav_mid, nav_right = st.columns([0.13, 0.79, 0.08])

            with nav_left:
                if st.button("◀ 이전", disabled=(st.session_state["tpl_carousel_page"] <= 0), key="tpl_prev_btn"):
                    if st.session_state["tpl_carousel_page"] > 0:
                        st.session_state["tpl_carousel_page"] -= 1
                    st.rerun()

            with nav_mid:
                cur_page = st.session_state["tpl_carousel_page"]
                # 사람 눈에는 1페이지부터 보이도록 +1
                st.markdown(
                    f"<div style='text-align:center; font-size:0.85rem; color:#4B5563;'>"
                    f"Page {cur_page+1} / {total_pages}"
                    f"</div>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    "<div style='text-align:center; font-size:0.7rem; color:#9CA3AF;'>"
                    "</div>",
                    unsafe_allow_html=True
                )

            with nav_right:
                if st.button("다음 ▶", disabled=(st.session_state["tpl_carousel_page"] >= total_pages-1), key="tpl_next_btn"):
                    if st.session_state["tpl_carousel_page"] < total_pages-1:
                        st.session_state["tpl_carousel_page"] += 1
                    st.rerun()

            st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)


            st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

            # -------------------------------------------------
            # 템플릿별 실제 사례 상세 보기
            # -------------------------------------------------
            c1, c3, c2=st.columns([0.9,0.1,1])
            with c1:
                st.markdown("#### 📺 템플릿별 실제 영상 제목 모음")
                st.caption("해당 템플릿으로 실제 업로드된 영상 제목 예시입니다.")

                # 🔹 상위 15개 템플릿만 선택
                top15_grp = grp.head(15)

                for i, row_tpl in top15_grp.iterrows():
                    tpl_str = row_tpl["template"]
                    titles_list = row_tpl["original_title"]
                    use_cnt = int(row_tpl["count"])

                    with st.expander(f"{i+1}. {tpl_str}"):
                        # 너무 많으면 상위 30개만
                        ex_titles = titles_list[:30]

                        for rank, t in enumerate(ex_titles, start=1):
                            st.markdown(
                                f"""
                                <div style="
                                    border:1px solid #E5E7EB;
                                    border-radius:8px;
                                    padding:8px 10px;
                                    margin-bottom:6px;
                                    background-color:#FFFFFF;
                                    box-shadow:0 4px 8px rgba(0,0,0,0.02);
                                    font-size:0.8rem;
                                    line-height:1.4;
                                    color:#111827;
                                ">
                                    <div style="font-weight:500;color:#111;">
                                        {rank}. {t}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        if len(titles_list) > 30:
                            st.caption(f"... 그 외 {len(titles_list)-30}개 더 있음")
        else:
            st.info("템플릿 분석 데이터(title_template.json)가 없어 트렌드 캐러셀을 표시할 수 없습니다.")

        # 키워드 영향력
        with c2:
            if kw_stats is not None and len(kw_stats) > 0:
                st.markdown("#### 🧠 키워드 영향력")
                st.caption("예측 모델에서 강하게 작용한 키워드일수록 더 크고 진하게 표시됩니다.")

                if kw_stats is not None and len(kw_stats) > 0:
                    # 1) 점수 기준 선택 (지금은 score_sum 하나지만 라디오 유지 가능)
                    kw_mode = st.radio(
                        "정렬 기준",
                        ["score_sum"],
                        horizontal=True,
                        key="kw_sort_trend_sidebar_wordcloud"
                    )

                    # 2) 상위 키워드 추출
                    top_kw = kw_stats.sort_values(kw_mode, ascending=False).head(50).copy()
                    # 너무 많으면 복잡해 보이니까 50개 정도까지만 사용

                    # 3) 점수 정규화해서 시각화 스타일 계산
                    #    점수 최댓값/최솟값 기준으로 폰트 크기와 불투명도(alpha) 스케일링
                    vals = pd.to_numeric(top_kw[kw_mode], errors="coerce").fillna(0.0)
                    max_v = vals.max() if len(vals) else 1.0
                    min_v = vals.min() if len(vals) else 0.0
                    span = max(max_v - min_v, 1e-6)

                    # 각 키워드별 스타일 만들기
                    chips_html = ""
                    for _, r in top_kw.iterrows():
                        word = str(r.get("keyword", "")).strip()
                        score_raw = float(r.get(kw_mode, 0.0))

                        # 0.0~1.0 로 스케일
                        norm = (score_raw - min_v) / span
                        # 폰트 크기: 0.0 -> 11px, 1.0 -> 28px 사이
                        font_px = 11 + norm * (28 - 11)
                        # 투명도: 0.0 -> 0.4, 1.0 -> 1.0
                        alpha = 0.4 + norm * (1.0 - 0.4)

                        # 살짝 회전 줄 수도 있지만, 가독성 떨어지면 빼는 게 낫다.
                        # 여기선 회전은 안 넣고 대신 약간의 색 변화를 줘도 된다.
                        # hue는 고정하고 alpha만 조절하는 방식으로 충분히 "구름" 느낌 남.

                        chips_html += f"""
                    <span style="
                    display:inline-block;
                    margin:6px 8px;
                    line-height:1.2;
                    font-weight:600;
                    color:rgba(31,41,55,{alpha});
                    font-size:{font_px:.1f}px;
                    white-space:nowrap;
                    " title="{word} • 점수 {score_raw:.1f}">
                    {word}
                    </span>
                        """

                    # 4) 전체 워드클라우드 블럭으로 렌더
                    st.markdown(
                        f"""
                        <div style="
                        border:1px solid #E5E7EB;
                        border-radius:12px;
                        background:#FFFFFF;
                        box-shadow:0 12px 24px rgba(0,0,0,0.03);
                        padding:16px;
                        min-height:260px;
                        display:flex;
                        flex-wrap:wrap;
                        align-content:flex-start;
                        align-items:flex-start;
                        font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica Neue,Arial,sans-serif;
                    ">
                    {chips_html}
                    </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:
                    st.info("키워드 통계(kw_stats)가 없습니다.")


            else:
                st.info("템플릿 분석 데이터(title_template.json)가 없어 트렌드 분석을 표시할 수 없습니다.")

    else:
        st.warning("채널 데이터를 찾을 수 없습니다. CSV 경로/파일을 확인해 주세요.")
else:
    # 아직 분석 전
    st.info("채널 ID를 입력하고 '추천 제목 생성 🔍' 버튼을 누르면 분석이 시작됩니다.")
