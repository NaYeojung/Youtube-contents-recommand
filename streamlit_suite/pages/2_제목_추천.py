import os, json, numpy as np, pandas as pd, streamlit as st, altair as alt

st.set_page_config(page_title="TubeBoost", layout="wide")
st.title("채널 기반 추천 콘텐츠")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# -------------------------------------------------
# 공용 유틸
# -------------------------------------------------
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


# -------------------------------------------------
# 문자열 형태의 수치(조회수/좋아요/댓글 등)를 float로 변환
#    예: "12,345", "1.2만", "3.4천", "982", "1,234회"
# -------------------------------------------------
def parse_count_series(series):
    if series is None:
        return pd.Series(dtype="float64")

    def _parse_one(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        s = s.replace(",", "").replace("회", "").strip()

        # "1.2만" -> 1.2 * 10000
        if s.endswith("만"):
            try:
                return float(s[:-1]) * 10000
            except:
                return np.nan

        # "3.4천" -> 3.4 * 1000
        if s.endswith("천"):
            try:
                return float(s[:-1]) * 1000
            except:
                return np.nan

        # 기본 숫자
        try:
            return float(s)
        except:
            return np.nan

    return series.apply(_parse_one)


# -------------------------------------------------
# 채널 데이터 로더 + 파생 컬럼 생성
#  - channel(@KoreanCryingGuy)_all_videos.csv 사용
#  - 기대 컬럼:
#      title, published_date, view_count, like_count, comment_count
# -------------------------------------------------
def load_channel_df():
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

    # 업로드 요일/시간
    df_ch["_dow"] = df_ch["_published_dt"].dt.dayofweek  # 월=0 ... 일=6
    df_ch["_hour"] = df_ch["_published_dt"].dt.hour      # 업로드 시각 (없으면 NaN)

    return df_ch


# -------------------------------------------------
# 채널 KPI 요약 계산
# -------------------------------------------------
def build_channel_summary(df_ch: pd.DataFrame):
    """
    반환 summary:
      total_videos : 총 업로드 수
      recent_30_cnt : 최근 30일 업로드 수
      avg_views_all : 전체 평균 조회수
      median_views_all : 전체 중앙값 조회수
      avg_views_recent5 : 최근 5개 영상 평균 조회수
      avg_like_rate : 평균 좋아요율
      top_hits : [{'title': 제목, 'views': 조회수}, ...] 상위 3개
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

    # 안전하게 키 기본값 보장 (UI에서 KeyError 안 나도록)
    for k in ["avg_like_rate", "median_views_all", "avg_views_recent5"]:
        summary.setdefault(k, None)

    return summary


# -------------------------------------------------
# 데이터 로드 (추천제목/템플릿 통계 등)
# -------------------------------------------------
reco      = first_non_none(load_csv("reco_results.csv"),           load_csv("reco_results_demo.csv"))
tpl_stats = first_non_none(load_csv("template_keyword_stats.csv"), load_csv("template_keyword_stats_demo.csv"))
kw_stats  = first_non_none(load_csv("keyword_stats.csv"),          load_csv("keyword_stats_demo.csv"))
feat_imp  = first_non_none(load_csv("feature_importances.csv"),    load_csv("feature_importances_demo.csv"))
p10_style = first_non_none(load_json("p10_style.json"),            load_json("p10_style_demo.json"))
candidates_df = first_non_none(load_csv("title_recommend_candidates.csv"),None)


if reco is None:
    st.error("reco_results.csv가 필요합니다.")
    st.stop()
    
# 세션 상태 초기화
if "analysis_mode" not in st.session_state:
    st.session_state["analysis_mode"] = False

if "idx_range" not in st.session_state:
    st.session_state["idx_range"] = None



# -------------------------------------------------
# 사이드바 필터 (조회수 슬라이더 제거 버전)
# -------------------------------------------------
with st.sidebar:
    st.header("필터")

    tpl_choices = ["(전체)"] + sorted(reco["template"].dropna().unique().tolist())
    tpl_sel = st.selectbox("템플릿", tpl_choices)

df = reco.copy()
if tpl_sel != "(전체)":
    df = df[df["template"] == tpl_sel]

# 숫자화
df["predicted_views"] = pd.to_numeric(df["predicted_views"], errors="coerce")
df = df.dropna(subset=["predicted_views"])


# -------------------------------------------------
# 입력 폼
# -------------------------------------------------
c1, c2 = st.columns([2, 1])
with c1:
    with st.form("predict_form"):
        title_input = st.text_input("채널 ID", value="@KoreanCryingGuy")
        run = st.form_submit_button("추천 제목 생성")

    if run:
        # 버튼 눌렀으면 분석 모드 진입
        st.session_state["analysis_mode"] = True


# -------------------------------------------------
# 실행 버튼 눌렀을 때 화면
# -------------------------------------------------
if st.session_state["analysis_mode"]:
    ch_df = load_channel_df()

    if ch_df is not None and len(ch_df) > 0:

        # 1) 업로드 타임라인 정렬 & 인덱스 부여
        timeline_df = ch_df.dropna(subset=["_published_dt"]).copy()
        timeline_df = timeline_df.sort_values("_published_dt").reset_index(drop=True)
        timeline_df["_t_idx"] = timeline_df.index

        if len(timeline_df) > 0:
            min_idx = int(timeline_df["_t_idx"].min())
            max_idx = int(timeline_df["_t_idx"].max())
        else:
            min_idx = 0
            max_idx = 0
            
        with c2:    
            if len(timeline_df) > 0:
                default_start = min_idx
                default_end   = max_idx
                
                idx_range = st.slider(
                    "분석할 업로드 구간",
                    min_value=min_idx,
                    max_value=max_idx,
                    value=(default_start, default_end),
                    step=1
                )
                start_idx, end_idx = idx_range

                start_dt = timeline_df.loc[timeline_df["_t_idx"] == start_idx, "_published_dt"].min()
                end_dt   = timeline_df.loc[timeline_df["_t_idx"] == end_idx,   "_published_dt"].max()
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
                        f"{start_dt.date()} ~ {end_dt.date()}"
                    )
            else:
                st.info("날짜 정보가 없어 전체 영상을 분석합니다.")
                ch_sub = ch_df.copy()
                st.caption(f"(전체 {len(ch_sub)}개 영상 분석)")

        # 이후 전체 KPI/차트/테이블은 ch_sub 기준
        summary = build_channel_summary(ch_sub)

        # ... (KPI 카드, TOP5, 차트들 등 기존 코드 계속)


        st.subheader("📊 채널 분석")
        st.markdown("")

        # --- KPI 카드들 ---
        kpi_cols = st.columns(3)

        # KPI 카드 1: 업로드 현황
        with kpi_cols[0]:
            with st.container(border=False):
                st.markdown("**업로드 현황**")
                st.markdown(f"### 총 {summary['total_videos']}개")
                recent30 = summary.get("recent_30_cnt")
                if recent30 is not None:
                    st.caption(f"최근 30일 업로드: {recent30}개")
                else:
                    st.caption("최근 30일 업로드: -")

        # KPI 카드 2: 조회수 퍼포먼스
        def fmt_views(v):
            if v is None:
                return "-"
            return f"{int(v):,}회"

        with kpi_cols[1]:
            with st.container(border=False):
                st.markdown("**조회수 평균**")
                st.markdown(f"### {fmt_views(summary.get('avg_views_all'))}")
                st.caption(
                    # f"중앙값 {fmt_views(summary.get('median_views_all'))}"
                    f"최근 5개 평균 {fmt_views(summary.get('avg_views_recent5'))}"
                )

        # KPI 카드 3: 시청자 반응도
        with kpi_cols[2]:
            with st.container(border=False):
                st.markdown("**시청자 반응도**")
                avg_like_rate = summary.get("avg_like_rate")
                if avg_like_rate is not None and not np.isnan(avg_like_rate):
                    like_pct = avg_like_rate * 100.0
                    st.markdown(f"### {like_pct:.2f}%")
                else:
                    st.markdown("### 평균 좋아요율 -")
                st.caption("좋아요율 = 좋아요수 / 조회수")

        # --- 상위 퍼포먼스 영상 TOP 5 테이블 ---
        st.markdown(" ")
        st.markdown("#### 상위 퍼포먼스 영상 TOP 5")
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
        top5_df["조회수"] = top5_df["조회수"].map(lambda x: f"{int(x):,}" if pd.notna(x) else "-")
        if "좋아요" in top5_df.columns:
            top5_df["좋아요"] = top5_df["좋아요"].map(lambda x: f"{int(x):,}" if pd.notna(x) else "-")
        if "댓글" in top5_df.columns:
            top5_df["댓글"] = top5_df["댓글"].map(lambda x: f"{int(x):,}" if pd.notna(x) else "-")

        st.dataframe(top5_df, use_container_width=True)

        # --- 조회수 추이 라인 차트 ---
        st.markdown(" ")
        st.markdown("#### 최근 조회수 추이")

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
            st.caption("업로드일/조회수 정보가 부족해 조회수 추이를 표시할 수 없습니다.")

        # --- 반응 vs 조회수 산점도 ---
        st.markdown("#### 반응 vs 조회수")
        corr_cols = st.columns(2)

        # 조회수 vs 좋아요수
        with corr_cols[0]:
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

        # 조회수 ↔ 댓글수
        with corr_cols[1]:
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

        # --- 업로드 타이밍 성과 ---
        st.markdown(" ")
        st.markdown("#### 업로드 타이밍 성과")
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

        st.divider()

        # ==========================
        # 2) 추천 제목 카드 섹션
        # ==========================
        st.subheader("📝 추천 제목")
        
        import ast
        import numpy as np

        template_info = {}

        if candidates_df is not None and len(candidates_df) > 0:
            for tpl_name, g in candidates_df.groupby("template"):

                # 후보 제목 목록
                cand_titles = (
                    g["title_candidate"]
                    .dropna()
                    .drop_duplicates()
                    .tolist()
                )

                # 첫 행 기준 슬롯 정보
                first_row = g.iloc[0]

                # slot_mapping / slot_scores 파싱
                def safe_parse(v):
                    try:
                        return ast.literal_eval(v) if isinstance(v, str) else v
                    except Exception:
                        return {}
                slot_mapping = safe_parse(first_row.get("slot_mapping", "{}"))
                slot_scores  = safe_parse(first_row.get("slot_scores", "{}"))

                # 슬롯별 상위 키워드/점수 구조화
                slot_top = {}
                for col in g.columns:
                    if col.endswith("_top_keywords"):
                        slot_name = col.replace("_top_keywords", "")
                        kw_list = safe_parse(first_row.get(col, []))
                        sc_list = safe_parse(first_row.get(f"{slot_name}_top_scores", []))
                        if isinstance(kw_list, (list, tuple)):
                            items = []
                            for i, kw in enumerate(kw_list):
                                if kw is None or (isinstance(kw, float) and np.isnan(kw)):
                                    continue
                                score_val = None
                                if isinstance(sc_list, (list, tuple)) and i < len(sc_list):
                                    score_val = sc_list[i]
                                items.append({"keyword": kw, "score": score_val})
                            if items:
                                slot_top[slot_name] = items

                template_info[tpl_name] = {
                    "candidates": cand_titles,
                    "slot_mapping": slot_mapping,
                    "slot_scores": slot_scores,
                    "slot_top": slot_top,
                }
        else:
            template_info = {}


        # 예측 조회수 높은 순으로 정렬
        df_view = df.sort_values(["predicted_views"], ascending=False).reset_index(drop=True)

        for _, row in df_view.iterrows():
            with st.container(border=True):

                rec_title = row.get("recommended_title", "")
                tpl_name  = row.get("template", "-")

                st.markdown(f"##### {rec_title}")
                st.write(f"**템플릿:** {tpl_name}")

                with st.expander("자세히 알아보기"):

                    # tinfo: 이 템플릿에 대한 전체 정보
                    tinfo = template_info.get(tpl_name, {})
                    cand_list = tinfo.get("candidates", []) if isinstance(tinfo, dict) else []
                    slot_top = tinfo.get("slot_top", {}) if isinstance(tinfo, dict) else {}

                    # 레이아웃 2컬럼
                    left_col, right_col = st.columns([0.45, 0.55])

                    # -------------------------------------------------
                    # (A) 같은 템플릿에서 생성된 후보 제목들 (왼쪽)
                    # -------------------------------------------------
                    import re

                    with left_col:
                        st.markdown("**같은 템플릿 후보 제목들**")

                        if len(cand_list) == 0:
                            st.caption("후보 없음")

                        else:
                            # 안전한 세션 키 생성
                            safe_tpl_key = re.sub(r"[^a-zA-Z0-9_-]", "_", tpl_name)
                            session_key = f"show_count_{safe_tpl_key}"

                            # 세션 상태 초기화
                            if session_key not in st.session_state:
                                st.session_state[session_key] = 10

                            show_n = st.session_state[session_key]

                            # 만약 세션에 잘못된 값(>전체 길이)이 있다면 보정
                            if show_n > len(cand_list):
                                show_n = len(cand_list)

                            # 실제로 보여줄 후보 리스트
                            visible_candidates = cand_list[:show_n]

                            # 카드 렌더
                            for i_c, cand_title in enumerate(visible_candidates, start=1):
                                is_main = (cand_title == rec_title)

                                card_style = (
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
                                    card_style += "border:1px solid #A78BFA;background-color:#F5F3FF;"

                                label_html = (
                                    '<div style="font-size:0.7rem; color:#6B21A8; font-weight:500; margin-top:4px;">(현재 추천 제목)</div>'
                                    if is_main else ""
                                )

                                card_html = f"""
                                <div style="{card_style}">
                                <div style="font-weight:600;">{i_c}. {cand_title}</div>
                                {label_html}
                                </div>
                                """
                                st.markdown(card_html, unsafe_allow_html=True)

                            # 더 보기 버튼 (정확히 작동)
                            if show_n < len(cand_list):
                                if st.button("더 보기", key=f"morebtn_{safe_tpl_key}"):
                                    st.session_state[session_key] = min(show_n + 10, len(cand_list))
                                    st.rerun()

                                        
                    # -------------------------------------------------
                    # (B) 슬롯별 상위 키워드 후보들 (오른쪽)
                    # -------------------------------------------------
                    with right_col:
                        st.markdown("**슬롯별 상위 키워드 후보**")

                        if len(slot_top) == 0:
                            st.caption("슬롯 후보 키워드 정보 없음")
                        else:
                            # slot_top: { "SITUATION": [ {keyword, score}, ... ], "TOPIC": [...] }
                            import numpy as _np
                            import pandas as _pd

                            for slot_name, items in slot_top.items():
                                # DataFrame으로 변환하고 정렬/정규화
                                slot_df = _pd.DataFrame(items)

                                # score 숫자화
                                slot_df["score"] = _pd.to_numeric(slot_df["score"], errors="coerce").fillna(0.0)

                                # 상대 스케일: 이 슬롯 안에서 최고 점수를 100으로
                                max_score = slot_df["score"].max()
                                if max_score > 0:
                                    slot_df["rel_score"] = (slot_df["score"] / max_score) * 100.0
                                else:
                                    slot_df["rel_score"] = 0.0

                                # 보기 좋게 상위 5개만
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

                                # 키워드별 게이지 바 렌더
                                # rel_score를 막대 너비로 시각화 (0~100%)
                                # 각각을 작은 카드처럼 반복 출력
                                for _, rslot in slot_df.iterrows():
                                    kw_text   = str(rslot.get("keyword", ""))
                                    rel_score = float(rslot.get("rel_score", 0.0))

                                    # 점수 라벨은 퍼센트 느낌
                                    # 예: 87.5 → "87"
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

                            # 슬롯 섹션 전체 끝나고 약간 여백
                            st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    # ==========================
    # 3) GLOBAL INSIGHTS
    # ==========================
    st.divider()
    st.subheader("📈 템플릿/키워드 트렌드 패턴")

    # 템플릿 활용 비중
    if tpl_stats is not None:
        st.markdown("**템플릿 활용 비중 TOP**")
        top_tpl = tpl_stats.sort_values("share", ascending=False).head(8)
        chart_tpl = (
            alt.Chart(top_tpl)
            .mark_bar()
            .encode(
                x=alt.X("share:Q", axis=alt.Axis(format="%"), title="활용 비중"),
                y=alt.Y("template:N", sort="-x", title="템플릿"),
                tooltip=["template", "share", "avg_pred_views", "count"]
            )
        )
        st.altair_chart(chart_tpl, use_container_width=True)

    # 키워드 영향력
    if kw_stats is not None:
        st.markdown("**키워드 영향력**")
        kw_mode = st.radio("정렬 기준", ["score_sum"], horizontal=True, key="kw_sort")
        top_kw = kw_stats.sort_values(kw_mode, ascending=False).head(15)
        chart_kw = (
            alt.Chart(top_kw)
            .mark_bar()
            .encode(
                x=alt.X(f"{kw_mode}:Q", title=kw_mode),
                y=alt.Y("keyword:N", sort="-x", title="키워드"),
                tooltip=["keyword", "score_avg", "score_sum", "freq"]
            )
        )
        st.altair_chart(chart_kw, use_container_width=True)
