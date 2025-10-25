import os, json, numpy as np, pandas as pd, streamlit as st, altair as alt

st.set_page_config(page_title="추천제목 대시보드", layout="wide")
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
    ch_path = os.path.join(DATA_DIR, "channel(@KoreanCryingGuy)_all_videos.csv")
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
        top5_df["조회수"] = top5_df["조회수"].map(lambda x: f"{int(x):,}" if pd.notna(x) else "-")
        if "좋아요" in top5_df.columns:
            top5_df["좋아요"] = top5_df["좋아요"].map(lambda x: f"{int(x):,}" if pd.notna(x) else "-")
        if "댓글" in top5_df.columns:
            top5_df["댓글"] = top5_df["댓글"].map(lambda x: f"{int(x):,}" if pd.notna(x) else "-")

        st.dataframe(top5_df, use_container_width=True)

        # --- 조회수 추이 라인 차트 ---
        st.markdown(" ")
        st.markdown("#### 📈 최근 조회수 추이")

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
        st.markdown("#### 🔁 반응 vs 조회수")
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

        st.divider()

    # ==========================
    # 2) 추천 제목 카드 섹션
    # ==========================
    st.subheader("📝 추천 제목")

    # 추천 제목 근거 문구 생성
    def build_rationale(row, style):
        parts = []

        # 템플릿/키워드
        if isinstance(row.get("keyword_topk"), str):
            parts.append(
                f"템플릿 {row.get('template','?')}에서 상위 키워드 `{row['keyword_topk']}`를 반영했습니다."
            )
        else:
            parts.append(
                f"템플릿 {row.get('template','?')}의 상위 키워드를 반영했습니다."
            )

        feats = []
        # 제목 길이 vs 상위10% 평균
        if "title_len" in row and "p10_len" in row:
            try:
                if abs(row["title_len"] - row["p10_len"]) <= 3:
                    feats.append("제목 길이가 상위 10% 평균과 유사")
            except Exception:
                pass

        # 이모지 사용
        if "emoji_count" in row and p10_style:
            t = p10_style.get("emoji_target", None)
            if t is not None:
                try:
                    if abs(row["emoji_count"] - t) <= 1:
                        feats.append("이모지 사용이 상위 10% 범위")
                except Exception:
                    pass

        # 느낌표 사용
        if "exclaim_count" in row and p10_style:
            t = p10_style.get("exclaim_target", None)
            if t is not None:
                try:
                    if abs(row["exclaim_count"] - t) <= 1:
                        feats.append("느낌표 사용이 상위 10% 범위")
                except Exception:
                    pass

        # 신선도 / 썸네일 점수
        if "novelty_score" in row and row["novelty_score"] >= 0.4:
            feats.append("제목 신선도(표현 다양성)가 높음")
        if "thumbnail_score" in row and row["thumbnail_score"] >= 0.65:
            feats.append("썸네일 가독성/대비 점수가 높음")

        if feats:
            parts.append(" / ".join(feats))

        return " ".join(parts)

    def safe_rationale(row, style):
        try:
            return str(build_rationale(row, style))
        except Exception:
            return ""

    # 추천 제목들 정렬 (예측 조회수 높은 순)
    df_view = df.sort_values(["predicted_views"], ascending=False).reset_index(drop=True)
    df_view["rationale"] = df_view.apply(lambda r: safe_rationale(r, p10_style), axis=1)

    # 카드 렌더
    for _, row in df_view.iterrows():
        with st.container(border=True):
            st.markdown(f"##### {row['recommended_title']}")
            cols_block = st.columns(1)
            cols_block[0].write(f"**템플릿:** {row.get('template','-')}")
            st.caption(row["rationale"])

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
