import os, json, numpy as np, pandas as pd, streamlit as st, altair as alt

st.set_page_config(page_title="추천제목 대시보드", layout="wide")
st.title("채널 기반 추천 콘텐츠")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

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

reco      = first_non_none(load_csv("reco_results.csv"),           load_csv("reco_results_demo.csv"))
tpl_stats = first_non_none(load_csv("template_keyword_stats.csv"), load_csv("template_keyword_stats_demo.csv"))
kw_stats  = first_non_none(load_csv("keyword_stats.csv"),          load_csv("keyword_stats_demo.csv"))
feat_imp  = first_non_none(load_csv("feature_importances.csv"),    load_csv("feature_importances_demo.csv"))
p10_style = first_non_none(load_json("p10_style.json"),            load_json("p10_style_demo.json"))

if reco is None:
    st.error("reco_results.csv가 필요합니다.")
    st.stop()

with st.sidebar:
    st.header("필터")
    tpl_choices = ["(전체)"] + sorted(reco["template"].dropna().unique().tolist())
    tpl_sel = st.selectbox("템플릿", tpl_choices)

    pv = pd.to_numeric(reco["predicted_views"], errors="coerce").dropna()
    if pv.empty:
        min_views, max_views = 0, 1
    else:
        min_views = int(np.floor(pv.min()))
        max_views = int(np.ceil(pv.max()))
    if min_views == max_views:
        min_views = max(0, min_views - 1); max_views = max_views + 1
    step_auto = max(1, int(round((max_views - min_views) / 20)))
    view_range = st.slider("예측 조회수 범위",
                           min_value=min_views, max_value=max_views,
                           value=(min_views, max_views), step=step_auto)

df = reco.copy()
if tpl_sel != "(전체)":
    df = df[df["template"] == tpl_sel]
df["predicted_views"] = pd.to_numeric(df["predicted_views"], errors="coerce")
df = df.dropna(subset=["predicted_views"])
df = df[(df["predicted_views"] >= view_range[0]) & (df["predicted_views"] <= view_range[1])]

# (이하: KPI·차트·Per-item 카드·테이블 렌더링 코드는 기존 app.py 그대로)

# =============== 입력 폼 ===============
with st.form("predict_form"):
    title = st.text_input("채널 ID", value="@KoreanCryingGuy")
    run = st.form_submit_button("추천 제목 생성")

# =============== 실행 ===============
if run:
        
    # =====================
    # RECOMMENDATIONS TABLE + PER-TITLE RATIONALE
    # =====================
    st.subheader("📝 추천 제목")

    # Helper: 한 줄 설명 자동 생성
    def build_rationale(row, style):
        parts = []
        # 템플릿/키워드
        if isinstance(row.get("keyword_topk"), str):
            parts.append(f"템플릿 {row.get('template','?')}에서 상위 키워드 `{row['keyword_topk']}`를 반영했습니다.")
        else:
            parts.append(f"템플릿 {row.get('template','?')}의 상위 키워드를 반영했습니다.")
        # 모델 피쳐 근거
        feats = []
        if "title_len" in row and "p10_len" in row:
            if abs(row["title_len"] - row["p10_len"]) <= 3:
                feats.append("제목 길이가 상위 10% 평균과 유사")
        if "emoji_count" in row and style:
            t = style.get("emoji_target", None)
            if t is not None and abs(row["emoji_count"] - t) <= 1:
                feats.append("이모지 사용이 상위 10% 범위")
        if "exclaim_count" in row and style:
            t = style.get("exclaim_target", None)
            if t is not None and abs(row["exclaim_count"] - t) <= 1:
                feats.append("느낌표 사용이 상위 10% 범위")
        if "novelty_score" in row and row["novelty_score"] >= 0.4:
            feats.append("제목 신선도(표현 다양성)가 높음")
        if "thumbnail_score" in row and row["thumbnail_score"] >= 0.65:
            feats.append("썸네일 가독성/대비 점수가 높음")

        if feats:
            parts.append(" / ".join(feats))

        # # 모델 예측
        # if "predicted_views" in row:
        #     parts.append(f"예측 조회수 **{int(row['predicted_views']):,}**")

        # # 스타일 매칭
        # if "p10_style_match" in row:
        #     parts.append(f"상위 10% 스타일 일치도 **{row['p10_style_match']:.2f}**")

        return " ".join(parts)

    # Show table
    show_cols = ["recommended_title","template","predicted_views","p10_style_match","title_len","emoji_count","exclaim_count","keyword_topk"]
    show_cols = [c for c in show_cols if c in df.columns]

    # Ranking
    df_view = df.sort_values(["predicted_views"], ascending=False).reset_index(drop=True)
    def safe_rationale(row, style):
        try:
            out = build_rationale(row, style)
            return str(out)
        except Exception:
            return ""

    df_view["rationale"] = df_view.apply(lambda r: safe_rationale(r, p10_style), axis=1)

    # Display
    for i, row in df_view.iterrows():
        with st.container(border=True):
            st.markdown(f"##### {row['recommended_title']}")
            cols = st.columns(1)
            cols[0].write(f"**템플릿:** {row.get('template','-')}")
            # cols[1].write(f"**예측 조회수:** {int(row.get('predicted_views',0)):,}")
            # if 'p10_style_match' in row:
            #     cols[2].write(f"**스타일 일치도:** {row.get('p10_style_match'):.2f}")
            st.caption(row["rationale"])

    # =====================
    # GLOBAL INSIGHTS
    # =====================
    st.divider()

    # 1) 템플릿 활용도 & 성과
    if tpl_stats is not None:
        st.markdown("**템플릿 활용 비중 TOP**")
        top_tpl = tpl_stats.sort_values("share", ascending=False).head(8)
        chart = alt.Chart(top_tpl).mark_bar().encode(
            x=alt.X("share:Q", axis=alt.Axis(format="%"), title="활용 비중"),
            y=alt.Y("template:N", sort="-x", title="템플릿"),
            tooltip=["template","share","avg_pred_views","count"]
        )
        st.altair_chart(chart, use_container_width=True)

    # 2) 키워드 점수
    if kw_stats is not None:
        st.markdown("**키워드 영향력**")
        kw_mode = st.radio("정렬 기준", ["score_sum"], horizontal=True, key="kw_sort")
        top_kw = kw_stats.sort_values(kw_mode, ascending=False).head(15)
        chart3 = alt.Chart(top_kw).mark_bar().encode(
            x=alt.X(f"{kw_mode}:Q", title=kw_mode),
            y=alt.Y("keyword:N", sort="-x", title="키워드"),
            tooltip=["keyword","score_avg","score_sum","freq"]
        )
        st.altair_chart(chart3, use_container_width=True)



