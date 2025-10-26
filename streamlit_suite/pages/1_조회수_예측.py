# -*- coding: utf-8 -*-
import os, sys, tempfile, json
import numpy as np
import pandas as pd
import streamlit as st
import io
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = 'Malgun Gothic'   # 한글 폰트
plt.rcParams['axes.unicode_minus'] = False      # 마이너스 기호 깨짐 방지

# =============== 경로/모듈 로드 ===============
WEB_DIR = r"C:\web"  # 필요 시 수정
if WEB_DIR not in sys.path:
    sys.path.append(WEB_DIR)

try:
    from feature import process_user_input
except Exception as e:
    st.set_page_config(page_title="TubeBoost", layout="wide")
    st.error(f"feature.py import 실패: {e}")
    st.stop()

st.set_page_config(page_title="조회수 예측", layout="wide")
st.title("📈 조회수 예측")

# =============== 유틸 ===============
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
    """영문 feature명을 한국어로 변환"""
    return FEATURE_LABELS.get(k, k)

def _safe_mean(series_like):
    try:
        return float(pd.to_numeric(series_like, errors="coerce").mean())
    except Exception:
        return np.nan

def _predict_views_by_regressor(model, df, cols):
    pred = model.predict(df[cols].values)
    # 로그 스케일 학습이었다면 expm1, 아니라면 생략
    try:
        return int(np.expm1(pred[0]))
    except Exception:
        try:
            return int(float(pred[0]))
        except Exception:
            return None

def load_csv_compat(path: str) -> pd.DataFrame:
    """pandas 버전/인코딩 이슈를 피해서 안전하게 CSV를 읽는다."""
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
    """object_details(dict/list/str)에서 라벨 분포 추출"""
    try:
        v = obj_details
        if isinstance(v, str):
            v = json.loads(v)
    except Exception:
        try:
            import ast
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
    """text_details(list/str)에서 확률 높은 텍스트 샘플 추출"""
    try:
        v = text_details
        if isinstance(v, str):
            v = json.loads(v)
    except Exception:
        try:
            import ast
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
    return f"""<span style="display:inline-block;padding:4px 10px;margin:4px;border-radius:999px;background:#F1F5F9;border:1px solid #E2E8F0;font-size:12px">{text}</span>"""

def color_chip(name: str):
    # 간단 매핑 (Streamlit에서 CSS 이름 대부분 렌더 가능)
    bg = name
    return f"""<span style="display:inline-block;padding:6px 12px;margin:4px;border-radius:10px;border:1px solid #e5e7eb;background:{bg};color:#111827;font-weight:600">{name}</span>"""

def metric_row(df_row, cols, title):
    vals = [(k, df_row.get(k, 0)) for k in cols]
    table = pd.DataFrame({"지표": [k for k,_ in vals],
                          "값":  [vals[i][1] for i in range(len(vals))]})
    st.subheader(title)
    st.dataframe(table, use_container_width=True)
    
def get_peer_subset(md: pd.DataFrame, my_subs: int, pct: float, min_rows: int = 100) -> pd.DataFrame:
    """
    md에서 내 구독자수와 ±pct% 이내인 행만 추출.
    표본 수가 min_rows 미만이면 범위를 점진적으로 확대(최대 ±50%).
    """
    if "subscriber_count" not in md.columns:
        return pd.DataFrame()
    subs = pd.to_numeric(md["subscriber_count"], errors="coerce")
    md2 = md.copy()
    md2["subscriber_count"] = subs

    # 초기 범위
    p = max(0.01, pct / 100.0)
    my = float(my_subs)
    lo, hi = my * (1 - p), my * (1 + p)

    # 점진 확장
    for _ in range(10):  # 최대 10번(대략 ±50%까지)
        peer = md2[(md2["subscriber_count"] >= lo) & (md2["subscriber_count"] <= hi)]
        if len(peer) >= min_rows or p >= 0.5:
            return peer
        p *= 1.25
        lo, hi = my * (1 - p), my * (1 + p)
    return peer
def plot_sensitivity_barh(delta_df: pd.DataFrame, topn: int = 5, title: str = ""):
    """
    delta_df: columns => ['지표','증감(예측)']
    topn개 상위 지표를 가로 막대 그래프로 그린다.
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
    label: 지표 이름(str)
    cur_val: 현재 값
    tgt_val: 추천/평균 값
    gain: 예상 조회수 증가량 (정수 or None)
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
        # 거의 동일
        base_phrase = "가 평균과 유사한 수준입니다."
    else:
        if diff > 0:
            # 현재값이 더 큼
            if "길이" in label:
                base_phrase = "가 긴 편입니다."
            elif "수" in label:
                base_phrase = "가 많은 편입니다."
            elif "밝기" in label or "대비" in label:
                base_phrase = "가 높은 편입니다."
            else:
                base_phrase = "가 높은 편입니다."
        else:
            # 현재값이 더 작음
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



# =============== 사이드바 ===============
with st.sidebar:
    st.subheader("구독자수 범위 설정")
    peer_pct = st.slider("구독자 범위(±%)", min_value=1, max_value=50, value=15, step=1,
                     help="내 구독자 수의 ±범위로 유사 구독자 채널을 선택합니다.")
    peer_min_rows = st.number_input("최소 표본 수", min_value=20, value=100, step=10,
                                    help="범위 내 표본이 부족하면 자동으로 범위를 넓힙니다.")
    st.divider()
    st.subheader("모델/데이터 경로")
    saved_models_dir = st.text_input("saved_models 경로", value=os.path.join(WEB_DIR, "saved_models"))
    model_datas_path = st.text_input("model_datas.csv 경로", value=os.path.join(WEB_DIR, "model_datas.csv"))

# =============== 입력 폼 ===============
with st.form("predict_form"):
    c1, c2 = st.columns([3, 2])
    with c1:
        title = st.text_input("제목", value="인생여행지였던 뉴욕에 다시 혼자 입장! 근데 왜 이렇게 변함...? (찐 뉴요커가 추천해준 빵집, 맛집, 카페 다 뿌시긴함ㅣ스킴스 엉뽕 대난리)")
        thumbnail_url = st.text_input("썸네일 URL", value="https://i.ytimg.com/vi/D-jOG2ybV1s/hq720.jpg")
        thumbnail_file = st.file_uploader("또는 이미지 업로드", type=["png","jpg","jpeg","webp"])
    with c2:
        duration = st.text_input("영상 길이 (MM:SS)", value="36:40")
        subscriber = st.number_input("구독자 수", min_value=0, value=230000, step=1000)
        total_videos = st.number_input("누적 업로드 수", min_value=0, value=600, step=1)
        category = st.text_input("카테고리", value="General")
    run = st.form_submit_button("예측 실행")

# =============== 실행 ===============
if run:
    # 1) 썸네일 인풋 정리
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
        # 2) feature.py로 피처 생성 + 클러스터 예측
        df, cluster = process_user_input(
            title, thumb_input, duration, subscriber, total_videos, category
        )
        row = df.iloc[0].to_dict()

        

        # 3) (선택) 클러스터별 회귀모델로 조회수 예측
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

        # 4) 클러스터 통계 (평균/상위10%)
        md = None
        if os.path.exists(model_datas_path):
            try:
                md = load_csv_compat(model_datas_path)
                # cluster 컬럼이 문자열일 수도 있으므로 숫자화
                md["cluster"] = pd.to_numeric(md.get("cluster"), errors="coerce")
                filt = md[md["cluster"] == cluster]
                if len(filt):
                    cluster_mean = int(_safe_mean(filt.get("view_count")))
                    top10 = filt.nlargest(max(1, int(len(filt)*0.1)), "view_count")
                    cluster_top10_mean = int(_safe_mean(top10.get("view_count")))
            except Exception as e:
                st.warning(f"model_datas.csv 읽기 실패: {e}")
                
                
                # === 피어 그룹(유사 구독자) 평균 ===
        peer_mean_views = None
        peer_top10_mean_views = None
        peer_avg_core = None

        if md is not None and len(md):
            peer_df = get_peer_subset(md, int(row.get("subscriber_count", 0)), peer_pct, peer_min_rows)
            if len(peer_df):
                # 조회수 평균들
                if "view_count" in peer_df.columns:
                    peer_mean_views = int(_safe_mean(peer_df["view_count"]))
                    top10_peer = peer_df.nlargest(max(1, int(len(peer_df) * 0.1)), "view_count")
                    peer_top10_mean_views = int(_safe_mean(top10_peer["view_count"]))

                # 코어 피처 평균
                core_cols = [
                    "duration","title_length","word_count","emoji_count","special_char_count",
                    "has_question_mark","has_exclamation","brightness","contrast",
                    "person_count","object_count","has_text"
                ]
                peer_avg_core = {}
                for c in core_cols:
                    if c in peer_df.columns:
                        peer_avg_core[c] = round(_safe_mean(peer_df[c]), 3)
                        
                dist_cols = [
                    "text_left","text_middle","text_right",
                    "text_small","text_medium","text_large",
                    "person_left","person_middle","person_right",
                    "person_small","person_medium","person_large",
                ]

                peer_avg_dict = {}
                for c in dist_cols:
                    if c in peer_df.columns:
                        peer_avg_dict[c] = float(_safe_mean(peer_df[c]))
                    else:
                        peer_avg_dict[c] = 0.0  # 없는 컬럼은 0으로 채워도 안전

        # === 시뮬레이션 미리 계산 ===
        current_pred_views, improved_pred_views, lift_abs, lift_pct = None, None, None, None
        top_change_rows, drivers_text = pd.DataFrame(), ""

        if reg is not None and peer_avg_core:
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
                if k in ['emoji_count','special_char_count','person_count','has_text','word_count','object_count','title_length']:
                    peer_val = int(round(peer_val))
                pred_peer = simulate({k: peer_val})
                diff_val = None
                if predicted_views and pred_peer:
                    diff_val = int(pred_peer) - int(predicted_views)
                sims.append({"지표":k,"현재값":cur_val,"평균값":peer_val,"평균으로 맞출 때 예측":pred_peer,"증감(예측)":diff_val})
            delta_views_df = pd.DataFrame(sims)

            improving_changes = {r["지표"]:r["평균값"] for _,r in delta_views_df.iterrows() if r["증감(예측)"] and r["증감(예측)"]>0}
            pred_combo = simulate(improving_changes) if improving_changes else None
            current_pred_views = int(predicted_views) if predicted_views else None
            improved_pred_views = int(pred_combo) if pred_combo else None
            if current_pred_views and improved_pred_views:
                lift_abs = improved_pred_views - current_pred_views
                lift_pct = (lift_abs / current_pred_views) * 100 if current_pred_views>0 else None
            top_change_rows = delta_views_df.dropna(subset=["증감(예측)"]).sort_values("증감(예측)",ascending=False).head(3)
            drivers_text = ", ".join(get_label(v) for v in top_change_rows["지표"].tolist()) if not top_change_rows.empty else ""


        st.success(f"분석이 완료됐습니다. 클러스터 {cluster}가 적용됩니다.")
        st.markdown(" ")
        
        # 상단 요약 + 썸네일 프리뷰
        cA, cB = st.columns([3, 2])
        with cA:
            
            st.markdown("### 📌 인사이트 요약")
            st.markdown(" ")

            summary_lines = []
            if current_pred_views:
                summary_lines.append(f"현재 구성으로 예상되는 조회수는 약 {current_pred_views:,}회입니다.")
            if improved_pred_views and lift_abs and lift_pct:
                summary_lines.append(f"핵심 지표를 평균 수준으로 조정하면 약 {improved_pred_views:,}회까지 기대할 수 있습니다.")
            if drivers_text:
                summary_lines.append(f"특히 {drivers_text}가 조회수에 큰 영향을 주는 것으로 보입니다.")
            st.write("\n".join(f"\n{s}" for s in summary_lines))

        with cB:
            try:
                st.image(thumb_input, caption="입력 썸네일", use_container_width=True)
            except Exception:
                pass
            
        # 6) 제목 분석 영역 — 더 많은 지표
        st.markdown("---")
        st.header("📝 제목 분석")
        tcol1, tcol2, tcol3 = st.columns([1,1,1])
        tcol1.metric("제목 길이", int(row.get("title_length", 0)))
        tcol2.metric("단어 수", int(row.get("word_count", 0)))
        tcol3.metric("특수문자 수", int(row.get("special_char_count", 0)))

        tcol1.metric("이모지 수", int(row.get("emoji_count", 0)))
        tcol2.metric("? 포함", "Yes" if int(row.get("has_question_mark", 0)) else "No")
        tcol3.metric("! 포함", "Yes" if int(row.get("has_exclamation", 0)) else "No")

        # top_noun chips
        nouns = [row.get("top_noun_1",""), row.get("top_noun_2",""), row.get("top_noun_3","")]
        nouns = [n for n in nouns if str(n).strip()]
        if nouns:
            st.markdown("**상위 명사(Top Nouns)**", help="제목에서 추출한 주요 명사 3개")
            st.markdown("".join(chip(n) for n in nouns), unsafe_allow_html=True)

        # 7) 썸네일 분석 — 색상/밝기/대비/객체/텍스트
        st.markdown("---")
        st.header("🖼️ 썸네일 분석")

        scol1, scol2, scol3 = st.columns([1,1,1])
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

        # dominant colors → 칩
        st.markdown("**주요 색상**")
        dom_colors = []
        if isinstance(row.get("dominant_colors"), str):
            dom_colors = [c.strip() for c in row["dominant_colors"].split(",") if c.strip()]
        if dom_colors:
            st.markdown("".join(color_chip(c) for c in dom_colors), unsafe_allow_html=True)

        # 객체 라벨 / 텍스트 분포 시각화
        lbl_counts = parse_object_labels(row.get("object_details", {}))
        text_details_raw = row.get("text_details", [])
        if isinstance(text_details_raw, str):
            import ast
            try:
                text_details_raw = ast.literal_eval(text_details_raw)
            except Exception:
                text_details_raw = []

        # OCR 텍스트에서 단어별 빈도 추출
        from collections import Counter
        text_tokens = []
        for item in text_details_raw:
            txt = str(item.get("text", "")).strip()
            if txt:
                text_tokens.extend(txt.split())

        text_counts = pd.Series(Counter(text_tokens)).sort_values(ascending=False).head(10)

        
        st.markdown(" ")
        with st.expander("🔍 썸네일 지표 더 알아보기"):
            # 두 개의 열로 배치
            c1, c2 = st.columns([1, 2])

            # --- 객체 라벨 분포 ---
            if not lbl_counts.empty:
                c1.markdown("**객체 라벨 분포**")
                c1.dataframe(lbl_counts.rename("개수").to_frame(), use_container_width=True)
            else:
                c1.info("객체 인식 결과가 없습니다.")

            def safe_parse_details(raw_val):
                import ast, json
                out = raw_val
                if isinstance(out, str):
                    try:
                        out = ast.literal_eval(out)
                    except Exception:
                        try:
                            out = json.loads(out)
                        except Exception:
                            out = None
                return out

            def clamp(v, lo, hi):
                return max(lo, min(hi, v))

            # 1) object_details만 사용
            object_details_raw = safe_parse_details(row.get("object_details", {}))
            if object_details_raw is None:
                object_details_raw = {}
            if isinstance(object_details_raw, dict):
                objects_list = object_details_raw.get("objects", [])
            elif isinstance(object_details_raw, list):
                objects_list = object_details_raw
            else:
                objects_list = []

            # 2) 좌표계 크기 추정 (객체만으로)
            all_boxes = []
            for obj in objects_list:
                if not isinstance(obj, dict):
                    continue
                try:
                    x = float(obj.get("x", 0))
                    y = float(obj.get("y", 0))
                    w = float(obj.get("width", 0) or obj.get("w", 0))
                    h = float(obj.get("height", 0) or obj.get("h", 0))
                except Exception:
                    continue
                if w > 0 and h > 0:
                    all_boxes.append((x, y, w, h))

            if len(all_boxes) == 0:
                c2.caption("객체 박스를 표시할 수 없습니다.")
            else:
                max_x = max(x+w for (x,y,w,h) in all_boxes)
                max_y = max(y+h for (x,y,w,h) in all_boxes)
                if max_x <= 0: max_x = 1.0
                if max_y <= 0: max_y = 1.0

                CANVAS_W = 1280.0
                CANVAS_H = 720.0

                def scale_box_any(x, y, w, h):
                    sx = CANVAS_W / max_x
                    sy = CANVAS_H / max_y
                    return {
                        "x": x * sx,
                        "y": y * sy,
                        "w": w * sx,
                        "h": h * sy,
                    }

                # figure
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

                # 면적 TOP5
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

                sized_objects = sorted(sized_objects, key=lambda d: d["area"], reverse=True)[:5]

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
                c2.image(buf, caption="객체 배치 맵 (정규화된 좌표)", use_container_width=True)
                plt.close(fig)

            def safe_parse_details(raw_val):
                """
                문자열, dict, list 형태의 입력을 안전하게 파싱하여 Python 객체로 반환.
                JSON, literal_eval, 원본 형태를 모두 지원.
                실패 시 None 반환.
                """
                # 이미 dict나 list면 그대로 반환
                if isinstance(raw_val, (dict, list)):
                    return raw_val

                # 비어있거나 None이면 None 반환
                if raw_val is None:
                    return None

                # 문자열일 경우 JSON 또는 Python literal 형태 시도
                if isinstance(raw_val, str):
                    raw_val = raw_val.strip()
                    if not raw_val:
                        return None

                    # JSON 형식 시도
                    try:
                        return json.loads(raw_val)
                    except json.JSONDecodeError:
                        pass

                    # literal_eval (ex: "[{'text':'Hi'}]" 같은 Python literal)
                    try:
                        return ast.literal_eval(raw_val)
                    except Exception:
                        pass

                # 그 외 타입은 그대로 반환 (예: int, float 등)
                return raw_val


            # --- 텍스트 단어 분포 ---
            if not text_counts.empty:
                st.markdown("**OCR 텍스트 단어**")

                # text_details_raw 안에서 단어별 평균 신뢰도 계산
                text_details_raw = safe_parse_details(row.get("text_details", []))
                word_conf_map = {}
                if isinstance(text_details_raw, list):
                    for item in text_details_raw:
                        text = str(item.get("text", "")).strip()
                        conf = float(item.get("probability", 0.0))
                        if not text:
                            continue
                        # 단어 단위로 분리
                        for w in text.split():
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
                    conf_df = conf_df.sort_values(["빈도", "신뢰도(%)"], ascending=[False, False]).reset_index(drop=True)

                    st.dataframe(conf_df, use_container_width=True)
                else:
                    st.info("OCR 텍스트 인식 결과가 없습니다.")
            else:
                st.info("OCR 텍스트 인식 결과가 없습니다.")
            
            st.markdown(" ")
            st.markdown("**객체/텍스트 특성**")
            c1, c2 = st.columns([1, 1])
            
            if "peer_avg_dict" not in locals():
                peer_avg_dict = {}

            # 시각화 공통 스타일 값
            bar_width = 0.35
            my_color = "#80D2FF"
            peer_color = "#5C6AFF"

            ############################
            # 1. 텍스트 위치 분포
            ############################
            c1, c2 = st.columns([1, 1])
            with c1:
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

            ############################
            # 2. 텍스트 크기 분포
            ############################
            with c2:
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


            ############################
            # 3. 사람 위치 분포
            ############################
            c3, c4 = st.columns([1, 1])
            with c3:
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

            ############################
            # 4. 사람 크기 분포
            ############################
            with c4:
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
        
        # === 피어 그룹 평균 핵심 지표 표 ===
        with st.expander("🔍 핵심 지표 더 알아보기"):
            if peer_avg_core:
                st.subheader("핵심 지표 비교")
                peer_df_show = pd.DataFrame({
                    "지표": [get_label(k) for k in peer_avg_core.keys()],
                    "내 값": [row.get(k, np.nan) for k in peer_avg_core.keys()],
                    "평균": list(peer_avg_core.values())
                })
                st.dataframe(peer_df_show, use_container_width=True)

       
        # 8) 피어 평균으로 맞추기 시뮬레이션 (회귀모델 + 피어 평균이 있을 때)
        if 'reg' in locals() and reg is not None and peer_avg_core:
            st.markdown("---")
            st.header("🔧 A/B 테스트")

            # 회귀 예측 도우미
            def simulate(modified: dict):
                new_df = df.copy()
                for k, v in modified.items():
                    new_df.at[0, k] = v
                return _predict_views_by_regressor(reg, new_df, feat_cols)

            # 어떤 지표를 시뮬할지 후보
            target_keys = [
                'title_length','emoji_count','special_char_count','person_count','has_text',
                'brightness','contrast','word_count','object_count'
            ]

            sims = []
            sims_k = []
            for k in target_keys:
                if k not in peer_avg_core:
                    continue

                cur_val = float(df.iloc[0].get(k, 0))
                peer_val = float(peer_avg_core[k])

                # 정수형으로 보는 게 자연스러운 지표는 반올림
                if k in ['emoji_count','special_char_count','person_count','has_text',
                        'word_count','object_count','title_length']:
                    peer_val = int(round(peer_val))

                pred_peer = simulate({k: peer_val})
                diff_val = None
                if predicted_views is not None and pred_peer is not None:
                    diff_val = int(pred_peer) - int(predicted_views)

                sims.append({
                    "지표": k,
                    "현재값": cur_val,
                    "평균값": peer_val,
                    "평균으로 맞출 때 예측": pred_peer,
                    "증감(예측)": diff_val
                })
                sims_k.append({
                    "지표": get_label(k),
                    "현재값": cur_val,
                    "평균값": peer_val,
                    "평균으로 맞출 때 예측": pred_peer,
                    "증감(예측)": diff_val
                })


            # ===== [중요] 증감이 양수인 애들만 묶어서 한번에 바꾸면? =====
            improving_changes = {}
            for row_sim in sims:
                if row_sim["증감(예측)"] is not None and row_sim["증감(예측)"] > 0:
                    # 이 지표는 바꿀 가치가 있음 → 피어 평균값 사용
                    k = row_sim["지표"]
                    v = row_sim["평균값"]
                    improving_changes[k] = v

            if improving_changes:
                pred_combo = simulate(improving_changes)
            else:
                pred_combo = None
                
            # 현재 예측 조회수와 개선 후 예측 조회수 계산
            current_pred_views = int(predicted_views) if predicted_views is not None else None
            improved_pred_views = int(pred_combo) if pred_combo is not None else None

            lift_abs = None
            lift_pct = None
            if current_pred_views is not None and improved_pred_views is not None:
                lift_abs = improved_pred_views - current_pred_views
                lift_pct = (lift_abs / current_pred_views) * 100 if current_pred_views > 0 else None

            
            st.markdown(" ")
            st.markdown("### 핵심 성과 요약")

            kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

            # 카드 1: 현재 예측
            kpi_col1.metric(
                label="현재 예상 조회수",
                value=(f"{current_pred_views:,}" if current_pred_views is not None else "—")
            )

            # 카드 2: 개선 시 예상 조회수
            kpi_col2.metric(
                label="개선 후 예상 조회수",
                value=(f"{improved_pred_views:,}" if improved_pred_views is not None else "—"),
                delta=(
                    f"+{lift_abs:,}" if (lift_abs is not None and lift_abs > 0) else None
                )
            )

            # 카드 3: 예상 향상률
            kpi_col3.metric(
                label="향상률(%)",
                value=(f"{lift_pct:.1f}%" if lift_pct is not None else "—"),
                delta=(
                    f"+{lift_pct:.1f}%" if (lift_pct is not None and lift_pct > 0) else None
                )
            )
            
    
            st.markdown("### 핵심 지표별 조정 방향")
            top_change_rows = (
                pd.DataFrame(sims)
                .dropna(subset=["증감(예측)"])
                .sort_values("증감(예측)", ascending=False)
                .head(3)
            )

            for _, r in top_change_rows.iterrows():
                render_change_row(
                    label   = get_label(r["지표"]),
                    cur_val = r["현재값"],
                    tgt_val = r["평균값"],
                    gain    = r["증감(예측)"]
                )
            st.markdown(" ") 
            st.markdown(" ")       
            # 시뮬 결과 표 보여주기
            if sims_k:
                st.dataframe(pd.DataFrame(sims_k), use_container_width=True)

                

    except Exception as e:
        st.exception(e)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

