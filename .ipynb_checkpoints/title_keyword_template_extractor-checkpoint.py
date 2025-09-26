
"""
title_keyword_template_extractor.py

Utilities to extract keywords (TF‑IDF + KeyBERT) and generate
"templates" from Korean YouTube titles by masking variable parts.

Requirements (install in your environment):
    pip install pandas scikit-learn konlpy keybert sentence-transformers rapidfuzz

Recommended Korean embedding models for KeyBERT:
    - "snunlp/KR-SBERT-V40K-klueNLI-augSTS"  (good Korean SBERT)
    - or multilingual: "paraphrase-multilingual-MiniLM-L12-v2"

If you want a faster tokenizer for Korean, install Mecab and change POS tagger.
This script defaults to KoNLPy's Okt to avoid system dependency hurdles.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable, Dict, Any

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from konlpy.tag import Okt
from rapidfuzz import fuzz, process

# -------------------------
# Text normalization utils
# -------------------------

_num_re   = re.compile(r'\d+([.,]\d+)*')
_url_re   = re.compile(r'https?://\S+|www\.\S+')
_email_re = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
_hash_re  = re.compile(r'#\S+')
_emoji_re = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)

def normalize_korean(text: str) -> str:
    """Basic cleanup while keeping meaningful Korean tokens."""
    if not isinstance(text, str):
        return ""
    t = text.strip()
    t = _url_re.sub(' {URL} ', t)
    t = _email_re.sub(' {EMAIL} ', t)
    t = _hash_re.sub(' {HASHTAG} ', t)
    t = _num_re.sub(' {NUM} ', t)
    t = _emoji_re.sub(' ', t)
    # normalize brackets
    t = t.replace('【','[').replace('】',']').replace('（','(').replace('）',')')
    # collapse spaces
    t = re.sub(r'\s+', ' ', t).strip()
    return t

# -------------------------
# TF-IDF keyword extraction
# -------------------------

def tfidf_keywords(
    titles: Iterable[str],
    top_k: int = 5,
    ngram_range: Tuple[int, int] = (1,2),
    stop_words: Optional[List[str]] = None
) -> Tuple[List[List[Tuple[str, float]]], TfidfVectorizer]:
    """
    Compute per-title TF-IDF top keywords/phrases.
    Returns (per_title_keywords, vectorizer).
    """
    titles = [normalize_korean(t) for t in titles]
    if stop_words is None:
        stop_words = [
            # Common Korean stopwords for titles (extend as needed)
            "영상", "채널", "그리고", "하지만", "그러나", "합니다", "합니다.", 
            "하는", "에게", "에서", "으로", "같은", "이번", "오늘", "그것", "저것",
            "정말", "진짜", "완전", "추천", "후기", "리뷰", "모음", "총정리",
            "꿀팁", "가이드", "방법", "비법", "브이로그", "브이로그.", "데이트",
        ]
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',  # char_wb + ngrams works well for Korean titles
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
    )
    X = vectorizer.fit_transform(titles)
    X = normalize(X, norm='l2', copy=False)

    feature_names = vectorizer.get_feature_names_out()
    results: List[List[Tuple[str,float]]] = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        if row.nnz == 0:
            results.append([])
            continue
        # get top indices by score
        top_idx = row.indices[row.data.argsort()[::-1][:max(top_k*6, top_k)]]  # grab more then filter
        cand = [(feature_names[j], float(row[0, j])) for j in top_idx]
        # filter trivial fragments and pad to top_k
        cleaned = []
        for token, score in cand:
            tok = token.strip()
            if len(tok) < 2: 
                continue
            if any(sym in tok for sym in ['{','}','[',']','(',')']):
                continue
            # discard mostly punctuation
            if re.fullmatch(r'[\W_]+', tok):
                continue
            cleaned.append((tok, score))
            if len(cleaned) >= top_k:
                break
        results.append(cleaned)
    return results, vectorizer

# -------------------------
# KeyBERT keyword extraction
# -------------------------

@dataclass
class KeyBERTConfig:
    model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
    top_k: int = 5
    use_maxsum: bool = True
    nr_candidates: int = 20
    use_mmr: bool = False
    diversity: float = 0.5 # used when use_mmr is True
    keyphrase_ngram_range: Tuple[int, int] = (1, 3)
    stop_words: Optional[List[str]] = None

def keybert_keywords(
    titles: Iterable[str],
    config: KeyBERTConfig = KeyBERTConfig()
) -> List[List[Tuple[str, float]]]:
    """
    KeyBERT-based keywords/keyphrases per title.
    """
    titles = [normalize_korean(t) for t in titles]
    model = SentenceTransformer(config.model_name)
    kw_model = KeyBERT(model=model)

    results = []
    for text in titles:
        if not text:
            results.append([])
            continue
        if config.use_mmr:
            kws = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=config.keyphrase_ngram_range,
                stop_words=config.stop_words,
                use_mmr=True,
                diversity=config.diversity,
                top_n=config.top_k,
            )
        else:
            kws = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=config.keyphrase_ngram_range,
                stop_words=config.stop_words,
                use_maxsum=config.use_maxsum,
                nr_candidates=config.nr_candidates,
                top_n=config.top_k,
            )
        # list of (phrase, score)
        results.append([(k, float(s)) for k, s in kws])
    return results

# -------------------------
# Title template generation
# -------------------------

class KoreanTemplater:
    """
    Build a 'template' by masking variable parts while keeping the core pattern.
    Rules (heuristic):
      - URLs -> {URL}
      - Emails -> {EMAIL}
      - Hashtags -> {HASHTAG}
      - Numbers -> {NUM}
      - English words -> {ENG}
      - Proper nouns and long nouns -> {NOUN}
      - Quoted / bracketed contents -> keep brackets, mask inside if mostly nouns/numbers
    """
    def __init__(self, noun_placeholder: str = "{NOUN}"):
        self.okt = Okt()
        self.noun_placeholder = noun_placeholder
        self.eng_re = re.compile(r'[A-Za-z]{2,}')
        self.bracket_re = re.compile(r'(\[[^\]]+\]|\([^)]+\)|\{[^}]+\})')

    def _mask_tokens(self, text: str) -> str:
        text = normalize_korean(text)
        # English words
        text = self.eng_re.sub(' {ENG} ', text)
        # bracketed content: decide masking if most content is nouns/numbers
        def replace_bracket(m):
            content = m.group(0)[1:-1]
            cnt_alpha = len(re.findall(r'[A-Za-z가-힣]', content))
            cnt_digit = len(re.findall(r'\d', content))
            if cnt_alpha + cnt_digit == 0:
                return m.group(0)  # leave as is
            # if heavy digits or nouns -> mask
            return m.group(0)[0] + ' {X} ' + m.group(0)[-1]
        text = self.bracket_re.sub(replace_bracket, text)

        # POS tagging with Okt
        tokens = self.okt.pos(text, norm=True, stem=False)
        templated = []
        for tok, pos in tokens:
            if tok in {"{URL}", "{EMAIL}", "{HASHTAG}", "{NUM}", "{ENG}"}:
                templated.append(tok)
                continue
            if pos in {"Number"}:
                templated.append("{NUM}")
            elif pos in {"Noun", "ProperNoun"}:
                # mask long nouns or nouns adjacent to other nouns (compound entities)
                if len(tok) >= 2:
                    templated.append(self.noun_placeholder)
                else:
                    templated.append(tok)
            else:
                templated.append(tok)
        out = ''.join(templated)
        out = re.sub(r'\s+', ' ', out).strip()
        # canonical tidy-up: collapse repeated placeholders
        out = re.sub(r'(?:\{NOUN\}\s*){2,}', '{NOUN} ', out)
        out = re.sub(r'(?:\{NUM\}\s*){2,}', '{NUM} ', out)
        return out

    def template(self, title: str) -> str:
        return self._mask_tokens(title or "")

def aggregate_templates(templates: Iterable[str], top_n: int = 15) -> List[Tuple[str,int]]:
    """Return most frequent templates with counts."""
    s = pd.Series([t for t in templates if t])
    if s.empty:
        return []
    counts = s.value_counts()
    return list(zip(counts.index[:top_n].tolist(), counts.values[:top_n].tolist()))

# -------------------------
# High-level pipeline
# -------------------------

@dataclass
class PipelineConfig:
    title_col: str = "title"
    tfidf_top_k: int = 5
    tfidf_ngram_range: Tuple[int,int] = (2,4)
    keybert_model: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
    keybert_top_k: int = 5
    keybert_use_maxsum: bool = True
    keybert_ngram_range: Tuple[int,int] = (1,3)

def analyze_titles(
    df: pd.DataFrame,
    config: PipelineConfig = PipelineConfig()
) -> pd.DataFrame:
    """
    Adds the following columns to df:
        - tfidf_keywords: List[(keyword, score)]
        - keybert_keywords: List[(keyphrase, score)]
        - title_template: masked template string
    Returns a new DataFrame (original + columns).
    """
    assert config.title_col in df.columns, f"'{config.title_col}' not in DataFrame"
    titles = df[config.title_col].fillna("").astype(str).tolist()

    # TF-IDF
    tfidf_res, _ = tfidf_keywords(
        titles,
        top_k=config.tfidf_top_k,
        ngram_range=config.tfidf_ngram_range
    )

    # KeyBERT
    kb_conf = KeyBERTConfig(
        model_name=config.keybert_model,
        top_k=config.keybert_top_k,
        use_maxsum=config.keybert_use_maxsum,
        keyphrase_ngram_range=config.keybert_ngram_range
    )
    keybert_res = keybert_keywords(titles, kb_conf)

    # Templates
    templater = KoreanTemplater()
    templates = [templater.template(t) for t in titles]

    out = df.copy()
    out["tfidf_keywords"] = tfidf_res
    out["keybert_keywords"] = keybert_res
    out["title_template"] = templates
    return out

def summarize_templates(df: pd.DataFrame, template_col: str = "title_template", top_n: int = 20) -> pd.DataFrame:
    pairs = aggregate_templates(df[template_col].tolist(), top_n=top_n)
    return pd.DataFrame(pairs, columns=["template", "count"])

# -------------------------
# Example usage (commented)
# -------------------------
if __name__ == "__main__":
    # Example mini demo (requires the listed pip installs)
    data = {
        "title": [
            "무편집 브이로그 | 서울 데이트 코스 추천 (맛집 5곳)",
            "초보도 가능한 파이썬 크롤링 꿀팁 7가지",
            "아이폰 16 출시일, 가격 총정리! 사전예약 정보",
            "[ENG] Korea Travel Vlog – Best Food in Seoul 2025",
            "월세 50만? 1인가구 절약법 모음 (진짜 도움됨)",
        ]
    }
    demo = pd.DataFrame(data)
    cfg = PipelineConfig(
        tfidf_top_k=5,
        tfidf_ngram_range=(2,4),
        keybert_model="snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        keybert_top_k=5,
        keybert_ngram_range=(1,3),
    )
    enriched = analyze_titles(demo, cfg)
    print(enriched[["title","tfidf_keywords","keybert_keywords","title_template"]])

    summary = summarize_templates(enriched)
    print("\nTop templates:\n", summary)
