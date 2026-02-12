import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from reddit_scene_crawler.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL  # noqa: E402


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _compact_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _safe_int(v) -> int:
    try:
        if v is None:
            return 0
        if isinstance(v, (int, float)):
            return int(v)
        s = str(v).strip()
        if not s:
            return 0
        return int(float(s))
    except Exception:
        return 0


@dataclass(frozen=True)
class Doc:
    doc_type: str  # "post" | "comment"
    url: str
    text: str
    votes: int
    created_iso: str


def load_docs(posts_csv: Path, comments_csv: Path) -> List[Doc]:
    posts = pd.read_csv(posts_csv)
    comments = pd.read_csv(comments_csv)

    out: List[Doc] = []

    for _, r in posts.iterrows():
        title = _compact_ws(str(r.get("title", "") or ""))
        content = _compact_ws(str(r.get("content", "") or ""))
        text = _compact_ws(f"{title}\n\n{content}")
        url = str(r.get("url", "") or "").strip()
        if not text or not url:
            continue
        out.append(
            Doc(
                doc_type="post",
                url=url,
                text=text,
                votes=_safe_int(r.get("votes")),
                created_iso=str(r.get("created_iso", "") or "").strip(),
            )
        )

    for _, r in comments.iterrows():
        content = _compact_ws(str(r.get("content", "") or ""))
        url = str(r.get("url", "") or "").strip()
        if not content or not url:
            continue
        out.append(
            Doc(
                doc_type="comment",
                url=url,
                text=content,
                votes=_safe_int(r.get("votes")),
                created_iso=str(r.get("created_iso", "") or "").strip(),
            )
        )

    return out


def extract_candidates_tfidf(
    texts: List[str],
    *,
    sample_size: int = 8000,
    top_n: int = 350,
    min_df: int = 3,
    max_df: float = 0.35,
    max_features: int = 20000,
    seed: int = 7,
) -> List[Tuple[str, float]]:
    if not texts:
        return []

    rng = random.Random(seed)
    if len(texts) > sample_size:
        texts = rng.sample(texts, sample_size)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 3),
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
    )
    X = vectorizer.fit_transform(texts)
    scores = X.sum(axis=0).A1  # type: ignore[attr-defined]
    feats = vectorizer.get_feature_names_out()

    pairs = list(zip(feats, scores))
    pairs.sort(key=lambda x: x[1], reverse=True)

    def ok(term: str) -> bool:
        t = term.strip()
        if len(t) < 3:
            return False
        if t in {"im", "ive", "dont", "cant"}:
            return False
        if re.fullmatch(r"\d+", t):
            return False
        # avoid pure punctuation-ish
        if not re.search(r"[a-zA-Z]", t):
            return False
        return True

    out = [(t, float(s)) for (t, s) in pairs if ok(t)]
    return out[:top_n]


LLM_SYSTEM_PROMPT = """You are an expert analyst. You will take a list of candidate phrases extracted from Reddit posts+comments.

Goal: produce a structured keyword taxonomy for ONE SCENE.

We need 5 categories:
1) cognition: beliefs/attributions/understandings about causes, constraints, tradeoffs, why something happens.
2) feeling: subjective feelings/emotions/bodily sensations (tired, anxious, brain fog, motivated, etc.)
3) behavior: habitual patterns, routines, lifestyle behaviors (sleeping late, fasting, scrolling, etc.)
4) action: concrete actions or steps people do (take a nap, drink coffee, walk outside, open a window, etc.)
5) solution_principle: their perceived mechanism/principle behind solutions (blood sugar stability, caffeine tolerance, circadian rhythm, posture ergonomics, etc.)

Rules:
- Use ONLY English in canonical/variants.
- Canonical should be 2-6 words when possible (avoid too generic single words).
- Merge synonyms into one canonical keyword and provide variants for matching.
- Keep variants short, lowercase, and without punctuation.
- Return JSON ONLY with the schema described by the user.
"""


def build_taxonomy_with_openai(
    candidates: List[Tuple[str, float]],
    *,
    max_keywords_per_category: int = 60,
) -> Dict:
    try:
        from openai import OpenAI  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'openai'. Install requirements."
        ) from e

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is empty. Set env var OPENAI_API_KEY.")

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    # Keep payload compact: only send top phrases + scores
    payload_lines = [f"- {t} ({s:.4f})" for (t, s) in candidates]
    user_prompt = (
        "Candidate phrases (higher score = more salient):\n"
        + "\n".join(payload_lines)
        + "\n\n"
        + "Output JSON schema:\n"
        + "{\n"
        + '  "scene_taxonomy_version": "v1",\n'
        + '  "generated_at": "...",\n'
        + '  "categories": [\n'
        + '    {"name":"cognition","keywords":[{"canonical":"...","variants":["..."],"note":"..."}]},\n'
        + '    {"name":"feeling","keywords":[...]},\n'
        + '    {"name":"behavior","keywords":[...]},\n'
        + '    {"name":"action","keywords":[...]},\n'
        + '    {"name":"solution_principle","keywords":[...]}\n'
        + "  ]\n"
        + "}\n\n"
        + f"Constraints: max {max_keywords_per_category} keywords per category.\n"
        + "Only output JSON."
    )

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    content = resp.choices[0].message.content or "{}"
    data = json.loads(content)
    if not isinstance(data, dict) or "categories" not in data:
        raise RuntimeError("Unexpected LLM JSON. Missing 'categories'.")

    data["generated_at"] = _now_iso()
    data.setdefault("scene_taxonomy_version", "v1")
    return data


def _variant_patterns(variants: Iterable[str]) -> List[Tuple[str, Optional[re.Pattern]]]:
    """
    Return list of (variant_lower, pattern_or_None).
    - single token: regex word-boundary
    - phrase: simple substring match (pattern None)
    """
    out: List[Tuple[str, Optional[re.Pattern]]] = []
    for v in variants:
        vv = _compact_ws(v).lower()
        if not vv:
            continue
        if " " in vv:
            out.append((vv, None))
        else:
            out.append((vv, re.compile(rf"\b{re.escape(vv)}\b", re.IGNORECASE)))
    # de-dup preserve order
    seen = set()
    dedup: List[Tuple[str, Optional[re.Pattern]]] = []
    for item in out:
        key = item[0]
        if key in seen:
            continue
        seen.add(key)
        dedup.append(item)
    return dedup


_NON_WORD_RE = re.compile(r"[^a-zA-Z']+")


def _norm_en_text(s: str) -> str:
    """
    Normalize English-ish text for vectorizer matching:
    - lowercase
    - keep letters and apostrophes
    - collapse everything else to spaces
    """
    s = (s or "").lower()
    s = _NON_WORD_RE.sub(" ", s)
    return _compact_ws(s)


def count_and_examples(
    docs: List[Doc],
    taxonomy: Dict,
    *,
    top_k: int = 30,
    examples_per_keyword: int = 10,
    min_total_count: int = 3,
) -> Dict:
    # flatten keywords
    categories = taxonomy.get("categories", [])
    if not isinstance(categories, list):
        raise ValueError("taxonomy.categories must be a list")

    # Normalize all docs once (fast matching)
    docs_norm = [_norm_en_text(d.text) for d in docs]

    results = {
        "generated_at": _now_iso(),
        "top_k": top_k,
        "examples_per_keyword": examples_per_keyword,
        "categories": [],
    }

    # Build a global vocabulary from all variants (across all categories)
    all_terms: List[str] = []
    for cat in categories:
        items = (cat or {}).get("keywords", [])
        if not isinstance(items, list):
            continue
        for it in items:
            canonical = _compact_ws(str((it or {}).get("canonical", ""))).lower()
            variants = (it or {}).get("variants", [])
            if not isinstance(variants, list) or not variants:
                variants = [canonical]
            for v in variants:
                nv = _norm_en_text(str(v))
                if nv:
                    all_terms.append(nv)

    # de-dup preserve order
    all_terms = list(dict.fromkeys(all_terms))

    # If vocabulary is empty, return empty report
    if not all_terms:
        for cat in categories:
            name = str((cat or {}).get("name", "")).strip()
            if name:
                results["categories"].append({"name": name, "total_hits": 0, "keywords": []})
        return results

    vectorizer = CountVectorizer(
        vocabulary=all_terms,
        ngram_range=(1, 3),
        lowercase=False,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b",
    )
    X = vectorizer.fit_transform(docs_norm)
    # Use CSC for fast column (term) access
    Xc = X.tocsc()
    vocab = vectorizer.vocabulary_
    term_counts = X.sum(axis=0).A1  # type: ignore[attr-defined]

    for cat in categories:
        name = str((cat or {}).get("name", "")).strip()
        items = (cat or {}).get("keywords", [])
        if not name or not isinstance(items, list):
            continue

        kw_rows = []
        total_hits = 0

        # First pass: compute counts from vectorized matrix
        for it in items:
            canonical = _compact_ws(str((it or {}).get("canonical", ""))).lower()
            if not canonical:
                continue
            variants = (it or {}).get("variants", [])
            if not isinstance(variants, list) or not variants:
                variants = [canonical]
            norm_variants = []
            for v in variants:
                nv = _norm_en_text(str(v))
                if nv:
                    norm_variants.append(nv)
            norm_variants = list(dict.fromkeys(norm_variants))
            if not norm_variants:
                continue

            hit_count = 0
            for v in norm_variants:
                idx = vocab.get(v)
                if idx is None:
                    continue
                hit_count += int(term_counts[idx])

            if hit_count < min_total_count:
                continue

            total_hits += hit_count
            kw_rows.append(
                {
                    "canonical": canonical,
                    "variants": norm_variants,
                    "count": int(hit_count),
                    "note": _compact_ws(str((it or {}).get("note", ""))),
                }
            )

        # Guard: no keywords
        if not kw_rows or total_hits <= 0:
            results["categories"].append({"name": name, "total_hits": 0, "keywords": []})
            continue

        # Percent + sort
        for r in kw_rows:
            r["percent"] = float(r["count"]) / float(total_hits)

        kw_rows.sort(key=lambda x: (x["count"], x["canonical"]), reverse=True)
        kw_rows = kw_rows[: max(top_k, 1)]

        # Second pass: collect examples for selected keywords (vote-sorted)
        for r in tqdm(kw_rows, desc=f"examples:{name}", leave=False):
            # Get doc indices that match ANY variant
            doc_idx_set = set()
            for v in r["variants"]:
                idx = vocab.get(v)
                if idx is None:
                    continue
                col = Xc.getcol(idx)
                if col.nnz:
                    doc_idx_set.update(col.indices.tolist())

            if not doc_idx_set:
                r["examples"] = []
                continue

            # Pick top docs by votes
            # (avoid sorting huge lists by trimming first)
            doc_indices = list(doc_idx_set)
            doc_indices.sort(key=lambda i: docs[i].votes, reverse=True)
            doc_indices = doc_indices[: max(examples_per_keyword * 6, examples_per_keyword)]

            ex = []
            for i in doc_indices[:examples_per_keyword]:
                d = docs[i]
                snippet = _compact_ws(d.text)
                if len(snippet) > 360:
                    snippet = snippet[:357] + "..."
                ex.append(
                    {
                        "doc_type": d.doc_type,
                        "votes": d.votes,
                        "created_iso": d.created_iso,
                        "text": snippet,
                        "url": d.url,
                    }
                )
            r["examples"] = ex

        results["categories"].append({"name": name, "total_hits": total_hits, "keywords": kw_rows})

    return results


def render_pdf(report: Dict, *, out_pdf: Path, title: str, subtitle: str) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (  # type: ignore
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'reportlab'. Install requirements."
        ) from e

    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    h3 = styles["Heading3"]
    body = styles["BodyText"]
    body_small = ParagraphStyle("BodySmall", parent=body, fontSize=9, leading=11)
    mono = ParagraphStyle("MonoSmall", parent=body_small, fontName="Courier")

    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title=title,
        author="reddit_scene_crawler",
    )

    story = []
    story.append(Paragraph(title, h1))
    story.append(Paragraph(subtitle, body_small))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"Generated at: {report.get('generated_at','')}", body_small))
    story.append(Spacer(1, 14))

    cats = report.get("categories", [])
    if isinstance(cats, list):
        story.append(Paragraph("Contents", h2))
        for c in cats:
            story.append(Paragraph(f"- {c.get('name','')}", body))
        story.append(PageBreak())

    for c in cats if isinstance(cats, list) else []:
        cname = c.get("name", "")
        story.append(Paragraph(str(cname), h2))
        story.append(Paragraph(f"Total keyword hits (denominator): {c.get('total_hits', 0)}", body_small))
        story.append(Spacer(1, 10))

        kws = c.get("keywords", [])
        if not kws:
            story.append(Paragraph("No keywords found for this category (after filtering).", body))
            story.append(PageBreak())
            continue

        # Table: Top keywords
        table_data = [["Rank", "Keyword", "Count", "Percent"]]
        for i, r in enumerate(kws, start=1):
            table_data.append(
                [
                    str(i),
                    r.get("canonical", ""),
                    str(r.get("count", 0)),
                    f"{float(r.get('percent', 0.0)) * 100:.2f}%",
                ]
            )

        tbl = Table(table_data, colWidths=[1.2 * cm, 8.6 * cm, 2.2 * cm, 2.4 * cm])
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ]
            )
        )
        story.append(tbl)
        story.append(Spacer(1, 12))

        # Details + examples
        for r in kws:
            canonical = r.get("canonical", "")
            story.append(Paragraph(str(canonical), h3))
            story.append(
                Paragraph(
                    f"Count: {r.get('count', 0)} | Percent: {float(r.get('percent', 0.0)) * 100:.2f}%",
                    body_small,
                )
            )
            variants = r.get("variants", [])
            if variants:
                story.append(Paragraph("Variants: " + ", ".join(map(str, variants[:20])), mono))
            note = str(r.get("note", "") or "").strip()
            if note:
                story.append(Paragraph("Note: " + note, body_small))
            story.append(Spacer(1, 6))

            exs = r.get("examples", []) or []
            if not exs:
                story.append(Paragraph("No examples found.", body_small))
                story.append(Spacer(1, 10))
                continue

            for j, ex in enumerate(exs, start=1):
                txt = str(ex.get("text", "") or "")
                url = str(ex.get("url", "") or "")
                votes = ex.get("votes", 0)
                story.append(Paragraph(f"{j}. ({ex.get('doc_type','')}, votes={votes}) {txt}", body_small))
                if url:
                    story.append(Paragraph(url, mono))
                story.append(Spacer(1, 4))

            story.append(Spacer(1, 10))

        story.append(PageBreak())

    doc.build(story)


def discover_scene_csvs(scene_dir: Path) -> Tuple[Path, Path]:
    # support both crawler default names and user's customized names
    posts = sorted(scene_dir.glob("posts*.csv"))
    comments = sorted(scene_dir.glob("comments*.csv"))
    if not posts or not comments:
        raise FileNotFoundError(
            f"Cannot find posts*.csv and comments*.csv under: {scene_dir}"
        )
    return posts[0], comments[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scene-dir",
        type=str,
        default=str(Path("exports") / "日常生活与全天能量维持"),
        help="Directory containing posts*.csv and comments*.csv",
    )
    ap.add_argument("--posts-csv", type=str, default="", help="Override posts CSV path")
    ap.add_argument("--comments-csv", type=str, default="", help="Override comments CSV path")
    ap.add_argument(
        "--out-pdf",
        type=str,
        default="",
        help="Output PDF path (default: <scene-dir>/scene_insights.pdf)",
    )
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--examples-per-keyword", type=int, default=10)
    ap.add_argument("--tfidf-top-n", type=int, default=350)
    ap.add_argument("--tfidf-sample-size", type=int, default=8000)
    ap.add_argument("--min-total-count", type=int, default=3)
    ap.add_argument(
        "--refresh-taxonomy",
        action="store_true",
        help="Force re-call OpenAI even if cache exists",
    )
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir)
    if args.posts_csv and args.comments_csv:
        posts_csv = Path(args.posts_csv)
        comments_csv = Path(args.comments_csv)
    else:
        posts_csv, comments_csv = discover_scene_csvs(scene_dir)

    out_pdf = Path(args.out_pdf) if args.out_pdf else (scene_dir / "scene_insights.pdf")
    cache_tax = scene_dir / "analysis_cache_taxonomy.json"
    cache_report = scene_dir / "analysis_cache_report.json"

    print(f"[{_now_iso()}] Loading docs...")
    docs = load_docs(posts_csv, comments_csv)
    if not docs:
        raise RuntimeError("No docs loaded. Check CSV paths/columns.")

    print(f"[{_now_iso()}] Loaded docs: {len(docs)} (posts+comments).")

    # TF-IDF candidates
    texts = [d.text for d in docs]
    print(f"[{_now_iso()}] Extracting TF-IDF candidates...")
    candidates = extract_candidates_tfidf(
        texts,
        sample_size=args.tfidf_sample_size,
        top_n=args.tfidf_top_n,
        seed=args.seed,
    )
    if not candidates:
        raise RuntimeError("No TF-IDF candidates extracted.")

    # Taxonomy (LLM) with cache
    taxonomy: Dict
    if cache_tax.exists() and not args.refresh_taxonomy:
        print(f"[{_now_iso()}] Using cached taxonomy: {cache_tax}")
        taxonomy = json.loads(cache_tax.read_text(encoding="utf-8"))
    else:
        print(f"[{_now_iso()}] Calling OpenAI to build taxonomy (S1)...")
        taxonomy = build_taxonomy_with_openai(candidates)
        cache_tax.write_text(json.dumps(taxonomy, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{_now_iso()}] Saved taxonomy cache: {cache_tax}")

    # Count + examples (full corpus matching)
    print(f"[{_now_iso()}] Counting keyword hits and collecting examples...")
    report = count_and_examples(
        docs,
        taxonomy,
        top_k=args.top_k,
        examples_per_keyword=args.examples_per_keyword,
        min_total_count=args.min_total_count,
    )
    cache_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{_now_iso()}] Saved report cache: {cache_report}")

    # Render PDF
    title = "Scene Insights Report"
    subtitle = (
        f"Scene dir: {scene_dir} | docs={len(docs)} | top_k={args.top_k} | "
        f"examples_per_keyword={args.examples_per_keyword} | model={OPENAI_MODEL}"
    )
    print(f"[{_now_iso()}] Rendering PDF: {out_pdf}")
    render_pdf(report, out_pdf=out_pdf, title=title, subtitle=subtitle)
    print(f"[{_now_iso()}] Done. Output: {out_pdf}")


if __name__ == "__main__":
    main()

