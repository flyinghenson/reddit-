#!/usr/bin/env python3
import argparse
import asyncio
import hashlib
import json
import sys
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List

# Allow running without installing the package
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from reddit_scene_crawler.keyword_decomposer import decompose_keywords  # noqa: E402
from reddit_scene_crawler.models import RedditPost  # noqa: E402
from reddit_scene_crawler.reddit_crawler import RedditCrawler  # noqa: E402
from reddit_scene_crawler.scene_store import KeywordProgress, SceneStore  # noqa: E402


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def scene_id_from_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def looks_english(text: str) -> bool:
    """
    Lightweight English filter:
    - reject if too much CJK
    - require some ASCII letters
    """
    if not text:
        return False
    s = text.strip()
    if len(s) < 20:
        return False

    total = len(s)
    cjk = 0
    ascii_letters = 0
    for ch in s:
        code = ord(ch)
        if (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0x3040 <= code <= 0x30FF
            or 0xAC00 <= code <= 0xD7AF
        ):
            cjk += 1
        if ch.isascii() and ch.isalpha():
            ascii_letters += 1

    if cjk / max(total, 1) > 0.02:
        return False
    if ascii_letters < 10:
        return False
    if ascii_letters / max(total, 1) < 0.20:
        return False
    return True


def parse_args():
    p = argparse.ArgumentParser(
        description="Crawl Reddit posts/comments for a Chinese scene description (CSV + incremental resume)."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene", type=str, help="Chinese scene description (single text).")
    g.add_argument("--scene-file", type=str, help="Path to a UTF-8 text file containing the scene description.")

    p.add_argument("--keywords", type=str, default="", help="Comma-separated English search keywords (skip LLM).")
    p.add_argument("--keywords-file", type=str, default="", help="Path to JSON or line-based keywords file (skip LLM).")
    p.add_argument("--keyword-count", type=int, default=30, help="How many English search keywords to generate (LLM).")

    p.add_argument("--target-posts", type=int, default=10000, help="Stop after collecting at least this many posts (0=no limit).")
    p.add_argument("--max-pages-per-keyword", type=int, default=120, help="Max search pages per keyword.")
    p.add_argument("--time-range", type=str, default="all", help="Reddit time filter: hour/day/week/month/year/all.")
    p.add_argument("--post-sort", type=str, default="relevance", help="Reddit search sort (default: relevance).")
    p.add_argument("--exact-phrase", action="store_true", help="Quote multi-word keywords for exact phrase match (lower recall).")

    p.add_argument("--comment-sorts", type=str, default="best,top", help="Comma-separated comment sorts: best,top,new,controversial,qa,old.")
    p.add_argument("--expand-more", action="store_true", default=True, help="Expand 'more' nodes via morechildren (default: enabled).")
    p.add_argument("--no-expand-more", action="store_false", dest="expand_more", help="Disable morechildren expansion.")
    p.add_argument("--max-comment-posts", type=int, default=0, help="Max number of posts to process comments for (total cap). 0=all.")

    p.add_argument("--min-post-votes", type=int, default=0, help="Filter posts by minimum votes.")
    p.add_argument("--min-post-comments", type=int, default=0, help="Filter posts by minimum comments count.")

    p.add_argument("--output-dir", type=str, default="", help="Output directory (default: exports/<scene_id>/).")
    p.add_argument("--fresh", action="store_true", help="Delete output-dir first (start from scratch).")
    p.add_argument("--scene-name", type=str, default="", help="Short scene label used in output filenames.")
    return p.parse_args()


def _load_keywords_file(path: Path) -> List[str]:
    raw = path.read_text(encoding="utf-8").strip()
    out: List[str] = []
    if not raw:
        return out
    if raw.startswith("["):
        data = json.loads(raw)
        for item in data:
            if isinstance(item, str):
                out.append(item.strip())
            elif isinstance(item, dict) and item.get("keyword"):
                out.append(str(item["keyword"]).strip())
    else:
        for line in raw.splitlines():
            line = line.strip()
            if line:
                out.append(line)
    out = [k for k in out if k]
    out = list(dict.fromkeys(out))
    return out


async def main():
    args = parse_args()

    scene_description = ""
    if args.scene_file:
        scene_description = Path(args.scene_file).read_text(encoding="utf-8").strip()
    else:
        scene_description = (args.scene or "").strip()
    if not scene_description:
        raise SystemExit("Empty scene description.")

    scene_id = scene_id_from_text(scene_description)
    scene_name = (args.scene_name or "").strip()
    scene_dir_label = scene_name if scene_name else scene_description
    # Keep directory name human-readable; SceneStore will also put scene_name into CSV filenames
    safe_dir = "".join(ch if ch not in "/\\\n\r\t" else "_" for ch in scene_dir_label).strip()[:40]
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (PROJECT_ROOT / "exports" / f"{scene_id}__{safe_dir}")
    )

    if args.fresh and output_dir.exists():
        shutil.rmtree(output_dir)

    store = SceneStore(
        output_dir=output_dir,
        scene_id=scene_id,
        scene_description=scene_description,
        scene_name=scene_name,
    )
    store.load()

    def log(msg: str, level: str = "info"):
        prefix = {"info": "[*]", "success": "[+]", "warning": "[!]", "error": "[x]"}.get(level, "[*]")
        print(f"{prefix} {msg}")

    log(f"scene_id = {scene_id}")
    log(f"output_dir = {output_dir}")
    log(f"seen posts={len(store.seen_post_ids)}, seen comments={len(store.seen_comment_ids)}, processed posts={len(store.processed_post_ids)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "scene_id": scene_id,
                "scene_name": scene_name,
                "scene_description": scene_description,
                "started_at_iso": utc_now_iso(),
                "args": vars(args),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # keywords
    keyword_texts: List[str] = []
    if args.keywords:
        keyword_texts = [k.strip() for k in args.keywords.split(",") if k.strip()]
        log(f"using provided keywords = {len(keyword_texts)}")
    elif args.keywords_file:
        keyword_texts = _load_keywords_file(Path(args.keywords_file))
        log(f"using keywords from file = {len(keyword_texts)}")
    else:
        log(f"generating {args.keyword_count} keywords via LLM...")
        keyword_texts = await decompose_keywords(scene_description, keyword_count=args.keyword_count)
        log(f"LLM generated keywords = {len(keyword_texts)}", "success")

    keyword_texts = list(dict.fromkeys([k for k in keyword_texts if k]))
    (output_dir / "keywords.json").write_text(
        json.dumps([{"keyword": k} for k in keyword_texts], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    crawler = RedditCrawler(log_callback=log)

    try:
        # crawl posts and append to CSV (also enqueue todo posts)
        for i, kw in enumerate(keyword_texts):
            kp = store.keyword_progress.get(kw) or KeywordProgress()
            if kp.done:
                continue

            log(f"keyword [{i+1}/{len(keyword_texts)}]: {kw}")
            posts, after, pages = await crawler.search_posts_advanced(
                keyword=kw,
                sort=args.post_sort,
                time_range=args.time_range,
                max_pages=args.max_pages_per_keyword,
                min_votes=args.min_post_votes,
                min_comments=args.min_post_comments,
                exact_phrase=bool(args.exact_phrase),
                start_after=kp.after,
            )

            kept = 0
            for p in posts:
                text = (p.title or "") + "\n" + (p.content or "")
                if not looks_english(text):
                    continue
                if store.record_post(keyword=kw, post=p):
                    kept += 1
                if args.target_posts and len(store.seen_post_ids) >= args.target_posts:
                    break

            kp.after = after
            kp.pages_fetched += int(pages or 0)
            # Mark keyword as done only if at least one page was fetched successfully
            # (pages==0 usually means request failed / blocked, so allow retry next run)
            kp.done = (pages > 0 and after is None) or (pages >= args.max_pages_per_keyword)
            store.keyword_progress[kw] = kp
            store.save()

            log(f"kept posts for keyword = {kept}, total posts={len(store.seen_post_ids)}", "success")
            if args.target_posts and len(store.seen_post_ids) >= args.target_posts:
                log("target_posts reached", "success")
                break

        # crawl comments+replies for todo posts (resume-safe)
        comment_sorts = [s.strip() for s in args.comment_sorts.split(",") if s.strip()]
        log(f"processing todo posts: sorts={comment_sorts}, expand_more={args.expand_more}")

        processed_now = 0
        already_processed = len(store.processed_post_ids)
        for item in store.iter_unprocessed_todo_posts():
            post_id = str(item.get("post_id", "") or "")
            if not post_id or post_id in store.processed_post_ids:
                continue

            post = RedditPost(
                id=post_id,
                title=item.get("title", "") or "",
                content="",
                url=item.get("url", "") or f"https://www.reddit.com/comments/{post_id}",
                votes=int(item.get("votes", 0) or 0),
                comments_count=int(item.get("comments_count", 0) or 0),
                created_utc=float(item.get("created_utc", 0) or 0),
                search_keyword=item.get("keyword", "") or "",
                subreddit=item.get("subreddit", "") or "",
                author=item.get("author", "") or "[deleted]",
            )

            for csort in comment_sorts:
                comments = await crawler.fetch_post_comments(post, sort=csort, expand_more=args.expand_more)
                appended = 0
                for c in comments:
                    if not looks_english(c.content or ""):
                        continue
                    if store.record_comment(keyword=post.search_keyword, comment=c, post_id=post.id):
                        appended += 1
                log(f"post {post.id}: sort={csort} fetched={len(comments)} appended={appended}")

            store.mark_post_processed(post.id)
            processed_now += 1
            # If max_comment_posts is set, treat it as TOTAL cap per scene (not per run)
            if args.max_comment_posts and (already_processed + processed_now) >= args.max_comment_posts:
                log(
                    f"reached max-comment-posts(total)={args.max_comment_posts}, stop comment phase (resume next run)",
                    "warning",
                )
                break
            if processed_now % 20 == 0:
                store.save()
                log(f"checkpoint: processed_now={processed_now}, total_processed={len(store.processed_post_ids)}", "success")

        store.save()
        log("done", "success")
        log(f"posts.csv: {store.posts_csv_path}")
        log(f"comments.csv: {store.comments_csv_path}")

    finally:
        await crawler.close()
        store.close_writers()


if __name__ == "__main__":
    asyncio.run(main())

