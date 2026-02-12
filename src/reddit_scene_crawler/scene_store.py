import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def utc_iso(ts: float) -> str:
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except Exception:
        return ""


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


class CsvAppendWriter:
    def __init__(self, path: Path, fieldnames: List[str]):
        self.path = path
        self.fieldnames = fieldnames
        ensure_dir(path.parent)

        file_exists = path.exists() and path.stat().st_size > 0
        self._fh = path.open("a", newline="", encoding="utf-8-sig")
        self._writer = csv.DictWriter(
            self._fh,
            fieldnames=self.fieldnames,
            lineterminator="\n",
            quoting=csv.QUOTE_MINIMAL,
        )
        if not file_exists:
            self._writer.writeheader()
            self._fh.flush()

    def append_rows(self, rows: Iterable[Dict]):
        for row in rows:
            filtered = {k: row.get(k, "") for k in self.fieldnames}
            self._writer.writerow(filtered)
        self._fh.flush()

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


@dataclass
class KeywordProgress:
    after: Optional[str] = None
    pages_fetched: int = 0
    done: bool = False


@dataclass
class SceneStore:
    output_dir: Path
    scene_id: str
    scene_description: str
    scene_name: str = ""

    state_path: Path = field(init=False)
    seen_posts_path: Path = field(init=False)
    seen_comments_path: Path = field(init=False)
    todo_posts_path: Path = field(init=False)
    posts_csv_path: Path = field(init=False)
    comments_csv_path: Path = field(init=False)

    seen_post_ids: Set[str] = field(default_factory=set, init=False)
    seen_comment_ids: Set[str] = field(default_factory=set, init=False)
    processed_post_ids: Set[str] = field(default_factory=set, init=False)
    keyword_progress: Dict[str, KeywordProgress] = field(default_factory=dict, init=False)

    _posts_writer: Optional[CsvAppendWriter] = field(default=None, init=False, repr=False)
    _comments_writer: Optional[CsvAppendWriter] = field(default=None, init=False, repr=False)

    POSTS_COLUMNS: List[str] = field(
        default_factory=lambda: [
            "scene_id",
            "keyword",
            "post_id",
            "title",
            "content",
            "url",
            "votes",
            "comments_count",
            "created_utc",
            "created_iso",
            "fetched_at_iso",
        ],
        init=False,
    )

    COMMENTS_COLUMNS: List[str] = field(
        default_factory=lambda: [
            "scene_id",
            "keyword",
            "post_id",
            "comment_id",
            "parent_fullname",
            "depth",
            "content",
            "url",
            "votes",
            "comments_count",  # = direct reply count
            "created_utc",
            "created_iso",
            "source_sort",
            "fetched_at_iso",
        ],
        init=False,
    )

    def __post_init__(self):
        ensure_dir(self.output_dir)
        self.state_path = self.output_dir / "state.json"
        self.seen_posts_path = self.output_dir / "seen_posts.txt"
        self.seen_comments_path = self.output_dir / "seen_comments.txt"
        self.todo_posts_path = self.output_dir / "todo_posts.jsonl"
        label = _sanitize_filename_component(self.scene_name) if self.scene_name else ""
        suffix = f"_{label}" if label else ""
        self.posts_csv_path = self.output_dir / f"posts{suffix}.csv"
        self.comments_csv_path = self.output_dir / f"comments{suffix}.csv"

    def _load_seen_ids(self, path: Path) -> Set[str]:
        if not path.exists():
            return set()
        ids: Set[str] = set()
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                v = line.strip()
                if v:
                    ids.add(v)
        return ids

    def _append_seen_id(self, path: Path, _id: str) -> None:
        if not _id:
            return
        with path.open("a", encoding="utf-8") as f:
            f.write(_id + "\n")

    def load(self) -> None:
        self.seen_post_ids = self._load_seen_ids(self.seen_posts_path)
        self.seen_comment_ids = self._load_seen_ids(self.seen_comments_path)

        if not self.state_path.exists():
            self.save()
            return

        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.processed_post_ids = set(data.get("processed_post_ids", []) or [])

        kp = data.get("keyword_progress", {}) or {}
        self.keyword_progress = {}
        for k, v in kp.items():
            self.keyword_progress[k] = KeywordProgress(
                after=v.get("after"),
                pages_fetched=int(v.get("pages_fetched", 0) or 0),
                done=bool(v.get("done", False)),
            )

    def save(self) -> None:
        payload = {
            "version": 1,
            "scene_id": self.scene_id,
            "scene_description": self.scene_description,
            "updated_at_iso": utc_now_iso(),
            "processed_post_ids": sorted(self.processed_post_ids),
            "keyword_progress": {
                k: {"after": v.after, "pages_fetched": v.pages_fetched, "done": v.done}
                for k, v in self.keyword_progress.items()
            },
        }
        self.state_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def open_writers(self) -> None:
        if not self._posts_writer:
            self._posts_writer = CsvAppendWriter(self.posts_csv_path, self.POSTS_COLUMNS)
        if not self._comments_writer:
            self._comments_writer = CsvAppendWriter(
                self.comments_csv_path, self.COMMENTS_COLUMNS
            )

    def close_writers(self) -> None:
        if self._posts_writer:
            self._posts_writer.close()
            self._posts_writer = None
        if self._comments_writer:
            self._comments_writer.close()
            self._comments_writer = None

    def record_post(self, *, keyword: str, post) -> bool:
        post_id = getattr(post, "id", "") or ""
        if not post_id or post_id in self.seen_post_ids:
            return False

        self.seen_post_ids.add(post_id)
        self._append_seen_id(self.seen_posts_path, post_id)

        self.open_writers()
        row = {
            "scene_id": self.scene_id,
            "keyword": keyword,
            "post_id": post_id,
            "title": _sanitize_csv_text(getattr(post, "title", "") or ""),
            "content": _sanitize_csv_text(getattr(post, "content", "") or ""),
            "url": getattr(post, "url", "") or "",
            "votes": getattr(post, "votes", 0) or 0,
            "comments_count": getattr(post, "comments_count", 0) or 0,
            "created_utc": getattr(post, "created_utc", 0) or 0,
            "created_iso": utc_iso(getattr(post, "created_utc", 0) or 0),
            "fetched_at_iso": utc_now_iso(),
        }
        self._posts_writer.append_rows([row])

        self.enqueue_post(keyword=keyword, post=post)
        return True

    def record_comment(self, *, keyword: str, comment, post_id: str) -> bool:
        comment_id = getattr(comment, "id", "") or ""
        if not comment_id or comment_id in self.seen_comment_ids:
            return False

        self.seen_comment_ids.add(comment_id)
        self._append_seen_id(self.seen_comments_path, comment_id)

        self.open_writers()
        row = {
            "scene_id": self.scene_id,
            "keyword": keyword,
            "post_id": post_id,
            "comment_id": comment_id,
            "parent_fullname": getattr(comment, "parent_fullname", "") or "",
            "depth": getattr(comment, "depth", "") if getattr(comment, "depth", None) is not None else "",
            "content": _sanitize_csv_text(getattr(comment, "content", "") or ""),
            "url": getattr(comment, "url", "") or "",
            "votes": getattr(comment, "votes", 0) or 0,
            "comments_count": getattr(comment, "reply_count", 0) or 0,
            "created_utc": getattr(comment, "created_utc", 0) or 0,
            "created_iso": utc_iso(getattr(comment, "created_utc", 0) or 0),
            "source_sort": getattr(comment, "source_sort", "") or "",
            "fetched_at_iso": utc_now_iso(),
        }
        self._comments_writer.append_rows([row])
        return True

    def mark_post_processed(self, post_id: str) -> None:
        if post_id:
            self.processed_post_ids.add(post_id)

    def enqueue_post(self, *, keyword: str, post) -> None:
        post_id = getattr(post, "id", "") or ""
        if not post_id:
            return
        payload = {
            "keyword": keyword,
            "post_id": post_id,
            "title": _sanitize_csv_text(getattr(post, "title", "") or ""),
            "url": getattr(post, "url", "") or "",
            "subreddit": getattr(post, "subreddit", "") or "",
            "author": getattr(post, "author", "") or "",
            "votes": getattr(post, "votes", 0) or 0,
            "comments_count": getattr(post, "comments_count", 0) or 0,
            "created_utc": getattr(post, "created_utc", 0) or 0,
        }
        with self.todo_posts_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def iter_unprocessed_todo_posts(self) -> Iterator[Dict]:
        if not self.todo_posts_path.exists():
            return iter(())

        def _gen():
            with self.todo_posts_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    pid = str(item.get("post_id", "") or "")
                    if pid and pid not in self.processed_post_ids:
                        yield item

        return _gen()


def _sanitize_csv_text(s: str) -> str:
    """
    Make CSV "one record per line" friendly:
    - Convert real newlines to literal "\\n"
    - Strip NUL bytes
    """
    if not s:
        return ""
    s = s.replace("\x00", "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s.replace("\n", "\\n")


def _sanitize_filename_component(s: str) -> str:
    """
    Keep filenames readable and safe across platforms.
    - Trim whitespace
    - Replace path separators and control chars
    - Limit length
    """
    if not s:
        return ""
    s = s.strip()
    s = s.replace("\x00", "")
    for ch in ["/", "\\", ":", "*", "?", "\"", "<", ">", "|"]:
        s = s.replace(ch, "_")
    s = " ".join(s.split())
    return s[:60]

