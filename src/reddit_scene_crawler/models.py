from dataclasses import dataclass
from typing import Optional


@dataclass
class RedditPost:
    id: str
    title: str
    content: str
    url: str
    votes: int
    comments_count: int
    created_utc: float
    search_keyword: str
    subreddit: str = ""
    author: str = "[deleted]"


@dataclass
class RedditComment:
    id: str
    post_id: str
    content: str
    url: str
    votes: int
    reply_count: int
    created_utc: float
    search_keyword: str
    source_sort: str
    parent_fullname: Optional[str] = None  # e.g. "t3_<post_id>" or "t1_<comment_id>"
    depth: Optional[int] = None            # 1 for top-level comments (best-effort)

