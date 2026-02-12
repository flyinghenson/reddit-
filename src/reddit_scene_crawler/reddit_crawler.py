import asyncio
import json
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    import httpx  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    httpx = None  # type: ignore

import urllib.error
import urllib.parse
import urllib.request

from .config import (
    COMMENT_MIN_VOTES,
    MIN_REQUEST_INTERVAL,
    POST_MIN_COMMENTS,
    POST_MIN_VOTES,
    PULLPUSH_BASE_URL,
    REDDIT_BASE_URL,
    REDDIT_SEARCH_LIMIT,
    REDDIT_WWW_URL,
    USE_PULLPUSH,
)
from .models import RedditComment, RedditPost


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


class RedditCrawler:
    """
    Crawl Reddit via public JSON endpoints (non-official):
    - Search:   /search.json
    - Comments: /comments/<post_id>.json
    - Expand:   /api/morechildren.json
    """

    def __init__(self, log_callback=None):
        self._log = log_callback or (lambda msg, lvl="info": None)
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "DNT": "1",
        }
        self._headers = headers
        self._use_httpx = httpx is not None
        self.client = None
        if self._use_httpx:
            self.client = httpx.AsyncClient(
                headers=headers,
                timeout=30.0,
                follow_redirects=True,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )

        self._rate_lock = asyncio.Lock()
        self._last_request_time = 0.0
        self._min_interval = float(MIN_REQUEST_INTERVAL)
        self._use_pullpush = bool(USE_PULLPUSH)

    async def close(self):
        if self.client is not None:
            await self.client.aclose()

    async def _rate_limit(self):
        async with self._rate_lock:
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_request_time = time.time()

    async def _fetch_json(
        self,
        url: str,
        params: Optional[dict] = None,
        retries: int = 3,
    ) -> Optional[dict]:
        for attempt in range(retries):
            await self._rate_limit()
            try:
                if self._use_httpx and self.client is not None:
                    resp = await self.client.get(url, params=params)
                    if resp.status_code == 429:
                        wait = 10 * (attempt + 1)
                        self._log(f"rate limited, sleep {wait}s then retry", "warning")
                        await asyncio.sleep(wait)
                        continue
                    if resp.status_code in (403, 404):
                        self._log(f"http {resp.status_code}: {url[:120]}", "warning")
                        return None
                    resp.raise_for_status()
                    return resp.json()
                else:
                    return await asyncio.to_thread(
                        self._fetch_json_stdlib, url, params
                    )
            except Exception as e:
                msg = str(e)
                lmsg = msg.lower()
                if "rate limited" in lmsg or "429" in lmsg:
                    wait = 10 * (attempt + 1)
                    self._log(f"rate limited, sleep {wait}s then retry", "warning")
                    await asyncio.sleep(wait)
                    continue
                if "timed out" in lmsg or "timeout" in lmsg:
                    await asyncio.sleep(3)
                    continue

                self._log(f"request error: {e}", "error")
                if attempt < retries - 1:
                    await asyncio.sleep(2)
        return None

    def _fetch_json_stdlib(self, url: str, params: Optional[dict]) -> Optional[dict]:
        """Synchronous JSON fetch via urllib (used when httpx is unavailable)."""
        full_url = url
        if params:
            qs = urllib.parse.urlencode(params, doseq=True)
            sep = "&" if "?" in full_url else "?"
            full_url = f"{full_url}{sep}{qs}"
        req = urllib.request.Request(full_url, headers=self._headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                status = getattr(resp, "status", 200)
                if status in (403, 404):
                    self._log(f"http {status}: {full_url[:120]}", "warning")
                    return None
                raw = resp.read()
                return json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                # mimic caller retry loop by raising
                self._log(f"http 429: {full_url[:120]}", "warning")
                raise RuntimeError("rate limited (429)") from e
            if e.code in (403, 404):
                self._log(f"http {e.code}: {full_url[:120]}", "warning")
                return None
            raise
        except Exception:
            raise

    # ---------------------------
    # Search
    # ---------------------------

    async def _pullpush_search_submission(
        self,
        *,
        q: str,
        size: int = 100,
        sort: str = "desc",
        sort_type: str = "score",
        before: Optional[int] = None,
    ) -> Optional[dict]:
        params: Dict[str, Any] = {
            "q": q,
            "size": min(100, max(1, int(size))),
            "sort": sort,
            "sort_type": sort_type,
        }
        if before is not None:
            params["before"] = int(before)
        return await self._fetch_json(f"{PULLPUSH_BASE_URL}/reddit/search/submission/", params=params)

    async def _pullpush_search_comment(
        self,
        *,
        link_id: str,
        size: int = 100,
        sort: str = "desc",
        sort_type: str = "created_utc",
        before: Optional[int] = None,
    ) -> Optional[dict]:
        params: Dict[str, Any] = {
            "link_id": link_id,
            "size": min(100, max(1, int(size))),
            "sort": sort,
            "sort_type": sort_type,
        }
        if before is not None:
            params["before"] = int(before)
        return await self._fetch_json(f"{PULLPUSH_BASE_URL}/reddit/search/comment/", params=params)

    async def search_posts_advanced(
        self,
        *,
        keyword: str,
        sort: str = "relevance",
        time_range: str = "all",  # hour/day/week/month/year/all
        max_pages: int = 120,
        min_votes: int = POST_MIN_VOTES,
        min_comments: int = POST_MIN_COMMENTS,
        exact_phrase: bool = True,
        start_after: Optional[str] = None,
    ) -> Tuple[List[RedditPost], Optional[str], int]:
        if self._use_pullpush:
            # PullPush does not provide "relevance". We approximate with score-desc.
            # We paginate by moving "before" backward on created_utc.
            posts: List[RedditPost] = []
            pages_fetched = 0
            before: Optional[int] = None

            q = f"\"{keyword}\"" if exact_phrase and " " in keyword else keyword
            self._log(f"[pullpush] search '{keyword}' sort=score_desc")

            seen_ids: Set[str] = set()
            while pages_fetched < max_pages:
                data = await self._pullpush_search_submission(
                    q=q, size=100, sort="desc", sort_type="score", before=before
                )
                if not data or "data" not in data:
                    break
                items = data.get("data") or []
                if not items:
                    break
                pages_fetched += 1

                min_created: Optional[int] = None
                kept = 0
                for pd in items:
                    pid = str(pd.get("id", "") or "")
                    if not pid or pid in seen_ids:
                        continue
                    seen_ids.add(pid)

                    score = int(pd.get("score", 0) or 0)
                    num_comments = int(pd.get("num_comments", 0) or 0)
                    if score < min_votes or num_comments < min_comments:
                        continue

                    created_utc = int(pd.get("created_utc", 0) or 0)
                    if created_utc and (min_created is None or created_utc < min_created):
                        min_created = created_utc

                    subreddit = pd.get("subreddit", "") or ""
                    permalink = pd.get("permalink") or ""
                    if permalink:
                        url = f"{REDDIT_WWW_URL}{permalink}"
                    else:
                        url = f"{REDDIT_WWW_URL}/comments/{pid}"

                    posts.append(
                        RedditPost(
                            id=pid,
                            title=pd.get("title", "") or "",
                            content=pd.get("selftext", "") or "",
                            url=url,
                            votes=score,
                            comments_count=num_comments,
                            created_utc=float(created_utc),
                            search_keyword=keyword,
                            subreddit=subreddit,
                            author=pd.get("author", "[deleted]") or "[deleted]",
                        )
                    )
                    kept += 1

                self._log(f"[pullpush] page {pages_fetched}: results={len(items)} kept={kept} total_posts={len(posts)}")

                if min_created is None:
                    break
                # Move window backward
                before = min_created - 1

            return posts, None, pages_fetched

        posts: List[RedditPost] = []
        after = start_after
        pages_fetched = 0

        self._log(f"search '{keyword}' sort={sort} t={time_range}")

        while True:
            q = f"\"{keyword}\"" if exact_phrase and " " in keyword else keyword
            params = {
                "q": q,
                "sort": sort,
                "t": time_range,
                "limit": REDDIT_SEARCH_LIMIT,
                "type": "link",
                "raw_json": 1,
            }
            if after:
                params["after"] = after

            # Try old.reddit first; fallback to www if blocked/empty.
            data = await self._fetch_json(f"{REDDIT_BASE_URL}/search.json", params=params)
            if not data or "data" not in data or not data.get("data", {}).get("children"):
                data2 = await self._fetch_json(f"{REDDIT_WWW_URL}/search.json", params=params)
                if data2 and "data" in data2 and data2.get("data", {}).get("children"):
                    data = data2
            if not data or "data" not in data:
                break

            # Count this page as fetched (even if it has 0 children)
            pages_fetched += 1

            children = data["data"].get("children", [])
            if not children:
                break

            kept = 0
            for child in children:
                if child.get("kind") != "t3":
                    continue
                pd = child.get("data", {}) or {}
                score = int(pd.get("score", 0) or 0)
                num_comments = int(pd.get("num_comments", 0) or 0)
                if score < min_votes or num_comments < min_comments:
                    continue

                pid = str(pd.get("id", "") or "")
                if not pid:
                    continue
                permalink = pd.get("permalink", "") or ""
                posts.append(
                    RedditPost(
                        id=pid,
                        title=pd.get("title", "") or "",
                        content=pd.get("selftext", "") or "",
                        url=f"{REDDIT_WWW_URL}{permalink}" if permalink else f"{REDDIT_WWW_URL}/comments/{pid}",
                        votes=score,
                        comments_count=num_comments,
                        created_utc=float(pd.get("created_utc", 0) or 0),
                        search_keyword=keyword,
                        subreddit=pd.get("subreddit", "") or "",
                        author=pd.get("author", "[deleted]") or "[deleted]",
                    )
                )
                kept += 1

            self._log(
                f"page {pages_fetched}: results={len(children)} kept={kept} total_posts={len(posts)}"
            )

            after = data["data"].get("after")
            if not after or pages_fetched >= max_pages:
                break

        return posts, after, pages_fetched

    # ---------------------------
    # Comments
    # ---------------------------

    def _count_direct_replies(self, replies_data) -> int:
        if not replies_data or isinstance(replies_data, str):
            return 0
        children = replies_data.get("data", {}).get("children", [])
        return sum(1 for c in children if c.get("kind") == "t1")

    def _collect_more_children_ids(self, children: list) -> List[str]:
        ids: List[str] = []
        for child in children:
            kind = child.get("kind")
            if kind == "more":
                cd = child.get("data", {}) or {}
                ids.extend([str(x) for x in (cd.get("children") or []) if x])
                continue
            if kind != "t1":
                continue
            replies = (child.get("data", {}) or {}).get("replies", "")
            if replies and not isinstance(replies, str):
                ids.extend(self._collect_more_children_ids(replies.get("data", {}).get("children", [])))
        return ids

    def _parse_comments_recursive(
        self,
        *,
        listing_children: list,
        post: RedditPost,
        source_sort: str,
        depth: int,
        parent_fullname: str,
    ) -> List[RedditComment]:
        out: List[RedditComment] = []
        for child in listing_children:
            kind = child.get("kind")
            if kind == "more":
                continue
            if kind != "t1":
                continue
            cd = child.get("data", {}) or {}
            cid = str(cd.get("id", "") or "")
            if not cid:
                continue

            score = int(cd.get("score", 0) or 0)
            if score < COMMENT_MIN_VOTES:
                # still recurse into replies to avoid losing children that qualify
                pass

            permalink = cd.get("permalink", "") or ""
            replies_data = cd.get("replies", "")
            reply_count = self._count_direct_replies(replies_data)

            if score >= COMMENT_MIN_VOTES:
                out.append(
                    RedditComment(
                        id=cid,
                        post_id=post.id,
                        content=cd.get("body", "") or "",
                        url=f"{REDDIT_WWW_URL}{permalink}" if permalink else post.url,
                        votes=score,
                        reply_count=reply_count,
                        created_utc=float(cd.get("created_utc", 0) or 0),
                        search_keyword=post.search_keyword,
                        source_sort=source_sort,
                        parent_fullname=str(cd.get("parent_id") or parent_fullname),
                        depth=depth,
                    )
                )

            if replies_data and not isinstance(replies_data, str):
                child_parent = f"t1_{cid}"
                out.extend(
                    self._parse_comments_recursive(
                        listing_children=replies_data.get("data", {}).get("children", []),
                        post=post,
                        source_sort=source_sort,
                        depth=depth + 1,
                        parent_fullname=child_parent,
                    )
                )
        return out

    async def _fetch_morechildren(
        self,
        *,
        post_id: str,
        children_ids: List[str],
        sort: str,
        retries: int = 3,
    ) -> Optional[Dict[str, Any]]:
        if not children_ids:
            return None
        params = {
            "link_id": f"t3_{post_id}",
            "children": ",".join(children_ids),
            "sort": sort,
            "api_type": "json",
            "raw_json": 1,
        }

        data = await self._fetch_json(f"{REDDIT_BASE_URL}/api/morechildren.json", params=params, retries=retries)
        if data is not None:
            return data
        return await self._fetch_json(f"{REDDIT_WWW_URL}/api/morechildren.json", params=params, retries=retries)

    def _parse_morechildren_things(
        self,
        *,
        things: Iterable[Dict[str, Any]],
        post: RedditPost,
        source_sort: str,
    ) -> Tuple[List[RedditComment], List[str]]:
        comments: List[RedditComment] = []
        more_ids: List[str] = []

        for thing in things:
            kind = thing.get("kind")
            data = thing.get("data", {}) or {}
            if kind == "t1":
                cid = str(data.get("id", "") or "")
                if not cid:
                    continue
                score = int(data.get("score", 0) or 0)
                if score < COMMENT_MIN_VOTES:
                    continue
                permalink = data.get("permalink", "") or ""
                replies_data = data.get("replies", "")
                reply_count = self._count_direct_replies(replies_data)
                comments.append(
                    RedditComment(
                        id=cid,
                        post_id=post.id,
                        content=data.get("body", "") or "",
                        url=f"{REDDIT_WWW_URL}{permalink}" if permalink else post.url,
                        votes=score,
                        reply_count=reply_count,
                        created_utc=float(data.get("created_utc", 0) or 0),
                        search_keyword=post.search_keyword,
                        source_sort=source_sort,
                        parent_fullname=str(data.get("parent_id") or ""),
                        depth=None,  # not always available
                    )
                )
                if replies_data and not isinstance(replies_data, str):
                    more_ids.extend(
                        self._collect_more_children_ids(replies_data.get("data", {}).get("children", []))
                    )
            elif kind == "more":
                more_ids.extend([str(x) for x in (data.get("children") or []) if x])

        return comments, more_ids

    async def fetch_post_comments(
        self,
        post: RedditPost,
        *,
        sort: str = "best",
        expand_more: bool = True,
    ) -> List[RedditComment]:
        if self._use_pullpush:
            # PullPush comments endpoint returns all comments (incl. replies) for link_id.
            # Paginate by created_utc descending via `before`.
            comments: List[RedditComment] = []
            seen: Set[str] = set()
            before: Optional[int] = None
            pages = 0
            max_pages = 5000  # safety

            while pages < max_pages:
                data = await self._pullpush_search_comment(
                    link_id=post.id,
                    size=100,
                    sort="desc",
                    sort_type="created_utc",
                    before=before,
                )
                if not data or "data" not in data:
                    break
                items = data.get("data") or []
                if not items:
                    break
                pages += 1

                min_created: Optional[int] = None
                new_count = 0
                for cd in items:
                    cid = str(cd.get("id", "") or "")
                    if not cid or cid in seen:
                        continue
                    seen.add(cid)

                    score = int(cd.get("score", 0) or 0)
                    if score < COMMENT_MIN_VOTES:
                        continue

                    created_utc = int(cd.get("created_utc", 0) or 0)
                    if created_utc and (min_created is None or created_utc < min_created):
                        min_created = created_utc

                    body = cd.get("body") or cd.get("body_html") or ""
                    permalink = cd.get("permalink") or ""
                    url = f"{REDDIT_WWW_URL}{permalink}" if permalink else f"{REDDIT_WWW_URL}/comments/{post.id}/_/{cid}"
                    parent_id = cd.get("parent_id") or ""

                    comments.append(
                        RedditComment(
                            id=cid,
                            post_id=post.id,
                            content=str(body),
                            url=url,
                            votes=score,
                            reply_count=0,  # not provided by PullPush
                            created_utc=float(created_utc),
                            search_keyword=post.search_keyword,
                            source_sort=f"pullpush_{sort}",
                            parent_fullname=str(parent_id) if parent_id else None,
                            depth=None,
                        )
                    )
                    new_count += 1

                # if we didn't move, stop
                if min_created is None:
                    break
                before = min_created - 1

                # If fewer than requested or no new, likely end
                if len(items) < 100 or new_count == 0:
                    break

            self._log(f"[pullpush] post {post.id}: comments fetched={len(comments)} pages={pages}")
            return comments

        url = f"{REDDIT_BASE_URL}/comments/{post.id}.json"
        params = {"sort": sort, "limit": 500, "raw_json": 1}

        data = await self._fetch_json(url, params=params)
        if not data or not isinstance(data, list) or len(data) < 2:
            return []

        listing_children = data[1].get("data", {}).get("children", [])
        comments = self._parse_comments_recursive(
            listing_children=listing_children,
            post=post,
            source_sort=sort,
            depth=1,
            parent_fullname=f"t3_{post.id}",
        )

        if not expand_more:
            return comments

        seen: Set[str] = set(c.id for c in comments if c.id)
        pending_more = self._collect_more_children_ids(listing_children)
        pending_more = list(dict.fromkeys([x for x in pending_more if x]))

        batch_size = 80
        rounds = 0
        while pending_more:
            rounds += 1
            batch = pending_more[:batch_size]
            pending_more = pending_more[batch_size:]

            resp = await self._fetch_morechildren(post_id=post.id, children_ids=batch, sort=sort)
            if not resp:
                continue

            things = resp.get("json", {}).get("data", {}).get("things", [])
            new_comments, new_more = self._parse_morechildren_things(
                things=things,
                post=post,
                source_sort=sort,
            )

            for c in new_comments:
                if c.id and c.id not in seen:
                    seen.add(c.id)
                    comments.append(c)

            for mid in new_more:
                if mid:
                    pending_more.append(mid)

            if rounds >= 5000:
                self._log(f"morechildren expansion hit safety cap for post={post.id}", "warning")
                break

        return comments

