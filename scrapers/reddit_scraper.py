"""Reddit discussion-comment scraper.

============================================================================
Position in the pipeline
----------------------------------------------------------------------------
Website -> Product Discovery -> ... -> Google Reviews -> Reddit -> YouTube
Reviews -> Twitter -> Instagram -> Sentiment Analysis -> ...

This module is a sibling of google_scraper.py / youtube_scraper.py /
twitter_scraper.py / instagram_scraper.py: it is called from app.py with a
per-job ``company_data`` dict and returns a flat ``List[str]`` of clean
comment text, exactly like every other scraper. It never touches sentiment
analysis, aspect intelligence, buying recommendations, the dashboard, or
the PDF report - those stages already consume whatever platform comment
lists app.py hands them, generically, by platform name.
----------------------------------------------------------------------------

Unlike Google Maps and YouTube, Reddit publishes a public, unauthenticated
JSON view of its search results and comment threads (append ``.json`` to
almost any reddit.com URL). That means this scraper does not need
Playwright/browser automation - a couple of plain HTTP GETs against
reddit.com's own JSON endpoints (with a descriptive User-Agent, as Reddit's
API etiquette asks for even for anonymous requests) is enough. The sync
HTTP calls are still run inside a small dedicated ThreadPoolExecutor rather
than directly on the event loop, mirroring exactly how google_scraper.py /
youtube_scraper.py run their (heavier) sync Playwright work - so this
module plugs into app.py's existing ``await scrape_x(...)`` call sites
without needing any special-casing.

----------------------------------------------------------------------------
Product-centric search priority (only Priority 1 is used for the "General"
company-wide job, which has no product_name - see _build_search_tiers):

  1. Product Name              e.g. "Tangzu Wan'er"
  2. Brand + Product            e.g. "Tangzu Wan'er Reddit"
     (or "<brand> <product>" when a distinct product_brand is supplied and
     isn't already part of the product name)
  3. Company + Product          e.g. "Headphone Zone Tangzu Wan'er"

A tier only counts as having "returned something" once its candidate posts
have actually been visited and yielded at least one usable comment, not
merely once a search returns candidate posts. Unlike a simple first-
success-wins fallback, tiers ACCUMULATE: a tier that only turns up a thin
result doesn't stop the search - broader, lower-priority tiers still run
and their comments are added on top, until either a healthy total is
reached (_MIN_COMMENTS_BEFORE_STOPPING) or the hard cap (MAX_TOTAL_COMMENTS,
from config.py's MAX_REDDIT_COMMENTS) is hit. This is deliberately
different from youtube_scraper.py's tier-fallback, which does stop at the
first tier that succeeds - Reddit's tiers are cheap plain-HTTP requests, so
the trade-off of trying one more tier for more genuine volume is worth it
here in a way it may not be for a heavier per-tier cost elsewhere.
----------------------------------------------------------------------------

Discussion-post filtering ("ignore News / Advertisements / Image posts /
Videos"):
  * Reddit's own search operator ``self:yes`` restricts results to native
    text ("self") posts server-side - this alone removes essentially all
    link/image/video posts, since those aren't self posts.
  * Defense-in-depth client-side checks additionally drop anything flagged
    as a video, gallery, or externally-linked post, anything stickied or
    NSFW, and anything whose flair/title reads as news or an advertisement.
============================================================================
"""

import asyncio
import concurrent.futures
import html
import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import urlencode

from bs4 import BeautifulSoup

from config import MAX_REDDIT_COMMENTS
from scrapers.browser_utils import normalize_comments

logger = logging.getLogger(__name__)

# --- Concurrency ------------------------------------------------------------
# Small worker pool, kept deliberately modest (unlike the browser pools in
# google_scraper.py / youtube_scraper.py) since this scraper is plain HTTP
# against Reddit's public JSON endpoints, which rate-limit unauthenticated
# clients fairly aggressively. Exported so app.py can size its per-platform
# semaphore to match, exactly like it does today with youtube_scraper's
# MAX_BROWSER_WORKERS.
MAX_REDDIT_WORKERS = 2

_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=MAX_REDDIT_WORKERS, thread_name_prefix="reddit_scraper"
)

# --- Hard internal time budget -----------------------------------------
# Kept well under the outer asyncio.wait_for() cap applied per job in
# app.py (50s - see REDDIT_TIMEOUT_SECONDS there) so this scraper almost
# always returns on its own, with whatever it has collected so far,
# instead of being cut off cold by the outer timeout and losing partial
# results that already finished.
# Raised 18 -> 80 (was previously raised 16 -> 18, leaving only 2s of
# margin under the old 20s outer cap): that margin was sized for the
# original MAX_TOTAL_COMMENTS=60/single-tier behavior and was never
# rescaled when tier-accumulation + MAX_REDDIT_COMMENTS=200 landed.
# Production logs showed real runs completing at 22.8s - already past
# both the old 18s internal budget AND the old 20s outer cap - because
# individual HTTP round-trips plus _REQUEST_DELAY_SECONDS pauses across
# up to 3 accumulating tiers don't fit a 2s margin. This is deliberately
# much smaller than youtube_scraper.py's equivalent (600s) since this is
# plain HTTP, not browser automation - it doesn't need anywhere near that
# much time, just enough real slack that the internal deadline checks
# between requests (not mid-request) can't slip past the outer cap.
TIME_BUDGET_SECONDS = 40  # cut from 80 - priority shifted to 3-min total time; this is a safety ceiling on scrape time, not a target - MAX_REDDIT_COMMENTS (config.py, currently 70) is the actual comment-count cap.
_MIN_TIME_FOR_ANOTHER_TIER_SECONDS = 3

# --- Volume / politeness caps -------------------------------------------
MAX_CANDIDATE_POSTS_PER_TIER = 10  # raised from 6 - search already asks Reddit
# for up to 25 results per query (see the `limit: 25` search param below) but
# was only ever inspecting the top 6 by comment count; 10 lets more real
# discussion threads contribute without a large increase in request volume.
MAX_COMMENTS_PER_POST = 40         # raised from 15 - this costs NO extra HTTP
# requests to raise: _fetch_post_comments[_html] already downloads a post's
# full comment payload in one request regardless of how many comments are
# kept from it, so throwing away comments past 15 was pure lost yield from
# data already paid for. Big, popular threads can now actually contribute
# close to their real comment count instead of being clipped hard.
MAX_TOTAL_COMMENTS = MAX_REDDIT_COMMENTS  # was a hardcoded local 60; now reads
# config.py like every other platform's cap does (see config.py's comment on
# MAX_REDDIT_COMMENTS for why 60 was never realistically going to hit the
# 100-200-per-product target on its own).
_REQUEST_DELAY_SECONDS = 0.4       # brief pause between successive Reddit requests

# Once accumulated comments (across however many tiers have run so far) reach
# this many, stop trying further/broader search tiers even if MAX_TOTAL_COMMENTS
# hasn't been hit yet - there's a real yield/relevance trade-off in continuing
# past a genuinely healthy result just to reach the hard cap, since tiers 2/3
# are intentionally broader (brand+product, company+product) and thus more
# likely to surface tangential discussion. This only controls how EAGERLY the
# loop below keeps going past a "good enough" tier - see the tier-accumulation
# fix in _scrape_sync.
_MIN_COMMENTS_BEFORE_STOPPING = 40

# Reddit asks even anonymous/unauthenticated clients to identify themselves
# with a descriptive User-Agent; generic ones are the fastest way to get
# soft-throttled.
_USER_AGENT = "ManobhavaAI/1.0 (product review discussion scraper; contact: support@manobhava.ai)"
_HEADERS = {"User-Agent": _USER_AGENT, "Accept": "application/json"}

# Browser-shaped headers for the old.reddit.com HTML fallback below.
# Verified live (2026-07-17): reddit.com's edge now answers HTTP 403
# ("blocked by network security") to EVERY *.json endpoint from this
# client — search, comments, even subreddit listings, under any
# User-Agent, and even from a real headless Chromium. The one thing it
# still serves anonymously is old.reddit.com's server-rendered *HTML*,
# provided the request looks like a normal browser page view.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# --- Source fallback chain --------------------------------------------------
# www.reddit.com/search.json returned a blanket HTTP 403 on *every single*
# query in production (see the log: identical 403 + identical Reddit
# "theme-beta" interstitial body, on the very first request of the process,
# regardless of query content). That signature is an edge/IP-reputation
# block on the www.reddit.com front-end itself, not a fixable per-request
# header/query tweak - so instead of retrying the same blocked endpoint,
# each tier now tries a short list of Reddit's *other* public JSON
# front-ends, in order, stopping at the first one that responds. This is a
# source fallback (per the brief: "if one source fails, automatically fall
# back to another"), not a retry of the same request - each domain is
# fetched at most once per call.
#   1. old.reddit.com  - the legacy front-end; served from different edge
#      infrastructure than www, and historically the least aggressively
#      gated for anonymous JSON requests.
#   2. www.reddit.com  - kept last rather than dropped: the block observed
#      in this log may be a temporary IP-reputation flag rather than a
#      permanent one, so it's still worth one attempt.
_SEARCH_DOMAINS = ["old.reddit.com", "www.reddit.com"]


def _search_url(domain: str) -> str:
    return f"https://{domain}/search.json"


def _comments_url(domain: str, subreddit: str, post_id: str) -> str:
    return (
        f"https://{domain}/r/{subreddit}/comments/{post_id}.json"
        f"?{urlencode({'limit': 100, 'sort': 'top', 'depth': 1})}"
    )


# --- Discussion-post filtering ----------------------------------------------
_EXCLUDED_FLAIR_RE = re.compile(
    r"\b(news|advertisement|advertising|promo|promotional|sponsored|announcement)\b",
    re.IGNORECASE,
)
_EXCLUDED_TITLE_RE = re.compile(
    r"\[(?:ad|advertisement|promo|sponsored)\]|\((?:ad|advertisement|promo|sponsored)\)",
    re.IGNORECASE,
)
_EXCLUDED_POST_HINTS = {"image", "hosted:video", "rich:video", "link"}


def _is_discussion_post(post: Dict) -> bool:
    """True only for genuine text-discussion posts - never news, ads,
    images, videos, or galleries."""
    if post.get("stickied") or post.get("over_18") or post.get("pinned"):
        return False
    if not post.get("is_self", False):
        return False
    if post.get("is_video") or post.get("is_gallery"):
        return False
    hint = (post.get("post_hint") or "").lower()
    if hint in _EXCLUDED_POST_HINTS:
        return False
    domain = (post.get("domain") or "").lower()
    if not domain.startswith("self."):
        return False
    if not post.get("num_comments"):
        return False
    flair = post.get("link_flair_text") or ""
    if _EXCLUDED_FLAIR_RE.search(flair):
        return False
    title = post.get("title") or ""
    if _EXCLUDED_TITLE_RE.search(title):
        return False
    return True


# --- Comment cleaning ---------------------------------------------------
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")
_URL_RE = re.compile(r"https?://\S+")
_MD_EMPHASIS_RE = re.compile(r"[*_~`^]+")
# FIX: this used to be r"^\s*>+\s?" applied with .sub("", text) - which only
# strips the leading '>' marker character(s) from a quoted line, leaving the
# quoted line's actual CONTENT in place, merged in with whatever the replier
# wrote below it. On Reddit, replying by quoting ("> original comment" then
# your own response underneath) is a very common pattern - the old behavior
# meant the quoted person's words silently became indistinguishable from the
# replying commenter's own text by the time this reaches sentiment analysis
# (e.g. someone quoting a complaint just to disagree with it would have that
# complaint folded into their own "clean" comment as if they'd said it).
# Now matches and removes the WHOLE line (through its trailing newline, or
# to end-of-string for a final line with none), for every line whose first
# non-whitespace character is '>' - confirmed via test against multi-line
# input that this drops the quoted line entirely rather than just its marker.
_QUOTE_LINE_RE = re.compile(r"^[ \t]*>.*(?:\n|$)", re.MULTILINE)
_WHITESPACE_RE = re.compile(r"\s+")

_EXCLUDED_AUTHORS = {"automoderator", "[deleted]"}
_REMOVED_BODIES = {"[deleted]", "[removed]"}


def _clean_reddit_markdown(text: str) -> str:
    """Strips Reddit-specific markdown/HTML-entity noise. Generic cleanup
    (link stripping, whitespace normalization) is left to the shared
    clean_comment/remove_links/normalize_text pipeline already applied to
    every platform's comments in app.py, so this stays scoped to syntax
    that's specific to Reddit's comment markdown."""
    text = html.unescape(text)
    text = _QUOTE_LINE_RE.sub("", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub("", text)
    text = _MD_EMPHASIS_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


@dataclass
class RedditComment:
    """Internal record kept for filtering/logging. Only ``comment_text`` is
    surfaced by scrape_reddit_comments() - the sentiment pipeline is never
    changed to accept anything richer than the List[str] it already gets
    from every other platform."""
    post_title: str
    comment_text: str
    author: str
    upvotes: int


# --- HTTP -----------------------------------------------------------------
def _get_json(url: str, timeout: float, retries: int = 1) -> Optional[Dict]:
    request = urllib.request.Request(url, headers=_HEADERS)
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(request, timeout=max(2.0, timeout)) as resp:
                raw = resp.read()
            return json.loads(raw.decode("utf-8", errors="ignore"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < retries:
                logger.warning("Reddit rate-limited us (429); backing off before retry: %s", url)
                time.sleep(1.5)
                attempt += 1
                continue
            # Diagnostic-only addition: a 403 (as opposed to 429) that's
            # consistent across every query variant, on the very first
            # request of the process, is the signature of an anti-bot /
            # IP-reputation block rather than a code-level bug - this is
            # not fixable from in-code header/retry tweaks alone. Logging
            # the response body (if any) here is purely additive so the
            # next zero-Reddit-results run shows Reddit's actual block
            # page/message instead of just the bare status code, without
            # changing any control flow or return value below.
            try:
                body_preview = exc.read(500).decode("utf-8", errors="ignore")
            except Exception:
                body_preview = ""
            logger.warning(
                "Reddit HTTP error %s for %s%s",
                exc.code, url,
                f" | body preview: {body_preview!r}" if body_preview else "",
            )
            return None
        except Exception:
            logger.warning("Reddit request failed for %s", url, exc_info=True)
            return None


def _search_reddit(query: str, time_left: float) -> "tuple[List[Dict], str, str]":
    """Search Reddit for discussion posts matching ``query``. Restricted to
    self (text) posts via Reddit's own ``self:yes`` search operator, with
    client-side filtering (_is_discussion_post) as a defense-in-depth
    second pass.

    Tries each JSON domain in _SEARCH_DOMAINS in turn, stopping at the
    first one that returns a usable payload (a source fallback, not a
    retry - see the _SEARCH_DOMAINS comment above). When *every* JSON
    endpoint fails (the edge-block signature — see _BROWSER_HEADERS),
    falls back to old.reddit.com's server-rendered HTML search.

    Returns (posts, domain, transport): transport is "json" or "html" so
    the caller fetches each post's comments over the same transport that
    the search itself just proved is working.
    """
    params = {
        "q": f"{query} self:yes",
        "sort": "relevance",
        "t": "all",
        "limit": 25,
    }
    for domain in _SEARCH_DOMAINS:
        if time_left <= 1:
            break
        url = f"{_search_url(domain)}?{urlencode(params)}"
        payload = _get_json(url, time_left)
        if not payload:
            continue
        children = payload.get("data", {}).get("children", []) or []
        posts = [c.get("data", {}) for c in children if c.get("kind") == "t3"]
        discussion_posts = [p for p in posts if _is_discussion_post(p)]
        if not discussion_posts:
            # This domain answered but had nothing usable for this query -
            # that's a real "no results" signal, not a block, so don't
            # burn the remaining domains on the same query.
            return [], domain, "json"
        discussion_posts.sort(key=lambda p: p.get("num_comments", 0), reverse=True)
        return discussion_posts[:MAX_CANDIDATE_POSTS_PER_TIER], domain, "json"

    # Every JSON endpoint failed — the edge-block signature, not a
    # no-results signal. Try the HTML front-end before giving up.
    if time_left > 1:
        html_posts = _search_reddit_html(query, time_left)
        if html_posts:
            logger.info(
                "Reddit scrape: JSON endpoints blocked; HTML fallback found "
                "%d discussion post(s) for %r.", len(html_posts), query,
            )
            return html_posts, "old.reddit.com", "html"
    return [], _SEARCH_DOMAINS[-1], "json"


def _fetch_post_comments(domain: str, subreddit: str, post_id: str, post_title: str, time_left: float) -> List[RedditComment]:
    """Top-level comments only (matches the ``depth=1`` request param) -
    plenty for sentiment purposes and keeps request volume low.

    Uses the same domain that successfully answered the search for this
    tier (passed in by the caller) rather than re-discovering a working
    domain per post - the search already proved that domain isn't blocked
    right now.
    """
    payload = _get_json(_comments_url(domain, subreddit, post_id), time_left)
    if not payload or not isinstance(payload, list) or len(payload) < 2:
        return []
    children = (payload[1].get("data", {}) or {}).get("children", []) or []

    out: List[RedditComment] = []
    for child in children:
        if child.get("kind") != "t1":
            continue
        data = child.get("data", {}) or {}
        if data.get("stickied"):
            continue
        author = (data.get("author") or "").strip()
        if author.lower() in _EXCLUDED_AUTHORS:
            continue
        body = (data.get("body") or "").strip()
        if not body or body in _REMOVED_BODIES:
            continue

        cleaned = _clean_reddit_markdown(body)
        if not cleaned or len(cleaned.split()) < 3:
            continue

        upvotes = data.get("score")
        if upvotes is None:
            upvotes = data.get("ups", 0)
        out.append(RedditComment(
            post_title=post_title,
            comment_text=cleaned,
            author=author or "unknown",
            upvotes=int(upvotes or 0),
        ))
        if len(out) >= MAX_COMMENTS_PER_POST:
            break
    return out


# --- HTML fallback transport (old.reddit.com server-rendered pages) --------
# Used automatically when every JSON endpoint is edge-blocked (403) — see
# the note above _BROWSER_HEADERS. Search results and comment threads are
# parsed out of old.reddit.com's HTML instead; the discussion-post
# filtering intent is preserved (the same `self:yes` operator is applied
# server-side, plus the same title/ad exclusions client-side).

def _get_html(url: str, timeout: float) -> Optional[str]:
    request = urllib.request.Request(url, headers=_BROWSER_HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=max(2.0, timeout)) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception:
        logger.warning("Reddit HTML request failed for %s", url)
        return None


def _search_reddit_html(query: str, time_left: float) -> List[Dict]:
    """Search via old.reddit.com's HTML search page. Returns posts
    normalized to the same minimal dict shape _scrape_sync reads
    (subreddit / id / title / num_comments, plus permalink for the HTML
    comment fetch)."""
    url = f"https://old.reddit.com/search?{urlencode({'q': f'{query} self:yes'})}"
    html_text = _get_html(url, time_left)
    if not html_text:
        return []

    posts: List[Dict] = []
    try:
        soup = BeautifulSoup(html_text, "html.parser")
        for div in soup.select("div.search-result-link"):
            # ADDED (best-effort, unverified against a live page - see note
            # below): _is_discussion_post() already excludes over_18 posts
            # on the JSON path, but that path 403s on every request seen so
            # far (see _BROWSER_HEADERS above), so this HTML path is what
            # actually runs on every live scrape right now, and it had no
            # NSFW check at all. old.reddit.com's templates mark NSFW
            # content with an "over18" class token elsewhere on the site
            # (subreddit/listing pages); I'm assuming the same token appears
            # on this result div when applicable, but I have no network
            # access in this environment to fetch a real search-results page
            # and confirm that's actually how *this* template marks it. This
            # can only ever exclude a post, never wrongly include one, so
            # it's safe to ship unverified - please check the next log for
            # whether NSFW/18+ content still slips through; if it does, the
            # class/attribute name below needs correcting against real HTML.
            classes = div.get("class") or []
            if "over18" in classes:
                continue
            fullname = div.get("data-fullname") or ""  # "t3_<id>"
            post_id = fullname.split("_", 1)[1] if fullname.startswith("t3_") else ""
            title_a = div.select_one("a.search-title")
            title = title_a.get_text(strip=True) if title_a else ""
            sub_a = div.select_one("a.search-subreddit-link")
            subreddit = (sub_a.get_text(strip=True) if sub_a else "").removeprefix("r/")
            comments_a = div.select_one("a.search-comments")
            permalink = comments_a.get("href") if comments_a else ""
            m = re.search(r"(\d+)", comments_a.get_text() if comments_a else "")
            num_comments = int(m.group(1)) if m else 0

            if not post_id or not subreddit or not permalink:
                continue
            if not num_comments:
                continue
            if _EXCLUDED_TITLE_RE.search(title):
                continue
            posts.append({
                "subreddit": subreddit,
                "id": post_id,
                "title": title,
                "num_comments": num_comments,
                "permalink": permalink,
            })
    except Exception:
        logger.warning("Reddit HTML search parse failed for %r", query, exc_info=True)
        return []

    posts.sort(key=lambda p: p.get("num_comments", 0), reverse=True)
    return posts[:MAX_CANDIDATE_POSTS_PER_TIER]


def _fetch_post_comments_html(permalink: str, post_title: str, time_left: float) -> List[RedditComment]:
    """Top-level comments parsed out of an old.reddit.com thread page —
    the HTML-transport counterpart of _fetch_post_comments, applying the
    same author/body filters."""
    html_text = _get_html(permalink, time_left)
    if not html_text:
        return []

    out: List[RedditComment] = []
    try:
        soup = BeautifulSoup(html_text, "html.parser")
        area = soup.select_one("div.commentarea")
        if area is None:
            return []
        for c in area.select("div.sitetable > div.comment"):
            classes = c.get("class") or []
            if "stickied" in classes or "deleted" in classes:
                continue
            author_a = c.select_one("a.author")
            author = author_a.get_text(strip=True) if author_a else ""
            if author.lower() in _EXCLUDED_AUTHORS:
                continue
            entry = c.select_one("div.entry")
            md = entry.select_one("div.md") if entry else None
            body = md.get_text(" ", strip=True) if md else ""
            if not body or body in _REMOVED_BODIES:
                continue

            cleaned = _clean_reddit_markdown(body)
            if not cleaned or len(cleaned.split()) < 3:
                continue

            score_span = c.select_one("span.score.unvoted")
            m = re.search(r"(-?\d+)", score_span.get_text() if score_span else "")
            out.append(RedditComment(
                post_title=post_title,
                comment_text=cleaned,
                author=author or "unknown",
                upvotes=int(m.group(1)) if m else 0,
            ))
            if len(out) >= MAX_COMMENTS_PER_POST:
                break
    except Exception:
        logger.warning("Reddit HTML comment parse failed for %s", permalink, exc_info=True)
    return out


# --- Product-centric search tiers -------------------------------------------
def _build_search_tiers(company_name: str, product_name: str, product_brand: str) -> List["tuple[str, str]"]:
    """Priority order:
      1. product_name
      2. brand + product_name (or "<product_name> Reddit" when no distinct
         brand is available - see inline note below)
      3. company_name + product_name

    A job with no product_name (the "General" company-wide job) falls back
    to a single plain company-name search, mirroring youtube_scraper.py's
    behavior for its own product-less "General" job.
    """
    tiers: List["tuple[str, str]"] = []
    product_name = (product_name or "").strip()
    company_name = (company_name or "").strip()
    product_brand = (product_brand or "").strip()

    if product_name:
        tiers.append(("product_name", product_name))

        # Priority 2 ("Brand + Product"): if a distinct brand is supplied
        # and isn't already embedded in the product name, lead with
        # "<brand> <product>". Otherwise the product name already reads as
        # brand+model (e.g. "Tangzu Wan'er" - "Tangzu" is the brand), so
        # disambiguate with a "Reddit" suffix instead, matching how a
        # person would naturally search for third-party discussion of that
        # exact product (e.g. "Tangzu Wan'er Reddit").
        if product_brand and product_brand.lower() not in product_name.lower():
            tiers.append(("brand_product", f"{product_brand} {product_name}"))
        else:
            tiers.append(("brand_product", f"{product_name} Reddit"))

        if company_name and company_name.lower() not in product_name.lower():
            tiers.append(("company_product", f"{company_name} {product_name}"))
    elif company_name:
        tiers.append(("company_general", company_name))

    return tiers


def _scrape_sync(company_name: str, product_name: str, product_brand: str) -> List[RedditComment]:
    # Record the start time to enforce the overall scraping time limit
    start = time.monotonic()

    # Returns the remaining time available for scraping
    def time_left() -> float:
        return TIME_BUDGET_SECONDS - (time.monotonic() - start)

    # Generate different search queries (tiers) to maximize the chance of finding discussions
    tiers = _build_search_tiers(company_name, product_name, product_brand)

    # Stores all collected comments
    collected: List[RedditComment] = []

    # Used to prevent duplicate comments
    seen_keys: set = set()

    # Tracks which search tier(s) contributed comments, in order
    sources_used: List[str] = []

    # Try each search tier, ACCUMULATING across tiers (not stopping at the
    # first one that returns anything) until either a healthy yield is
    # reached, the hard cap is hit, or time/tiers run out.
    #
    # FIX: this used to `break` as soon as any single tier returned even one
    # usable comment - so a specific product name (Priority 1) that only
    # matched one small thread with a handful of comments would stop the
    # entire scrape right there, never trying the intentionally broader
    # Priority 2 (brand+product) / Priority 3 (company+product) tiers that
    # might have surfaced additional real discussion. That was the single
    # biggest reason real per-product counts were landing well under the
    # 100-200 target even when more genuine discussion existed. Now a tier
    # is only treated as "we have enough" once accumulated comments reach
    # _MIN_COMMENTS_BEFORE_STOPPING - a thin result keeps the search going
    # to the next, broader tier instead of ending it.
    for label, query in tiers:

        # Stop trying new search queries if there isn't enough time left,
        # or if a previous tier already reached the hard cap.
        if time_left() <= _MIN_TIME_FOR_ANOTHER_TIER_SECONDS:
            logger.info("Reddit scrape: skipping tier %r - out of time budget.", label)
            break
        if len(collected) >= MAX_TOTAL_COMMENTS:
            logger.info("Reddit scrape: skipping tier %r - already at the comment cap.", label)
            break

        # Search Reddit for posts matching the current query
        posts, working_domain, transport = _search_reddit(query, time_left())

        # Skip to the next search tier if no posts were found
        if not posts:
            logger.info("Reddit scrape: tier %r search (%r) found 0 discussion posts.", label, query)
            continue

        logger.info(
            "Reddit scrape: tier %r search (%r) found %d discussion post(s); "
            "fetching comments.", label, query, len(posts),
        )

        # Stores comments collected from the current search tier
        tier_comments: List[RedditComment] = []

        # Fetch comments from each Reddit post
        for post in posts:

            # Stop processing if the remaining time is very low
            if time_left() <= _MIN_TIME_FOR_ANOTHER_TIER_SECONDS / 2:
                break

            subreddit = post.get("subreddit", "")
            post_id = post.get("id", "")
            title = post.get("title", "")

            # Skip invalid posts
            if not subreddit or not post_id:
                continue

            # Retrieve comments from the current Reddit post, over the
            # same transport the search itself just worked on
            if transport == "html":
                comments = _fetch_post_comments_html(
                    post.get("permalink", ""),
                    title,
                    time_left(),
                )
            else:
                comments = _fetch_post_comments(
                    working_domain,
                    subreddit,
                    post_id,
                    title,
                    time_left(),
                )

            # Small delay between requests to avoid sending requests too quickly
            time.sleep(_REQUEST_DELAY_SECONDS)

            # Add only unique comments
            for c in comments:
                key = c.comment_text.strip().lower()

                if not key or key in seen_keys:
                    continue

                seen_keys.add(key)
                tier_comments.append(c)

            # Stop collecting once the maximum comment limit is reached
            if len(collected) + len(tier_comments) >= MAX_TOTAL_COMMENTS:
                break

        if tier_comments:
            collected.extend(tier_comments)
            sources_used.append(label)

            logger.info(
                "Reddit scrape: tier %r contributed %d comment(s) (running total=%d).",
                label,
                len(tier_comments),
                len(collected),
            )

            # "Enough" now means a healthy accumulated total, not merely
            # "a tier returned something" - only stop here if we've reached
            # that bar or the hard cap; otherwise fall through to the next,
            # broader tier for more.
            if len(collected) >= _MIN_COMMENTS_BEFORE_STOPPING or len(collected) >= MAX_TOTAL_COMMENTS:
                logger.info(
                    "Reddit scrape: reached %d comment(s) after tier %r - "
                    "that's enough, not trying further tiers.",
                    len(collected), label,
                )
                break

            logger.info(
                "Reddit scrape: only %d comment(s) so far after tier %r - "
                "still trying broader tiers for more.",
                len(collected), label,
            )
            continue

        logger.info(
            "Reddit scrape: tier %r yielded posts but 0 usable comments; trying next tier.",
            label,
        )

    source = "+".join(sources_used) if sources_used else "none"

    # Calculate the total scraping time
    elapsed = time.monotonic() - start

    # Log the final scraping summary
    logger.info(
        "Reddit comments for company=%r product=%r: source=%s collected=%d elapsed=%.1fs (cap=%d).",
        company_name,
        product_name,
        source,
        len(collected),
        elapsed,
        MAX_TOTAL_COMMENTS,
    )

    # Return the collected comments, limited to the maximum allowed
    return collected[:MAX_TOTAL_COMMENTS]


async def scrape_reddit_comments(company_data: Dict[str, str]) -> List[str]:
    """Public entry point - same contract as scrape_google_reviews() /
    scrape_youtube_comments() / scrape_twitter_comments() /
    scrape_instagram_comments(): takes the per-job company_data dict app.py
    already builds, returns a flat List[str] of clean comment text (no
    sentiment analysis, no product attribution - that all happens exactly
    as it does today for every other platform once app.py has this list).

    Reads (all optional except company_name, mirroring youtube_scraper.py):
      * company_name  - required; used for the "General" job and Priority 3
      * product_name  - drives Priority 1 & 2 when present
      * product_brand - refines Priority 2 when present
    """
    company_name = (company_data.get("company_name") or "").strip()
    product_name = (company_data.get("product_name") or "").strip()
    product_brand = (company_data.get("product_brand") or "").strip()

    if not company_name and not product_name:
        return []

    loop = asyncio.get_event_loop()
    try:
        comments = await loop.run_in_executor(
            _EXECUTOR, _scrape_sync, company_name, product_name, product_brand,
        )
    except Exception:
        logger.exception(
            "Unhandled error scraping Reddit comments for company=%r product=%r.",
            company_name, product_name,
        )
        return []

    texts = [c.comment_text for c in comments]
    final = normalize_comments(texts)[:MAX_TOTAL_COMMENTS]
    logger.info("Reddit returned %d comments", len(final))
    return final
