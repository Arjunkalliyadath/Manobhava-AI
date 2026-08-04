"""
Module Name
-----------
scrapers/youtube_scraper.py

Purpose
-------
Collects genuine, product-specific comments from YouTube: comments left
on videos that actually discuss the selected product, prioritizing the
company's own official channel and review-style videos over unrelated
brand-wide content.

Responsibilities
-----------------
- `scrape_youtube_comments(company_data)`: the single public entry point.
  Runs a set of product-centric discovery tiers (a company-wide "General"
  search plus one per selected product), ranks candidate videos, then
  visits enough of them to reach the configured comment target
  (`config.MAX_YOUTUBE_COMMENTS`) while removing duplicates.
- Gives the brand-wide "General" job a shorter time budget than
  product-specific jobs (see `GENERAL_YOUTUBE_TIMEOUT_SECONDS` in
  app.py / the per-job timeout logic here) since it is lower priority
  than a genuine per-product result.
- Attaches to the shared Chromium process via `scrapers.browser_utils`
  rather than launching its own, and logs `BROWSER_TRACE` timing at each
  setup checkpoint, same as google_scraper.py / twitter_scraper.py /
  instagram_scraper.py.
- Enforces one real wall-clock deadline anchored to the moment
  `scrape_youtube_comments()` itself starts (`OUTER_HARD_TIMEOUT_SECONDS`
  / `SAFETY_MARGIN_SECONDS`), so a slow setup phase can never silently eat
  into the time a caller expects the result back within — see the
  in-file comment on `OUTER_HARD_TIMEOUT_SECONDS` for the specific
  production incident (real comments collected, discarded because the
  coroutine had already been cancelled) that this fixed.

Dependencies
------------
`playwright` (sync API), `httpx`, plus standard library `asyncio`, `json`,
`re`, `threading`, `time`.
"""

import asyncio
import atexit
import concurrent.futures
import json
import logging
import math
import re
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import httpx

from scrapers.browser_utils import browser_manager, normalize_comments, find_social_profile_url
from config import (
    MAX_YOUTUBE_COMMENTS,
    MAX_SCROLL_ITERATIONS,
    SCROLL_IDLE_LIMIT,
    NAVIGATION_RETRIES,
    NAV_TIMEOUT_MS_MAX,
    NAV_TIMEOUT_MS_MIN,
)

logger = logging.getLogger(__name__)

# --- Hard internal time budget -----------------------------------------
# PREVIOUS BUG (same class of bug already fixed in twitter_scraper.py and
# instagram_scraper.py - see twitter_scraper.py's OUTER_HARD_TIMEOUT_SECONDS
# block for the full writeup): `deadline = start + TIME_BUDGET_SECONDS`
# below used to start a FRESH clock only after browser/context/page setup
# had already finished, with no memory of how much of the OUTER app.py
# wait_for() budget setup itself had already spent. Setup was regularly
# taking 22-46s+ under this app's shared-Chromium contention at the start
# of an analysis run (up to 3 YouTube jobs - "General" plus one per
# selected product - all launching their own Playwright driver at once,
# alongside Google/Twitter/Instagram doing the same - see BROWSER_TRACE
# logs), so "setup + a fresh TIME_BUDGET_SECONDS" routinely exceeded
# app.py's real 30s YOUTUBE_TIMEOUT_SECONDS cap. Production logs show
# exactly this: app.py logged "YouTube Scraper job for 'boAt Rockerz 255
# Pro+' exceeded its 30s hard timeout — returning []" (i.e. app.py's
# outer wait_for() already gave up and used [] as the result) while the
# *actual* _run_sync() kept executing unseen on its executor thread and
# later logged "YouTube comments for 'boAt Lifestyle' (product='boAt
# Rockerz 255 Pro+' ...): final=59" - 59 real comments, correctly
# collected, that never reached the dashboard because the coroutine that
# would have returned them had already been cancelled.
#
# Fixed the same way as the other two scrapers: one real wall-clock
# deadline anchored to the moment scrape_youtube_comments() itself starts
# (before the channel/handle resolution fallback), via
# OUTER_HARD_TIMEOUT_SECONDS / SAFETY_MARGIN_SECONDS, passed down into
# _run_sync() as `_overall_start` so it always leaves real margin for the
# result to make it back through run_in_executor()/wait_for(), no matter
# how long setup took. TIME_BUDGET_SECONDS is kept only as the
# "no contention at all" reference value the comments below still
# describe; deadline/time_left() (see _run_sync()) are what the code
# actually uses now.
OUTER_HARD_TIMEOUT_SECONDS = 90  # cut from 600 - priority shifted to 3-min total time. With MAX_YOUTUBE_COMMENTS now 70 (was 30 when this comment was first written), this should rarely be hit; kept as a safety ceiling, not a target.
# only ever enough time to visit ONE candidate video per product before the outer
# deadline hit - nowhere near enough to reach hundreds/thousands of comments across
# many videos. Getting real volume requires real wall-clock time; there's no way
# around that while scraping respectfully. Must be kept in sync with app.py's
# YOUTUBE_TIMEOUT_SECONDS (both call sites).
SAFETY_MARGIN_SECONDS = 20.0  # scaled up from 8.0 alongside the larger outer budget
MIN_USEFUL_BUDGET_SECONDS = 6.0  # below this, don't even attempt navigation

# Kept a few seconds under the outer asyncio.wait_for() cap applied in
# app.py (100s as of this fix - YOUTUBE_TIMEOUT_SECONDS there, kept in
# sync with OUTER_HARD_TIMEOUT_SECONDS=90 above plus buffer) so this
# scraper almost always returns on its own,
# with whatever it has collected so far, instead of being cut off cold by
# the outer timeout and losing partial results. NOTE: this budget is
# shared across all 3 product-centric search tiers (see "Product-centric
# discovery tiers" below) — a run that falls through 1-2 dead tiers has
# meaningfully less time left for the tier that finally succeeds than a
# single-tier flow would. Revisit this value if logs show later tiers
# frequently being skipped due to time (`_MIN_TIME_FOR_ANOTHER_TIER_SECONDS`
# below) even on runs that started with a full budget. NOTE: this constant
# itself is no longer what enforces the real deadline (see the
# OUTER_HARD_TIMEOUT_SECONDS block above) — it's kept only as the
# "no contention at all" reference value these comments describe.
TIME_BUDGET_SECONDS = 17

# How many candidate video URLs to gather from the channel/search page
# before visiting them for comments. Listing videos is cheap; visiting
# each one for comments is the expensive part, so we keep some candidates
# in reserve in case early videos have comments disabled or turn out to
# have little content.
VIDEO_CANDIDATES_TO_COLLECT = 50

# Overall cap on how many comments this scraper will try to collect across
# all videos for one company/channel, sourced from config so it can be
# tuned without touching this file.
MAX_TOTAL_COMMENTS = MAX_YOUTUBE_COMMENTS

# --- Product-centric discovery tiers ---------------------------------------
# When a product name is supplied, discovery/scraping proceeds through
# search tiers in priority order instead of going straight to the official
# company channel:
#   1. "<product name> review"
#   2. "<brand> <product name> review"   (brand = product_brand, or the
#      company name if no product-specific brand was supplied)
#   3. the official company channel, then (as today) a plain company-name
#      search if the channel itself has no videos.
#
# A tier only counts as successful once its videos have actually been
# visited and yielded at least one usable comment - not merely once a
# search has returned candidate video URLs. A tier whose videos all turn
# out to have comments disabled, be bot-walled, etc. is treated as a dead
# end and the next tier is tried automatically. This makes "stop at the
# first successful tier" cost more time in the worst case (a product with
# no scrapable comments anywhere may burn through all 3-4 tiers before
# giving up), which the existing time-budget checks below already bound
# gracefully - see TIME_BUDGET_SECONDS above.
_MIN_TIME_FOR_ANOTHER_TIER_SECONDS = 4

_VIDEO_ID_RE = re.compile(r'"videoId":"([\w-]{11})"')
# Shorts entries embed their video ID under a distinct "reelWatchEndpoint"
# key rather than the plain videoRenderer key that _VIDEO_ID_RE matches, so
# we can collect Shorts IDs separately and subtract them out. This is what
# lets the raw regex fallback (used when the /videos tab renders too little
# for the DOM-based selectors to find anything) avoid pulling in Shorts.
_SHORTS_ID_RE = re.compile(r'"reelWatchEndpoint":\{"videoId":"([\w-]{11})"')
_CONSENT_LABELS = ("Accept all", "I agree", "Accept the use of cookies", "Reject all")

# --- Structural selector fallbacks -----------------------------------------
# YouTube has shipped at least two different comment DOM shapes in
# production: the long-standing ytd-comment-renderer tree, and a newer
# "view model" rearchitecture (ytd-comment-view-model) that some sessions
# get instead. Relying on a single selector meant a whole scrape could
# silently return 0 comments on any session bucketed into the newer shape.
# Every comment-related locator below is a comma-joined CSS selector list
# (a native CSS/Playwright union — a match on any alternative counts)
# instead of one hardcoded string.
_COMMENT_TEXT_SELECTOR = (
    "ytd-comment-renderer #content-text, "
    "ytd-comment-view-model #content-text, "
    "ytd-comment-view-model #content-text span, "
    "ytd-comment-view-model .yt-core-attributed-string, "
    "ytd-comment-view-model yt-attributed-string, "
    "ytd-backstage-comment #content-text, "
    "#comment #content-text, "
    "ytd-comment-thread-renderer #content-text"
)
_COMMENTS_CONTAINER_SELECTOR = (
    "ytd-comments, #comments, ytd-item-section-renderer#sections, "
    "ytd-comment-thread-renderer, ytd-comments-header-renderer"
)

# "Read more" / "more replies" controls are scoped to *inside* the comments
# section (the leading "ytd-comments "). The unscoped selector this
# replaces, "ytd-expander #more", also matches the video description's own
# "...more" expander, which sits in an unrelated ytd-expander higher up the
# page — clicking it expanded the description panel instead of a truncated
# comment, i.e. exactly the "unrelated control" failure mode to avoid.
_READ_MORE_SELECTOR = (
    "ytd-comments ytd-expander #more, "
    "ytd-comments tp-yt-paper-button#more, "
    "ytd-comments ytd-comment-view-model ytd-expander #more, "
    "ytd-comments #more.ytd-expander"
)
_MORE_REPLIES_SELECTOR = (
    "ytd-comments ytd-comment-replies-renderer #more-replies, "
    "ytd-comments ytd-comment-replies-renderer tp-yt-paper-button#more-replies, "
    "ytd-comments ytd-comment-replies-renderer ytd-button-renderer#more-replies"
)

_VIDEO_TILE_WAIT_SELECTOR = "a#video-title, a#video-title-link, ytd-rich-grid-renderer"
_SEARCH_RESULT_WAIT_SELECTOR = "ytd-video-renderer a#video-title, a#video-title, ytd-item-section-renderer"

# Extra fallback locators for video discovery, tried in order inside
# _collect_video_urls after the original two. ytd-video-renderer is the
# search-results-page renderer (as opposed to the channel grid's
# ytd-rich-item-renderer); the href-based selector is a last-resort net for
# either page shape if YouTube renames the element/id again.
_VIDEO_LINK_SELECTORS = (
    "a#video-title",
    "a#video-title-link",
    "ytd-video-renderer a#video-title",
    "ytd-rich-item-renderer a#video-title-link",
    "a#thumbnail[href*='/watch']",
)

# Text markers used to tell a genuine "0 comments" video apart from a
# bot-check/login interstitial or an uploader-disabled comments section —
# all of which otherwise look identical from the outside, since our comment
# selectors simply find nothing either way.
_LOGIN_WALL_RE = re.compile(
    r"sign in to confirm|confirm you.?re not a bot|sign in to continue", re.I
)
_COMMENTS_DISABLED_RE = re.compile(
    r"comments (?:are|is) turned off|comments (?:are|is) disabled", re.I
)

# Text scraping never needs images or fonts; skipping them cuts page load
# time without affecting anything we read out of the DOM. JavaScript is
# left untouched since YouTube's comment feed is rendered client-side.
_BLOCKED_RESOURCE_TYPES = {"image", "font"}

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)
_LAUNCH_ARGS = ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"]

# --- Browser pool -----------------------------------------------------------
# One Playwright browser + context per worker thread, kept alive across
# calls instead of being relaunched every time. This avoids paying
# Chromium process-startup cost and the consent-cookie dance on every
# single request, and lets cookies/consent state persist between calls on
# the same worker thread.
MAX_BROWSER_WORKERS = 3
MAX_USES_BEFORE_RECYCLE = 50

_thread_local = threading.local()
_browser_trace = threading.local()
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=MAX_BROWSER_WORKERS, thread_name_prefix="youtube_scraper"
)

# --- Company-wide channel-resolution cache ----------------------------------
# scrape_youtube_comments() is invoked once per job in a job group - one
# "General" job plus one per selected product (see ARCHITECTURE.md §2/§3).
# The find_social_profile_url() search fallback below is entirely
# company-wide - it never depends on product_name - so every job in the same
# group resolves the exact same channel independently. Confirmed live
# (2026-07-30 Samsung India run, 3 concurrent jobs): the identical
# DuckDuckGo query ("site:youtube.com \"Samsung India\"") fired 3 times in
# the log, wasting 2 of the 3 network round trips on a miss - the same
# CPU/network contention concern as ARCHITECTURE.md §8, just on the network
# side instead of Playwright. This cache makes the search happen once per
# company per process lifetime; every other concurrent (or later) job for
# the same company reuses the result - hit or confirmed-miss - instead of
# re-querying. Safe to share across products (unlike comment content itself,
# which must stay per-product - see ARCHITECTURE.md §7): a channel handle is
# a property of the company, not of any one product.
_channel_resolution_lock = threading.Lock()
_channel_resolution_cache: Dict[str, str] = {}
_channel_resolution_events: Dict[str, threading.Event] = {}


class _BrowserHandle:
    """Holds one worker thread's Playwright/browser/context.

    Registered both in that thread's ``_thread_local`` (for reuse by the
    owning thread) and in the module-level ``_worker_handles`` list (so
    process-exit cleanup can reach it without running code back on the
    executor's worker threads).
    """

    __slots__ = ("playwright", "browser", "context", "uses")

    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.uses = 0


_worker_handles: List["_BrowserHandle"] = []
_worker_handles_lock = threading.Lock()


def _get_handle() -> "_BrowserHandle":
    handle = getattr(_thread_local, "handle", None)
    if handle is None:
        handle = _BrowserHandle()
        _thread_local.handle = handle
        with _worker_handles_lock:
            _worker_handles.append(handle)
    return handle


def _install_resource_blocking(context) -> None:
    def _route_handler(route):
        try:
            if route.request.resource_type in _BLOCKED_RESOURCE_TYPES:
                route.abort()
                return
        except Exception:
            pass
        try:
            route.continue_()
        except Exception:
            pass

    try:
        context.route("**/*", _route_handler)
    except Exception:
        logger.exception("Failed to install resource blocking; continuing without it.")


def _teardown_thread_browser() -> None:
    handle = _get_handle()
    try:
        if handle.context is not None:
            handle.context.close()
    except Exception:
        pass
    try:
        if handle.browser is not None:
            handle.browser.close()
    except Exception:
        pass
    handle.context = None
    handle.browser = None
    handle.uses = 0


def _close_handle_with_timeout(handle: "_BrowserHandle", timeout: float = 5.0) -> None:
    """Close one worker's Playwright objects, bounded by a short timeout.

    Runs on a fresh, throwaway ``threading.Thread`` rather than by
    submitting a job to ``_EXECUTOR``. Submitting to the same executor
    we're trying to shut down is what previously caused ``RuntimeError:
    cannot schedule new futures after shutdown`` — Python's interpreter
    shutdown sequence can mark the executor (or the process-wide threading
    state) as shut down before our atexit callback runs, and any
    ``executor.submit()`` after that point raises. A plain, unpooled thread
    has no such shutdown flag to race against.
    """

    def _job():
        try:
            if handle.context is not None:
                handle.context.close()
        except Exception:
            pass
        try:
            if handle.browser is not None:
                handle.browser.close()
        except Exception:
            pass
        try:
            if handle.playwright is not None:
                handle.playwright.stop()
        except Exception:
            pass
        handle.context = None
        handle.browser = None
        handle.playwright = None

    t = threading.Thread(target=_job, daemon=True)
    t.start()
    t.join(timeout=timeout)


def _shutdown_all_workers() -> None:
    """Best-effort cleanup of every worker's browser at process exit.

    Iterates the ``_worker_handles`` registry directly instead of
    submitting cleanup jobs back onto ``_EXECUTOR`` — see
    ``_close_handle_with_timeout`` for why that was unsafe.
    """
    with _worker_handles_lock:
        handles = list(_worker_handles)
    for handle in handles:
        _close_handle_with_timeout(handle)


atexit.register(_shutdown_all_workers)


def _ensure_context():
    """Get (creating or recycling as needed) this thread's browser context."""
    handle = _get_handle()
    browser_trace_started_at = time.monotonic()
    _browser_trace.ensure_context_started_at = browser_trace_started_at
    ctx = handle.context

    if ctx is not None:
        if handle.uses >= MAX_USES_BEFORE_RECYCLE:
            logger.info(
                "Recycling browser context on %s after %d uses.",
                threading.current_thread().name, handle.uses,
            )
            _teardown_thread_browser()
            ctx = None
        else:
            try:
                _ = ctx.pages  # cheap liveness check
                handle.uses += 1
                return ctx
            except Exception:
                logger.warning(
                    "Browser context on %s appears dead; recreating.",
                    threading.current_thread().name,
                )
                _teardown_thread_browser()
                ctx = None

    from playwright.sync_api import sync_playwright

    pw = handle.playwright
    if pw is None:
        logger.info(
            "BROWSER_TRACE scraper=YouTube event=before_sync_playwright_start elapsed=%.3fs timestamp=%.3f thread=%s",
            time.monotonic() - browser_trace_started_at, time.time(), threading.current_thread().name,
        )
        pw = sync_playwright().start()
        logger.info(
            "BROWSER_TRACE scraper=YouTube event=after_sync_playwright_start elapsed=%.3fs timestamp=%.3f thread=%s",
            time.monotonic() - browser_trace_started_at, time.time(), threading.current_thread().name,
        )
        handle.playwright = pw

    logger.info(
        "BROWSER_TRACE scraper=YouTube event=before_shared_browser_attach elapsed=%.3fs timestamp=%.3f thread=%s",
        time.monotonic() - browser_trace_started_at, time.time(), threading.current_thread().name,
    )
    # Attach to the ONE application-wide Chromium process instead of
    # launching a new one. ensure_shared_browser() only actually launches
    # Chromium on the very first call anywhere in the process; every call
    # after that - including this one, almost always - just returns the
    # existing CDP endpoint immediately. This (not a per-call launch slot)
    # is what removes the 44-46s startup stall: see browser_utils.py's
    # BrowserManager.ensure_shared_browser() docstring for the mechanics.
    cdp_endpoint = browser_manager.ensure_shared_browser()
    browser = pw.chromium.connect_over_cdp(cdp_endpoint)
    logger.info(
        "BROWSER_TRACE scraper=YouTube event=after_shared_browser_attach elapsed=%.3fs timestamp=%.3f thread=%s",
        time.monotonic() - browser_trace_started_at, time.time(), threading.current_thread().name,
    )
    logger.info(
        "BROWSER_TRACE scraper=YouTube event=before_new_context elapsed=%.3fs timestamp=%.3f thread=%s",
        time.monotonic() - browser_trace_started_at, time.time(), threading.current_thread().name,
    )
    context = browser.new_context(
        user_agent=_USER_AGENT,
        locale="en-US",
        viewport={"width": 1280, "height": 900},
        java_script_enabled=True,
        extra_http_headers={
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
        },
    )
    logger.info(
        "BROWSER_TRACE scraper=YouTube event=after_new_context elapsed=%.3fs timestamp=%.3f thread=%s",
        time.monotonic() - browser_trace_started_at, time.time(), threading.current_thread().name,
    )
    # Mask the webdriver flag that YouTube uses for bot detection
    context.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
        Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
        window.chrome = { runtime: {} };
    """)
    context.set_default_timeout(8000)
    context.set_default_navigation_timeout(12000)
    _install_resource_blocking(context)

    handle.browser = browser
    handle.context = context
    handle.uses = 1
    logger.info("Launched a new browser context on %s.", threading.current_thread().name)
    return context


# --- Startup pre-warm --------------------------------------------------
# Same fix as twitter_scraper.py's / instagram_scraper.py's identically-
# named block: setup alone (this thread's own sync_playwright().start() +
# attaching to the shared Chromium process + new_context()) was regularly
# taking 22-46s+ under this app's shared-browser contention at the start
# of an analysis run - see the OUTER_HARD_TIMEOUT_SECONDS block above.
# That cost is a ONE-TIME, per-worker-thread cost (_ensure_context above
# reuses a live context on later calls via a cheap `ctx.pages` check), so
# pay it once here, in the background, at module import time - i.e. when
# the FastAPI app boots up - well before the first real request in
# practice, since the user still has to go through Company/Product
# Discovery and pick a product before this module's
# scrape_youtube_comments() is ever called.
#
# MAX_BROWSER_WORKERS separate jobs are submitted (not just one) because
# a single analysis run can have up to 3 concurrent YouTube jobs in
# flight at once (one "General"/company-wide job plus one per selected
# product - see app.py's per-job asyncio.gather), each landing on its own
# worker thread. Submitting N tasks up front spreads across N distinct
# threads instead of piling onto one, so every thread a real job could be
# handed to is already warm.
#
# Best-effort only and never blocks import: each job runs on _EXECUTOR's
# own worker threads, and any failure is caught and logged - the first
# real request then simply falls back to paying the setup cost itself,
# exactly as it did before this change.
def _warm_up_browser_context() -> None:
    # Without this guard, this raises NotImplementedError on every warm-up
    # thread on Windows: app.py sets WindowsSelectorEventLoopPolicy
    # process-wide at import time (see browser_utils.py), but only the
    # Proactor loop can launch subprocesses on Windows, and Playwright's
    # sync API spawns its driver as a subprocess. The real request path
    # (_run_sync(), below) already re-asserts Proactor for the same
    # reason; this warm-up needs its own copy since it runs on a separate
    # thread, before any real request ever calls _run_sync().
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    try:
        _ensure_context()
        logger.info(
            "YouTube: browser context pre-warmed on %s.",
            threading.current_thread().name,
        )
    except Exception:
        logger.exception(
            "YouTube: background browser context warm-up failed on %s "
            "(non-fatal - the first real request will pay the setup "
            "cost itself instead, same as before this change).",
            threading.current_thread().name,
        )


for _ in range(MAX_BROWSER_WORKERS):
    _EXECUTOR.submit(_warm_up_browser_context)


_CHANNEL_ID_RE = re.compile(r"^UC[\w-]{22}$")
_YT_PATH_RE = re.compile(
    r"(?:youtube\.com|youtu\.be)/(channel/[\w-]+|user/[\w.-]+|c/[\w.-]+|@[\w.-]+)",
    re.IGNORECASE,
)


def _channel_url(target: str) -> str:
    """Turn whatever the caller has on file for a channel into a canonical
    ``https://www.youtube.com/...`` URL.

    Handles, in order:
      * a full URL (any of channel/, user/, c/, @handle, or bare domain
        with no scheme),
      * a bare "channel/UC...", "user/name", or "c/name" path fragment,
      * a bare channel ID (starts with "UC", 24 chars),
      * anything else, treated as an @handle.

    The previous version only matched the "channel/", "user/", "c/"
    prefixes when they were the *very first* characters of the string, so
    a value like "youtube.com/channel/UCxxxx" (a URL missing its scheme,
    which is a common way this field gets populated) fell through to the
    "@handle" branch and produced a broken URL
    (``.../@youtube.com/channel/UCxxxx``). This version searches for the
    known path patterns anywhere in the string and normalizes them.
    """
    target = (target or "").strip()
    if not target:
        return "https://www.youtube.com/"

    if target.startswith(("http://", "https://")):
        # Already a full URL — just use it as-is (it may not even be a
        # youtube.com URL, e.g. a youtu.be short link to a specific video;
        # the caller-level fallback search handles that case downstream).
        return target

    m = _YT_PATH_RE.search(target)
    if m:
        return f"https://www.youtube.com/{m.group(1)}"

    if target.startswith(("channel/", "user/", "c/")):
        return f"https://www.youtube.com/{target}"

    stripped = target.lstrip("@")
    if _CHANNEL_ID_RE.match(stripped):
        return f"https://www.youtube.com/channel/{stripped}"

    return f"https://www.youtube.com/@{stripped}"


def _dismiss_consent(page, timeout_ms: int = 1500) -> None:
    """Dismiss a cookie/consent dialog if one is showing.

    Previously this relied on the *caller* sleeping ~1.5s first so the
    dialog had time to render, then doing an instant ``count() > 0`` check
    per label — meaning every single page load paid a fixed 1.5s whether or
    not a dialog ever appeared. This version waits on each label directly
    (via Playwright's own actionability wait) up to a shared
    ``timeout_ms`` budget, so it returns almost immediately when no dialog
    is present and only spends time waiting when one might still be
    rendering.
    """
    deadline = time.monotonic() + (timeout_ms / 1000)
    for label in _CONSENT_LABELS:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        try:
            btn = page.get_by_role("button", name=label).first
            btn.wait_for(state="visible", timeout=max(200, remaining * 1000))
            btn.click(timeout=600)
            page.wait_for_timeout(200)
            return
        except Exception:
            continue


def _consent_dialog_present(page) -> bool:
    """Non-waiting check for whether a consent/cookie dialog is currently
    showing. Used only for diagnostics after collection has already given
    up — not part of the dismissal attempt above, which needs the
    actionability wait instead of an instant count.
    """
    try:
        for label in _CONSENT_LABELS:
            if page.get_by_role("button", name=label).count() > 0:
                return True
    except Exception:
        pass
    return False


def _diagnose_blocking_state(page) -> str:
    """Classify why a *listing* page (channel videos tab or search
    results) rendered nothing — the same consent/login-wall checks used for
    a video's comments, minus the comments-specific container/selector
    logic that doesn't apply here.
    """
    if _consent_dialog_present(page):
        return "consent_dialog"
    try:
        html = page.content()
    except Exception:
        html = ""
    if html and _LOGIN_WALL_RE.search(html):
        return "login_wall"
    return "rendering_failure_or_no_results"


# Evidence directory for zero-result debug captures. Lives at
# <project_root>/debug (sibling of app.py's DOWNLOADS_DIR), since this file
# is itself at <project_root>/scrapers/youtube_scraper.py.
_DEBUG_DIR = Path(__file__).resolve().parent.parent / "debug"


def _save_debug_artifacts(page, identifier: str, reason: str, extra: Dict[str, str] = None) -> None:
    """Save HTML/screenshot/context ONLY on failure/zero-results for
    debugging. Never on the success path.

    Writes three files under ``_DEBUG_DIR``, all sharing one base name
    stamped with date-time + milliseconds + thread id so concurrent or
    rapid-fire failures never overwrite each other:
      * <base>.png - screenshot
      * <base>.html - full page HTML
      * <base>.txt  - final URL, page title, reason, and whatever the
        caller passes in ``extra`` (e.g. consent-dialog appearance,
        video-grid load status).
    """
    try:
        _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        safe_id = re.sub(r'[^a-zA-Z0-9]', '_', identifier)[:50].strip('_') or "unknown"
        stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}_{threading.get_ident()}"
        base = _DEBUG_DIR / f"youtube_{safe_id}_{stamp}"

        try:
            final_url = page.url or ""
        except Exception:
            final_url = "<unavailable>"
        try:
            title = page.title()
        except Exception:
            title = "<unavailable>"

        try:
            page.screenshot(path=f"{base}.png", timeout=3000, full_page=True)
        except Exception:
            logger.warning("[YOUTUBE_DEBUG] could not capture screenshot for %s", identifier)

        try:
            with open(f"{base}.html", "w", encoding="utf-8") as f:
                f.write(page.content())
        except Exception:
            logger.warning("[YOUTUBE_DEBUG] could not capture HTML for %s", identifier)

        info = {
            "reason": reason,
            "identifier": identifier,
            "final_url": final_url,
            "page_title": title,
        }
        if extra:
            info.update(extra)
        try:
            with open(f"{base}.txt", "w", encoding="utf-8") as f:
                for k, v in info.items():
                    f.write(f"{k}: {v}\n")
        except Exception:
            pass

        logger.warning(
            "[YOUTUBE_DEBUG] saved zero-result evidence for %s reason=%s: %s.{png,html,txt}",
            identifier, reason, base,
        )
    except Exception:
        logger.exception("[YOUTUBE_DEBUG] failed to save debug artifacts for %s", identifier)


def _adaptive_nav_timeout(time_left) -> int:
    """Scale the navigation timeout to how much of the internal time
    budget is actually left, instead of a fixed 5000ms for every attempt.

    Navigation always receives at least NAV_TIMEOUT_MS_MIN (8s) as long as
    that much time is actually left in the budget - the budget clock only
    starts once browser/context/page setup has finished (see _run_sync
    below), so this floor is real, not eaten by Chromium startup. When
    plenty of budget remains the timeout can scale up to NAV_TIMEOUT_MS_MAX
    (12s) for a genuinely slow page; when the budget itself is under the
    floor, we hand over whatever is left rather than blocking past the
    scraper's own deadline.
    """
    remaining_ms = max(0.0, time_left()) * 1000
    if remaining_ms <= NAV_TIMEOUT_MS_MIN:
        return max(3000, int(remaining_ms))
    return int(min(NAV_TIMEOUT_MS_MAX, max(NAV_TIMEOUT_MS_MIN, remaining_ms * 0.5)))


def _capped_timeout(time_left, requested_ms: int, floor_ms: int = 500) -> int:
    """Cap a fixed wait/timeout value to what's actually left in the
    overall scraper time budget.

    Several ``wait_for_selector`` / ``scroll_into_view_if_needed`` /
    ``wait_for_timeout`` calls downstream used hardcoded millisecond
    values (5000/6000/9000/etc.) with no awareness of ``time_left()`` -
    unlike navigation, which already goes through
    ``_adaptive_nav_timeout``. A single one of those fixed waits could
    silently consume most or all of the remaining budget on a slow-
    rendering page, starving later search tiers even when the run
    started with plenty of time. This never asks for more than
    ``requested_ms``, but scales down as the budget runs low instead of
    blocking past it; ``floor_ms`` keeps it from asking Playwright for a
    near-zero/negative timeout.
    """
    remaining_ms = max(0.0, time_left()) * 1000
    if remaining_ms <= floor_ms:
        return floor_ms
    return int(min(requested_ms, remaining_ms))


_YT_NAV_RETRY_BACKOFF_SECONDS = 0.75


def _goto_with_retry(page, url: str, *, timeout: int, time_left, retries: int = NAVIGATION_RETRIES) -> bool:
    # ``timeout`` is accepted for call-site compatibility but is intentionally
    # ignored: we recompute an adaptive timeout before every attempt so that
    # retries never ask for more time than the scraper actually has left.
    last_exc = None
    total_attempts = retries + 1
    for attempt in range(total_attempts):
        remaining = time_left()
        if remaining <= 3:
            logger.warning(
                "Skipping navigation to %s: out of time budget "
                "(attempt=%d/%d remaining=%.1fs).",
                url, attempt + 1, total_attempts, remaining,
            )
            return False
        # Recompute adaptive timeout against the *current* remaining budget,
        # not the value frozen at call time.  This is the core fix: a failed
        # attempt burns real wall-clock time, so each retry must recalculate
        # rather than reuse a stale number that can now exceed what is left.
        attempt_timeout = _adaptive_nav_timeout(time_left)
        if attempt > 0:
            logger.info(
                "YouTube nav retry attempt=%d/%d url=%s remaining=%.1fs chosen_timeout=%dms",
                attempt + 1, total_attempts, url, remaining, attempt_timeout,
            )
        try:
            # Use "load" instead of "domcontentloaded": YouTube's comment
            # section is bootstrapped by JS that fires during the load event,
            # not at DOMContentLoaded. Using domcontentloaded meant we started
            # scrolling before the comment lazy-loader JS had even been parsed.
            page.goto(url, wait_until="load", timeout=attempt_timeout)
            return True
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "YouTube nav attempt %d/%d to %s failed: %s",
                attempt + 1, total_attempts, url, exc,
            )
            if attempt < retries:
                # Cap the backoff sleep to what is actually left minus the
                # minimum guard (3 s) so we never sleep past the deadline.
                backoff = min(
                    _YT_NAV_RETRY_BACKOFF_SECONDS * (attempt + 1),
                    max(0.0, time_left() - 3),
                )
                if backoff <= 0:
                    logger.warning(
                        "No time left for backoff before retry %d/%d to %s; aborting.",
                        attempt + 2, total_attempts, url,
                    )
                    break
                time.sleep(backoff)
    logger.error("Giving up on %s after %d attempts: %s", url, total_attempts, last_exc)
    return False


def _extract_video_urls_from_html(html: str, limit: int) -> List[str]:
    shorts_ids = set(_SHORTS_ID_RE.findall(html))
    seen: List[str] = []
    for vid in _VIDEO_ID_RE.findall(html):
        if vid in shorts_ids:
            continue
        url = f"https://www.youtube.com/watch?v={vid}"
        if url not in seen:
            seen.append(url)
        if len(seen) >= limit:
            break
    return seen


# Video renderers that live inside a Shorts shelf (as opposed to the normal
# uploads grid) are wrapped in one of these ancestor tags. We exclude
# anchors found inside them so a channel's Shorts don't get treated as
# regular videos to open for comments. Kept as bare tag names (no attribute
# selectors) since these are used inside an XPath ancestor:: check.
_SHORTS_CONTAINER_TAGS = (
    "ytd-reel-item-renderer",
    "ytd-reel-shelf-renderer",
    "ytm-shorts-lockup-view-model",
)


def _collect_video_urls(page, limit: int) -> List[str]:
    urls: List[str] = []
    shorts_skipped = 0

    for sel in _VIDEO_LINK_SELECTORS:
        for loc in page.locator(sel).all()[:limit]:
            try:
                href = loc.get_attribute("href")
            except Exception:
                continue
            if not href or "watch" not in href or "/shorts/" in href:
                if href and "/shorts/" in href:
                    shorts_skipped += 1
                continue
            try:
                # Skip anything whose ancestor chain marks it as a Shorts
                # shelf item, even though it happened to carry a /watch/
                # style href (rare, but seen on some Shorts renderer
                # variants).
                if any(
                    loc.locator(f"xpath=ancestor::{tag}").count() > 0
                    for tag in _SHORTS_CONTAINER_TAGS
                ):
                    shorts_skipped += 1
                    continue
            except Exception:
                pass
            full = "https://www.youtube.com" + href if href.startswith("/") else href
            if full not in urls:
                urls.append(full)
        if len(urls) >= limit:
            break

    if len(urls) < limit:
        try:
            html = page.content()
            for u in _extract_video_urls_from_html(html, limit * 2):
                if u not in urls:
                    urls.append(u)
                if len(urls) >= limit:
                    break
        except Exception:
            pass

    if shorts_skipped:
        logger.info("Video discovery: skipped %d Shorts entr(y/ies).", shorts_skipped)

    return urls[:limit]



# ============================================================================
# HTTP/InnerTube path — no browser required.
# ----------------------------------------------------------------------------
# The production log shows youtube.com/results navigations timing out on
# BOTH attempts every time ("Timeout 8491ms exceeded" / "Timeout 7062ms
# exceeded"), and a channel /videos page that DID render (video_grid_loaded
# = True, videos_found = 15) still yielding 0 comments, because the
# subsequent comment-scroll phase needed several more seconds of budget
# that had already been spent on Chromium contention before it started.
#
# youtube.com/results and youtube.com/watch are still plain server-rendered
# HTML - the video list and the first page of comments are embedded in the
# initial response as a JSON blob (`ytInitialData`), the same data the
# client-side JS reads to paint the DOM tiles we were otherwise scrolling
# for. Fetching that HTML with httpx and reading the JSON directly:
#   * needs no Chromium process at all, so it isn't affected by browser-
#     launch contention or subject to MAX_CONCURRENT_BROWSER_LAUNCHES,
#   * doesn't need to wait for lazy-loading/Intersection-Observer hydration,
#   * typically completes in well under a second per request.
#
# Comment *pages* beyond the first are fetched via the same InnerTube
# endpoint (`/youtubei/v1/next`) the page's own JS calls when you scroll -
# this is the same technique used by the widely-used
# `youtube-comment-downloader` tool. The API key and client version needed
# to call it are themselves public values embedded in every watch page's
# HTML (`INNERTUBE_API_KEY` / `clientVersion`), not a secret.
#
# This is tried FIRST for both search and comments. The existing
# Playwright DOM-scrape logic below is left completely intact as the
# fallback for the (hopefully rare) case where YouTube changes this JSON
# shape or blocks the plain-HTTP path specifically.
# ============================================================================

_YT_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
_YT_INITIAL_DATA_RE = re.compile(r'var ytInitialData\s*=\s*(\{.*?\});</script>', re.DOTALL)
_YT_INITIAL_DATA_RE_ALT = re.compile(r'ytInitialData"\]\s*=\s*(\{.*?\});', re.DOTALL)
_YT_INNERTUBE_KEY_RE = re.compile(r'"INNERTUBE_API_KEY":"([^"]+)"')
_YT_CLIENT_VERSION_RE = re.compile(r'"clientVersion":"([\d.]+)"')
_INNERTUBE_URL = "https://www.youtube.com/youtubei/v1/next"


def _fetch_yt_html(url: str, timeout: float = 6.0) -> Optional[str]:
    try:
        with httpx.Client(follow_redirects=True, headers=_YT_HTTP_HEADERS) as client:
            resp = client.get(url, timeout=timeout)
            if resp.status_code >= 400:
                return None
            return resp.text
    except Exception as exc:
        logger.info("YouTube httpx fetch failed for %s: %s", url, exc)
        return None


def _extract_yt_initial_data(html: str) -> Optional[dict]:
    for pattern in (_YT_INITIAL_DATA_RE, _YT_INITIAL_DATA_RE_ALT):
        m = pattern.search(html)
        if not m:
            continue
        try:
            return json.loads(m.group(1))
        except Exception:
            continue
    return None


def _video_ids_from_initial_data(data: dict, limit: int) -> List[str]:
    ids: List[str] = []
    seen = set()

    def _walk(node) -> None:
        if len(ids) >= limit:
            return
        if isinstance(node, dict):
            renderer = node.get("videoRenderer")
            if isinstance(renderer, dict):
                vid = renderer.get("videoId")
                if vid and vid not in seen:
                    seen.add(vid)
                    ids.append(vid)
            for value in node.values():
                if len(ids) >= limit:
                    return
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                if len(ids) >= limit:
                    return
                _walk(item)

    _walk(data)
    return ids[:limit]


def _httpx_video_urls(page_url: str, limit: int) -> List[str]:
    html = _fetch_yt_html(page_url)
    if not html:
        return []
    data = _extract_yt_initial_data(html)
    if not data:
        return []
    ids = _video_ids_from_initial_data(data, limit)
    return [f"https://www.youtube.com/watch?v={vid}" for vid in ids]


def _video_title_text(renderer: dict) -> str:
    try:
        title_obj = renderer.get("title") or {}
        runs = title_obj.get("runs")
        if runs:
            return "".join(r.get("text", "") for r in runs)
        return title_obj.get("simpleText", "") or ""
    except Exception:
        return ""


def _video_ids_and_titles_from_initial_data(data: dict, limit: int) -> List[Tuple[str, str]]:
    """Like ``_video_ids_from_initial_data``, but also captures each
    result's rendered title text so callers can apply relevance
    filtering (product-name token matching, see ``_title_matches_product``)
    BEFORE ever visiting a video - no extra fetch, since the title is
    already sitting right next to videoId in the same ``videoRenderer``
    node this walk was already reading.

    A renderer whose title text can't be extracted (unrecognized shape)
    still contributes its video ID with an empty-string title rather than
    being dropped - callers treat an empty title as "unknown, don't
    filter it out" (see ``_title_matches_product``), so a title-JSON
    shape change degrades to "trust the search" instead of silently
    zeroing out every candidate.
    """
    pairs: List[Tuple[str, str]] = []
    seen = set()

    def _walk(node) -> None:
        if len(pairs) >= limit:
            return
        if isinstance(node, dict):
            renderer = node.get("videoRenderer")
            if isinstance(renderer, dict):
                vid = renderer.get("videoId")
                if vid and vid not in seen:
                    seen.add(vid)
                    pairs.append((vid, _video_title_text(renderer)))
            for value in node.values():
                if len(pairs) >= limit:
                    return
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                if len(pairs) >= limit:
                    return
                _walk(item)

    _walk(data)
    return pairs[:limit]


def _httpx_video_urls_with_titles(page_url: str, limit: int) -> List[Tuple[str, str]]:
    html = _fetch_yt_html(page_url)
    if not html:
        return []
    data = _extract_yt_initial_data(html)
    if not data:
        return []
    pairs = _video_ids_and_titles_from_initial_data(data, limit)
    return [(f"https://www.youtube.com/watch?v={vid}", title) for vid, title in pairs]


# --- Product-title relevance gate (product-centric tiers only) -------------
# A search hit for "<product> review" is a good signal but not a guarantee -
# YouTube's own ranking also surfaces comparison/roundup videos ("Top 10
# budget earbuds") and wrong-model videos that happen to share enough
# keyword overlap to rank. Those videos are real and their comments are
# real, but most of those comments are NOT about the selected product -
# exactly the "generic content mislabeled as product-specific" failure
# principle #1 in ARCHITECTURE.md §3 warns about, just arriving through
# search-ranking noise instead of the (already-fixed, see the
# GATED TO product_name-LESS comment below) General-fallback route.
#
# This mirrors twitter_scraper.py's _mentions_product philosophy - biased
# toward false negatives over false positives - rather than reusing it
# directly, since a YouTube title is free-form creator copy (emoji, price
# call-outs, "review" phrasing) rather than a tweet, so requiring every
# token would likely reject a lot of genuinely-relevant videos. Instead
# this requires a clear majority (>=60%, rounded up) of the product's own
# meaningful tokens to appear in the title as whole words - not a
# substring match, so a product numbered "55" cannot match a title
# containing "255". Only applied to the httpx/InnerTube search path (the
# dominant path per _search_youtube's own comments - "what actually
# fixes" most searches); the Playwright DOM search fallback is
# deliberately left unfiltered, both to keep this change small and
# because that path only runs at all when httpx found zero candidates,
# i.e. there is nothing to filter yet.
_TITLE_MATCH_STOPWORDS = frozenset({
    "the", "a", "an", "for", "with", "and", "of", "in", "on", "to", "by",
})


def _product_tokens(product_name: str) -> List[str]:
    """Meaningful word/number tokens extracted from a product name.

    Strips only a small stopword list of connector words - deliberately
    NOT stripping things like "pro"/"plus"/model numbers, since those are
    exactly the tokens that distinguish one SKU from a related one (e.g.
    "Rockerz 255" vs "Rockerz 255 Pro+") and are the most important
    signal to keep.
    """
    raw = re.findall(r"[a-z0-9]+", (product_name or "").lower())
    tokens = [t for t in raw if t not in _TITLE_MATCH_STOPWORDS]
    # If the product name is somehow made up entirely of stopwords, fall
    # back to the raw tokens rather than returning an empty list - an
    # empty product_tokens list is treated as "nothing to check" (always
    # matches) by _title_matches_product, which would silently disable
    # filtering for that product rather than just being lenient.
    return tokens or raw


def _title_matches_product(title: str, product_tokens: List[str]) -> bool:
    """Conservative relevance gate for the product-centric search tiers
    only - never applied to the General/company-wide tiers, which have no
    specific product to be wrong about (see the GATED TO product_name-LESS
    comment further down this file).

    An empty ``product_tokens`` (nothing meaningful to check) or an empty
    ``title`` (extraction failed / unrecognized shape) both pass by
    default - fail OPEN, not closed, so this can only narrow an already-
    found candidate list, never silently zero one out because of its own
    internal parsing gap.
    """
    if not product_tokens or not title:
        return True
    title_tokens = set(re.findall(r"[a-z0-9]+", title.lower()))
    hits = sum(1 for t in product_tokens if t in title_tokens)
    required = max(1, math.ceil(len(product_tokens) * 0.6))
    return hits >= required


def _extract_innertube_context(html: str) -> Optional[Tuple[str, str]]:
    key_match = _YT_INNERTUBE_KEY_RE.search(html)
    if not key_match:
        return None
    version_match = _YT_CLIENT_VERSION_RE.search(html)
    client_version = version_match.group(1) if version_match else "2.20240101.00.00"
    return key_match.group(1), client_version


def _find_comments_continuation(data: dict) -> Optional[str]:
    """Locate the continuation token that opens the comments feed, scoped
    specifically to the comments engagement panel (rather than a blind
    full-tree walk) so this can't accidentally grab an unrelated
    continuation (e.g. related-videos pagination) that happens to share
    the same shape."""
    for panel in data.get("engagementPanels") or []:
        renderer = panel.get("engagementPanelSectionListRenderer", {}) or {}
        panel_id = (renderer.get("panelIdentifier") or "").lower()
        if "comment" not in panel_id:
            continue
        try:
            contents = renderer["content"]["sectionListRenderer"]["contents"]
        except Exception:
            continue
        for c in contents:
            item_section = (c or {}).get("itemSectionRenderer", {}) or {}
            for ic in item_section.get("contents", []):
                token = (
                    (ic.get("continuationItemRenderer") or {})
                    .get("continuationEndpoint", {})
                    .get("continuationCommand", {})
                    .get("token")
                )
                if token:
                    return token
    return None


def _innertube_comments_page(api_key: str, client_version: str, continuation: str, timeout: float) -> Optional[dict]:
    body = {
        "context": {"client": {"clientName": "WEB", "clientVersion": client_version}},
        "continuation": continuation,
    }
    try:
        with httpx.Client(headers=_YT_HTTP_HEADERS) as client:
            resp = client.post(
                f"{_INNERTUBE_URL}?key={api_key}",
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
            if resp.status_code >= 400:
                return None
            return resp.json()
    except Exception as exc:
        logger.info("YouTube InnerTube comments request failed: %s", exc)
        return None


def _comment_texts_and_next(payload: dict) -> Tuple[List[str], Optional[str]]:
    texts: List[str] = []
    next_token: Optional[str] = None

    # --- Current InnerTube format ---------------------------------------
    # YouTube no longer inlines comment text in each commentThreadRenderer;
    # it's stored separately in a parallel frameworkUpdates.entityBatchUpdate
    # .mutations array as commentEntityPayload entries, correlated back to
    # a thread only by an opaque key. Since this function only needs a flat
    # bag of comment text (not threaded structure/authorship), every
    # commentEntityPayload found here is taken directly - no key
    # correlation with commentThreadRenderer below is needed. Videos still
    # served in the legacy shape (handled below) simply have no
    # frameworkUpdates.entityBatchUpdate.mutations, so this contributes
    # nothing for them - no double-counting either way.
    mutations = (
        (payload.get("frameworkUpdates") or {})
        .get("entityBatchUpdate", {})
        .get("mutations", [])
        or []
    )
    for mutation in mutations:
        entity_payload = (mutation or {}).get("payload", {}).get("commentEntityPayload")
        if not entity_payload:
            continue
        content = (entity_payload.get("properties") or {}).get("content") or {}
        text = (content.get("content") or "").strip()
        if text:
            texts.append(text)

    # --- Legacy InnerTube format (kept intact) --------------------------
    for ep in payload.get("onResponseReceivedEndpoints") or []:
        action = ep.get("appendContinuationItemsAction") or ep.get("reloadContinuationItemsCommand")
        if not action:
            continue
        for item in action.get("continuationItems", []):
            thread = item.get("commentThreadRenderer")
            if thread:
                comment = thread.get("comment", {}).get("commentRenderer", {})
                runs = comment.get("contentText", {}).get("runs", []) or []
                text = "".join(r.get("text", "") for r in runs).strip()
                if text:
                    texts.append(text)
                continue
            cont = item.get("continuationItemRenderer")
            if cont:
                next_token = (
                    cont.get("continuationEndpoint", {})
                    .get("continuationCommand", {})
                    .get("token")
                )
    return texts, next_token


def _fetch_comments_via_innertube(video_url: str, limit: int, time_left) -> List[str]:
    """Pull top-level comments for ``video_url`` directly from YouTube's own
    InnerTube JSON API - no browser, no DOM scrolling. Returns [] on any
    failure (unrecognized page shape, blocked request, no continuation
    found, comments disabled, etc.) so the caller can fall back to the
    existing Playwright scroll-based path without special-casing."""
    if time_left() <= 2:
        return []
    html = _fetch_yt_html(video_url)
    if not html:
        return []
    ctx = _extract_innertube_context(html)
    data = _extract_yt_initial_data(html)
    if not ctx or not data:
        return []
    api_key, client_version = ctx
    continuation = _find_comments_continuation(data)
    if not continuation:
        return []

    comments: List[str] = []
    pages = 0
    while continuation and len(comments) < limit and pages < 80 and time_left() > 2:
        payload = _innertube_comments_page(
            api_key, client_version, continuation, timeout=min(6.0, max(2.0, time_left())),
        )
        pages += 1
        if not payload:
            break
        texts, continuation = _comment_texts_and_next(payload)
        comments.extend(texts)
    return comments[:limit]


def _search_youtube(page, query: str, time_left, product_tokens: Optional[List[str]] = None) -> List[str]:
    """Run one YouTube search for ``query`` and return candidate video URLs.

    Shared by every search-based discovery tier - the product-review and
    brand-product-review priority searches, as well as the plain
    company-name search that already existed as the fallback for a channel
    with 0 videos. Navigation, the search-results wait selector, and result
    collection are identical in every case; only the query string changes.

    Tries the plain-HTTP InnerTube path first (see the block above) - this
    is what actually fixes the repeated "Timeout 8491ms exceeded" /
    "Timeout 7062ms exceeded" search-navigation failures in the log, since
    it doesn't touch Chromium at all. Falls back to the existing
    Playwright-based search only if that path comes back empty.

    ``product_tokens``: only passed by the product-centric tiers (see
    ``_run_sync``). When given, httpx results are filtered through
    ``_title_matches_product`` before being returned - no extra fetch,
    since title text rides along in the same JSON response. Left as
    ``None`` (the default) for every other caller, in particular the
    General/company-wide tiers, which keep exactly their previous
    unfiltered behavior - there's no specific product to filter against
    there, and this default means this change can't affect that path.
    """
    try:
        if product_tokens:
            pairs = _httpx_video_urls_with_titles(
                f"https://www.youtube.com/results?search_query={quote_plus(query)}",
                VIDEO_CANDIDATES_TO_COLLECT,
            )
            if pairs:
                matched = [u for u, t in pairs if _title_matches_product(t, product_tokens)]
                logger.info(
                    "YouTube scrape: httpx search for %r found %d video(s), "
                    "%d passed the product-title relevance check (no "
                    "browser needed).",
                    query, len(pairs), len(matched),
                )
                # Candidates existed but (possibly) none looked genuinely
                # about this product - a relevance dead end, not a
                # discovery failure. Return as-is (even if empty) rather
                # than falling through to the unfiltered DOM search below,
                # which would just re-find this same set without titles to
                # check against; the caller's tier loop moves on to the
                # next tier instead (same "tier is a dead end" handling as
                # a search that found 0 videos at all).
                return matched
            video_urls = []
        else:
            video_urls = _httpx_video_urls(
                f"https://www.youtube.com/results?search_query={quote_plus(query)}",
                VIDEO_CANDIDATES_TO_COLLECT,
            )
    except Exception:
        logger.exception("YouTube httpx search failed for query=%r.", query)
        video_urls = []
    if video_urls:
        logger.info(
            "YouTube scrape: httpx search for %r found %d video(s) (no browser needed).",
            query, len(video_urls),
        )
        return video_urls

    logger.info(
        "YouTube scrape: httpx search found nothing for %r; falling back to "
        "browser-based search.", query,
    )
    search_url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
    if not _goto_with_retry(page, search_url, timeout=_adaptive_nav_timeout(time_left), time_left=time_left):
        return []
    _dismiss_consent(page)
    wait_ms = _capped_timeout(time_left, 5000)
    try:
        page.wait_for_selector(_SEARCH_RESULT_WAIT_SELECTOR, timeout=wait_ms)
    except Exception:
        logger.info(
            "YouTube scrape: no search results rendered for query=%r within "
            "%dms (reason=%s).", query, wait_ms, _diagnose_blocking_state(page),
        )
    return _collect_video_urls(page, VIDEO_CANDIDATES_TO_COLLECT)


def _expand_comment_extras(page, time_left, limit: int = 60) -> int:
    """Click "Read more" on truncated comments and expand reply threads.

    Both selectors are scoped to *inside* the comments section
    (``_READ_MORE_SELECTOR`` / ``_MORE_REPLIES_SELECTOR`` both lead with
    "ytd-comments "), so this never touches the video description's own
    "...more" expander, which shares the same underlying ytd-expander
    structure but lives higher up the page. Both are only clicked when
    Playwright reports the control as currently visible — a comment that's
    attached to the DOM but scrolled off-screen is left alone.
    """
    clicked = 0

    try:
        more_buttons = page.locator(_READ_MORE_SELECTOR).all()
    except Exception:
        more_buttons = []
    for btn in more_buttons[:limit]:
        if time_left() <= 2:
            return clicked
        try:
            if btn.is_visible():
                btn.click(timeout=600)
                clicked += 1
                page.wait_for_timeout(80)
        except Exception:
            continue

    try:
        reply_buttons = page.locator(_MORE_REPLIES_SELECTOR).all()
    except Exception:
        reply_buttons = []
    for btn in reply_buttons[:limit]:
        if time_left() <= 2:
            return clicked
        try:
            if btn.is_visible():
                btn.click(timeout=600)
                clicked += 1
                page.wait_for_timeout(150)
        except Exception:
            continue

    return clicked


def _wait_for_more(page, selector: str, prev_count: int, max_ms: int = 500) -> int:
    """Poll ``selector``'s match count in short steps until it exceeds
    ``prev_count`` or ``max_ms`` elapses, whichever comes first.

    Replaces an unconditional ``page.wait_for_timeout(max_ms)`` after a
    scroll: the worst-case wait is identical (this never runs longer than
    ``max_ms``), but as soon as new nodes attach — which is most of the
    time on a live feed — this returns immediately instead of always
    sitting out the full window. Returns the last observed count so
    callers can use it as the next call's baseline.
    """
    deadline = time.monotonic() + (max_ms / 1000)
    count = prev_count
    while time.monotonic() < deadline:
        try:
            count = page.locator(selector).count()
        except Exception:
            return count
        if count > prev_count:
            return count
        page.wait_for_timeout(100)
    return count


def _classify_comment_failure(page, time_left) -> str:
    """Classify why a video yielded zero comments.

    The previous version logged one generic "no comments rendered" message
    for every possible cause, which made a comments-disabled video, a
    bot-check interstitial, and an actual selector regression
    indistinguishable in the logs. This checks, in order: running out of
    time budget, a still-open consent dialog, a login/bot-check wall,
    comments explicitly turned off by the uploader, the comments container
    never attaching to the DOM at all (rendering failure), our selectors
    finding nothing despite the page itself reporting a nonzero comment
    count (selector mismatch — a real signal these selectors need
    updating), and finally a genuine absence of public comments.
    """
    if time_left() <= 2:
        return "timeout"

    if _consent_dialog_present(page):
        return "consent_dialog"

    try:
        html = page.content()
    except Exception:
        html = ""
    if html and _LOGIN_WALL_RE.search(html):
        return "login_wall"
    if html and _COMMENTS_DISABLED_RE.search(html):
        return "comments_disabled"

    try:
        has_container = page.locator(_COMMENTS_CONTAINER_SELECTOR).count() > 0
    except Exception:
        has_container = False
    if not has_container:
        return "rendering_failure"

    try:
        header_text = page.locator(
            "ytd-comments-header-renderer #count, "
            "ytd-comments-header-renderer yt-formatted-string"
        ).first.inner_text(timeout=1000)
    except Exception:
        header_text = ""
    if re.search(r"[1-9]\d*", header_text or ""):
        # The page itself claims a nonzero comment count, but none of our
        # known comment-text selectors matched anything inside the
        # container — a strong signal that YouTube changed the comment DOM
        # again rather than that this video truly has no comments.
        return "selector_mismatch"

    return "no_public_comments"


def _scroll_comments_until_idle(page, time_left, remaining_budget: int) -> List[str]:
    """Scroll a video's comments section, collecting text as it loads.

    Keeps scrolling (expanding "Read more" and reply threads along the
    way) until one of:
      * ``remaining_budget`` comments have been located on this video,
      * ``MAX_SCROLL_ITERATIONS`` scroll steps have happened,
      * ``SCROLL_IDLE_LIMIT`` consecutive scrolls produced no new comment
        nodes, or
      * the overall scraper time budget runs low.
    """
    comments: List[str] = []
    if remaining_budget <= 0:
        return comments
    if time_left() <= 2:
        # Matches _classify_comment_failure's own "timeout" threshold.
        # Entering the scroll/expand sequence below with essentially no
        # budget left just means burning the last of it on fixed waits
        # that can't complete anyway - bail out now so the caller (and
        # any later search tier) gets an accurate "out of time" signal
        # instead of a silent, budget-exhausting stall.
        return comments

    _dismiss_consent(page)

    # Phase 1: Scroll down past the video player to trigger YouTube's
    # Intersection Observer that bootstraps the comment section.
    # YouTube only hydrates comments when #comments enters the viewport;
    # scroll_into_view_if_needed alone is not reliable because the element
    # may not exist yet when we call it. We do a series of timed scrolls
    # first to let the page JS attach the container, then attempt targeting.
    try:
        # Quick scroll past the video description to where comments live
        page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight * 0.3)")
        page.wait_for_timeout(_capped_timeout(time_left, 600, floor_ms=100))
        page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight * 0.5)")
        page.wait_for_timeout(_capped_timeout(time_left, 600, floor_ms=100))
    except Exception:
        pass

    container_found = True
    try:
        page.locator(_COMMENTS_CONTAINER_SELECTOR).first.scroll_into_view_if_needed(
            timeout=_capped_timeout(time_left, 5000)
        )
        # Nudge a little further so the container sits clearly inside the
        # viewport rather than right at its edge — YouTube's lazy comment
        # loader is keyed off actual visibility, and landing exactly on
        # the boundary has been observed to leave it un-triggered.
        page.evaluate("window.scrollBy(0, 400)")
        page.wait_for_timeout(_capped_timeout(time_left, 400, floor_ms=100))
    except Exception:
        container_found = False

    if not container_found:
        # Container hasn't attached yet — blind scrolls to coax the lazy loader
        for step in (1200, 1500, 1800, 2000):
            if time_left() <= 2:
                break
            try:
                page.evaluate(f"window.scrollBy(0, {step})")
            except Exception:
                break
            page.wait_for_timeout(_capped_timeout(time_left, 400, floor_ms=100))
        try:
            page.locator(_COMMENTS_CONTAINER_SELECTOR).first.scroll_into_view_if_needed(
                timeout=_capped_timeout(time_left, 4000)
            )
            page.evaluate("window.scrollBy(0, 400)")
            page.wait_for_timeout(_capped_timeout(time_left, 400, floor_ms=100))
        except Exception:
            pass

    # Phase 2: Wait for at least one comment node to appear. Use a longer
    # timeout here because YouTube's comment loader fires asynchronously
    # after the Intersection Observer triggers — this is the most common
    # place where comments exist but 0 are collected.
    try:
        page.wait_for_selector(_COMMENT_TEXT_SELECTOR, timeout=_capped_timeout(time_left, 9000))
    except Exception:
        # One final aggressive scroll attempt before giving up
        try:
            page.evaluate("window.scrollBy(0, 600)")
            page.wait_for_timeout(_capped_timeout(time_left, 800, floor_ms=100))
            page.wait_for_selector(_COMMENT_TEXT_SELECTOR, timeout=_capped_timeout(time_left, 4000))
        except Exception:
            reason = _classify_comment_failure(page, time_left)
            logger.info(
                "Comment scroll: no comments collected for this video (reason=%s).",
                reason,
            )
            return comments

    iteration = 0
    idle = 0
    prev_count = -1
    start = time.monotonic()
    while True:
        iteration += 1
        if iteration > MAX_SCROLL_ITERATIONS:
            logger.info("Comment scroll: hit MAX_SCROLL_ITERATIONS (%d).", MAX_SCROLL_ITERATIONS)
            break
        if time_left() <= 2:
            logger.info("Comment scroll: stopping, low on overall time budget.")
            break

        try:
            page.evaluate("window.scrollBy(0, 1400)")
        except Exception:
            break
        _wait_for_more(page, _COMMENT_TEXT_SELECTOR, max(prev_count, 0), max_ms=800)

        if iteration % 3 == 0:
            _expand_comment_extras(page, time_left)

        try:
            count = page.locator(_COMMENT_TEXT_SELECTOR).count()
        except Exception:
            count = -1

        logger.info(
            "Comment scroll: iteration=%d current_count=%d idle_streak=%d elapsed=%.1fs",
            iteration, count, idle, time.monotonic() - start,
        )

        if count >= remaining_budget:
            logger.info("Comment scroll: reached this video's share of the overall target.")
            break
        if count <= prev_count:
            idle += 1
            if idle >= SCROLL_IDLE_LIMIT:
                logger.info(
                    "Comment scroll: no new comments after %d consecutive scrolls; stopping.",
                    SCROLL_IDLE_LIMIT,
                )
                break
        else:
            idle = 0
        prev_count = max(prev_count, count)

    _expand_comment_extras(page, time_left)

    for loc in page.locator(_COMMENT_TEXT_SELECTOR).all()[:remaining_budget]:
        try:
            txt = loc.inner_text().strip()
            if txt and len(txt.split()) >= 2:
                comments.append(txt)
        except Exception:
            pass
    return comments


def _run_sync(
    company_name: str,
    videos_url: str,
    product_name: str = "",
    product_brand: str = "",
    _overall_start: Optional[float] = None,
) -> List[str]:
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # Fallback for any direct/legacy caller that doesn't pass
    # _overall_start: behaves like a fresh clock starting right here,
    # same as before this fix - only scrape_youtube_comments() below (the
    # real entry point) passes the true, pre-setup anchor.
    if _overall_start is None:
        _overall_start = time.monotonic()

    setup_start = time.monotonic()

    try:
        context = _ensure_context()
    except Exception:
        logger.exception("Could not start/obtain a browser context for %r.", company_name)
        return []

    results: List[str] = []
    seen_keys = set()
    duplicates_removed = 0
    page = None
    source = "none"
    consent_page_appeared = False
    video_grid_loaded = False
    videos_attempted = 0
    last_videos_found = 0
    try:
        logger.info(
            "BROWSER_TRACE scraper=YouTube event=before_new_page elapsed=%.3fs timestamp=%.3f thread=%s",
            time.monotonic() - getattr(_browser_trace, "ensure_context_started_at", time.monotonic()), time.time(), threading.current_thread().name,
        )
        page = context.new_page()
        logger.info(
            "BROWSER_TRACE scraper=YouTube event=after_new_page elapsed=%.3fs timestamp=%.3f thread=%s",
            time.monotonic() - getattr(_browser_trace, "ensure_context_started_at", time.monotonic()), time.time(), threading.current_thread().name,
        )
        page.set_default_timeout(12000)

        setup_elapsed = time.monotonic() - setup_start
        start = time.monotonic()
        # Deadline anchored to the OUTER hard timeout (app.py's
        # YOUTUBE_TIMEOUT_SECONDS), counted from when this job actually
        # started (_overall_start) - not a fresh TIME_BUDGET_SECONDS clock
        # that ignores how much of the outer budget setup already used.
        # This guarantees _run_sync() always leaves SAFETY_MARGIN_SECONDS
        # of real slack for the result to make it back through
        # run_in_executor()/wait_for(), no matter how long setup took
        # under this app's shared-browser contention. See the
        # OUTER_HARD_TIMEOUT_SECONDS block above.
        deadline = _overall_start + OUTER_HARD_TIMEOUT_SECONDS - SAFETY_MARGIN_SECONDS
        remaining_budget = deadline - start

        logger.info(
            "YouTube scrape: browser/context/page setup took %.1fs "
            "(%.1fs elapsed since job start); %.1fs left for "
            "navigation+scrape before the safety-margined internal "
            "deadline (outer cap=%ds, safety margin=%.1fs).",
            setup_elapsed,
            start - _overall_start,
            remaining_budget,
            OUTER_HARD_TIMEOUT_SECONDS,
            SAFETY_MARGIN_SECONDS,
        )

        if remaining_budget <= MIN_USEFUL_BUDGET_SECONDS:
            logger.warning(
                "YouTube scrape: only %.1fs left for %r after a %.1fs "
                "setup (job start to now: %.1fs) - not enough time to "
                "attempt navigation; returning early instead of risking "
                "the outer hard timeout.",
                remaining_budget, company_name, setup_elapsed,
                time.monotonic() - _overall_start,
            )
            return normalize_comments(results)[:MAX_TOTAL_COMMENTS]

        def time_left() -> float:
            return deadline - time.monotonic()

        def scrape_tier(label: str, video_urls: List[str]) -> bool:
            """Visit each of ``video_urls`` (skipping any a previous tier
            already tried) and merge any comments found into ``results``.

            Returns True iff this tier contributed at least one usable
            comment - the actual "stop here" signal - and False if it's a
            dead end (comments disabled, bot wall, no public comments,
            etc.), in which case the caller moves on to try the next tier.
            """
            nonlocal duplicates_removed, videos_attempted
            # Scoped to THIS tier only (not shared across tiers): if an
            # earlier tier already visited a URL and it yielded zero
            # comments, that outcome may have been transient (slow
            # render, one-off nav hiccup) rather than a real dead end
            # (comments disabled, bot wall). A later, more targeted tier
            # gets its own fresh attempt at the same URL instead of being
            # silently blocked from ever revisiting it. This can never
            # duplicate a *successful* visit: any tier that lands even
            # one usable comment sets `succeeded = True` and halts the
            # whole search immediately (see the tier loops below), so a
            # URL only becomes eligible for a retry here after a prior
            # tier's attempt was already a zero-comment outcome.
            attempted_this_tier: set = set()
            count_before = len(results)
            for video_url in video_urls:
                if video_url in attempted_this_tier:
                    continue
                if len(results) >= MAX_TOTAL_COMMENTS or time_left() <= 3:
                    break
                attempted_this_tier.add(video_url)
                videos_attempted += 1
                remaining_budget = MAX_TOTAL_COMMENTS - len(results)
                try:
                    # Primary: InnerTube JSON path - no browser page.goto,
                    # no scroll/hydration wait. Only falls through to the
                    # existing Playwright scroll logic if this comes back
                    # empty (unrecognized page shape, comments disabled,
                    # or the request itself failed).
                    video_comments = _fetch_comments_via_innertube(video_url, remaining_budget, time_left)
                    if not video_comments:
                        if not _goto_with_retry(page, video_url, timeout=_adaptive_nav_timeout(time_left), time_left=time_left):
                            continue
                        video_comments = _scroll_comments_until_idle(page, time_left, remaining_budget)
                    before = len(results)
                    for c in video_comments:
                        # Normalized (stripped + lowercased) key, matching
                        # normalize_comments()'s own dedup rule exactly, so a
                        # comment isn't double-counted against the running
                        # budget here only to be merged away by that final
                        # pass anyway. A set lookup also replaces the previous
                        # O(n) "c in results" scan with an O(1) check.
                        key = c.strip().lower()
                        if not key or key in seen_keys:
                            duplicates_removed += 1
                            continue
                        seen_keys.add(key)
                        results.append(c)
                    logger.info(
                        "YouTube scrape [%s]: video=%s contributed %d new comment(s); "
                        "running total=%d/%d.",
                        label, video_url, len(results) - before, len(results), MAX_TOTAL_COMMENTS,
                    )
                except Exception:
                    logger.exception("Failed scraping comments for video=%s.", video_url)
                    continue
            return len(results) > count_before

        # --- Priority 1 & 2: product-centric review search ------------------
        # Search YouTube directly for review content about this specific
        # product before ever looking at the official company channel (the
        # channel is the brand's own uploads, not third-party reviews - the
        # "often returns zero comments" complaint this whole change
        # addresses). A tier is only abandoned - moving on to the next one -
        # once its videos have actually been visited and yielded zero
        # usable comments; a search simply returning candidate URLs is not
        # by itself enough to stop on.
        succeeded = False
        search_tiers = []
        product_tokens: List[str] = []
        if product_name:
            product_tokens = _product_tokens(product_name)
            search_tiers.append(("product_review", f"{product_name} review"))
            brand = (product_brand or company_name or "").strip()
            if brand and brand.lower() != product_name.strip().lower():
                search_tiers.append(("brand_product_review", f"{brand} {product_name} review"))

        for label, query in search_tiers:
            if time_left() <= _MIN_TIME_FOR_ANOTHER_TIER_SECONDS:
                logger.info("YouTube scrape: skipping tier %r — out of time budget.", label)
                break
            video_urls = _search_youtube(page, query, time_left, product_tokens=product_tokens)
            last_videos_found = len(video_urls) or last_videos_found
            if not video_urls:
                logger.info("YouTube scrape: tier %r search (%r) found 0 videos.", label, query)
                continue
            logger.info(
                "YouTube scrape: tier %r search (%r) found %d candidate video(s); "
                "scraping them for comments.", label, query, len(video_urls),
            )
            if scrape_tier(label, video_urls):
                source = label
                succeeded = True
                break
            logger.info(
                "YouTube scrape: tier %r yielded videos but 0 usable comments; "
                "trying next tier.", label,
            )

        # --- Priority 3: existing official-channel logic, then (as before)
        # a plain company-name search - only reached if the tiers above
        # never landed a single usable comment.
        #
        # GATED TO product_name-LESS (General/company-wide) JOBS ONLY.
        # Confirmed root cause of a real data-quality bug: when a
        # PER-PRODUCT job's targeted "<product> review" search came up
        # thin, this used to fall through to scraping the brand's general
        # channel (any recent video, unrelated to that product) and then a
        # plain company-name search - and whatever comments those found
        # got attributed to that specific product anyway. Production logs
        # showed a product whose own targeted search found only 17
        # candidate videos ending up with 1,405 "final" comments credited
        # to it, almost all of it generic brand-channel content that was
        # never actually about that product. These broad tiers are
        # legitimately appropriate for the General job (which has no
        # specific product to be wrong about - it's supposed to be
        # brand-wide, same as Twitter/Instagram), so they still run there;
        # a per-product job that exhausts its targeted tiers now simply
        # returns what it genuinely found (possibly little or nothing)
        # instead of backfilling with irrelevant content mislabeled as
        # product-specific.
        if not product_name and not succeeded and time_left() > _MIN_TIME_FOR_ANOTHER_TIER_SECONDS:
            video_urls: List[str] = []
            try:
                video_urls = _httpx_video_urls(videos_url, VIDEO_CANDIDATES_TO_COLLECT)
            except Exception:
                video_urls = []
            if video_urls:
                video_grid_loaded = True
                logger.info(
                    "YouTube scrape: httpx fetch of official channel %s found %d "
                    "video(s) (no browser needed).", videos_url, len(video_urls),
                )
            elif _goto_with_retry(page, videos_url, timeout=_adaptive_nav_timeout(time_left), time_left=time_left):
                consent_page_appeared = _consent_dialog_present(page)
                _dismiss_consent(page)
                # Explicitly wait for at least one video tile before scrolling,
                # instead of a fixed sleep beforehand. The previous version
                # went straight into scrolling/collecting behind only a fixed
                # 1500ms sleep, which on a slow-rendering channel page reads 0
                # videos before the grid has actually painted — the same class
                # of readiness bug the comment loop had. This gives slow
                # channels a real chance to render before we give up, without
                # paying a blind sleep on fast ones.
                _tile_wait_ms = _capped_timeout(time_left, 6000)
                try:
                    page.wait_for_selector(_VIDEO_TILE_WAIT_SELECTOR, timeout=_tile_wait_ms)
                    video_grid_loaded = True
                except Exception:
                    logger.info(
                        "YouTube scrape: no video tiles rendered on %s within %dms "
                        "(reason=%s).", videos_url, _tile_wait_ms, _diagnose_blocking_state(page),
                    )
                tile_count = 0
                for _ in range(2):
                    page.evaluate("window.scrollBy(0, 1200)")
                    tile_count = _wait_for_more(page, _VIDEO_TILE_WAIT_SELECTOR, tile_count, max_ms=500)
                video_urls = _collect_video_urls(page, VIDEO_CANDIDATES_TO_COLLECT)
            last_videos_found = len(video_urls) or last_videos_found

            if video_urls:
                logger.info(
                    "YouTube scrape: tier 'official_channel' found %d candidate "
                    "video(s); scraping them for comments.", len(video_urls),
                )
                if scrape_tier("official_channel", video_urls):
                    source = "official_channel"
                    succeeded = True

            if not succeeded and time_left() > _MIN_TIME_FOR_ANOTHER_TIER_SECONDS:
                logger.info(
                    "YouTube scrape: official channel yielded no usable comments — "
                    "falling back to a single general YouTube search for %r.",
                    company_name,
                )
                video_urls = _search_youtube(page, company_name, time_left)
                last_videos_found = len(video_urls) or last_videos_found
                if video_urls:
                    if scrape_tier("channel_name_search", video_urls):
                        source = "channel_name_search"
                        succeeded = True
                    else:
                        logger.info(
                            "YouTube scrape: search fallback also yielded 0 usable "
                            "comments for %r (reason=%s).",
                            company_name, _diagnose_blocking_state(page),
                        )
                else:
                    logger.info(
                        "YouTube scrape: search fallback also yielded 0 videos for %r "
                        "(reason=%s) — reliable YouTube coverage needs an "
                        "authenticated session or the official YouTube Data API.",
                        company_name, _diagnose_blocking_state(page),
                    )

        if not results:
            _save_debug_artifacts(
                page, videos_url, "zero_comments",
                extra={
                    "consent_page_appeared": str(consent_page_appeared),
                    "video_grid_loaded": str(video_grid_loaded),
                    "videos_found": str(last_videos_found),
                    "videos_attempted": str(videos_attempted),
                    "video_source": source,
                    "diagnosis": _diagnose_blocking_state(page),
                },
            )
    finally:
        if page is not None:
            try:
                page.close()
            except Exception:
                pass

    final = normalize_comments(results)[:MAX_TOTAL_COMMENTS]
    elapsed = time.monotonic() - start
    logger.info(
        "YouTube comments for %r (product=%r, source=%s): raw=%d "
        "duplicates_removed=%d final=%d elapsed=%.1fs (cap=%d).",
        company_name, product_name, source, len(results), duplicates_removed,
        len(final), elapsed, MAX_TOTAL_COMMENTS,
    )
    logger.info("YouTube returned %d comments", len(final))
    return final


async def scrape_youtube_comments(company_data: Dict[str, str]) -> List[str]:
    # Anchor point for the real wall-clock deadline (see the
    # OUTER_HARD_TIMEOUT_SECONDS block above). Captured before the
    # channel/handle resolution fallback and before this job is even
    # submitted to the executor, so _run_sync()'s deadline reflects the
    # FULL time app.py's outer wait_for() has already been counting
    # against this coroutine - not just the part after setup finishes.
    _overall_start = time.monotonic()

    target = (
        company_data.get("youtube_url")
        or company_data.get("youtube")
    )

    loop = asyncio.get_event_loop()

    if not target:
        company_name = (company_data.get("company_name") or "").strip()
        if company_name:
            cache_key = company_name.strip().lower()
            should_resolve = False
            wait_event: Optional[threading.Event] = None
            with _channel_resolution_lock:
                if cache_key in _channel_resolution_cache:
                    target = _channel_resolution_cache[cache_key]
                else:
                    wait_event = _channel_resolution_events.get(cache_key)
                    if wait_event is None:
                        wait_event = threading.Event()
                        _channel_resolution_events[cache_key] = wait_event
                        should_resolve = True

            if should_resolve:
                try:
                    target = await loop.run_in_executor(
                        _EXECUTOR, find_social_profile_url, company_name, "youtube",
                    )
                except Exception:
                    target = ""
                finally:
                    # Always cache and always signal waiters, even on
                    # exception - a raised find_social_profile_url() must
                    # not leave concurrent jobs for the same company
                    # blocked on wait_event.wait() forever.
                    with _channel_resolution_lock:
                        _channel_resolution_cache[cache_key] = target
                    wait_event.set()
                if target:
                    logger.info(
                        "YouTube scrape: no channel from the site scan; "
                        "search fallback found %r for %r.", target, company_name,
                    )
            elif wait_event is not None:
                # Another job (General or a sibling product) is already
                # resolving this exact company's channel - wait for its
                # result instead of firing a duplicate search. This is what
                # collapses the 3 identical DuckDuckGo queries seen in the
                # log down to 1.
                await loop.run_in_executor(_EXECUTOR, wait_event.wait)
                with _channel_resolution_lock:
                    target = _channel_resolution_cache.get(cache_key, "")
                if target:
                    logger.info(
                        "YouTube scrape: reused channel %r for %r, resolved "
                        "moments ago by a concurrent job in the same run "
                        "(no duplicate search fired).", target, company_name,
                    )
        if not target:
            logger.info(
                "YouTube scrape: no channel from the site scan and the "
                "search fallback found nothing for %r; skipping rather "
                "than guessing a handle from the name.",
                company_data.get("company_name"),
            )
            return []

    videos_url = _channel_url(target).rstrip("/") + "/videos"
    company_name = company_data.get("company_name", target)
    # Both optional and purely additive: any existing caller that doesn't
    # set these (e.g. a plain company-wide job) gets exactly today's
    # behavior - discovery goes straight to the official channel logic.
    product_name = (company_data.get("product_name") or "").strip()
    product_brand = (company_data.get("product_brand") or "").strip()

    try:
        return await loop.run_in_executor(
            _EXECUTOR, _run_sync, company_name, videos_url, product_name, product_brand, _overall_start,
        )
    except Exception:
        logger.exception("Unhandled error scraping YouTube comments for %r.", company_name)
        return []
