"""
Module Name
-----------
scrapers/instagram_scraper.py

Purpose
-------
Collects genuine, product-specific captions and comments from Instagram —
the company's own posts about a selected product, and the comments left
on them — while avoiding generic brand-wide content where possible.

Responsibilities
-----------------
- `scrape_instagram_comments(company_data)`: the single public entry
  point. Locates relevant posts from the company's Instagram profile,
  filters for product relevance, and collects caption/comment text up to
  `config.MAX_INSTAGRAM_COMMENTS`.
- Attaches to the shared Chromium process via `scrapers.browser_utils`
  rather than launching its own, and logs `BROWSER_TRACE` timing at each
  setup checkpoint, same as google_scraper.py / twitter_scraper.py /
  youtube_scraper.py.
- Instagram has no strict quantity target (per the project's own
  priorities) — this module optimizes for reliability of what it can
  collect rather than chasing a fixed volume, given how aggressively
  Instagram gates unauthenticated/automated access.

Dependencies
------------
`playwright` (sync API), plus standard library `asyncio`, `re`,
`threading`, `time`.
"""

import asyncio
import atexit
import concurrent.futures
import json
import logging
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from scrapers.browser_utils import browser_manager, normalize_comments, find_social_profile_url
from config import (
    MAX_INSTAGRAM_COMMENTS,
    MAX_SCROLL_ITERATIONS,
    SCROLL_IDLE_LIMIT,
    NAVIGATION_RETRIES,
    NAV_TIMEOUT_MS_MAX,
    NAV_TIMEOUT_MS_MIN,
)

logger = logging.getLogger(__name__)

# Kept as an alias for backwards compatibility with any code importing the
# old name directly; the real cap now lives in config.py.
MAX_RESULTS = MAX_INSTAGRAM_COMMENTS

MIN_RESULTS_BEFORE_FALLBACK = 5

# How many individual post permalinks to try pulling captions/comments
# from once we're past the profile grid itself.
MAX_POSTS_TO_VISIT = 20

# How many grid thumbnails to click through on the profile embed page to
# *discover* real post URLs (see the comment above the click loop in
# scrape_instagram_comments for why this is necessary). Kept smaller than
# MAX_POSTS_TO_VISIT since each discovery click costs a real navigation
# round-trip, unlike the old (broken) static href scan it replaces.
MAX_POSTS_TO_DISCOVER = 8

# --- Hard internal time budget -----------------------------------------
# PREVIOUS BUG (same class of bug already fixed in twitter_scraper.py -
# see that file's OUTER_HARD_TIMEOUT_SECONDS block for the full writeup):
# `deadline = start + TIME_BUDGET_SECONDS` below used to start a FRESH
# clock only after browser/context/page setup had already finished, with
# no memory of how much of the OUTER app.py wait_for() budget setup
# itself had already spent. Setup alone was regularly taking ~25-32s
# under this app's shared-Chromium contention at the start of an analysis
# run (see BROWSER_TRACE logs), so "setup (32s) + a fresh 20s budget"
# routinely added up to ~50s+ of real work against app.py's actual 34s
# INSTAGRAM_TIMEOUT_SECONDS cap. The stale comment this block used to
# carry ("kept under app.py's 26s cap") was itself evidence of the bug:
# app.py's real value is 34s and had drifted out of sync with this file.
# Concretely, this meant: app.py's outer wait_for() would hard-cancel the
# coroutine at 34s and use [] as the result, while the *actual* _run()
# kept executing to completion unseen on its executor thread - sometimes
# finding several real comments (see "Instagram items ... final=6" in
# production logs) that were computed correctly but never made it back to
# the dashboard, because the coroutine that would have returned them had
# already been cancelled.
#
# Fixed below the same way as twitter_scraper.py: one real wall-clock
# deadline anchored to the moment scrape_instagram_comments() itself
# starts (before the httpx preflight), via OUTER_HARD_TIMEOUT_SECONDS /
# SAFETY_MARGIN_SECONDS, so _run() always leaves real margin for its
# result to make it back through run_in_executor()/wait_for() no matter
# how long setup took. TIME_BUDGET_SECONDS is kept only as the
# "no contention at all" reference value the comments above still
# describe; deadline/time_left() (see _run()) are what the code actually
# uses now.
OUTER_HARD_TIMEOUT_SECONDS = 30  # cut from 45 - priority shifted to 3-min total time. must be kept in sync with app.py's INSTAGRAM_TIMEOUT_SECONDS
SAFETY_MARGIN_SECONDS = 8.0
MIN_USEFUL_BUDGET_SECONDS = 6.0  # below this, don't even attempt navigation

TIME_BUDGET_SECONDS = 20

# Ceiling on how long the zero-result debug-artifact screenshot is allowed
# to spend, no matter how unresponsive the page is. Mirrors
# twitter_scraper.py's constant of the same name/purpose.
DEBUG_CAPTURE_TIMEOUT_MS = 1500

_CHROME_MARKERS = (
    "followers", " posts", "view full profile", "following",
    "view profile", "log in", "sign up", "profile picture",
    "this account is private",
)

_NON_REVIEW_MARKERS = (
    "job description", "years of experience", "apply now", "we're hiring",
    "we are hiring", "job opening", "career opportunit", "job title",
    "questions about benefits", "employee review", "employer review",
    "work-life balance", "interview questions", "glassdoor",
    "salary range", "job posting", "now hiring", "currently hiring",
)

_MORE_COMMENTS_SELECTORS = [
    # Structural-first: real <button> elements matched by their
    # accessible/visible text. Instagram's embed markup renames its
    # generated CSS classes across deploys, but these render as semantic
    # buttons with stable, human-readable labels.
    "button:has-text('View more comments')",
    "button:has-text('Load more comments')",
    "button:has-text('View replies')",
    "button:has-text('Continue this thread')",
    # Instagram sometimes uses a styled <div role="button"> instead of a
    # native <button> for the same controls.
    "[role='button']:has-text('View more comments')",
    "[role='button']:has-text('Load more comments')",
    "[role='button']:has-text('View replies')",
    "[role='button']:has-text('Continue this thread')",
    # Class-based fallback, kept last, only for the one control that has
    # historically had no reliable text/role signature on some embed
    # variants.
    "span:has-text('View all')",
]

# Structural-first selector chain for captions/comment text. Each entry is
# tried in order via _first_matching_locator(); we stop at the first one
# that actually matches something on the page. Instagram rotates its
# generated CSS class names frequently, so leading with layout/structure
# (rows inside the comments list, elements near a <time> element) survives
# class churn far better than a single class-based selector. The
# class-based entry is kept last purely as a last-resort fallback for
# older embed markup that still uses stable "Caption"/"caption" class
# fragments.
_CAPTION_SELECTOR_CHAIN = [
    "article time ~ div",
    "article ul > li div[dir='auto']",
    "[class*='Caption'], [class*='caption']",
]

# Text scraping never needs images or fonts; skipping them cuts page load
# time without affecting anything we read out of the DOM.
_BLOCKED_RESOURCE_TYPES = {"image", "font"}

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
_LAUNCH_ARGS = [
    "--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage",
    "--disable-blink-features=AutomationControlled",
]

# --- Browser pool -----------------------------------------------------------
# One Playwright browser + context per worker thread, kept alive across
# calls instead of being relaunched every time. This avoids paying
# Chromium process-startup cost on every single request and lets cookies
# persist between calls on the same worker thread.
MAX_BROWSER_WORKERS = 3
MAX_USES_BEFORE_RECYCLE = 50

_thread_local = threading.local()
_browser_trace = threading.local()
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=MAX_BROWSER_WORKERS, thread_name_prefix="instagram_scraper"
)


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
            "BROWSER_TRACE scraper=Instagram event=before_sync_playwright_start elapsed=%.3fs timestamp=%.3f thread=%s",
            time.monotonic() - browser_trace_started_at, time.time(), threading.current_thread().name,
        )
        pw = sync_playwright().start()
        logger.info(
            "BROWSER_TRACE scraper=Instagram event=after_sync_playwright_start elapsed=%.3fs timestamp=%.3f thread=%s",
            time.monotonic() - browser_trace_started_at, time.time(), threading.current_thread().name,
        )
        handle.playwright = pw

    logger.info(
        "BROWSER_TRACE scraper=Instagram event=before_shared_browser_attach elapsed=%.3fs timestamp=%.3f thread=%s",
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
        "BROWSER_TRACE scraper=Instagram event=after_shared_browser_attach elapsed=%.3fs timestamp=%.3f thread=%s",
        time.monotonic() - browser_trace_started_at, time.time(), threading.current_thread().name,
    )
    logger.info(
        "BROWSER_TRACE scraper=Instagram event=before_new_context elapsed=%.3fs timestamp=%.3f thread=%s",
        time.monotonic() - browser_trace_started_at, time.time(), threading.current_thread().name,
    )
    context = browser.new_context(
        user_agent=_USER_AGENT,
        locale="en-US",
        viewport={"width": 1280, "height": 900},
    )
    logger.info(
        "BROWSER_TRACE scraper=Instagram event=after_new_context elapsed=%.3fs timestamp=%.3f thread=%s",
        time.monotonic() - browser_trace_started_at, time.time(), threading.current_thread().name,
    )
    context.set_default_timeout(8000)
    context.set_default_navigation_timeout(10000)
    _install_resource_blocking(context)

    handle.browser = browser
    handle.context = context
    handle.uses = 1
    logger.info("Launched a new browser context on %s.", threading.current_thread().name)
    return context


# --- Startup pre-warm --------------------------------------------------
# Same fix as twitter_scraper.py's identically-named block: setup alone
# (this thread's own sync_playwright().start() + attaching to the shared
# Chromium process + new_context()) was regularly taking 25-32s+ under
# this app's shared-browser contention at the start of an analysis run -
# see the OUTER_HARD_TIMEOUT_SECONDS block above. That cost is a ONE-TIME,
# per-worker-thread cost (_ensure_context above reuses a live context on
# later calls via a cheap `ctx.pages` check), so pay it once here, in the
# background, at module import time - i.e. when the FastAPI app boots up
# - well before the first real request in practice, since the user still
# has to go through Company/Product Discovery and pick a product before
# this module's scrape_instagram_comments() is ever called.
#
# MAX_BROWSER_WORKERS separate jobs are submitted (not just one) so every
# thread this module's own _EXECUTOR could later hand a real _run() to
# gets warmed, not just whichever one happens to run first.
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
    # (_run_sync()/_run(), below) already re-asserts Proactor for the
    # same reason; this warm-up needs its own copy since it runs on a
    # separate thread, before any real request ever calls it.
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    try:
        _ensure_context()
        logger.info(
            "Instagram: browser context pre-warmed on %s.",
            threading.current_thread().name,
        )
    except Exception:
        logger.exception(
            "Instagram: background browser context warm-up failed on %s "
            "(non-fatal - the first real request will pay the setup "
            "cost itself instead, same as before this change).",
            threading.current_thread().name,
        )


for _ in range(MAX_BROWSER_WORKERS):
    _EXECUTOR.submit(_warm_up_browser_context)


def _is_non_review(text: str) -> bool:
    low = text.lower()
    return _is_profile_chrome(text) or any(marker in low for marker in _NON_REVIEW_MARKERS)


def _is_profile_chrome(text: str) -> bool:
    low = f" {text.lower()} "
    hits = sum(1 for marker in _CHROME_MARKERS if marker in low)
    return hits >= 2


def _profile_url(target: str) -> str:
    target = (target or "").strip()
    if target.startswith(("http://", "https://")):
        return target
    return f"https://www.instagram.com/{target.lstrip('@')}/"


_BLOCK_TEXT_MARKERS = (
    "log in to see", "log in to continue", "this content isn't available",
    "sorry, this page isn't available",
)


def _first_matching_locator(page, selector_chain):
    """Try each selector in ``selector_chain`` in order; return the first
    one that matches at least one element on the page.

    Falls back to the *last* entry in the chain (the broadest, typically
    class-based selector) if none of the earlier, more structural
    selectors match anything, so behavior never silently regresses to
    "found nothing" just because a structural selector didn't apply to a
    particular page variant. Returns (locator_or_None, selector_used).
    """
    for sel in selector_chain[:-1]:
        try:
            loc = page.locator(sel)
            if loc.count() > 0:
                return loc, sel
        except Exception:
            continue
    fallback_sel = selector_chain[-1]
    try:
        return page.locator(fallback_sel), fallback_sel
    except Exception:
        return None, fallback_sel


def _diagnose_page_state(page) -> str:
    """Classify why a page isn't yielding content, instead of logging a
    generic empty result.

    Returns one of: "ok", "login_wall", "checkpoint", "private_account",
    "no_public_comments", "rendering_failure", "selector_failure".

    URL-based checks run first since they're cheap and reliable; the
    bounded body-text check is the fallback for cases (common on the
    /embed/ pages) where Instagram overlays a login prompt or checkpoint
    without changing the URL at all.
    """
    try:
        url = page.url or ""
    except Exception:
        url = ""
    if "challenge" in url:
        return "checkpoint"
    if "accounts/login" in url or "/login" in url:
        return "login_wall"

    try:
        body_text = page.locator("body").inner_text(timeout=500).lower()
    except Exception:
        return "rendering_failure"

    if "this account is private" in body_text:
        return "private_account"
    if "checkpoint" in body_text:
        return "checkpoint"
    if any(marker in body_text for marker in _BLOCK_TEXT_MARKERS):
        return "login_wall"

    try:
        has_content = page.locator(
            "article, [class*='Caption'], [class*='caption']"
        ).count() > 0
    except Exception:
        return "selector_failure"
    if not has_content:
        return "no_public_comments"

    return "ok"


# --- Browser-free preflight ------------------------------------------------
# The log shows Instagram's embed page (the existing code's own
# login-wall-avoidance attempt) still landing on login_wall_present=True
# for every profile in this run - Instagram has tightened the embed
# endpoint enough that it's no longer a reliable dodge. Instagram is
# optional per the pipeline's priority order, so checking for that same
# login wall with one plain HTTP GET, BEFORE ever calling _ensure_context(),
# means a blocked profile never spends a worker thread on a CDP attach +
# new_context() + navigation at all - freeing that worker sooner for the
# next job, even though attaching to the shared browser (see
# browser_utils.ensure_shared_browser()) is no longer the scarce resource
# it used to be now that Chromium itself is only launched once,
# application-wide.
_PREFLIGHT_TIMEOUT_SECONDS = 4.0
_IG_LOGIN_MARKERS = ("/accounts/login", "/challenge")
_IG_LOGIN_TEXT_MARKERS = (
    "log in • instagram", "login • instagram", "loginform",
    "log in to see photos and videos",
)


def _httpx_preflight(url: str) -> "Tuple[bool, str]":
    """One plain GET, no retry. Returns (blocked, html):

    * blocked - True if the response is already an unambiguous
      login-wall/redirect, False if it looks navigable (or if the check
      itself is inconclusive - in which case the existing Playwright path
      decides, unchanged).
    * html - whatever HTML came back, so the caller can mine the
      contextJSON payload out of it (present even on login-walled embed
      pages) instead of throwing the response away.

    Headers below were widened from a bare User-Agent (the previous
    version) to a fuller, realistic browser set - a request carrying
    ONLY a User-Agent is itself an anomaly no real browser produces, and
    a captured live run showed this exact URL serving a completely
    different, data-free response (the same client-side Polaris app
    shell used for the main site, no contextJSON at all) to that
    single-header request. This won't necessarily change Instagram's
    behavior - see _looks_like_anonymous_shell() below for the fallback
    if it doesn't - but it costs nothing and removes one obvious tell.
    """
    try:
        with httpx.Client(follow_redirects=True) as client:
            resp = client.get(
                url,
                timeout=_PREFLIGHT_TIMEOUT_SECONDS,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                    ),
                    "Accept": (
                        "text/html,application/xhtml+xml,application/xml;q=0.9,"
                        "image/avif,image/webp,*/*;q=0.8"
                    ),
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Upgrade-Insecure-Requests": "1",
                    "Sec-Fetch-Dest": "document",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Site": "none",
                    "Sec-Fetch-User": "?1",
                    "Sec-CH-UA": '"Chromium";v="124", "Not-A.Brand";v="99"',
                    "Sec-CH-UA-Mobile": "?0",
                    "Sec-CH-UA-Platform": '"Windows"',
                    "Connection": "keep-alive",
                },
            )
    except Exception:
        return False, ""  # inconclusive - let the browser path try properly

    html = resp.text or ""
    final_url = str(resp.url)
    if any(marker in final_url for marker in _IG_LOGIN_MARKERS):
        return True, html
    body_sample = html[:20000].lower()
    return any(marker in body_sample for marker in _IG_LOGIN_TEXT_MARKERS), html


# A live capture on 2026-07-22 showed the embed URL serving a 600KB+
# response that returns 200 OK, isn't caught by _IG_LOGIN_TEXT_MARKERS
# (no "log in" text anywhere), and yet carries zero post/caption data -
# it's Instagram's normal logged-out "Polaris" web-app shell (React
# app config + CSS variables + feature flags only), not the lighter
# legacy embed page contextJSON relies on. The browser fallback that
# ran immediately afterward hit an actual rendered login wall on BOTH
# the embed and profile URLs for the same profile in that same run -
# so when this shell shows up, spending ~20-30s on Playwright
# navigation has (so far, empirically) never once recovered anything
# it didn't already know was unavailable. Recognizing it lets the
# caller skip straight to returning early, same as a normal blocked
# preflight, instead of paying for a browser attempt already shown
# unlikely to help.
#
# Deliberately conservative: only fires when BOTH the Polaris shell's
# own marker is present AND every known real-data marker is absent, so
# a page that happens to be a Polaris page but still carries data isn't
# misclassified.
_POLARIS_SHELL_MARKERS = ("PolarisProfilePage", "PolarisSEO")
_REAL_DATA_MARKERS = (
    "contextJSON", "graphql_media", "edge_media_to_caption",
    "biography", "edge_owner_to_timeline_media",
)


def _looks_like_anonymous_shell(html: str) -> bool:
    if not html:
        return False
    return (
        any(m in html for m in _POLARIS_SHELL_MARKERS)
        and not any(m in html for m in _REAL_DATA_MARKERS)
    )




def _looks_blocked(page) -> bool:
    """Kept for internal call-site compatibility; now backed by the fuller
    ``_diagnose_page_state`` classification above."""
    return _diagnose_page_state(page) != "ok"


# --- Embedded-JSON extraction ----------------------------------------------
# Instagram's profile /embed/ page now login-walls the *rendered* DOM for
# unauthenticated sessions (every recent run's debug capture shows
# login_wall_present=True with zero caption elements), but the very same
# HTML still ships the profile's recent posts as structured JSON inside a
# "contextJSON" script payload — including each post's caption text and
# shortcode (verified against a saved debug capture: 6 posts, all with
# captions). Parsing that JSON out of the raw HTML sidesteps the login
# wall entirely and needs no browser at all.
_CONTEXT_JSON_RE = re.compile(r'"contextJSON":"((?:[^"\\]|\\.)*)"')


def _extract_context_media_nodes(html: str) -> List[dict]:
    """Parse the contextJSON payload out of a profile/post /embed/ page's
    raw HTML and return the raw list of per-post ``shortcode_media`` node
    dicts (one entry per post Instagram included in that page's payload).
    Returns [] when the payload is absent or unparseable — callers treat
    that as "nothing found", never as an error.

    Shared by _parse_embed_context() (flattened, for pages that only ever
    carry one post) and _parse_embed_context_grouped() (kept per-post, for
    the profile page - see that function's docstring for why the two
    differ)."""
    if not html:
        return []
    m = _CONTEXT_JSON_RE.search(html)
    if not m:
        return []
    try:
        # The payload is a JSON string *inside* a JSON document, so it
        # decodes in two steps: un-escape the string, then parse it.
        data = json.loads(json.loads('"' + m.group(1) + '"'))
    except Exception:
        return []
    ctx = (data or {}).get("context") or {}
    nodes: List[dict] = []
    for media in ctx.get("graphql_media") or []:
        node = (media or {}).get("shortcode_media") or {}
        if isinstance(node, dict) and node:
            nodes.append(node)
    return nodes


def _parse_embed_context(html: str) -> "Tuple[List[str], List[str]]":
    """Extract (caption/comment texts, post shortcodes) from a profile or
    post /embed/ page's raw HTML, flattened across every post the payload
    carries. Returns ([], []) when the payload is absent or unparseable —
    callers treat that as "nothing found", never as an error.

    Safe on a page that only ever describes ONE post (an individual
    post's /embed/captioned/ page) since there is no post boundary to
    lose there. On a page that can carry SEVERAL posts at once (the
    profile grid), use _parse_embed_context_grouped() instead — flattening
    there would mean a product-specific filter can no longer tell which
    caption a given comment's own post actually had."""
    texts: List[str] = []
    shortcodes: List[str] = []
    for node in _extract_context_media_nodes(html):
        sc = node.get("shortcode")
        if sc and sc not in shortcodes:
            shortcodes.append(sc)
        for key in ("edge_media_to_caption", "edge_media_to_parent_comment",
                    "edge_media_preview_comment"):
            for edge in ((node.get(key) or {}).get("edges")) or []:
                text = (((edge or {}).get("node") or {}).get("text") or "").strip()
                if text:
                    texts.append(text)
    return texts, shortcodes


def _parse_embed_context_grouped(html: str) -> "List[Dict[str, Any]]":
    """Same contextJSON payload as _parse_embed_context(), kept grouped by
    post instead of flattened. Returns one dict per post: {"shortcode":
    str|None, "caption": str, "comments": List[str]}.

    _parse_embed_context() throws away which caption a given comment's
    post actually had, which is harmless on a single-post page but not on
    the profile grid: that page's payload can describe several different
    recent posts at once, each with its own caption and its own comments.
    Preserving the grouping lets a product-specific job gate on each
    POST's own caption - the same "check the caption once, then trust the
    whole thread" rule the per-post browser visit further below already
    applies - instead of independently pattern-matching every individual
    comment's own text against the product name. Real commenters almost
    never restate the full product name in a comment ("love these!",
    "how's the battery on these?"), so per-comment-text filtering quietly
    discards most genuine, on-topic comments that a caption-level gate
    would correctly keep - see _select_relevant_texts() below, which
    consumes this function's output."""
    posts: "List[Dict[str, Any]]" = []
    for node in _extract_context_media_nodes(html):
        caption = ""
        for edge in ((node.get("edge_media_to_caption") or {}).get("edges")) or []:
            text = (((edge or {}).get("node") or {}).get("text") or "").strip()
            if text:
                caption = text
                break
        comments: List[str] = []
        for key in ("edge_media_to_parent_comment", "edge_media_preview_comment"):
            for edge in ((node.get(key) or {}).get("edges")) or []:
                text = (((edge or {}).get("node") or {}).get("text") or "").strip()
                if text:
                    comments.append(text)
        posts.append({
            "shortcode": node.get("shortcode"),
            "caption": caption,
            "comments": comments,
        })
    return posts


def _select_relevant_texts(posts: "List[Dict[str, Any]]", product_name: str) -> List[str]:
    """Given _parse_embed_context_grouped()'s per-post list, return the
    caption/comment texts worth keeping for this job.

    No product_name (the "General" job): every post is relevant by
    definition, so every caption and comment text is kept - mirrors
    _add_if_relevant()'s own "no product_name" branch below.

    With a product_name: gate per POST on that post's own caption; if it
    mentions the product, keep that caption plus ALL of that post's
    comments, regardless of whether each comment's own text happens to
    repeat the product name. If the caption doesn't mention the product,
    skip the whole post - including any comment on it that might
    coincidentally contain the product's tokens, since that comment's
    own post was never actually about this product."""
    out: List[str] = []
    for post in posts:
        caption = post.get("caption") or ""
        if product_name and not _mentions_product(caption, product_name):
            continue
        if caption:
            out.append(caption)
        out.extend(post.get("comments") or [])
    return out


def _clean_candidate(txt: str) -> str:
    """Shared filter for caption/comment candidates: strips trailing UI
    chrome and rejects short/non-review text. Returns "" when the text
    should be dropped. Used both by the browser path's ``_add`` and by the
    browser-free embedded-JSON path."""
    txt = (txt or "").strip()
    # Embed-DOM captions arrive wrapped in widget chrome: a leading bare
    # username line and a trailing "View all N comments" link. Stripping
    # both makes the DOM copy identical to the contextJSON copy of the
    # same caption, so the two dedupe instead of double-counting.
    txt = re.sub(r"^[A-Za-z0-9._]{2,30}\s*\n+", "", txt)
    txt = re.sub(r"\s*View all(\s+\d+)?\s+comments\s*$", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\s*(Reply|See translation)\s*$", "", txt).strip()
    if not txt or len(txt.split()) < 6 or _is_non_review(txt):
        return ""
    return txt


# Evidence directory for zero-result debug captures. Lives at
# <project_root>/debug (sibling of app.py's DOWNLOADS_DIR), since this file
# is itself at <project_root>/scrapers/instagram_scraper.py.
_DEBUG_DIR = Path(__file__).resolve().parent.parent / "debug"


def _title_bounded(page, timeout_ms: int, default: str = "<unavailable>") -> str:
    """Get the page title with a timeout that Playwright actually honors.

    ``page.title()`` takes no ``timeout`` argument and is not covered by
    ``page.set_default_timeout()`` either - verified against Playwright's
    own ``Frame.title()``, which sends its protocol call with no timeout
    calculator at all, so on a page left in a half-navigated/unresponsive
    state it can hang indefinitely waiting for a browser response that
    never comes. That is the actual mechanism behind "gave up on
    navigation with time to spare, still got cut off by the outer timeout"
    in production logs (same root cause found and fixed once already in
    twitter_scraper.py, which this scraper independently had too).

    ``Locator.text_content(timeout=...)``, unlike ``page.title()``, DOES
    route through Playwright's real timeout machinery. Querying the
    ``<title>`` element this way gets the same information with an
    actually-enforced bound, and does it on the calling thread - a
    cross-thread watchdog was tried for this in twitter_scraper.py first
    and abandoned: Playwright's sync API is greenlet-bound to whichever
    thread started it, and calling it from another thread breaks that
    binding (surfaces as "greenlet.error: cannot switch to a different
    thread" and orphaned "Task exception was never retrieved" warnings).
    """
    try:
        return page.locator("title").text_content(timeout=timeout_ms) or default
    except Exception:
        return default


def _content_bounded(page, timeout_ms: int) -> Optional[str]:
    """Get an HTML dump with a timeout Playwright actually honors, for the
    same reason ``_title_bounded`` exists: ``page.content()`` takes no
    timeout and isn't covered by ``set_default_timeout()``.
    ``Locator.inner_html(timeout=...)`` on the ``html`` element IS
    genuinely bounded. Returns ``None`` (rather than a watered-down guess)
    if it can't get the HTML within the timeout, so callers can treat that
    as "no content available" instead of silently hanging or getting an
    empty string that reads as "the page really was empty".
    """
    try:
        inner = page.locator("html").inner_html(timeout=timeout_ms)
    except Exception:
        return None
    return f"<html>{inner}</html>"


def _save_debug_artifacts(page, identifier: str, reason: str, extra: Dict[str, str] = None) -> None:
    """Save HTML/screenshot/context ONLY on failure/zero-results for
    debugging. Never on the success path.

    Writes three files under ``_DEBUG_DIR``, all sharing one base name
    stamped with date-time + milliseconds + thread id so concurrent or
    rapid-fire failures never overwrite each other:
      * <base>.png - screenshot
      * <base>.html - full page HTML
      * <base>.txt  - final URL, page title, reason, and whatever the
        caller passes in ``extra`` (e.g. embed/profile page load status,
        login-wall presence).
    """
    try:
        _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        safe_id = re.sub(r'[^a-zA-Z0-9]', '_', identifier)[:50].strip('_') or "unknown"
        stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}_{threading.get_ident()}"
        base = _DEBUG_DIR / f"instagram_{safe_id}_{stamp}"

        # This capture only ever runs on a failure/zero-result path, right
        # after the scraper has already decided to give up - it must never
        # be allowed to itself eat into the SAFETY_MARGIN_SECONDS the
        # deadline math budgeted for returning cleanly.
        #
        # page.screenshot() below takes an explicit timeout= and is
        # genuinely bounded by it. Title/HTML capture go through
        # _title_bounded()/_content_bounded() instead of raw
        # page.title()/page.content(), which take no timeout parameter and
        # are not covered by set_default_timeout() either - on a page left
        # in exactly the half-navigated/unresponsive state this function
        # exists to diagnose, either raw call can hang indefinitely waiting
        # for a browser response that never comes, blowing through the
        # safety margin and getting the whole job cancelled by app.py's
        # outer hard timeout before _run() can return - which throws away
        # results that were otherwise fine. That is the concrete mechanism
        # behind "gave up on navigation with time to spare, still got cut
        # off by the outer timeout" in production logs.
        try:
            page.set_default_timeout(1000)
        except Exception:
            pass

        try:
            final_url = page.url or ""
        except Exception:
            final_url = "<unavailable>"
        title = _title_bounded(page, 1000)

        try:
            # viewport-only + a short bounded timeout, not full_page: a
            # feed page whose comments never finished settling (exactly
            # the case this runs on) can make a full_page screenshot keep
            # scrolling/re-laying-out for the whole 3s this used to allow,
            # for a debug artifact that never needed the full scroll
            # history anyway. Mirrors the same fix already applied to
            # twitter_scraper.py's DEBUG_CAPTURE_TIMEOUT_MS.
            page.screenshot(path=f"{base}.png", timeout=DEBUG_CAPTURE_TIMEOUT_MS, full_page=False)
        except Exception:
            logger.warning("[INSTAGRAM_DEBUG] could not capture screenshot for %s", identifier)

        html = _content_bounded(page, 1500)
        if html is not None:
            try:
                with open(f"{base}.html", "w", encoding="utf-8") as f:
                    f.write(html)
            except Exception:
                logger.warning("[INSTAGRAM_DEBUG] could not capture HTML for %s", identifier)
        else:
            logger.warning(
                "[INSTAGRAM_DEBUG] could not capture HTML for %s (timed out or errored)", identifier
            )

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
            "[INSTAGRAM_DEBUG] saved zero-result evidence for %s reason=%s: %s.{png,html,txt}",
            identifier, reason, base,
        )
    except Exception:
        logger.exception("[INSTAGRAM_DEBUG] failed to save debug artifacts for %s", identifier)


def _find_scroll_container(page):
    """Find the element that actually holds the scrollable comments feed,
    instead of assuming either "the whole page" or a blindly-supplied
    locator is correct.

    Scrolling the wrong (non-scrollable) container is indistinguishable
    from "nothing new loaded" and makes the idle-detection in
    ``_scroll_until_idle`` trigger for the wrong reason. Returns ``None``
    (meaning "fall back to page-level scroll") if nothing on the page is
    actually scrollable.
    """
    for sel in ("article", "[role='dialog']", "main", "body"):
        try:
            loc = page.locator(sel).first
            if loc.count() == 0:
                continue
            scrollable = loc.evaluate("el => el.scrollHeight > el.clientHeight + 20")
            if scrollable:
                return loc
        except Exception:
            continue
    return None


NAV_RETRY_BACKOFF_SECONDS = 0.75


def _adaptive_nav_timeout(time_left) -> int:
    """Scale the navigation timeout to how much of the internal time
    budget is actually left, instead of a fixed 5000ms for every attempt.

    Navigation always receives at least NAV_TIMEOUT_MS_MIN (8s) as long as
    that much time is actually left in the budget - the budget clock only
    starts once browser/context/page setup has finished (see
    scrape_instagram_comments below), so this floor is real, not eaten by
    Chromium startup. When plenty of budget remains the timeout can scale
    up to NAV_TIMEOUT_MS_MAX (12s) for a genuinely slow page; when the
    budget itself is under the floor, we hand over whatever is left rather
    than blocking past the scraper's own deadline.
    """
    remaining_ms = max(0.0, time_left()) * 1000
    if remaining_ms <= NAV_TIMEOUT_MS_MIN:
        return max(3000, int(remaining_ms))
    return int(min(NAV_TIMEOUT_MS_MAX, max(NAV_TIMEOUT_MS_MIN, remaining_ms * 0.5)))


def _goto_with_retry(page, url: str, *, timeout: int, time_left, retries: int = NAVIGATION_RETRIES) -> bool:
    """Navigate with retries, logging, and a short backoff between attempts.

    ``timeout`` is accepted for call-site compatibility but is intentionally
    ignored: we recompute an adaptive timeout before every attempt so that
    retries never ask for more time than the scraper actually has left.
    """
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
                "Instagram nav retry attempt=%d/%d url=%s remaining=%.1fs chosen_timeout=%dms",
                attempt + 1, total_attempts, url, remaining, attempt_timeout,
            )
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=attempt_timeout)
            return True
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Instagram nav attempt %d/%d to %s failed: %s",
                attempt + 1, total_attempts, url, exc,
            )
            if attempt < retries:
                # Cap the backoff sleep to what is actually left minus the
                # minimum guard (3 s) so we never sleep past the deadline.
                backoff = min(
                    NAV_RETRY_BACKOFF_SECONDS * (attempt + 1),
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


def _expand_more_comments(page, time_left, limit: int = 20) -> int:
    """Click "View more comments" / "Load more comments" style controls."""
    clicked = 0
    for sel in _MORE_COMMENTS_SELECTORS:
        if clicked >= limit or time_left() <= 2:
            break
        try:
            buttons = page.locator(sel).all()
        except Exception:
            continue
        for btn in buttons[: max(0, limit - clicked)]:
            if time_left() <= 2:
                break
            try:
                if btn.is_visible():
                    btn.click(timeout=600)
                    clicked += 1
                    page.wait_for_timeout(200)
            except Exception:
                continue
    return clicked


def _wait_for_more(page, count_fn, prev_count: int, max_ms: int = 500) -> int:
    """Poll ``count_fn()`` in short steps until it exceeds ``prev_count`` or
    ``max_ms`` elapses, whichever comes first.

    Replaces an unconditional ``page.wait_for_timeout(max_ms)`` after a
    scroll step: the worst-case wait is identical (this never runs longer
    than ``max_ms``), but as soon as new items attach - which is most of
    the time when content is actually available - this returns immediately
    instead of always sitting out the full window, leaving more of the
    scraper's time budget for additional scroll iterations. Mirrors the
    same helper in youtube_scraper.py / google_scraper.py / twitter_scraper.py,
    adapted here to reuse the ``count_fn`` already threaded through this
    function's caller instead of a fixed selector.
    """
    deadline = time.monotonic() + (max_ms / 1000)
    count = prev_count
    while time.monotonic() < deadline:
        try:
            count = count_fn()
        except Exception:
            return count
        if count > prev_count:
            return count
        page.wait_for_timeout(80)
    return count


def _scroll_until_idle(page, time_left, count_fn, max_items: int, scroll_target=None) -> int:
    """Generic idle-bounded scroll loop shared by the profile/post passes.

    Keeps scrolling (page-level, or a specific ``scroll_target`` locator if
    given) until ``max_items`` have been located, ``MAX_SCROLL_ITERATIONS``
    scroll steps have happened, ``SCROLL_IDLE_LIMIT`` consecutive scrolls
    produced no new items, or the overall scraper time budget runs low.

    Returns the last observed count.

    Note: Instagram does not serve most content (full comment threads,
    additional posts beyond the first screenful, etc.) to unauthenticated
    sessions — it redirects to a login wall instead. In practice this loop
    will often exit quickly via the idle-streak condition simply because
    nothing new is available to load without logging in, not because of a
    bug here.
    """
    iteration = 0
    idle = 0
    prev_count = -1
    start = time.monotonic()
    while True:
        iteration += 1
        if iteration > MAX_SCROLL_ITERATIONS:
            logger.info("Instagram scroll: hit MAX_SCROLL_ITERATIONS (%d).", MAX_SCROLL_ITERATIONS)
            break
        if time_left() <= 2:
            logger.info("Instagram scroll: stopping, low on overall time budget.")
            break

        count = count_fn()
        logger.info(
            "Instagram scroll: iteration=%d current_count=%d idle_streak=%d elapsed=%.1fs",
            iteration, count, idle, time.monotonic() - start,
        )
        if count >= max_items:
            logger.info("Instagram scroll: reached target of %d item(s).", max_items)
            break
        if count <= prev_count:
            idle += 1
            if idle >= SCROLL_IDLE_LIMIT:
                logger.info(
                    "Instagram scroll: no new items after %d consecutive scrolls; stopping.",
                    SCROLL_IDLE_LIMIT,
                )
                break
        else:
            idle = 0
        prev_count = max(prev_count, count)

        _expand_more_comments(page, time_left)
        try:
            if scroll_target is not None:
                scroll_target.evaluate("el => el.scrollTo(0, el.scrollHeight)")
            else:
                page.mouse.wheel(0, 1200)
        except Exception:
            break
        # Readiness-based wait: return as soon as new items attach instead
        # of always sitting out the full 450ms window. Same worst-case
        # bound as the previous fixed sleep.
        _wait_for_more(page, count_fn, prev_count, max_ms=450)

    return max(prev_count, 0)


def _mentions_product(text: str, product_name: str) -> bool:
    """Same precise/strict matcher as twitter_scraper.py's version: every
    meaningful token of ``product_name`` must appear in ``text``. Used to
    decide, per-post, whether a caption is actually about this specific
    product before spending time budget expanding its comments.
    """
    if not text or not product_name:
        return False
    text_l = text.lower()
    tokens = [t for t in re.findall(r"[a-z0-9]+", product_name.lower()) if len(t) > 1]
    if not tokens:
        return False
    return all(t in text_l for t in tokens)


async def scrape_instagram_comments(company_data: Dict[str, str]) -> List[str]:
    # Anchor point for the real wall-clock deadline (see the
    # OUTER_HARD_TIMEOUT_SECONDS block above). Captured before the httpx
    # preflight check and before this job is even submitted to the
    # executor, so _run()'s deadline reflects the FULL time app.py's outer
    # wait_for() has already been counting against this coroutine - not
    # just the part after browser/context/page setup finishes.
    _overall_start = time.monotonic()

    # When set, this call is for one specific selected product (not the
    # brand-wide "General" job) - every content source below gets filtered
    # down to posts/captions that actually mention it, instead of dumping
    # the brand's whole recent-posts feed.
    product_name = (company_data.get("product_name") or "").strip()

    target = (
        company_data.get("instagram_url")
        or company_data.get("instagram")
    )

    loop = asyncio.get_event_loop()

    if not target:
        company_name = (company_data.get("company_name") or "").strip()
        if company_name:
            try:
                target = await loop.run_in_executor(
                    _EXECUTOR, find_social_profile_url, company_name, "instagram",
                )
            except Exception:
                target = ""
            if target:
                target = target.split("?")[0].split("#")[0]
                logger.info(
                    "Instagram scrape: no handle from the site scan; "
                    "search fallback found %r for %r.", target, company_name,
                )
        if not target:
            logger.info(
                "Instagram scrape: no handle from the site scan and the "
                "search fallback found nothing for %r; skipping rather "
                "than guessing a handle from the name.",
                company_data.get("company_name"),
            )
            return []

    handle = target
    for prefix in ("https://www.instagram.com/", "https://instagram.com/"):
        if handle.startswith(prefix):
            handle = handle[len(prefix):].strip("/")
    handle = handle.lstrip("@")

    profile_url = _profile_url(handle)
    company_name = company_data.get("company_name", handle)

    embed_preflight_url = profile_url.rstrip("/") + "/embed/"
    try:
        preflight_blocked, preflight_html = await loop.run_in_executor(
            _EXECUTOR, _httpx_preflight, embed_preflight_url,
        )
    except Exception:
        preflight_blocked, preflight_html = False, ""

    # TEMP DIAGNOSTIC: the live preflight response is currently invisible
    # once this function returns - only the *browser's* failed page state
    # gets written to debug/. Log + save it unconditionally so a run that
    # comes back with zero seed_texts tells us WHY (short/blocked body?
    # contextJSON missing entirely? present but unparseable?) instead of
    # just that it happened. Safe to remove once Instagram is confirmed
    # working again.
    logger.info(
        "Instagram scrape: preflight GET %s -> %d byte(s), "
        "contextJSON_substring_present=%s, blocked=%s.",
        embed_preflight_url, len(preflight_html or ""),
        "contextJSON" in (preflight_html or ""), preflight_blocked,
    )
    if preflight_html:
        try:
            _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            with open(_DEBUG_DIR / "instagram_preflight_last.html", "w", encoding="utf-8") as _f:
                _f.write(preflight_html)
        except Exception:
            logger.warning("Instagram scrape: could not save preflight diagnostic HTML.")

    # Mine the embed page's contextJSON payload (recent posts' captions +
    # shortcodes) out of the plain-HTTP response before any browser work.
    # This survives the login wall: Instagram walls the rendered DOM but
    # still ships this JSON in the same HTML.
    seed_texts, seed_shortcodes = _parse_embed_context(preflight_html)
    # Kept grouped by post too (same payload, see _parse_embed_context_grouped's
    # docstring) so a product-specific job can gate on each post's own
    # caption instead of pattern-matching every individual caption/comment
    # text - both the early-return branch just below and _run() further
    # down use this instead of the flat seed_texts for relevance decisions.
    seed_posts = _parse_embed_context_grouped(preflight_html)
    if seed_texts or seed_shortcodes:
        logger.info(
            "Instagram scrape: embedded contextJSON on %s yielded %d "
            "caption/comment text(s) across %d post(s) without a browser.",
            embed_preflight_url, len(seed_texts), len(seed_posts),
        )

    anonymous_shell = (
        not preflight_blocked
        and not seed_texts and not seed_shortcodes
        and _looks_like_anonymous_shell(preflight_html)
    )

    if preflight_blocked or anonymous_shell:
        # Gate per-post on each post's own caption before cleaning, rather
        # than cleaning every flattened text and then separately
        # pattern-matching each one against the product name - the latter
        # discards genuine comments that don't happen to repeat the
        # product name back (see _select_relevant_texts()'s docstring).
        relevant_texts = _select_relevant_texts(seed_posts, product_name)
        seed_clean = normalize_comments(
            [t for t in (_clean_candidate(c) for c in relevant_texts) if t]
        )[:MAX_INSTAGRAM_COMMENTS]
        if anonymous_shell:
            logger.info(
                "Instagram scrape: preflight for %s came back as the "
                "logged-out Polaris app shell (no contextJSON, no post "
                "data of any kind) rather than a login-wall redirect - "
                "returning %d embedded item(s) immediately instead of "
                "spending the browser budget on a page shown (in every "
                "run checked so far) to also render a login wall "
                "(reason=anonymous_shell, no Chromium launch spent on this).",
                embed_preflight_url, len(seed_clean),
            )
        else:
            logger.info(
                "Instagram scrape: preflight detected a login wall for %s before "
                "touching the browser pool - returning %d embedded-JSON item(s) "
                "immediately (reason=login_wall, no Chromium launch spent on this).",
                embed_preflight_url, len(seed_clean),
            )
        return seed_clean

    def _run() -> List[str]:
        if sys.platform.startswith("win"):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

        setup_start = time.monotonic()

        try:
            context = _ensure_context()
        except Exception:
            logger.exception("Could not start/obtain a browser context for %r.", profile_url)
            return []

        results: List[str] = []
        duplicates_removed = 0
        seen = set()

        def _add(txt: str) -> None:
            nonlocal duplicates_removed
            # _clean_candidate strips trailing UI affordances that
            # sometimes get bundled into the same text node as the
            # caption/comment itself (a lingering "Reply" button label,
            # "See translation" link), so two otherwise-identical comments
            # aren't treated as unique just because one has this trailing
            # chrome and the other doesn't.
            txt = _clean_candidate(txt)
            if not txt:
                return
            key = " ".join(txt.split()).lower()
            if key in seen:
                duplicates_removed += 1
                return
            seen.add(key)
            results.append(txt)

        def _add_if_relevant(txt: str) -> None:
            # For the "General" job (no product_name) every candidate is
            # relevant by definition. For a product-specific job, only
            # keep text that actually mentions the product - genuine over
            # volume, no falling back to unrelated brand content.
            if not product_name or _mentions_product(txt, product_name):
                _add(txt)

        # Captions/comments recovered browser-free from the embed page's
        # contextJSON payload go in first — the browser passes below only
        # need to add to them (and the dedupe in _add keeps overlap
        # harmless). Already gated per-post by _select_relevant_texts(), so
        # this calls _add() directly rather than _add_if_relevant(), which
        # would re-run the per-text relevance check and wrongly reject a
        # genuine comment just because it doesn't itself repeat the
        # product name.
        for txt in _select_relevant_texts(seed_posts, product_name):
            _add(txt)

        page = None
        try:
            logger.info(
                "BROWSER_TRACE scraper=Instagram event=before_new_page elapsed=%.3fs timestamp=%.3f thread=%s",
                time.monotonic() - getattr(_browser_trace, "ensure_context_started_at", time.monotonic()), time.time(), threading.current_thread().name,
            )
            page = context.new_page()
            logger.info(
                "BROWSER_TRACE scraper=Instagram event=after_new_page elapsed=%.3fs timestamp=%.3f thread=%s",
                time.monotonic() - getattr(_browser_trace, "ensure_context_started_at", time.monotonic()), time.time(), threading.current_thread().name,
            )
            page.set_default_timeout(7000)

            setup_elapsed = time.monotonic() - setup_start
            start = time.monotonic()
            # Deadline anchored to the OUTER hard timeout (app.py's
            # INSTAGRAM_TIMEOUT_SECONDS), counted from when this coroutine
            # actually started (_overall_start) - not a fresh
            # TIME_BUDGET_SECONDS clock that ignores how much of the outer
            # budget setup already used. This guarantees _run() always
            # leaves SAFETY_MARGIN_SECONDS of real slack for the result to
            # make it back through run_in_executor()/wait_for(), no matter
            # how long setup took under this app's shared-browser
            # contention. See the OUTER_HARD_TIMEOUT_SECONDS block above.
            deadline = _overall_start + OUTER_HARD_TIMEOUT_SECONDS - SAFETY_MARGIN_SECONDS
            remaining_budget = deadline - start

            logger.info(
                "Instagram scrape: browser/context/page setup took %.1fs "
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
                    "Instagram scrape: only %.1fs left for %s after a "
                    "%.1fs setup (job start to now: %.1fs) - not enough "
                    "time to attempt navigation; returning early (with "
                    "whatever embedded-JSON seed content was already "
                    "collected) instead of risking the outer hard timeout.",
                    remaining_budget, profile_url, setup_elapsed,
                    time.monotonic() - _overall_start,
                )
                return normalize_comments(results)[:MAX_INSTAGRAM_COMMENTS]

            def time_left() -> float:
                return deadline - time.monotonic()

            # --- Primary attempt: public embed page (no login wall) -------
            embed_url = profile_url.rstrip("/") + "/embed/"
            # Post permalinks discovered browser-free from contextJSON
            # shortcodes come first; the click-through discovery below only
            # runs if this seeding produced nothing.
            post_links: List[str] = [
                f"https://www.instagram.com/p/{sc}/" for sc in seed_shortcodes
            ]
            profile_page_loaded = None  # None = fallback never attempted
            embed_page_loaded = _goto_with_retry(page, embed_url, timeout=_adaptive_nav_timeout(time_left), time_left=time_left)
            if embed_page_loaded:
                # Readiness-based wait: wait for actual content to show up
                # rather than sleeping a fixed amount of time regardless of
                # whether the page is ready sooner or slower than that.
                try:
                    page.wait_for_selector(
                        "article, [class*='Caption'], [class*='caption'], "
                        "a[href*='/p/'], a[href*='/reel/']",
                        timeout=4000,
                    )
                except Exception:
                    reason = _diagnose_page_state(page)
                    logger.info(
                        "Instagram scrape: embed page for %s never rendered "
                        "caption/post content within 4s (reason=%s).",
                        profile_url, reason,
                    )

                def _caption_count() -> int:
                    try:
                        loc, _sel = _first_matching_locator(page, _CAPTION_SELECTOR_CHAIN)
                        return loc.count() if loc is not None else -1
                    except Exception:
                        return -1

                scroll_container = _find_scroll_container(page)
                _scroll_until_idle(
                    page, time_left, _caption_count, MAX_INSTAGRAM_COMMENTS,
                    scroll_target=scroll_container,
                )
                caption_loc, used_sel = _first_matching_locator(page, _CAPTION_SELECTOR_CHAIN)
                if caption_loc is not None:
                    logger.info(
                        "Instagram scrape: embed page for %s using caption "
                        "selector %r.", profile_url, used_sel,
                    )
                    for loc in caption_loc.all():
                        try:
                            _add_if_relevant(loc.inner_text())
                        except Exception:
                            pass
                # The rendered DOM is usually login-walled, but the page
                # source still carries the contextJSON payload — mine it
                # here too in case the browser was served a different
                # variant than the httpx preflight (or the preflight failed).
                try:
                    embed_html = _content_bounded(page, 1500)
                    # Grouped (not the flat _parse_embed_context) since this
                    # is the profile grid - it can describe several posts
                    # in one payload, and _add_if_relevant() on the
                    # flattened text would filter each caption/comment
                    # independently, losing which post a given comment
                    # actually belongs to (see _parse_embed_context_grouped's
                    # docstring).
                    page_posts = _parse_embed_context_grouped(embed_html)
                    for txt in _select_relevant_texts(page_posts, product_name):
                        _add(txt)
                    for post in page_posts:
                        sc = post.get("shortcode")
                        if not sc:
                            continue
                        link = f"https://www.instagram.com/p/{sc}/"
                        if link not in post_links:
                            post_links.append(link)
                except Exception:
                    pass
                try:
                    # Instagram's profile embed grid renders every
                    # thumbnail as <a href="#" role="link"> - the real
                    # navigation is wired up via a JS click handler, not a
                    # normal href, so scanning for a[href*='/p/'] (the old
                    # approach) can never find anything here and
                    # post_links stayed empty. Click each thumbnail
                    # instead - that fires the real handler - and read
                    # the resulting page URL, which the widget updates to
                    # the individual post's own permalink.
                    thumbs_sel = "a[role='link'][href='#']"
                    # Skip click-through discovery entirely when contextJSON
                    # already gave us real permalinks — each discovery click
                    # costs a navigation round-trip out of the time budget.
                    thumb_count = 0 if post_links else page.locator(thumbs_sel).count()
                    for idx in range(min(thumb_count, MAX_POSTS_TO_DISCOVER)):
                        if time_left() <= 4:
                            break
                        try:
                            thumbs = page.locator(thumbs_sel)
                            if idx >= thumbs.count():
                                break
                            thumbs.nth(idx).click(timeout=2000)
                            page.wait_for_timeout(400)
                            new_url = page.url or ""
                            if (
                                ("/p/" in new_url or "/reel/" in new_url)
                                and new_url not in post_links
                            ):
                                post_links.append(new_url)
                            else:
                                # No real navigation happened - the click
                                # may have opened an in-page overlay
                                # instead. Grab whatever caption/comment
                                # content is visible right now before
                                # closing it, so this thumbnail isn't a
                                # total loss either way.
                                try:
                                    overlay_loc, _ = _first_matching_locator(
                                        page, _CAPTION_SELECTOR_CHAIN
                                    )
                                    if overlay_loc is not None:
                                        # _CAPTION_SELECTOR_CHAIN matches
                                        # both the caption and comment
                                        # list items (see its definition),
                                        # and every element it finds here
                                        # comes from the ONE post this
                                        # overlay opened for. Gate once
                                        # across everything found - the
                                        # same "check once, keep the whole
                                        # thread" rule the per-post browser
                                        # visit below uses - instead of
                                        # re-testing each element on its
                                        # own text, which would silently
                                        # drop genuine comments that don't
                                        # individually repeat the product
                                        # name (the same issue the
                                        # contextJSON grouping fix
                                        # addressed for the profile-level
                                        # payload).
                                        overlay_texts = []
                                        for loc in overlay_loc.all():
                                            try:
                                                overlay_texts.append(loc.inner_text())
                                            except Exception:
                                                pass
                                        if not product_name or any(
                                            _mentions_product(t, product_name)
                                            for t in overlay_texts
                                        ):
                                            for t in overlay_texts:
                                                _add(t)
                                except Exception:
                                    pass
                            page.go_back(timeout=3000)
                            page.wait_for_selector(thumbs_sel, timeout=3000)
                        except Exception:
                            # Best-effort: if the click/back cycle leaves
                            # the page in a bad state, reload the grid
                            # fresh rather than abandoning the rest of
                            # the thumbnails.
                            try:
                                _goto_with_retry(
                                    page, embed_url,
                                    timeout=_adaptive_nav_timeout(time_left),
                                    time_left=time_left,
                                )
                            except Exception:
                                break
                except Exception:
                    pass

            # --- Fallback: direct profile page, skipped instantly if it
            # redirects to a login/checkpoint wall ------------------------
            if len(results) < MIN_RESULTS_BEFORE_FALLBACK and time_left() > 4:
                profile_page_loaded = _goto_with_retry(page, profile_url, timeout=_adaptive_nav_timeout(time_left), time_left=time_left)
                if profile_page_loaded:
                    try:
                        page.wait_for_selector(
                            "h1, span._aacl, div._aacl, [role='main']", timeout=3000,
                        )
                    except Exception:
                        pass
                    reason = _diagnose_page_state(page)
                    if reason != "ok":
                        logger.info(
                            "Instagram scrape: profile fallback for %s "
                            "unavailable (reason=%s); skipping extraction.",
                            profile_url, reason,
                        )
                    else:
                        # Header/bio text here is profile-level chrome, not
                        # post content - it's essentially never genuinely
                        # about one specific product, so skip adding it at
                        # all for a product-specific job rather than
                        # running it through the relevance filter (which
                        # would almost always just reject it anyway).
                        # Post-link discovery just below is unaffected -
                        # it's still useful either way.
                        if not product_name:
                            for sel in ["span._aacl", "div._aacl", "h1", "span"]:
                                for loc in page.locator(sel).all()[:50]:
                                    try:
                                        txt = loc.inner_text()
                                        if txt and len(txt.split()) >= 6:
                                            _add(txt)
                                    except Exception:
                                        pass
                        try:
                            for a in page.locator("a[href*='/p/'], a[href*='/reel/']").all()[:MAX_POSTS_TO_VISIT]:
                                href = a.get_attribute("href")
                                if href:
                                    full = "https://www.instagram.com" + href if href.startswith("/") else href
                                    if full not in post_links:
                                        post_links.append(full)
                        except Exception:
                            pass

            # --- Visit individual posts (captioned embeds) to pick up
            # additional captions/top comments beyond the profile grid,
            # continuing until the cap is hit, the links run out, or we
            # run low on time. For a product-specific job, each post's
            # caption is checked BEFORE spending time expanding its
            # comments - only posts that are actually about this product
            # get their comment threads expanded at all.
            for link in post_links:
                if len(results) >= MAX_INSTAGRAM_COMMENTS or time_left() <= 4:
                    break
                post_embed = link.rstrip("/") + "/embed/captioned/"
                if not _goto_with_retry(page, post_embed, timeout=_adaptive_nav_timeout(time_left), time_left=time_left):
                    continue
                try:
                    page.wait_for_selector(
                        "article, [class*='Caption'], [class*='caption']", timeout=3000,
                    )
                except Exception:
                    pass
                reason = _diagnose_page_state(page)
                if reason != "ok":
                    logger.info(
                        "Instagram scrape: post %s unavailable (reason=%s); skipping.",
                        post_embed, reason,
                    )
                    continue

                caption_loc, used_sel = _first_matching_locator(page, _CAPTION_SELECTOR_CHAIN)
                caption_texts: List[str] = []
                if caption_loc is not None:
                    for loc in caption_loc.all():
                        try:
                            caption_texts.append(loc.inner_text())
                        except Exception:
                            pass
                try:
                    peek_html = _content_bounded(page, 1500)
                    peek_texts, _ = _parse_embed_context(peek_html)
                except Exception:
                    peek_texts = []

                if product_name:
                    relevant = any(
                        _mentions_product(t, product_name)
                        for t in caption_texts + peek_texts
                    )
                    if not relevant:
                        logger.info(
                            "Instagram scrape: post=%s does not mention "
                            "product=%r; skipping (comments not expanded).",
                            post_embed, product_name,
                        )
                        continue

                before = len(results)
                _expand_more_comments(page, time_left)
                for t in caption_texts:
                    _add(t)
                for t in peek_texts:
                    _add(t)
                # Re-query after expansion: more comment nodes may now be
                # attached under the same selector chain.
                caption_loc2, _ = _first_matching_locator(page, _CAPTION_SELECTOR_CHAIN)
                if caption_loc2 is not None:
                    for loc in caption_loc2.all():
                        try:
                            _add(loc.inner_text())
                        except Exception:
                            pass
                # Post embeds sometimes carry their own contextJSON payload
                # (caption + preview comments) in the page source even when
                # the rendered DOM shows nothing extractable, and expanding
                # comments can add to it too.
                try:
                    post_html = _content_bounded(page, 1500)
                    post_texts, _ = _parse_embed_context(post_html)
                    for txt in post_texts:
                        _add(txt)
                except Exception:
                    pass
                logger.info(
                    "Instagram scrape: post=%s contributed %d new item(s); "
                    "running total=%d/%d.",
                    post_embed, len(results) - before, len(results), MAX_INSTAGRAM_COMMENTS,
                )

            # --- One search fallback (single query), only if still short
            # on data and there's time left. Product-specific jobs narrow
            # the query to that product so the snippets it finds (if any)
            # are still on-topic. ------------------------------
            if len(results) < MIN_RESULTS_BEFORE_FALLBACK and time_left() > 4:
                if product_name:
                    q = f'site:instagram.com "{company_name}" "{product_name}"'
                else:
                    q = f'site:instagram.com "{company_name}"'
                search_url = f"https://www.google.com/search?q={q.replace(' ', '+')}&hl=en&num=20"
                if _goto_with_retry(page, search_url, timeout=_adaptive_nav_timeout(time_left), time_left=time_left):
                    try:
                        page.wait_for_selector(
                            "div.VwiC3b, span.aCOpRe, div.IsZvec, div.g", timeout=3000,
                        )
                    except Exception:
                        pass
                    for sel in ["div.VwiC3b", "span.aCOpRe", "div.IsZvec",
                                "div.lyLwlc", "span.MUxGbd"]:
                        for loc in page.locator(sel).all()[:25]:
                            try:
                                _add_if_relevant(loc.inner_text())
                            except Exception:
                                pass

            if not results:
                _save_debug_artifacts(
                    page, profile_url, "zero_comments",
                    extra={
                        "embed_page_loaded": str(embed_page_loaded),
                        "profile_page_loaded": str(profile_page_loaded),
                        "login_wall_present": str(_looks_blocked(page)),
                        "diagnosis": _diagnose_page_state(page),
                    },
                )
        except Exception:
            logger.exception("Instagram scrape failed for %r.", profile_url)
        finally:
            if page is not None:
                try:
                    page.close()
                except Exception:
                    pass

        final = normalize_comments(results)[:MAX_INSTAGRAM_COMMENTS]
        elapsed = time.monotonic() - start
        logger.info(
            "Instagram items for %r: raw=%d duplicates_removed=%d final=%d "
            "elapsed=%.1fs (cap=%d).",
            company_name, len(results), duplicates_removed, len(final), elapsed,
            MAX_INSTAGRAM_COMMENTS,
        )
        logger.info("Instagram returned %d comments", len(final))
        return final

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(_EXECUTOR, _run)
    except Exception:
        logger.exception("Unhandled error scraping Instagram comments for %r.", profile_url)
        return []
