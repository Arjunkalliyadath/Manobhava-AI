"""
Module Name
-----------
scrapers/twitter_scraper.py

Purpose
-------
Collects genuine, product-specific posts from Twitter/X — posts that
discuss the selected product specifically, rather than the company or
brand in general — while remaining reliable in the face of the
platform's access restrictions.

Responsibilities
-----------------
- `scrape_twitter_comments(company_data)`: the single public entry point.
  Runs product-centric search tiers similar to youtube_scraper.py /
  reddit_scraper.py, giving a brand-wide "General" job a shorter budget
  (`GENERAL_TWITTER_TIMEOUT_SECONDS` in app.py) than genuine per-product
  jobs.
- Attaches to the shared Chromium process via `scrapers.browser_utils`
  rather than launching its own, and logs `BROWSER_TRACE` timing at each
  setup checkpoint, same as google_scraper.py / instagram_scraper.py /
  youtube_scraper.py.
- Twitter/X has no strict quantity target (per the project's own
  priorities) — this module optimizes for reliability of what it can
  collect rather than chasing a fixed volume.

Dependencies
------------
`playwright` (sync API), plus standard library `asyncio`, `re`,
`threading`, `time`.
"""

import asyncio
import atexit
import concurrent.futures
import html
import logging
import re
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote_plus

import httpx

from scrapers.browser_utils import browser_manager, normalize_comments, find_social_profile_url
from config import (
    MAX_TWITTER_POSTS,
    MAX_SCROLL_ITERATIONS,
    SCROLL_IDLE_LIMIT,
    NAVIGATION_RETRIES,
    NAV_TIMEOUT_MS_MAX,
    NAV_TIMEOUT_MS_MIN,
)

logger = logging.getLogger(__name__)

# --- Hard internal time budget -----------------------------------------
# PREVIOUS BUG: this scraper's internal deadline (`deadline = start +
# TIME_BUDGET_SECONDS`, inside _run()) was always a FRESH clock started
# only after browser/context/page setup had already finished, with no
# memory of how much of the OUTER 32s app.py wait_for() budget the httpx
# preflight check (up to _PREFLIGHT_TIMEOUT_SECONDS) and setup itself
# (sync_playwright start + shared-Chromium attach + new_context +
# new_page) had already spent. Setup alone varies from ~3s up to ~14s
# depending on how many other scraper tabs are rendering concurrently on
# this app's single shared Chromium process (see browser_utils.py), so a
# fixed post-setup allowance could - and did, per production logs showing
# "total scraper runtime 31.66s" against the 32s outer cap, with the
# preflight check's own time not even counted in that 31.66s - leave only
# a sliver of real margin. When that margin ran out, app.py's outer
# wait_for() hard-cancelled the coroutine and threw away everything
# already collected, instead of letting the scraper return gracefully
# (with whatever partial results it had) on its own.
#
# Fixed below by anchoring one real wall-clock deadline to the moment
# scrape_twitter_comments() itself starts (before the preflight check),
# via OUTER_HARD_TIMEOUT_SECONDS/SAFETY_MARGIN_SECONDS, and having _run()
# spend whatever of that is actually left after setup - rather than always
# assuming a fresh TIME_BUDGET_SECONDS is available no matter how long
# getting there took. TIME_BUDGET_SECONDS itself is kept only as the
# "no contention at all" reference value a couple of comments below still
# describe; deadline/time_left() (see _run()) are what everything else in
# this file actually uses.
#
# X's guest/logged-out timeline can genuinely take much longer than a
# handful of seconds to hydrate under this app's shared-browser setup
# (several other scraper tabs rendering concurrently on the one shared
# Chromium process) - see _adaptive_content_timeout below, which is the
# main beneficiary of a generous budget. DEBUG_CAPTURE_TIMEOUT_MS further
# down keeps the zero-result screenshot/HTML capture from eating into
# whatever margin is left under the real 45s ceiling.
OUTER_HARD_TIMEOUT_SECONDS = 45  # must be kept in sync with app.py's TWITTER_TIMEOUT_SECONDS

# PREVIOUS GAP: unlike youtube_scraper.py (GENERAL_YOUTUBE_TIMEOUT_SECONDS=25
# vs a ~90s per-product internal budget) and reddit_scraper.py
# (GENERAL_REDDIT_TIMEOUT_SECONDS=20 vs ~40s per-product), this file had only
# ONE deadline for every job - the brand-wide "General" job silently got the
# exact same 45s as a genuine per-product search+filter job, even though
# cross-cutting principle #1 (ARCHITECTURE.md §3) says General should be
# lower-priority. Concretely this meant: with MAX_BROWSER_WORKERS=3 threads
# shared across up to 4 jobs (General + up to 3 selected products), a
# General job that used its full budget could occupy a real OS thread for as
# long as any product job - directly competing with the jobs that actually
# matter for the same limited thread pool and CPU (see §8 on why
# asyncio.wait_for() in app.py alone can't fix this: only shortening the
# internal deadline actually stops the underlying Playwright work sooner).
#
# Fixed by giving `_run()` a General-aware effective outer timeout below
# (`product_name` is already the correct signal - General jobs never carry
# one). app.py's own GENERAL_TWITTER_TIMEOUT_SECONDS must be kept equal to
# this value, the same "outer app.py timeout == this file's OUTER_HARD_TIMEOUT_
# SECONDS" convention already used for the per-product path above. Chosen to
# match youtube_scraper.py's General budget (25s) since Twitter's General job
# is architecturally the same shape (browser-based profile-timeline fallback)
# rather than Reddit's httpx-only one - not yet verified against a live log
# for this file specifically; revisit with real timing data before raising it
# back up, per this codebase's own stated practice of live-log-driven tuning.
GENERAL_OUTER_HARD_TIMEOUT_SECONDS = 25  # must be kept in sync with app.py's GENERAL_TWITTER_TIMEOUT_SECONDS
# Wider than a bare "return-path overhead" allowance would suggest: production
# logs show individual Playwright waits (e.g. wait_for_selector("article"))
# actually taking 15-25% longer in wall-clock time than the nominal timeout
# passed to them (17.45s measured against a ~14s requested timeout) when many
# other scraper tabs are rendering concurrently on this app's one shared
# Chromium process (see browser_utils.py) - almost certainly Python-side
# GIL/event-loop contention delaying delivery of the timeout itself, not the
# timeout logic being wrong. A margin sized only for clean scheduling
# overhead would still get eaten by that kind of overrun, so this is
# deliberately generous.
SAFETY_MARGIN_SECONDS = 8.0
MIN_USEFUL_BUDGET_SECONDS = 6.0  # below this, don't even attempt navigation

# Reserve carved out of the PRODUCT-SPECIFIC search attempt's content-wait
# specifically (see its wait_for_selector call below), sized larger than the
# 4.0/6.0 used elsewhere in this file. Live-log evidence (three real
# companies, 2026-07-30) showed the old reserve_seconds=4.0 here left the
# httpx profile re-check (attempt 2a) starved of its own MIN_USEFUL_BUDGET_
# SECONDS floor more often than not: _adaptive_content_timeout's 15000ms
# ceiling means the search wait can burn up to ~15s on its own whenever
# time_left() was already >=19s going in (15 + this file's old 4.0), and a
# genuine "no tweets rendered" timeout regularly used the FULL allotted wait.
# Raising the reserve to 10.0 pulls that worst case down to time_left()-10
# (still comfortably above the ~7s this app's guest timeline can need to
# hydrate, per _adaptive_content_timeout's own docstring), which reliably
# leaves MIN_USEFUL_BUDGET_SECONDS plus a few seconds of margin for the
# recheck to actually run instead of being silently skipped by the time-
# check that gates it. Not yet re-verified against a second live log - if a
# future run shows genuine search results now timing out before real tweets
# have a chance to render, this is the first number to reconsider, in the
# other direction.
SEARCH_CONTENT_WAIT_RESERVE_SECONDS = 10.0

TIME_BUDGET_SECONDS = 20

# Overall cap on how many tweets/replies this scraper will try to collect
# for one company/handle, sourced from config so it can be tuned without
# touching this file.
MAX_TOTAL_POSTS = MAX_TWITTER_POSTS

MAX_REPLIES_PER_TWEET = 20

_LOGIN_MARKERS = (
    "/login", "/i/flow/login", "/account/access", "/i/flow/", "/logout",
    "/error",
)

# Text-based signals that the page is a login wall even when the URL itself
# didn't change (X sometimes renders an in-page "Sign in" prompt on top of
# a timeline that never redirected).
_LOGIN_TEXT_MARKERS = (
    "sign in to x", "log in to x", "don't miss what's happening",
)

# Additional text-based signals used only for diagnostics (never to change
# control flow beyond "give up and return gracefully"). These let logs say
# *why* nothing was collected instead of a generic empty result, which is
# the difference between "the scraper is broken" and "this profile is
# suspended" when someone is debugging a run later.
_SUSPENDED_MARKERS = ("account suspended",)
_NOT_FOUND_MARKERS = (
    "this account doesn't exist", "this account doesn’t exist",
    "page doesn't exist", "page doesn’t exist", "hmm...this page doesn",
)
_PROTECTED_MARKERS = ("these posts are protected", "these tweets are protected")
_RATE_LIMIT_MARKERS = ("rate limit exceeded", "try again later", "something went wrong")

# Text of publicly-visible "expand" controls X shows on tweet detail pages.
# Clicking these (when present) renders replies that are collapsed by
# default; anything gated behind an actual login wall is left alone.
_EXPAND_BUTTON_TEXTS = ("Show more replies", "Show replies", "Continue thread", "Show additional replies")

# Markers used to recognize and skip pinned/promoted posts so they don't
# pollute the sentiment sample with non-organic or repeated content.
_NOISE_MARKERS = ("promoted", "pinned")

NAV_RETRY_BACKOFF_SECONDS = 0.75

# Ceiling on how long the zero-result debug-artifact screenshot is allowed
# to spend, no matter how unresponsive the page is. This used to be an
# unbounded full_page capture: on a page whose timeline never finished
# settling, that screenshot could itself take several more seconds *after*
# TIME_BUDGET_SECONDS had already run out, pushing the whole job past
# app.py's outer hard timeout and losing the run to a cold cancel instead
# of returning [] cleanly on its own. Debugging a zero-result page never
# needs the full scroll history either, so this also switches that
# screenshot to viewport-only instead of full_page.
DEBUG_CAPTURE_TIMEOUT_MS = 1500

# Text scraping never needs images or fonts; skipping them cuts page load
# time without affecting anything we read out of the DOM.
_BLOCKED_RESOURCE_TYPES = {"image", "font"}

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
_LAUNCH_ARGS = ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"]

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
    max_workers=MAX_BROWSER_WORKERS, thread_name_prefix="twitter_scraper"
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
            "BROWSER_TRACE scraper=Twitter event=before_sync_playwright_start elapsed=%.3fs timestamp=%.3f thread=%s",
            time.monotonic() - browser_trace_started_at, time.time(), threading.current_thread().name,
        )
        pw = sync_playwright().start()
        logger.info(
            "BROWSER_TRACE scraper=Twitter event=after_sync_playwright_start elapsed=%.3fs timestamp=%.3f thread=%s",
            time.monotonic() - browser_trace_started_at, time.time(), threading.current_thread().name,
        )
        handle.playwright = pw

    logger.info(
        "BROWSER_TRACE scraper=Twitter event=before_shared_browser_attach elapsed=%.3fs timestamp=%.3f thread=%s",
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
        "BROWSER_TRACE scraper=Twitter event=after_shared_browser_attach elapsed=%.3fs timestamp=%.3f thread=%s",
        time.monotonic() - browser_trace_started_at, time.time(), threading.current_thread().name,
    )
    logger.info(
        "BROWSER_TRACE scraper=Twitter event=before_new_context elapsed=%.3fs timestamp=%.3f thread=%s",
        time.monotonic() - browser_trace_started_at, time.time(), threading.current_thread().name,
    )
    context = browser.new_context(
        user_agent=_USER_AGENT,
        viewport={"width": 1280, "height": 1800},
        locale="en-US",
    )
    logger.info(
        "BROWSER_TRACE scraper=Twitter event=after_new_context elapsed=%.3fs timestamp=%.3f thread=%s",
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
# ROOT CAUSE of "Twitter never collects anything": production logs show
# setup alone (this thread's own sync_playwright().start() + attaching to
# the shared Chromium process via browser_manager.ensure_shared_browser()
# + new_context()) regularly taking 8s-25s+ - and on a cold/contended run
# (several other scrapers all launching their own Playwright driver at
# the same instant) it can eat the ENTIRE 32s OUTER_HARD_TIMEOUT_SECONDS
# by itself. When that happens the deadline math further up (anchored to
# _overall_start, correctly - see that block's docstring) has nothing
# left to spend, so scrape_twitter_comments() returns [] before a single
# navigation is ever attempted. That's not a navigation/selector bug;
# it's setup cost eating 100% of the time budget before scraping starts.
#
# That setup cost is a ONE-TIME, per-worker-thread cost - _ensure_context
# above already reuses a live context on later calls via a cheap
# `ctx.pages` check instead of redoing setup. So instead of a real,
# time-boxed scrape request paying that cost for the first time, pay it
# once here in the background at module import time, i.e. when the
# FastAPI app boots up (mirrors the existing "Sentiment pipeline
# preloaded at startup" pattern used elsewhere in this project) - well
# before the first real request in practice, since the user still has to
# go through Company/Product Discovery and pick a product before this
# module's scrape_twitter_comments() is ever called.
#
# MAX_BROWSER_WORKERS separate jobs are submitted (not just one) so every
# thread this module's own _EXECUTOR could later hand a real _run() to
# gets warmed, not just whichever one happens to run first: a
# ThreadPoolExecutor spins up one new worker thread per queued task (up
# to max_workers) whenever none are idle yet, so submitting N tasks back
# to back spreads across N distinct threads instead of piling onto one.
#
# Best-effort only and never blocks import: each job runs on _EXECUTOR's
# own worker threads, not the importing thread, and any failure (e.g. a
# dev machine without Playwright's browsers installed yet) is caught and
# logged - the first real request then simply falls back to paying the
# setup cost itself, exactly as it did before this change.
def _warm_up_browser_context() -> None:
    # Without this guard, this raises NotImplementedError on every warm-up
    # thread on Windows: app.py sets WindowsSelectorEventLoopPolicy
    # process-wide at import time (see browser_utils.py), but only the
    # Proactor loop can launch subprocesses on Windows, and Playwright's
    # sync API spawns its driver as a subprocess. The real request path
    # (_run(), below) already re-asserts Proactor for the same reason;
    # this warm-up needs its own copy since it runs on a separate thread,
    # before any real request ever calls _run().
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    try:
        _ensure_context()
        logger.info(
            "Twitter: browser context pre-warmed on %s.",
            threading.current_thread().name,
        )
    except Exception:
        logger.exception(
            "Twitter: background browser context warm-up failed on %s "
            "(non-fatal - the first real request will pay the setup "
            "cost itself instead, same as before this change).",
            threading.current_thread().name,
        )


for _ in range(MAX_BROWSER_WORKERS):
    _EXECUTOR.submit(_warm_up_browser_context)


def _profile_url(target: str) -> str:
    """Resolve ``target`` (already a full URL, an "@handle", or a bare
    company name) into a usable x.com profile URL.

    Previously this only ever did ``target.lstrip('@')`` and slapped the
    result onto ``https://x.com/``, so anything that wasn't already a
    clean handle - a company name with spaces, punctuation, a
    twitter.com link, a URL with tracking query params - produced an
    invalid or 404-prone URL before navigation ever got a chance to run.
    This is still a heuristic (there's no public search API to confirm
    the real handle without authentication), but it resolves the common
    shapes correctly instead of only the already-correct one.
    """
    target = (target or "").strip()
    if not target:
        return ""

    if target.startswith(("http://", "https://")):
        # Normalize the legacy twitter.com domain to x.com and strip
        # anything (query string, fragment, trailing slash) that isn't
        # part of the actual profile path.
        normalized = target.replace("twitter.com", "x.com")
        normalized = normalized.split("?")[0].split("#")[0]
        return normalized.rstrip("/")

    handle = target.lstrip("@").strip()
    # A valid X handle is letters/digits/underscore only. A bare company
    # name ("Acme Corp", "Acme, Inc.") fails that check, so collapse it to
    # the closest handle-safe candidate instead of building a URL that's
    # guaranteed to 404. This won't always land on the *correct* handle,
    # but it stops obviously-broken URLs (with spaces/punctuation) from
    # ever being requested.
    if not re.fullmatch(r"[A-Za-z0-9_]+", handle):
        handle = re.sub(r"[^A-Za-z0-9_]", "", handle.replace(" ", ""))
    return f"https://x.com/{handle}" if handle else ""


# --- Browser-free preflight ------------------------------------------------
# X/Twitter is optional per the pipeline's own priority order, and the log
# shows every single attempt in this run either failing to navigate at all
# (browser-launch contention ate the time budget before Chromium could even
# open the page) or landing on a login wall once it did load. Since
# _looks_blocked's own URL/text markers work just as well against a plain
# HTTP response as a rendered page, checking them BEFORE ever calling
# _ensure_context() means an obviously-blocked target never spends a
# worker thread on a CDP attach + new_context() + navigation at all -
# freeing that worker sooner for the next job, even though attaching to
# the shared browser (see browser_utils.ensure_shared_browser()) is no
# longer the scarce resource it used to be now that Chromium itself is
# only launched once, application-wide.
_PREFLIGHT_TIMEOUT_SECONDS = 4.0


# --- Browser-free extraction -------------------------------------------
# Manual testing (curl -4 against a profile URL) showed X's own server
# already returns every visible tweet as schema.org SocialMediaPosting
# microdata (itemProp="text", "datePublished", interactionStatistic,
# etc.) in the PLAIN, un-rendered HTML response - the same crawler/SEO
# markup search engines and link-unfurling bots consume. No JavaScript
# execution is needed to read it.
#
# This matters because every plain HTTP client tested against x.com in
# this app (httpx from company_discovery.py's metadata scan, and a
# manual curl -4) gets a fast, complete response, while
# Playwright/CDP-driven Chromium's navigation reliably times out before
# receiving a single response byte (wait_until="commit" never fires) -
# in both headless and headful mode, so this isn't the "headless" flag
# specifically; it's automation being detected at a layer no
# --disable-blink-features flag reaches. Fighting that detection
# adversarially isn't a real fix, but skipping the browser for the part
# of the job a plain GET already does is.
#
# Scope: this recovers the account's OWN tweets (brand voice/posts) from
# the profile page. It deliberately does NOT try to fetch each
# individual tweet's detail page for public replies - that would need
# its own markup confirmed the same way the profile page's was, and
# guessing at unverified structure is how fragile scrapers get built.
# If reply-level data is wanted, the individual tweet permalinks are
# already extracted below (see ``_extract_tweet_permalinks_from_html``)
# and this can be extended once that markup is checked.
_TWEET_TEXT_RE = re.compile(r'content="([^"]*)"\s+itemProp="text"')
_TWEET_PERMALINK_RE = re.compile(
    r'content="(https://x\.com/[^"/]+/status/\d+)"\s+itemProp="url"'
)


def _extract_tweets_from_html(html_text: str) -> List[str]:
    """Pulls tweet body text out of X's server-rendered schema.org
    microdata. Returns [] (never raises) if the markup doesn't match -
    callers must treat that exactly like "nothing found here" and fall
    back to the browser path, not as an error."""
    if not html_text:
        return []
    try:
        return [
            html.unescape(match).strip()
            for match in _TWEET_TEXT_RE.findall(html_text)
            if match and match.strip()
        ]
    except Exception:
        return []


def _extract_tweet_permalinks_from_html(html_text: str) -> List[str]:
    """Pulls individual tweet permalink URLs (.../status/<id>) out of the
    same microdata, in case a future extension wants to fetch each one
    for replies. Order-preserving, de-duplicated. Never raises."""
    if not html_text:
        return []
    try:
        seen = set()
        ordered = []
        for link in _TWEET_PERMALINK_RE.findall(html_text):
            if link not in seen:
                seen.add(link)
                ordered.append(link)
        return ordered
    except Exception:
        return []


def _httpx_fetch_profile(url: str) -> "tuple[Optional[str], Optional[str]]":
    """One plain GET (no browser). Returns (final_url, body_text), or
    (None, None) on any failure/timeout.

    Single fetch shared by the blocked-check and the tweet-extraction
    path below, so a working response is only ever requested once
    instead of the blocked-check and the content-read each paying for
    their own separate GET."""
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
                },
            )
    except Exception:
        return None, None
    return str(resp.url), resp.text



def _looks_blocked(page) -> bool:
    """Return True only for a real login redirect."""
    try:
        url = page.url or ""
    except Exception:
        return False

    return any(marker in url for marker in _LOGIN_MARKERS)


def _diagnose_empty_page(page) -> str:
    """Classify why no tweets were found, for logging only - never used
    to change control flow beyond "give up gracefully", which happens
    regardless of the reason.

    Returns one of: "login_wall", "suspended", "not_found", "protected",
    "rate_limited", "rendering_failure", "no_tweets". Bounded to a single
    short body-text read so this never adds meaningful latency on top of
    the empty-result path it runs on.
    """
    try:
        url = page.url or ""
    except Exception:
        url = ""
    if any(marker in url for marker in _LOGIN_MARKERS):
        return "login_wall"

    try:
        body_text = page.locator("body").inner_text(timeout=800).lower()
    except Exception:
        return "rendering_failure"

    if any(m in body_text for m in _LOGIN_TEXT_MARKERS):
        return "login_wall"
    if any(m in body_text for m in _SUSPENDED_MARKERS):
        return "suspended"
    if any(m in body_text for m in _NOT_FOUND_MARKERS):
        return "not_found"
    if any(m in body_text for m in _PROTECTED_MARKERS):
        return "protected"
    if any(m in body_text for m in _RATE_LIMIT_MARKERS):
        return "rate_limited"
    if not body_text.strip():
        return "rendering_failure"
    return "no_tweets"


# Evidence directory for zero-result debug captures. Lives at
# <project_root>/debug (sibling of app.py's DOWNLOADS_DIR), since this file
# is itself at <project_root>/scrapers/twitter_scraper.py.
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
    in production logs.

    ``Locator.text_content(timeout=...)``, unlike ``page.title()``, DOES
    route through Playwright's real timeout machinery (verified against
    Playwright's own Frame.text_content, which is called with a genuine
    timeout calculator). Querying the ``<title>`` element this way gets the
    same information with an actual, enforced bound - and, critically, it
    runs on the calling thread instead of needing a separate thread the way
    a manual watchdog would: Playwright's sync API is greenlet-bound to
    whichever thread started it, and calling it from any other thread
    breaks that binding (surfaces as "greenlet.error: cannot switch to a
    different thread" and orphaned "Task exception was never retrieved"
    warnings) - which a first version of this fix ran into before this one.
    """
    try:
        return page.locator("title").text_content(timeout=timeout_ms) or default
    except Exception:
        return default


def _content_bounded(page, timeout_ms: int) -> Optional[str]:
    """Get a debug-artifact-worthy HTML dump with a timeout Playwright
    actually honors, for the same reason ``_title_bounded`` exists:
    ``page.content()`` takes no timeout and isn't covered by
    ``set_default_timeout()``. ``Locator.inner_html(timeout=...)`` on the
    ``html`` element IS genuinely bounded. Returns ``None`` (rather than a
    watered-down guess) if it can't get the HTML within the timeout, so the
    caller can log that plainly instead of writing a misleading empty file.
    """
    try:
        inner = page.locator("html").inner_html(timeout=timeout_ms)
    except Exception:
        return None
    return f"<html>{inner}</html>"


def _save_debug_artifacts(page, url: str, reason: str, extra: Dict[str, str] = None) -> None:
    """Save HTML/screenshot/context ONLY on failure/zero-results for
    debugging. Never on the success path.

    Writes three files under ``_DEBUG_DIR``, all sharing one base name
    stamped with date-time + milliseconds + thread id so concurrent or
    rapid-fire failures never overwrite each other:
      * <base>.png - screenshot
      * <base>.html - full page HTML
      * <base>.txt  - final URL, page title, reason, and whatever the
        caller passes in ``extra`` (e.g. login-wall / article-existence
        checks).
    """
    try:
        _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        safe_url = re.sub(r'[^a-zA-Z0-9]', '_', url)[:50].strip('_') or "unknown"
        stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}_{threading.get_ident()}"
        base = _DEBUG_DIR / f"twitter_{safe_url}_{stamp}"

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
            page.screenshot(path=f"{base}.png", timeout=DEBUG_CAPTURE_TIMEOUT_MS, full_page=False)
        except Exception:
            logger.warning("[TWITTER_DEBUG] could not capture screenshot for %s", url)

        html = _content_bounded(page, 1500)
        if html is not None:
            try:
                with open(f"{base}.html", "w", encoding="utf-8") as f:
                    f.write(html)
            except Exception:
                logger.warning("[TWITTER_DEBUG] could not capture HTML for %s", url)
        else:
            logger.warning(
                "[TWITTER_DEBUG] could not capture HTML for %s (timed out or errored)", url
            )


        info = {
            "reason": reason,
            "target_url": url,
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
            "[TWITTER_DEBUG] saved zero-result evidence for %s reason=%s: %s.{png,html,txt}",
            url, reason, base,
        )
    except Exception:
        logger.exception("[TWITTER_DEBUG] failed to save debug artifacts for %s", url)


def _adaptive_content_timeout(time_left, reserve_seconds: float) -> int:
    """Scale how long we wait for the first tweet/reply `<article>` to
    appear to how much budget is actually left, instead of a small fixed
    number (previously 5000/7000ms) unrelated to the scraper's real
    TIME_BUDGET_SECONDS.

    This was the core reason Twitter kept returning zero comments: X's
    guest/logged-out timeline can genuinely take well over 7s to hydrate
    under this app's shared-browser setup (several other scraper tabs -
    Google, YouTube x3, Instagram - can be rendering concurrently on the
    same shared Chromium process), so the fixed short wait was regularly
    timing out before a single tweet had rendered, even when the page
    would have produced results given a bit more time. ``reserve_seconds``
    is carved out up front for whatever runs after this wait (scrolling,
    reply expansion, or just the zero-result diagnostics), so a generous
    wait here never eats into the time those steps need.
    """
    available = max(0.0, time_left() - reserve_seconds)
    return int(max(3000, min(15000, available * 1000)))


def _adaptive_nav_timeout(time_left) -> int:
    """Scale the navigation timeout to how much of the internal time
    budget is actually left, instead of a fixed 5000ms for every attempt.

    Navigation always receives at least NAV_TIMEOUT_MS_MIN (8s) as long as
    that much time is actually left in the budget - the budget clock only
    starts once browser/context/page setup has finished (see _run below),
    so this floor is real, not eaten by Chromium startup. When plenty of
    budget remains the timeout can scale up to NAV_TIMEOUT_MS_MAX (12s) for
    a genuinely slow page; when the budget itself is under the floor, we
    hand over whatever is left rather than blocking past the scraper's own
    deadline.
    """
    remaining_ms = max(0.0, time_left()) * 1000
    if remaining_ms <= NAV_TIMEOUT_MS_MIN:
        return max(3000, int(remaining_ms))
    return int(min(NAV_TIMEOUT_MS_MAX, max(NAV_TIMEOUT_MS_MIN, remaining_ms * 0.5)))


def _goto_with_retry(page, url: str, *, timeout: int, time_left, retries: int = NAVIGATION_RETRIES) -> bool:
    """Navigate with retries, logging, and a short backoff between attempts.

    The backoff gives transient network blips or momentary rate-limiting a
    moment to clear instead of hammering X with back-to-back retries.

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
                "Twitter/X nav retry attempt=%d/%d url=%s remaining=%.1fs chosen_timeout=%dms",
                attempt + 1, total_attempts, url, remaining, attempt_timeout,
            )
        try:
            page.goto(url, wait_until="commit", timeout=attempt_timeout,)
            return True
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Twitter/X nav attempt %d/%d to %s failed: %s",
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


def _wait_for_more(page, selector: str, prev_count: int, max_ms: int = 500) -> int:
    """Poll ``selector``'s match count in short steps until it exceeds
    ``prev_count`` or ``max_ms`` elapses, whichever comes first.

    Replaces an unconditional ``page.wait_for_timeout(max_ms)`` after a
    scroll step: the worst-case wait is identical (this never runs longer
    than ``max_ms``), but as soon as new article nodes attach - which is
    most of the time on a live timeline - this returns immediately instead
    of always sitting out the full window, leaving more of the scraper's
    time budget for additional scroll iterations. Mirrors the same helper
    in youtube_scraper.py / google_scraper.py.
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


def _get_articles(page):
    """Fetch tweet/reply elements using a structural selector first.

    X renders each tweet as ``<article data-testid="tweet">``, which is
    far less brittle than a bare ``article`` tag (that attribute is tied
    to the component's role in the UI, not to styling that changes with
    every redesign). Falls back to the bare tag only if the structural
    selector finds nothing, so a markup change doesn't silently return
    zero results.
    """
    try:
        structural = page.locator('article[data-testid="tweet"]')
        if structural.count() > 0:
            return structural.all()
    except Exception:
        pass
    try:
        return page.locator("article").all()
    except Exception:
        return []


def _is_noise_article(article) -> bool:
    """Best-effort skip of pinned/promoted posts.

    Both are non-organic (an ad) or non-representative of current
    sentiment (a pinned post can be old/evergreen), so letting them into
    the sentiment sample skews results. Checked via the small
    "social context" line X renders above a tweet (e.g. "Pinned Tweet",
    "Promoted") rather than scanning the whole article for performance.
    """
    try:
        marker = article.locator('[data-testid="socialContext"]').first.inner_text(timeout=300)
    except Exception:
        return False
    marker_lower = marker.lower()
    return any(m in marker_lower for m in _NOISE_MARKERS)


def _expand_reply_threads(page, time_left, max_clicks: int = 4) -> None:
    """Click publicly-visible "show more" controls on a tweet detail page
    so replies that are collapsed by default get rendered before
    extraction.

    Only expands controls that are already visible on the unauthenticated
    page (nothing here logs in or bypasses a wall). Best-effort: missing
    buttons or click failures are silently skipped, and it stops early if
    the scraper's overall time budget is running low.
    """
    clicks = 0
    for text in _EXPAND_BUTTON_TEXTS:
        if clicks >= max_clicks or time_left() <= 3:
            break
        try:
            buttons = page.get_by_text(text, exact=False)
            count = min(buttons.count(), max_clicks - clicks)
        except Exception:
            continue
        for i in range(count):
            if time_left() <= 3:
                break
            try:
                buttons.nth(i).click(timeout=1500)
                clicks += 1
                page.wait_for_timeout(300)
            except Exception:
                continue


def _scroll_timeline_until_idle(page, time_left, max_items: int):
    """Scroll the timeline collecting unique tweet/reply text as it loads.

    Keeps scrolling until ``max_items`` tweets have been located,
    ``MAX_SCROLL_ITERATIONS`` scroll steps have happened,
    ``SCROLL_IDLE_LIMIT`` consecutive scrolls produced no new articles, or
    the overall scraper time budget runs low.

    Returns a ``(texts, tweet_links, text_link_map)`` tuple. ``text_link_map``
    maps a tweet's normalized text key to its permalink (when one was found
    on the SAME article element the text came from), so a caller that wants
    to filter tweets by content (e.g. "only ones mentioning this product")
    can still find the right permalink to expand replies on.
    """
    seen_text: List[str] = []
    seen_set = set()
    links: List[str] = []
    seen_links = set()
    text_link_map: Dict[str, str] = {}

    iteration = 0
    idle = 0
    prev_count = -1
    start = time.monotonic()
    while True:
        iteration += 1
        if iteration > MAX_SCROLL_ITERATIONS:
            logger.info("Timeline scroll: hit MAX_SCROLL_ITERATIONS (%d).", MAX_SCROLL_ITERATIONS)
            break
        if time_left() <= 2 or len(seen_text) >= max_items:
            break

        try:
            for article in _get_articles(page):
                if _is_noise_article(article):
                    continue
                try:
                    text = article.inner_text()
                except Exception:
                    continue
                key = None
                if text and len(text.split()) > 4:
                    key = " ".join(text.split()).lower()
                    if key not in seen_set:
                        seen_set.add(key)
                        seen_text.append(text)
                try:
                    href = article.locator("a[href*='/status/']").first.get_attribute("href")
                    if href:
                        link = "https://x.com" + href if href.startswith("/") else href
                        if key:
                            text_link_map.setdefault(key, link)
                        if link not in seen_links:
                            seen_links.add(link)
                            links.append(link)
                except Exception:
                    pass
        except Exception:
            pass

        count = len(seen_text)
        logger.info(
            "Timeline scroll: iteration=%d current_count=%d idle_streak=%d elapsed=%.1fs",
            iteration, count, idle, time.monotonic() - start,
        )

        if count >= max_items:
            logger.info("Timeline scroll: reached target of %d tweets.", max_items)
            break
        if count <= prev_count:
            idle += 1
            if idle >= SCROLL_IDLE_LIMIT:
                logger.info(
                    "Timeline scroll: no new tweets after %d consecutive scrolls; stopping.",
                    SCROLL_IDLE_LIMIT,
                )
                break
        else:
            idle = 0
        prev_count = max(prev_count, count)

        try:
            dom_count_before = page.locator('article[data-testid="tweet"], article').count()
        except Exception:
            dom_count_before = -1
        try:
            page.mouse.wheel(0, 1800)
        except Exception:
            break
        # Readiness-based wait: return as soon as new article nodes attach
        # instead of always sitting out the full 500ms window. Same
        # worst-case bound as the previous fixed sleep.
        _wait_for_more(page, 'article[data-testid="tweet"], article', dom_count_before, max_ms=500)

    return seen_text, links, text_link_map


def _mentions_product(text: str, product_name: str) -> bool:
    """True only if every meaningful token of ``product_name`` (words and
    standalone numbers, e.g. "airdopes" and "131") appears somewhere in
    ``text``. Deliberately strict/precise rather than fuzzy: a false
    negative here just means one fewer (still-genuine) comment collected,
    but a false positive would mean mislabeling a generic/unrelated tweet
    as being about this specific product - the thing we're trying hardest
    to avoid.
    """
    if not text or not product_name:
        return False
    text_l = text.lower()
    tokens = [t for t in re.findall(r"[a-z0-9]+", product_name.lower()) if len(t) > 1]
    if not tokens:
        return False
    return all(t in text_l for t in tokens)


async def scrape_twitter_comments(company_data: Dict[str, str]) -> List[str]:
    # Anchor point for the real wall-clock deadline (see the
    # OUTER_HARD_TIMEOUT_SECONDS block above). Captured before the httpx
    # preflight check and before this job is even submitted to the
    # executor, so _run()'s deadline reflects the FULL time app.py's outer
    # wait_for() has already been counting against this coroutine - not
    # just the part after setup finishes.
    _overall_start = time.monotonic()

    # When set, this call is for one specific selected product (not the
    # brand-wide "General" job) - see the PRODUCT-SPECIFIC block inside
    # _run() below for how this changes the scrape strategy.
    product_name = (company_data.get("product_name") or "").strip()
    company_name_for_query = (company_data.get("company_name") or "").strip()

    if product_name:
        # Diagnostic only - does not change matching behaviour at all.
        # _mentions_product() requires EVERY token in product_name to
        # appear in a candidate tweet. That's a reasonable bar for a short,
        # brand+model style name ("Airdopes 131" - 2 tokens), but some
        # sites hand this scraper long, formulaic catalogue titles instead
        # (marketplace listings like "Brand Women's A-Line Cotton Kurta -
        # Navy Blue", or electronics variants like "Galaxy S26 Ultra 256
        # GB｜12 GB Black") where a genuine, casual mention would only ever
        # contain the first 2-3 "core identity" tokens, never the trailing
        # fabric/fit/color/variant ones. Requiring all of them there would
        # make a genuine match essentially impossible, for reasons that
        # have nothing to do with X or this scraper. Flagging the token
        # count here (without changing anything) so a real log makes it
        # obvious whether that's actually happening, rather than guessing -
        # the fix, if this shows up, most likely belongs in whatever
        # extracts/selects product_name upstream (a shorter "core name" for
        # social matching, distinct from the full catalogue title used
        # elsewhere), not in loosening the matching rule here blind.
        _diag_tokens = [t for t in re.findall(r"[a-z0-9]+", product_name.lower()) if len(t) > 1]
        if len(_diag_tokens) > 5:
            logger.info(
                "Twitter/X PRODUCT diagnostic: product_name=%r tokenizes to "
                "%d token(s) %r - _mentions_product() requires ALL of them "
                "in a single tweet, which is a demanding bar for a name "
                "this long; watch for this job returning 0 matches even "
                "when genuine mentions likely exist.",
                product_name, len(_diag_tokens), _diag_tokens,
            )

    target = (
        company_data.get("twitter_url")
        or company_data.get("twitter")
    )

    loop = asyncio.get_event_loop()

    if not target:
        company_name = (company_data.get("company_name") or "").strip()
        if company_name:
            try:
                target = await loop.run_in_executor(
                    _EXECUTOR, find_social_profile_url, company_name, "twitter",
                )
            except Exception:
                target = ""
            if target:
                logger.info(
                    "Twitter/X scrape: no handle from the site scan; "
                    "search fallback found %r for %r.", target, company_name,
                )
        if not target:
            logger.info(
                "Twitter/X scrape: no handle from the site scan and the "
                "search fallback found nothing for %r; skipping rather "
                "than guessing a handle from the name.",
                company_data.get("company_name"),
            )
            return []

    url = _profile_url(target)

    if not url:
        logger.warning(
            "Twitter/X scrape: could not resolve a usable profile URL from %r.",
            target,
        )
        return []

    # Fast httpx-only path: only safe for the "General"/brand-wide job.
    # It reads whatever the profile's schema.org microdata happens to
    # contain, with no way to filter by product and no way to expand
    # replies - so a PRODUCT-SPECIFIC call always goes through the full
    # browser path below instead, where it can actually search/filter.
    if not product_name:
        try:
            final_url, profile_html = await loop.run_in_executor(
                _EXECUTOR, _httpx_fetch_profile, url,
            )
        except Exception:
            final_url, profile_html = None, None

        preflight_blocked = bool(final_url and any(
            marker in final_url for marker in _LOGIN_MARKERS
        )) or bool(profile_html and any(
            marker in profile_html[:20000].lower() for marker in _LOGIN_TEXT_MARKERS
        ))

        if preflight_blocked:
            logger.info(
                "Twitter/X scrape: preflight detected a login wall for %s before "
                "touching the browser pool.",
                url,
            )
            return []

        if profile_html:
            httpx_tweets = _extract_tweets_from_html(profile_html)
            if httpx_tweets:
                final = normalize_comments(httpx_tweets)[:MAX_TOTAL_POSTS]
                logger.info(
                    "Twitter/X scrape: got %d post(s) for %s via a plain httpx "
                    "GET (schema.org microdata) - no browser needed.",
                    len(final), url,
                )
                return final
            logger.info(
                "Twitter/X scrape: httpx GET for %s succeeded but no tweet "
                "microdata matched (schema may have changed); falling back "
                "to the browser path.",
                url,
            )

    def _run() -> List[str]:

        if sys.platform.startswith("win"):
            asyncio.set_event_loop_policy(
                asyncio.WindowsProactorEventLoopPolicy()
            )

        setup_start = time.monotonic()

        try:
            context = _ensure_context()
        except Exception:
            logger.exception(
                "Could not start/obtain a browser context for %r.",
                url,
            )
            return []

        results: List[str] = []
        duplicates_removed = 0
        seen = set()

        def _add(text: str):
            nonlocal duplicates_removed

            key = " ".join(text.split()).lower()

            if key in seen:
                duplicates_removed += 1
                return

            seen.add(key)
            results.append(text)

        page = None

        try:
            logger.info(
                "BROWSER_TRACE scraper=Twitter event=before_new_page elapsed=%.3fs timestamp=%.3f thread=%s",
                time.monotonic() - getattr(_browser_trace, "ensure_context_started_at", time.monotonic()), time.time(), threading.current_thread().name,
            )
            page = context.new_page()
            logger.info(
                "BROWSER_TRACE scraper=Twitter event=after_new_page elapsed=%.3fs timestamp=%.3f thread=%s",
                time.monotonic() - getattr(_browser_trace, "ensure_context_started_at", time.monotonic()), time.time(), threading.current_thread().name,
            )
            page.set_default_timeout(8000)

            setup_elapsed = time.monotonic() - setup_start

            # General (brand-wide, no product_name) gets a materially
            # shorter budget than a real per-product job - see the
            # GENERAL_OUTER_HARD_TIMEOUT_SECONDS comment above for why this
            # has to happen HERE (the scraper's own deadline), not only in
            # app.py's outer asyncio.wait_for(), for it to actually reduce
            # real Playwright/CPU work rather than just app.py's patience.
            _effective_outer_timeout = (
                OUTER_HARD_TIMEOUT_SECONDS if product_name
                else GENERAL_OUTER_HARD_TIMEOUT_SECONDS
            )

            start = time.monotonic()
            deadline = _overall_start + _effective_outer_timeout - SAFETY_MARGIN_SECONDS
            remaining_budget = deadline - start

            logger.info(
                "Twitter/X scrape: browser/context/page setup took %.1fs "
                "(%.1fs elapsed since job start); %.1fs left for "
                "navigation+scrape before the safety-margined internal "
                "deadline (job=%s outer cap=%ds, safety margin=%.1fs).",
                setup_elapsed,
                start - _overall_start,
                remaining_budget,
                product_name or "General",
                _effective_outer_timeout,
                SAFETY_MARGIN_SECONDS,
            )

            if remaining_budget <= MIN_USEFUL_BUDGET_SECONDS:
                logger.warning(
                    "Twitter/X scrape: only %.1fs left for %s after a "
                    "%.1fs setup (job start to now: %.1fs) - not enough "
                    "time to attempt navigation; returning early instead "
                    "of risking the outer hard timeout.",
                    remaining_budget, url, setup_elapsed,
                    time.monotonic() - _overall_start,
                )
                return normalize_comments(results)

            def time_left():
                return deadline - time.monotonic()

            timeline_texts: List[str] = []
            tweet_links: List[str] = []
            text_link_map: Dict[str, str] = {}
            content_source = "none"  # "search" | "profile_httpx" | "profile_browser" - diagnostics only

            if product_name:
                # --- PRODUCT-SPECIFIC attempt 1: X's own search results
                # for "<company> <product>". This is the only source that
                # can, in principle, surface genuine OTHER users' tweets
                # about this exact product (not just the brand's own
                # timeline). X frequently login-walls search for logged-
                # out sessions, so this is a best-effort attempt - if it's
                # blocked or empty we fall through to attempt 2 below
                # rather than giving up. ---
                search_query = f"{company_name_for_query} {product_name}".strip()
                search_url = f"https://x.com/search?q={quote_plus(search_query)}&src=typed_query"
                _search_nav_start = time.monotonic()
                _search_nav_ok = _goto_with_retry(
                    page, search_url,
                    timeout=_adaptive_nav_timeout(time_left),
                    time_left=time_left,
                )
                logger.info(
                    "Twitter/X PRODUCT search: navigation to %r took %.2fs (ok=%s).",
                    search_query, time.monotonic() - _search_nav_start, _search_nav_ok,
                )
                if _search_nav_ok and not _looks_blocked(page):
                    try:
                        page.wait_for_selector(
                            "article",
                            timeout=_adaptive_content_timeout(
                                time_left, reserve_seconds=SEARCH_CONTENT_WAIT_RESERVE_SECONDS,
                            ),
                        )
                        timeline_texts, tweet_links, text_link_map = _scroll_timeline_until_idle(
                            page, time_left, MAX_TOTAL_POSTS,
                        )
                        content_source = "search"
                    except Exception:
                        logger.info(
                            "Twitter/X PRODUCT search: no tweets rendered for "
                            "query=%r.", search_query,
                        )
                else:
                    logger.info(
                        "Twitter/X PRODUCT search: blocked/login-walled or "
                        "navigation failed for query=%r.", search_query,
                    )

            # PREVIOUS BUG: the fallback below used to be gated on
            # `if not timeline_texts:` - i.e. "did search return ANYTHING at
            # all", not "did search return anything that actually mentions
            # the product". X search for "<company> <product>" can render
            # articles that are off-topic (retweets, loosely-matched terms,
            # unrelated brand mentions) without a single one containing a
            # genuine product mention - the old check treated that the same
            # as a successful search, silently skipping attempts 2a/2b below
            # even though either might hold the real, genuine match. The
            # PRODUCT filter log a bit further down already claimed both
            # search AND the profile timeline were tried; this makes that
            # actually true instead of only sometimes true.
            search_has_match = bool(product_name) and any(
                _mentions_product(t, product_name) for t in timeline_texts
            )

            if not search_has_match:
                # Discard whatever attempt 1 left behind - empty, or
                # non-empty-but-irrelevant search results - so "do we have
                # anything yet" (checked going into attempt 2b below) means
                # "do we have a genuine match", not "did search render
                # SOMETHING, relevant or not". Leaving stale irrelevant
                # search text sitting in timeline_texts here would silently
                # block attempt 2b the same way it used to block this whole
                # fallback before the fix above.
                timeline_texts = []
                tweet_links = []
                text_link_map = {}
                # --- PRODUCT-SPECIFIC attempt 2a: a cheap, no-browser
                # re-check of the brand's own profile page, before paying
                # for a full navigation+scroll of that exact same page in
                # attempt 2b below. Reuses the identical schema.org-microdata
                # extraction already proven out for the General job's fast
                # path (_httpx_fetch_profile / _extract_tweets_from_html,
                # see the docstrings above scrape_twitter_comments) - the
                # only difference is filtering the result through
                # _mentions_product instead of accepting it unfiltered.
                # Deliberately does NOT attempt reply expansion for matches
                # found this way (tweet_links/text_link_map are left
                # untouched, so the PRODUCT-filter block below naturally
                # derives an empty tweet_links for them) - reply-level
                # fetching from plain HTML would need permalink-to-text
                # pairing confirmed against real markup the same way the
                # text extraction itself was (see
                # _extract_tweet_permalinks_from_html's docstring), which
                # hasn't been done; better to under-deliver on reply depth
                # here than guess at unverified structure. Skipped for the
                # General job itself (product_name empty) - General already
                # made this exact call, before _run() was ever dispatched,
                # in the async fast-path above scrape_twitter_comments(). ---
                if product_name and time_left() > MIN_USEFUL_BUDGET_SECONDS:
                    _httpx_recheck_start = time.monotonic()
                    try:
                        _profile_final_url, _profile_html = _httpx_fetch_profile(url)
                    except Exception:
                        _profile_final_url, _profile_html = None, None
                    _profile_httpx_blocked = bool(_profile_final_url) and any(
                        m in _profile_final_url for m in _LOGIN_MARKERS
                    )
                    if _profile_html and not _profile_httpx_blocked:
                        _httpx_matches = [
                            t for t in _extract_tweets_from_html(_profile_html)
                            if _mentions_product(t, product_name)
                        ]
                        logger.info(
                            "Twitter/X PRODUCT httpx profile re-check for %r: "
                            "%d matching tweet(s) found via plain HTML in "
                            "%.2fs (no browser thread spent on this step).",
                            product_name, len(_httpx_matches),
                            time.monotonic() - _httpx_recheck_start,
                        )
                        if _httpx_matches:
                            timeline_texts = _httpx_matches
                            content_source = "profile_httpx"
                elif product_name:
                    # Previously silent: with no log line here, a log
                    # couldn't distinguish "recheck ran and found nothing"
                    # from "recheck never got a chance to run at all" - the
                    # exact ambiguity that made the Headphone Zone log hard
                    # to diagnose (search failed for all 3 products, but
                    # this step's own success/failure log never appeared).
                    logger.info(
                        "Twitter/X PRODUCT httpx profile re-check for %r: "
                        "skipped - only %.1fs left (need > %.1fs) after the "
                        "search attempt's own content-wait.",
                        product_name, time_left(), MIN_USEFUL_BUDGET_SECONDS,
                    )

            if not timeline_texts:
                # --- Attempt 2b (and the ONLY attempt for the "General"
                # brand-wide job): the brand's own profile timeline via a
                # full browser navigation + scroll. For a product-specific
                # call, results here get filtered down to tweets that
                # actually mention the product below - this is not a "just
                # show brand tweets" fallback. ---
                _nav_start = time.monotonic()
                _nav_ok = _goto_with_retry(
                    page, url,
                    timeout=_adaptive_nav_timeout(time_left),
                    time_left=time_left,
                )
                logger.info(
                    "Twitter/X scrape: navigation took %.2fs (ok=%s).",
                    time.monotonic() - _nav_start, _nav_ok,
                )
                if not _nav_ok:
                    _save_debug_artifacts(page, url, "nav_failure")
                    return normalize_comments(results)

                if _looks_blocked(page):
                    try:
                        page.wait_for_selector(
                            "article", timeout=_adaptive_content_timeout(time_left, reserve_seconds=6.0)
                        )
                        logger.info(
                            "Login prompt detected, but tweets are visible. Continuing."
                        )
                    except Exception:
                        logger.info("Actual login wall detected.")
                        _save_debug_artifacts(page, url, "login_wall")
                        return normalize_comments(results)

                got_articles = True
                _select_start = time.monotonic()
                try:
                    page.wait_for_selector(
                        "article", timeout=_adaptive_content_timeout(time_left, reserve_seconds=4.0)
                    )
                    page.wait_for_timeout(300)
                except Exception:
                    got_articles = False
                    reason = _diagnose_empty_page(page)
                    title = _title_bounded(page, 1000, default="<unknown>")
                    logger.info(
                        "Twitter/X scrape: no tweets rendered for %s "
                        "(title=%r diagnosis=%s)",
                        url, title, reason,
                    )
                logger.info(
                    "Twitter/X scrape: wait_for_selector(article) took %.2fs (found=%s).",
                    time.monotonic() - _select_start, got_articles,
                )

                if got_articles and time_left() > 2:
                    _scroll_start = time.monotonic()
                    timeline_texts, tweet_links, text_link_map = _scroll_timeline_until_idle(
                        page, time_left, MAX_TOTAL_POSTS,
                    )
                    content_source = "profile_browser"
                    logger.info(
                        "Twitter/X scrape: timeline scrolling took %.2fs (collected=%d).",
                        time.monotonic() - _scroll_start, len(timeline_texts),
                    )

            if product_name:
                # Precise product filter, applied uniformly regardless of
                # which of the three sources above (search, a cheap httpx
                # profile re-check, or a full browser profile scroll)
                # timeline_texts ended up coming from: genuine-over-volume
                # means it's better to return fewer comments here than to
                # mislabel a generic/unrelated tweet as being about this
                # specific product.
                matching_texts = [t for t in timeline_texts if _mentions_product(t, product_name)]
                matching_keys = {" ".join(t.split()).lower() for t in matching_texts}
                tweet_links = [text_link_map[k] for k in matching_keys if k in text_link_map]
                for t in matching_texts:
                    _add(t)
                logger.info(
                    "Twitter/X PRODUCT filter for %r: %d/%d candidate tweet(s) "
                    "actually mention the product (source=%s).",
                    product_name, len(matching_texts), len(timeline_texts),
                    content_source,
                )
                if not matching_texts:
                    logger.info(
                        "Twitter/X: no tweets mentioning product=%r found via "
                        "search, a profile re-check, or a full profile scroll; "
                        "returning [] rather than falling back to unrelated "
                        "brand content.",
                        product_name,
                    )
            else:
                for t in timeline_texts:
                    _add(t)

            _reply_expand_total = 0.0

            for link in tweet_links:

                if len(results) >= MAX_TOTAL_POSTS or time_left() <= 3:
                    break

                try:

                    if not _goto_with_retry(
                        page,
                        link,
                        timeout=_adaptive_nav_timeout(time_left),
                        time_left=time_left,
                    ):
                        continue

                    try:
                        page.wait_for_selector(
                            'article[data-testid="tweet"], article',
                            timeout=1000,
                        )
                    except Exception:
                        pass

                    page.mouse.wheel(0, 1200)
                    page.wait_for_timeout(600)

                    _expand_start = time.monotonic()
                    _expand_reply_threads(page, time_left)
                    _reply_expand_total += time.monotonic() - _expand_start

                    remaining = MAX_TOTAL_POSTS - len(results)

                    replies_added = 0

                    articles = _get_articles(page)

                    # BUG FIX: this was `max(remaining, MAX_REPLIES_PER_TWEET)`,
                    # which - whenever real budget remained (the common case
                    # early in the tweet_links loop, e.g. remaining=90) -
                    # picked the LARGER number and so effectively disabled the
                    # per-tweet cap entirely, letting a single reply-heavy
                    # tweet consume close to the WHOLE job's MAX_TOTAL_POSTS
                    # quota before any other matched tweet in tweet_links ever
                    # got a turn. MAX_REPLIES_PER_TWEET is a ceiling on one
                    # tweet's contribution (so genuine replies get sourced
                    # from multiple matched tweets, not just whichever is
                    # visited first) - min() is what actually enforces that,
                    # while still never asking for more than `remaining` when
                    # the overall quota is nearly full.
                    for reply in articles[
                        1 : 1 + min(remaining, MAX_REPLIES_PER_TWEET)
                    ]:

                        if len(results) >= MAX_TOTAL_POSTS:
                            break

                        if _is_noise_article(reply):
                            continue

                        try:

                            text = reply.inner_text()

                            if text and len(text.split()) > 4:

                                before = len(results)

                                _add(text)

                                if len(results) > before:
                                    replies_added += 1

                        except Exception:
                            continue

                    logger.info(
                        "Twitter/X scrape: tweet=%s contributed %d replies.",
                        link,
                        replies_added,
                    )

                except Exception:
                    continue

            logger.info(
                "Twitter/X scrape: reply expansion took %.2fs total across %d link(s).",
                _reply_expand_total, len(tweet_links),
            )

            if not results:
                _save_debug_artifacts(
                    page,
                    url,
                    "zero_tweets",
                    extra={
                        "login_wall_present": str(_looks_blocked(page)),
                        "articles_exist": str(len(_get_articles(page)) > 0),
                        "diagnosis": _diagnose_empty_page(page),
                    },
                )

        finally:

            if page is not None:
                try:
                    page.close()
                except Exception:
                    pass

        final = normalize_comments(results)[:MAX_TOTAL_POSTS]

        elapsed = time.monotonic() - start

        logger.info(
            "Twitter/X posts for %s: raw=%d duplicates_removed=%d final=%d "
            "elapsed=%.1fs",
            url,
            len(results),
            duplicates_removed,
            len(final),
            elapsed,
        )

        logger.info(
            "Twitter/X scrape: total scraper runtime %.2fs (setup+navigation+scrape).",
            time.monotonic() - setup_start,
        )

        logger.info("Twitter returned %d posts", len(final))

        return final

    logger.info("Twitter: submitting _run() to executor")

    try:
        result = await loop.run_in_executor(_EXECUTOR, _run)
        logger.info("Twitter: executor returned %d posts", len(result))
        return result

    except Exception:
        logger.exception(
            "Unhandled error scraping Twitter/X comments for %r.",
            url,
        )
        return []
