"""
Module Name
-----------
scrapers/google_scraper.py

Purpose
-------
Collects genuine, product-specific customer feedback from Google: reviews
left on a company's Google Business Profile (via Google Maps), matched
and disambiguated against the actual company being analyzed, plus (where
a product page exposes its own on-page Google-sourced reviews)
per-product review text.

Responsibilities
-----------------
- `scrape_google_reviews(company_data)`: locate the correct Google Maps
  business listing for a company (disambiguating common/generic names by
  appending the company's own website domain — see
  `_disambiguate_maps_query`, fixed after a real mismatch in production),
  open its Reviews panel, and collect review text, translating
  non-English reviews where Maps offers a translation.
- `scrape_google_product_reviews(product_data)`: collect reviews specific
  to a single product page, when available, kept separate from the
  business-level review cache described below.
- Attaches to the single shared Chromium process via
  `scrapers.browser_utils.ensure_shared_browser()` rather than launching
  its own — see that module's header for why this matters on this
  project's resource-constrained target hardware.
- Emits `BROWSER_TRACE`-prefixed timing logs at each setup/navigation
  checkpoint (playwright start, shared-browser attach, context/page
  creation, the Maps search navigation itself) so a slow or failed run
  can be diagnosed from a plain log without a debugger attached.

Dependencies
------------
`playwright` (sync API, run on a dedicated worker thread/pool via
browser_utils), plus standard library `asyncio`, `re`, `threading`,
`time`.
"""

import asyncio
import atexit
import concurrent.futures
import logging
import re
import sys
import threading
import time
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import quote_plus

from scrapers.browser_utils import browser_manager, normalize_comments
from config import (
    MAX_GOOGLE_REVIEWS,
    MAX_SCROLL_ITERATIONS,
    SCROLL_IDLE_LIMIT,
    NAVIGATION_RETRIES,
    NAV_TIMEOUT_MS_MAX,
    NAV_TIMEOUT_MS_MIN,
    GOOGLE_REVIEW_CACHE_TTL_SECONDS,
)

logger = logging.getLogger(__name__)

# Kept as an alias for backwards compatibility with any code importing the
# old name directly; the real cap now lives in config.py so it can be tuned
# alongside the other platform volume caps.
MAX_RESULTS = MAX_GOOGLE_REVIEWS

# --- Hard internal time budget -----------------------------------------
# Kept a few seconds under the outer asyncio.wait_for() cap applied per
# job in app.py (48s) so this scraper almost always returns on its own,
# with whatever it has collected so far, instead of being cut off cold by
# the outer timeout and losing partial results. Now only the FIRST call for
# a given business actually reaches this budget - every other product job
# for the same company is served from the cache in _CACHE_LOCK-protected
# dict below and returns near-instantly.
#
# WIDENED 18 -> 35 based on direct production-log evidence: a single failed
# then slow-but-successful navigation retry sequence was observed consuming
# ~24s by itself (attempt 1 using its full ~9s timeout, attempt 2 using
# most of its ~7.8s allotment) - before the ~2.1s of fixed settle waits or
# _PLACE_RESULT_LOOP_MIN_TIME_SECONDS's own 8s floor are even accounted
# for. Both NAV_TIMEOUT_MS_MIN/MAX (config.py) scale UP as more budget is
# available, so simply adding a few seconds back (as an earlier pass at
# this fix did, 18->22) does not reliably solve it: worst case nav alone
# can still consume up to ~21s (two attempts at the 12s ceiling minus
# backoff), which left only ~1s of real margin at 22s - not enough to
# clear the 8s place-result-loop floor. 35s leaves ~14s of margin after
# that worst-case nav for settle waits, layout detection, and the loop's
# own minimum, which is enough headroom based on what's been observed -
# but page-load timing is network-dependent and should be validated
# against real traffic; treat this as a strong first pass, not a
# guaranteed-final number.
#
# NOTE: this value is kept only as the "how long a clean run needs once
# setup is already finished" reference the paragraph above describes - it
# is NOT what _scrape_sync/_scrape_sync_product actually use for their
# deadline. See OUTER_HARD_TIMEOUT_SECONDS immediately below for why, and
# for the real fix.
TIME_BUDGET_SECONDS = 35

# --- Real (outer-timeout-anchored) deadline -----------------------------
# MECHANISM (still current): a fresh TIME_BUDGET_SECONDS clock started
# after setup can legally push the total past app.py's outer
# asyncio.wait_for() cap once setup cost is added in - and that cap is a
# hard cliff, not a soft budget. Production logs once showed a caller
# giving up at the outer cap while the background scrape kept running
# unseen and finished moments later with real reviews that never reached
# the dashboard, because the coroutine that would have returned them had
# already been cancelled. Fixed the same way as twitter_scraper.py/
# instagram_scraper.py/youtube_scraper.py: one real wall-clock deadline
# anchored to `start` (== `queued_at`, the moment scrape_google_reviews()/
# scrape_google_product_reviews() itself started, which is also
# effectively when app.py's wait_for() began counting against this job),
# via OUTER_HARD_TIMEOUT_SECONDS / SAFETY_MARGIN_SECONDS, so _scrape_sync
# always leaves real margin for the result to make it back through
# run_in_executor()/asyncio.shield()/wait_for(), no matter how long setup
# took. This is also why pairing this with the browser pre-warm matters:
# with setup down to ~1-3s (warmed) instead of 17s+ (cold), the working
# budget this computes (OUTER_HARD_TIMEOUT_SECONDS - SAFETY_MARGIN_SECONDS
# - setup) comes out larger on a warm run than a cold/contended one - the
# anchoring only costs real working time on a still-cold/contended run,
# which is exactly the situation where the alternative was overrunning the
# outer cap anyway.
#
# VALUE HISTORY (kept for context - the number actually in effect is the
# line comment on OUTER_HARD_TIMEOUT_SECONDS itself, not this paragraph):
# originally 48s, matching app.py's old GOOGLE_JOB_TIMEOUT_SECONDS. Raised
# to 90s on the reasoning that Google Reviews is single-flight cached per
# business (see the cache near the bottom of this file), so a bigger
# timeout was a one-time cost per analysis run, not a multiplied one - and
# the old 48s ceiling was itself why setup+nav+Reviews-tab detection was
# routinely leaving only ~10-20s for the scroll loop that actually
# collects review text, capping volume even for a genuinely popular
# product. Subsequently cut to the current 45s once the whole-pipeline
# 1-3 minute ceiling took priority over Google's one-time cost being
# individually affordable - a ~90s stage is still a large share of a 180s
# total budget even though it only happens once per run. If a future
# session revisits this number, weigh both considerations explicitly
# rather than re-deriving just one side of the trade-off.
OUTER_HARD_TIMEOUT_SECONDS = 45  # cut from 90 (itself raised from 48) - priority shifted to the whole-pipeline 3-min ceiling. must be kept in sync with app.py's GOOGLE_JOB_TIMEOUT_SECONDS
SAFETY_MARGIN_SECONDS = 8.0

# Below this much remaining budget, don't even attempt the initial Maps
# navigation - _INITIAL_NAV_RESERVE_SECONDS (defined further down, = 11)
# is what the navigation+candidate-scan step alone needs, and there has to
# be a few seconds left over on top of that for the Reviews-tab hunt and
# feed wait that follow it to have any realistic chance, or this just
# burns time on a navigation that can never reach the part that actually
# collects reviews.
MIN_USEFUL_BUDGET_SECONDS = 15.0

# --- Navigation retry policy --------------------------------------------
# No backoff sleep between retry attempts: the observed failure mode is a
# slow-but-eventually-successful navigation, not a transient fast-fail that
# benefits from waiting before retrying (see _goto_with_retry).
NAV_RETRIES = NAVIGATION_RETRIES

# --- Browser pool ---------------------------------------------------------
# Instead of launching a brand-new Chromium process for every single
# company (slow: ~1-2s of pure process/context startup on top of network
# time, and it throws away any cookies/consent state we already earned),
# we keep a small pool of worker threads alive for the life of the process.
# Each worker thread owns exactly one Playwright/browser/context (Sync
# Playwright objects are not thread-safe, so "one thread -> one browser" is
# the safe unit of reuse). Concurrent calls to scrape_google_reviews are
# spread across these workers, so we still get parallelism, but each
# worker's browser/context (and its cookies/consent state) survives across
# many calls instead of being torn down every time.
MAX_BROWSER_WORKERS = 3

# FOUND THIS SESSION (from code review; not yet confirmed against a live
# log - worth checking the next fresh log for this specifically): app.py's
# google_semaphore (config.MAX_PARALLEL_TASKS = 5 slots) lets every Google
# job start at once from app.py's side - the one "General" job plus one
# per selected product, so up to 4 with MAX_SELECTABLE_PRODUCTS=3. All 4
# then submit to THIS module's own _EXECUTOR, which only has
# MAX_BROWSER_WORKERS=3 threads. With 4 jobs and 3 workers, one job always
# sits queued inside the ThreadPoolExecutor waiting for a thread to free
# up - and since queued_at (captured before submission - see
# scrape_google_reviews()/scrape_google_product_reviews()) is what
# OUTER_HARD_TIMEOUT_SECONDS anchors its deadline to, that queueing delay
# comes directly out of the queued job's own working budget before a
# single line of its Playwright code runs. Raising this to 4 would remove
# the queueing but adds a 4th concurrent Chromium context specifically for
# Google, on top of Twitter/Instagram/YouTube/Website Review's own worker
# pools sharing the same 2-core/4-thread CPU (see ARCHITECTURE.md's
# CPU-contention section) - not a clear win without live-log evidence that
# queueing, rather than Google's own anti-bot posture, is the actual
# bottleneck for whichever job runs last. To check: compare "elapsed since
# job start" against "setup took" in _scrape_sync's "browser/context/page
# setup took..." log line - a gap much bigger than setup_elapsed itself is
# queueing delay, not setup cost.
#
# Recycle a worker's browser context after this many scrapes even if it's
# healthy, so long-running server processes don't slowly accumulate memory
# / detached listeners inside a single long-lived Chromium tab-set.
MAX_USES_BEFORE_RECYCLE = 50

# Resource types we never need for text scraping. Blocking them cuts page
# load time meaningfully. We deliberately do NOT block stylesheets: some
# anti-bot / anti-scraping setups rely on CSS-driven visibility (e.g.
# honeypot elements hidden with display:none), and losing computed styles
# could cause us to scrape decoy content.
BLOCK_HEAVY_RESOURCES = True
_BLOCKED_RESOURCE_TYPES = {"image", "media", "font"}

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
_LAUNCH_ARGS = [
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-dev-shm-usage",
    "--disable-blink-features=AutomationControlled",
]

_CONSENT_BUTTON_TEXTS = ["Accept all", "I agree", "Accept All", "AGREE", "Reject all"]

_AGO_RE = re.compile(
    r"\b(?:\d+|a|an)\s+(?:second|minute|hour|day|week|month|year)s?\s+ago\b",
    re.IGNORECASE,
)

_LOCAL_GUIDE_PREFIX_RE = re.compile(r"^\s*Local Guide\b[^\n]*\n?", re.IGNORECASE)

_BIO_STATS_RE = re.compile(
    r"\d[\d,.]*\s*[kKmM]?\+?\s*(followers|following|posts|likes|subscribers)\b",
    re.IGNORECASE,
)

_BOT_CHECK_MARKERS = ("unusual traffic", "captcha", "recaptcha", "/sorry/")

_MORE_BUTTON_SELECTORS = [
    # Current Google Maps "More" / "Read more" expansion controls.
    # Maps uses both <button> and <div>/<span> with jsaction for these.
    "button.w8nwRe",
    "button[aria-label*='more' i]",
    "button:has-text('More')",
    "button:has-text('Read more')",
    "button:has-text('See more')",
    # div[role='button'] variant — same pattern as the tab control,
    # Maps increasingly uses div[role='button'] for inline controls too.
    "div[role='button'][aria-label*='more' i]",
    # jsaction-based spans (most stable across class churn).
    "span[jsaction*='pane.review.expandReview']",
    "span[jsaction*='expandReview']",
    # Class-fragment patterns seen in current Maps builds.
    "span.review-more-link",
    "span[class*='kyuRq']",
    # XPath broad catch-all.
    "xpath=//button[contains(normalize-space(.), 'More')]",
    "xpath=//div[@role='button'][contains(normalize-space(.), 'More')]",
    "xpath=//span[@jsaction and contains(@jsaction,'expandReview')]",
]

# Buttons Google Maps shows on reviews it has auto-translated, letting the
# viewer flip between the translation and the original. We click through
# these too so translated reviews aren't left showing a truncated or
# machine-translated placeholder instead of their real text.
_TRANSLATION_BUTTON_SELECTORS = [
    "button:has-text('See original')",
    "button:has-text('Show original')",
    "button:has-text('Translate review')",
    "button[aria-label*='original' i]",
    "button[aria-label*='translat' i]",
    "span:has-text('See original')",
    "xpath=//button[contains(normalize-space(.), 'original')]",
]

_REVIEW_SELECTORS = [
    "span.wiI7pd",
    "div.jftiEf",
    "div[data-review-id]",
    "span.review-full-text",
    "div.review-snippet",
    "div.gws-localreviews__google-review",
]
_SNIPPET_SELECTORS = [
    "div.VwiC3b",
    "span.aCOpRe",
    "div.lyLwlc",
    "span.MUxGbd",
]
# Selectors for the scrollable reviews feed itself. ``div.m6QErb[aria-label]``
# is Google's generic scrollable-pane class and is reused for several
# different side panels (About, Q&A, Reviews, ...), so it's listed last and
# only used as a fallback; ``div[role='feed']`` is the more reliable,
# structure-based match since Maps consistently marks the reviews list with
# an ARIA feed role regardless of which obfuscated class name is current.
_REVIEW_PANEL_SELECTORS = [
    # Most reliable structural indicator — Maps consistently marks the
    # live reviews list with role='feed' regardless of class-name churn.
    "div[role='feed']",
    # Current Maps builds (2024-2026) use these specific m6QErb variants
    # for the reviews scroll pane. Ordered most-to-least specific.
    "div.m6QErb.WNBkOb",
    "div.m6QErb.XiKgde",
    "div.m6QErb.DxyBCb",
    "div.m6QErb.dmoGRc",
    # Aria-label on the panel container — present on most layouts.
    "div[aria-label*='Reviews' i]",
    "div[aria-label*='Review' i]",
    # Any m6QErb scrollable pane (broad — used as fallback since this
    # class is also used for other side panels, but after a Reviews tab
    # click the first one encountered should be the reviews feed).
    "div.m6QErb[aria-label]",
    # jsaction on the panel wrapper.
    "div[jsaction*='pane.reviewChart']",
    # Legacy review dialog list selector.
    "div.review-dialog-list",
    # Last resort: any scrollable region inside the main content area.
    "div[role='main'] div[tabindex='-1']",
]

# Selectors for the "Reviews" tab button on a place detail page, tried only
# after the ARIA role-based lookup (see _scrape_maps) comes up empty. Maps
# does not always expose this control with role="tab" - on narrower/mobile
# layouts and in some current builds it renders as a plain button, and the
# visible text can be swapped for an icon with the label only present in
# aria-label or a jsaction hook - so a role/name match alone can miss it
# even though the control is present and clickable.
_REVIEWS_TAB_SELECTORS = [
    # --- Tier 0: modern (2025-2026) tab bar --------------------------------
    # Current Maps builds render the place-page tab bar as <button
    # role='tab'> elements whose aria-label reads "<Tab> of <business>" /
    # "Reviews for <business>" (verified against a live 2026-07 capture:
    # the tablist held `button[role='tab'][aria-label='Overview of ...']`,
    # `... 'About of ...'` — plain <button role='tab'>, not div[role=
    # 'button']).
    "button[role='tab'][aria-label^='Reviews']",
    "button[role='tab'][aria-label*='Reviews' i]",
    "button[role='tab']:has-text('Reviews')",

    # --- Tier 1: aria-label on div[role='button'] --------------------------
    # Google Maps 2024-2026 renders the nav pills as `div[role='button']`
    # inside a `div[role='tablist']`, NOT as `<button>` elements. This is
    # the single most common reason the old list found nothing: every
    # `button[...]` selector silently matched zero nodes.
    "div[role='button'][aria-label='Reviews']",
    "div[role='button'][aria-label*='Reviews' i]",
    "div[role='button'][aria-label*='Review' i]",

    # --- Tier 2: jsaction attribute on div (not button) -------------------
    # jsaction hooks are more stable than class names across Maps redesigns.
    "div[jsaction*='pane.rating.moreReviews']",
    "div[jsaction*='moreReviews']",
    "div[jsaction*='reviewChart']",

    # --- Tier 3: data-value / data-tab-index attributes -------------------
    # Some Maps layouts tag the tab container with data-value or
    # data-tab-index that survive class-name churn.
    "[data-value='Reviews']",
    "[data-value*='Review' i]",
    "[data-tab-index][aria-label*='Reviews' i]",

    # --- Tier 4: any element carrying the Reviews aria-label --------------
    # No element-type constraint — catches the label whether it's on a
    # div, button, span, or anchor.
    "[aria-label='Reviews']",
    "[aria-label*='Reviews' i]",

    # --- Tier 5: button element variants (legacy / A-B test layouts) ------
    # Kept as fallback: Maps has historically used <button> on some builds.
    "button[aria-label*='Reviews' i]",
    "button[aria-label*='Review' i]",
    "button[jsaction*='moreReviews']",
    "button[jsaction*='pane.rating.moreReviews']",
    "button[jsaction*='reviewChart']",

    # --- Tier 6: role='tab' variants (older Maps builds) ------------------
    "div[role='tab'][aria-label*='Reviews' i]",
    "div[role='tab']:has-text('Reviews')",
    "[role='tablist'] [aria-label*='Reviews' i]",

    # --- Tier 7: anchors / star rating click-through ---------------------
    # On direct-redirect business pages there is sometimes no tab bar at
    # all; clicking the star-rating count instead jumps straight to the
    # reviews list.
    "a[aria-label*='Reviews' i]",
    "a[href*='reviews']",
    "span.fontBodyMedium a",

    # --- Tier 8: XPath (broadest catch-all) --------------------------------
    # Matches any interactive element whose accessible label OR visible text
    # is or contains "Reviews". Intentionally broad — if everything above
    # failed, we want a last-chance match before giving up.
    "xpath=//div[@role='button'][contains(@aria-label,'Reviews')]",
    "xpath=//div[@role='button'][normalize-space(.)='Reviews']",
    "xpath=//*[@jsaction and contains(@jsaction,'moreReviews')]",
    "xpath=//*[@jsaction and contains(@jsaction,'reviewChart')]",
    "xpath=//button[contains(@aria-label,'Review')]",
]

# --- ROOT-CAUSE FIX (see _scrape_maps' place-result loop below) ----------
# Minimum time_left() required to even ENTER the place-result selector
# loop. This used to be hardcoded to 15 inline, which - combined with the
# ~4-5s of fixed overhead that's always spent before this loop is reached
# (initial nav + the 1800ms/300ms settle waits + consent dismissal +
# bot-check + _detect_maps_layout) - meant time_left() was *already* at or
# below 15 on essentially every single call, every time, regardless of
# company. The loop's own `break` fired on its very first check, before a
# single selector was ever tried, and did so silently (no log line), which
# is exactly why every business in the production logs (Concept Kart, boAt
# Lifestyle, Xiaomi India, Sony - completely unrelated companies) failed
# identically with "no place-result selector matched in time": nothing was
# ever actually attempted. Every other time-budget guard in this file (see
# the `_wait_for_review_panel`, review-tab hunt, and scroll-loop checks
# below) uses 6-12 as its "give up" threshold, not 15 - this constant
# brings the place-result loop back in line with that convention.
#
# NOTE: lowering this threshold alone was not sufficient on its own - it
# was still observed going negative (-5.8s) in production even at 8,
# because TIME_BUDGET_SECONDS itself (see above) was too small for a
# failed-then-successful nav retry sequence to fit inside. Both fixes are
# required together: this threshold controls how much time the loop needs
# once reached; TIME_BUDGET_SECONDS controls whether that much time is
# realistically ever left over.
_PLACE_RESULT_LOOP_MIN_TIME_SECONDS = 8

# Reserved for the settle wait + layout-detection step that must run right
# after the initial Maps navigation, before the candidate scan even starts.
_SETTLE_RESERVE_SECONDS = 3.0

# Reserved, ahead of time, for everything downstream of the initial Maps
# navigation - the settle/layout-detection step AND the candidate-scan step
# itself (_PLACE_RESULT_LOOP_MIN_TIME_SECONDS). Without this, an adaptive
# nav timeout computed purely from "whatever budget is left" can (with
# retries) legally consume down to just a few seconds remaining - below
# what the candidate scan needs to even attempt - which is exactly the
# "Maps detected -> scan skipped -> BUSINESS_NOT_FOUND" failure seen in
# production logs. This constant is only applied to the *initial* Maps
# navigation call, since that is the one sitting in front of the scan.
_INITIAL_NAV_RESERVE_SECONDS = _PLACE_RESULT_LOOP_MIN_TIME_SECONDS + _SETTLE_RESERVE_SECONDS

# Selectors for a place result in the Maps search results list, used to
# click into a place's detail page. Listed in order of specificity: the
# structural href match is resilient to Google's periodic class-name
# churn, the aria-role match is a reasonable middle ground, and the bare
# class names are kept as a last-resort fallback since they're the ones
# most likely to silently stop matching after a Maps redesign.
_PLACE_RESULT_SELECTORS = [
    "a[href*='/maps/place/']",
    "div[role='feed'] a",
    "a.hfpxzc",
    "div[role='article']",
    "div.Nv2PK",
    # Broad structural fallbacks: any clickable result row inside a feed,
    # regardless of Google's current obfuscated class names.
    "div[role='feed'] div[role='button']",
    "div[jsaction*='pane.resultSection'] a",
]

# --- Layout detection -----------------------------------------------------
# For a strong single-match query, Maps often skips the search-results list
# entirely and navigates straight to the place detail page. The old code
# assumed a results list was always present and only ever tried to click
# into one, so on a direct-redirect it logged "could not click place
# result" and then blindly kept going anyway - which happened to still
# work sometimes, but gave no real signal about *why* review collection
# succeeded or failed. These selectors let us positively identify "we are
# already on a business detail page" instead of only inferring it by
# elimination.
_BUSINESS_PAGE_INDICATOR_SELECTORS = [
    # Most reliable structural indicators — present on place detail pages.
    # NOTE: "div[role='main'] h1" was intentionally removed. The search-results
    # page renders role="main" on its left-side panel, which contains
    # <h1 class="fontTitleLarge IFMGgb">Results</h1> as a direct descendant.
    # That made the selector fire as a false positive while still on the search
    # list, causing _navigated() and the step-1.5 guard to confirm "business page"
    # before any SPA navigation had actually occurred.  All remaining selectors
    # below return 0 matches on the search-results page (verified against the
    # captured debug HTML) and are therefore safe to use as business-page indicators.
    "h1.DUwDvf",
    "div.TIHn2 h1",
    # Tab/pill navigation bar present on every place detail page.
    "div[role='tablist']",
    # Business action buttons (address, call, etc.) are place-detail-only.
    "button[data-item-id='authority']",
    "button[data-item-id='address']",
    # Nav pill container class (stable 2023-2026).
    "div.PPCwl",
    # jsaction hook on the rating section (place-detail-only).
    "div[jsaction*='pane.rating']",
    "div.RWPxGd",
]

# Selectors that indicate we're looking at a search-results list (as
# opposed to a single business detail page), used alongside the indicators
# above to classify the page before deciding whether to click a result.
_SEARCH_RESULTS_INDICATOR_SELECTORS = [
    "div[role='feed']",
    "div.Nv2PK",
    "a.hfpxzc",
]

# --- Diagnostic codes -------------------------------------------------------
# Coarse, greppable failure categories so a 0-review run's log clearly
# distinguishes *why* nothing came back instead of forcing a human to
# re-read a wall of "continuing anyway" warnings. Every call site below
# that can plausibly explain a 0-review outcome logs exactly one of these.
DIAG_BUSINESS_NOT_FOUND = "BUSINESS_NOT_FOUND"
DIAG_BUSINESS_MISMATCH = "BUSINESS_MISMATCH"
DIAG_REVIEWS_TAB_MISSING = "REVIEWS_TAB_MISSING"
DIAG_FEED_MISSING = "FEED_MISSING"
DIAG_ZERO_REVIEWS = "ZERO_REVIEWS"
DIAG_BOT_PAGE = "BOT_PAGE"
DIAG_NAV_FAILURE = "NAV_FAILURE"
DIAG_TIMEOUT = "TIMEOUT"
DIAG_WRONG_SELECTOR = "WRONG_SELECTOR"


def _log_diag(code: str, query: str, detail: str = "") -> None:
    """Emit one structured, greppable diagnostic line.

    Kept as a single-line, fixed-prefix format (``[GOOGLE_SCRAPE_DIAG]``)
    so a log search/alert can key off ``code`` without parsing prose.
    """
    logger.warning(
        "[GOOGLE_SCRAPE_DIAG] code=%s query=%r detail=%s",
        code, query, detail or "-",
    )


# Short UI/action labels that occasionally end up in a text node next to
# real review content (e.g. a "Helpful" vote button or a "Report" link
# sitting inside the same review card). These are filtered out of
# extracted text as an extra safety net on top of selector precision.
_UI_LABEL_BLACKLIST = {
    "helpful", "report", "share", "like", "send", "more", "less",
    "translate", "see original", "show original", "translate review",
    "response from the owner", "reply", "flag as inappropriate",
}

# Per-worker-thread Playwright state. Never touched from more than one
# thread at a time because each thread only ever runs jobs handed to it by
# the ThreadPoolExecutor below.
_thread_local = threading.local()
_browser_trace = threading.local()

_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=MAX_BROWSER_WORKERS, thread_name_prefix="google_scraper"
)


class _BrowserHandle:
    """Holds one worker thread's Playwright/browser/context.

    A handle is created once per worker thread and stashed both in that
    thread's ``_thread_local`` (so the owning thread can find/reuse it) and
    in the module-level ``_worker_handles`` registry below (so process-exit
    cleanup can find every handle without needing to run code back on the
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


def _looks_like_review(text: str) -> bool:
    if len(_BIO_STATS_RE.findall(text)) >= 2:
        return False
    return True


def _strip_review_boilerplate(text: str) -> str:
    t = text
    t = _LOCAL_GUIDE_PREFIX_RE.sub("", t)
    m = _AGO_RE.search(t)
    if m:
        t = t[m.end():]
    t = re.split(r"\.\.\.\s*More\b", t)[0]
    t = re.split(r"\bLike\b\s*\bShare\b", t)[0]
    t = re.sub(r"Response from the owner.*$", "", t, flags=re.IGNORECASE | re.DOTALL)
    t = re.sub(r"\bShare\b\s*$", "", t)
    t = re.sub(r"\s+", " ", t).strip(" .|\u2022")
    return t


# Near-duplicate collapsing (see _dedupe) only ever activates for strings at
# least this long, so two short-but-similarly-phrased *distinct* reviews
# (e.g. "Great service, highly recommend!" from two different reviewers)
# are never at risk of being merged - only long, near-verbatim repeats are.
_NEAR_DUP_MIN_LEN = 40
_NEAR_DUP_RATIO = 0.93


def _dedupe(strings: List[str]) -> List[str]:
    """Case/whitespace-insensitive de-duplication, order-preserving.

    Also collapses near-duplicates for longer strings. ``raw_results`` (see
    ``_scrape_sync``) can merge text from Maps plus the Google-search and
    Bing fallbacks, which only run when Maps didn't return enough reviews -
    and it's common for a search-engine snippet to be a shorter excerpt of a
    review whose full text was already captured from Maps. An exact-match
    check alone doesn't catch that, since the wording/truncation differs
    between sources. Scoped to ``_NEAR_DUP_MIN_LEN``+ strings at a high
    similarity ratio, with a cheap length-ratio check before the more
    expensive comparison, so cost stays negligible at the review counts this
    module deals with.
    """
    seen = set()
    out: List[str] = []
    kept_long: List[str] = []  # normalized keys of already-kept long entries
    for s in strings:
        key = re.sub(r"\s+", " ", s).strip().lower()
        if not key or key in seen:
            continue
        if len(key) >= _NEAR_DUP_MIN_LEN:
            is_near_dup = False
            for other in kept_long:
                longer, shorter = (len(other), len(key)) if len(other) >= len(key) else (len(key), len(other))
                if shorter / longer < 0.7:
                    continue  # cheap pre-filter: too different in length to bother comparing
                if SequenceMatcher(None, key, other).ratio() >= _NEAR_DUP_RATIO:
                    is_near_dup = True
                    break
            if is_near_dup:
                continue
            kept_long.append(key)
        seen.add(key)
        out.append(s)
    return out


def _wait_for_more(page, selector: str, prev_count: int, max_ms: int = 500) -> int:
    """Poll ``selector``'s match count in short steps until it exceeds
    ``prev_count`` or ``max_ms`` elapses, whichever comes first.

    Replaces an unconditional ``page.wait_for_timeout(max_ms)`` after a
    scroll step: the worst-case wait is identical (this never runs longer
    than ``max_ms``), but as soon as new review nodes attach - which is
    most of the time on a live feed - this returns immediately instead of
    always sitting out the full window, leaving more of the scraper's time
    budget for additional scroll iterations. Mirrors the same helper in
    youtube_scraper.py.
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


def _bot_checked(page) -> bool:
    """Cheap, instant bot-check/captcha detection (no extra waiting)."""
    try:
        url = page.url or ""
    except Exception:
        url = ""
    if any(marker in url for marker in _BOT_CHECK_MARKERS):
        return True
    try:
        title = (page.title() or "").lower()
    except Exception:
        title = ""
    return any(marker in title for marker in _BOT_CHECK_MARKERS)


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


def _dismiss_consent(page) -> None:
    """Best-effort dismissal of Google's cookie/consent interstitial.

    Without this, the very first navigation in a fresh context can land on
    a consent page instead of the real content, and every selector below
    silently finds nothing. Once dismissed, the choice is stored as a
    cookie in the (reused) context, so this is normally only paid once per
    worker thread, not once per request.
    """
    for text in _CONSENT_BUTTON_TEXTS:
        try:
            btn = page.get_by_role("button", name=text, exact=False)
            if btn.count() > 0:
                btn.first.click(timeout=1500)
                page.wait_for_timeout(400)
                logger.debug("Dismissed a consent dialog (button=%r).", text)
                return
        except Exception:
            continue


def _adaptive_nav_timeout(time_left, reserve_seconds: float = 0.0) -> int:
    """Scale the navigation timeout to how much of the internal time
    budget is actually left, instead of a fixed 5000ms for every attempt.

    ``reserve_seconds`` carves out time for phases that must run *after*
    this navigation completes (settle/layout-detection, candidate scan) so
    navigation is budget-aware in both directions: it still won't ask for
    more than is actually left, but it also can no longer legally consume
    the entire remaining budget down to nothing, starving whatever runs
    next. Callers that have no downstream phase to protect (e.g. the
    click-through and fallback navigations, which run after the scan) pass
    the default of 0.0 and get the original behavior.

    Navigation always receives at least NAV_TIMEOUT_MS_MIN (8s) as long as
    that much *navigable* time (remaining minus the reserve) is available -
    the budget clock only starts once browser/context/page setup has
    finished (see _scrape_sync below), so this floor is real, not eaten by
    Chromium startup. When plenty of budget remains the timeout can scale
    up to NAV_TIMEOUT_MS_MAX (12s) for a genuinely slow page (Maps in
    particular can be slow to paint its results list); when the navigable
    time itself is under the floor, we hand over whatever is left rather
    than blocking past the scraper's own deadline.
    """
    navigable_ms = max(0.0, time_left() - reserve_seconds) * 1000
    if navigable_ms <= NAV_TIMEOUT_MS_MIN:
        return max(3000, int(navigable_ms))
    return int(min(NAV_TIMEOUT_MS_MAX, max(NAV_TIMEOUT_MS_MIN, navigable_ms * 0.5)))


def _goto_with_retry(
    page, url: str, *, timeout: int, time_left, retries: int = NAV_RETRIES,
    reserve_seconds: float = 0.0,
) -> bool:
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
        attempt_timeout = _adaptive_nav_timeout(time_left, reserve_seconds)
        if attempt > 0:
            logger.info(
                "Google nav retry attempt=%d/%d url=%s remaining=%.1fs chosen_timeout=%dms",
                attempt + 1, total_attempts, url, remaining, attempt_timeout,
            )
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=attempt_timeout)
            return True
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Navigation attempt %d/%d to %s failed: %s",
                attempt + 1, total_attempts, url, exc,
            )
            # No backoff sleep between attempts: the observed failure mode
            # is a slow-but-eventually-successful navigation, not a
            # transient fast-fail that benefits from waiting before retry -
            # sleeping here only spends shared budget that the candidate
            # scan needs later, for no measurable benefit.
            continue
    logger.error("Giving up on %s after %d attempts: %s", url, total_attempts, last_exc)
    return False


def _expand_translated_reviews(page, time_left, limit: int = 40) -> int:
    """Click "See original" / "Translate review" toggles.

    Without this, an auto-translated review can render as a placeholder or
    a shortened machine translation instead of the reviewer's real text.
    """
    clicked = 0
    for sel in _TRANSLATION_BUTTON_SELECTORS:
        if clicked >= limit or time_left() <= 8:
            break
        try:
            buttons = page.locator(sel).all()
        except Exception:
            continue
        for btn in buttons[: max(0, limit - clicked)]:
            if time_left() <= 8:
                break
            try:
                btn.click(timeout=800)
                clicked += 1
                page.wait_for_timeout(120)
            except Exception:
                continue
    return clicked


def _scroll_reviews_panel(page, time_left, max_results: int = None) -> int:
    """Scroll the actual reviews feed until it stops yielding new reviews.

    Replaces the old fixed-count scroll loop with an unbounded ``while
    True`` that keeps scrolling (and periodically expanding "More" /
    translated-review buttons so truncated text is captured) until one of:

      * ``max_results`` distinct review nodes have been located,
      * ``MAX_SCROLL_ITERATIONS`` scroll steps have been performed,
      * ``SCROLL_IDLE_LIMIT`` consecutive scrolls produced no new reviews, or
      * the overall scraper time budget is nearly exhausted.

    Returns the number of distinct review nodes found so callers/logs know
    how much was actually collected.
    """
    max_results = max_results or MAX_GOOGLE_REVIEWS
    start = time.monotonic()
    logger.info("Scrolling reviews (target up to %d review(s))...", max_results)

    def _count() -> int:
        """Count currently-rendered review nodes.

        ``div[data-review-id]`` is checked first since it's Google's current
        structural marker for a review card (cheap: a single locator call in
        the normal case, and behaves identically to before whenever it
        returns >0). If it ever reports 0 - e.g. a future Maps build
        drops/renames that attribute, the same kind of class-name churn this
        file's other selector lists are already built to survive - falling
        straight through to "no new reviews" would make the idle-scroll
        logic below stop after just a couple of iterations even though the
        feed is actively rendering content. As a safety net in that (today
        unseen) case only, we also check the other known review-card
        selectors from ``_REVIEW_SELECTORS`` and take whichever reports the
        highest count, so a single selector going stale can't silently stall
        collection.
        """
        try:
            primary = page.locator("div[data-review-id]").count()
        except Exception:
            primary = -1
        if primary > 0:
            return primary
        best = max(primary, 0)
        for sel in _REVIEW_SELECTORS:
            if sel == "div[data-review-id]":
                continue
            try:
                c = page.locator(sel).count()
            except Exception:
                continue
            if c > best:
                best = c
        return best

    scrollable = None
    for sel in _REVIEW_PANEL_SELECTORS:
        try:
            loc = page.locator(sel).first
            if loc.count() > 0:
                scrollable = loc
                break
        except Exception:
            continue

    if scrollable is None:
        # No dedicated feed found; fall back to page-level scrolling with
        # the same idle/iteration/time stopping conditions.
        logger.info("Reviews scroll: no scrollable feed found; using page-level scroll.")
        _log_diag(
            DIAG_WRONG_SELECTOR, "<n/a>",
            "none of _REVIEW_PANEL_SELECTORS matched a scrollable feed node; "
            "falling back to whole-page scrolling",
        )
        iteration = 0
        idle = 0
        prev_count = _count()
        while True:
            iteration += 1
            if iteration > MAX_SCROLL_ITERATIONS or time_left() <= 10:
                break
            try:
                page.mouse.wheel(0, 900)
            except Exception:
                break
            # Readiness-based wait: return as soon as new review nodes
            # attach instead of always sitting out the full 400ms window.
            # Same worst-case bound as the previous fixed sleep.
            _wait_for_more(page, "div[data-review-id]", prev_count, max_ms=400)
            count = _count()
            if count <= prev_count:
                idle += 1
            else:
                idle = 0
            prev_count = max(prev_count, count)
            if idle >= SCROLL_IDLE_LIMIT or prev_count >= max_results:
                break
        return max(prev_count, 0)

    # Scroll the feed panel into the viewport before starting the loop.
    # Maps uses an intersection-observer-based lazy-loader: if the feed
    # panel is off-screen, the XHR that fetches review cards never fires
    # and _count() stays at zero no matter how many times we scroll.
    try:
        scrollable.scroll_into_view_if_needed(timeout=2000)
        page.wait_for_timeout(600)
    except Exception:
        pass

    iteration = 0
    idle = 0
    prev_count = -1
    while True:
        iteration += 1
        if iteration > MAX_SCROLL_ITERATIONS:
            logger.info("Reviews scroll: hit MAX_SCROLL_ITERATIONS (%d).", MAX_SCROLL_ITERATIONS)
            break
        if time_left() <= 10:
            logger.info("Reviews scroll: stopping, low on overall time budget.")
            break

        try:
            scrollable.evaluate("el => el.scrollTo(0, el.scrollHeight)")
        except Exception:
            break
        # Readiness-based wait: return as soon as new review nodes attach
        # instead of always sitting out the full 500ms window. Same
        # worst-case bound as the previous fixed sleep.
        _wait_for_more(page, "div[data-review-id]", prev_count, max_ms=500)

        # Periodically expand truncated / translated reviews so their full
        # text is present in the DOM once we go collect it. Doing this
        # every iteration (rather than only once at the end) means reviews
        # loaded early aren't left un-expanded while we keep scrolling.
        if iteration % 3 == 0:
            _expand_more_buttons(page, time_left)
            _expand_translated_reviews(page, time_left)

        count = _count()
        logger.info(
            "Reviews scroll: iteration=%d current_count=%d idle_streak=%d elapsed=%.1fs",
            iteration, count, idle, time.monotonic() - start,
        )

        if count >= max_results:
            logger.info("Reviews scroll: reached target of %d reviews.", max_results)
            break

        if count <= prev_count:
            idle += 1
            if idle >= SCROLL_IDLE_LIMIT:
                logger.info(
                    "Reviews scroll: no new reviews after %d consecutive scrolls; stopping.",
                    SCROLL_IDLE_LIMIT,
                )
                break
        else:
            idle = 0
        prev_count = max(prev_count, count)

    # Final expansion pass once scrolling has settled, so anything loaded
    # on the very last scroll still gets its "More" / translation buttons
    # clicked before we read the text out.
    _expand_more_buttons(page, time_left)
    _expand_translated_reviews(page, time_left)
    return max(prev_count, 0)


def _expand_more_buttons(page, time_left, limit: int = 150) -> int:
    """Click "More" links so truncated review text isn't lost.

    The previous version only matched buttons whose aria-label contained
    "more", which misses Google Maps' actual "More" text-link (rendered as
    a plain button/span with visible text, not an aria-label) as well as
    third-party review widgets. We try several patterns and keep going
    until the limit or the time budget is hit.
    """
    clicked = 0
    for sel in _MORE_BUTTON_SELECTORS:
        if clicked >= limit or time_left() <= 8:
            break
        try:
            buttons = page.locator(sel).all()
        except Exception:
            continue
        for btn in buttons[: max(0, limit - clicked)]:
            if time_left() <= 8:
                break
            try:
                btn.click(timeout=800)
                clicked += 1
                page.wait_for_timeout(120)
            except Exception:
                continue
    return clicked


def _collect_texts(page, selectors: List[str], limit_per_selector: int = 25) -> List[str]:
    """Batch-collect visible text for a list of selectors.

    Using .all_inner_texts() issues one round-trip per selector instead of
    one per matched element (the old code called .inner_text() in a Python
    loop over every located element), which is both faster and removes a
    lot of per-element try/except noise.
    """
    texts: List[str] = []
    for sel in selectors:
        try:
            found = page.locator(sel).all_inner_texts()
        except Exception:
            continue
        for txt in found[:limit_per_selector]:
            txt = (txt or "").strip()
            if txt:
                texts.append(txt)
    return texts


def _ensure_context():
    """Get (creating or recycling as needed) this thread's browser context."""
    handle = _get_handle()
    browser_trace_started_at = time.monotonic()
    _browser_trace.ensure_context_started_at = browser_trace_started_at
    ctx = handle.context
    logger.info(
        "CONTEXT_STATE_TRACE thread_id=%s has_playwright=%s has_browser=%s has_context=%s uses=%d",
        threading.get_ident(),
        handle.playwright is not None,
        handle.browser is not None,
        handle.context is not None,
        handle.uses,
    )

    if ctx is not None:
        if handle.uses >= MAX_USES_BEFORE_RECYCLE:
            logger.info(
                "Recycling browser context on %s after %d uses.",
                threading.current_thread().name, handle.uses,
            )
            _teardown_thread_browser(reason="recycle")
            ctx = None
        else:
            try:
                _ = ctx.pages  # cheap liveness check
                handle.uses += 1
                logger.info("USING EXISTING CONTEXT")
                return ctx
            except Exception:
                logger.warning(
                    "Browser context on %s appears dead; recreating.",
                    threading.current_thread().name,
                )
                _teardown_thread_browser(reason="dead_context")
                ctx = None

    logger.info("CREATING NEW CONTEXT")
    from playwright.sync_api import sync_playwright

    pw = handle.playwright
    if pw is None:
        logger.info(
            "BROWSER_TRACE scraper=Google event=before_sync_playwright_start elapsed=%.3fs timestamp=%.3f thread=%s",
            time.monotonic() - browser_trace_started_at, time.time(), threading.current_thread().name,
        )
        pw = sync_playwright().start()
        logger.info(
            "BROWSER_TRACE scraper=Google event=after_sync_playwright_start elapsed=%.3fs timestamp=%.3f thread=%s",
            time.monotonic() - browser_trace_started_at, time.time(), threading.current_thread().name,
        )
        handle.playwright = pw

    logger.info(
        "BROWSER_TRACE scraper=Google event=before_shared_browser_attach elapsed=%.3fs timestamp=%.3f thread=%s",
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
        "BROWSER_TRACE scraper=Google event=after_shared_browser_attach elapsed=%.3fs timestamp=%.3f thread=%s",
        time.monotonic() - browser_trace_started_at, time.time(), threading.current_thread().name,
    )
    logger.info(
        "BROWSER_TRACE scraper=Google event=before_new_context elapsed=%.3fs timestamp=%.3f thread=%s",
        time.monotonic() - browser_trace_started_at, time.time(), threading.current_thread().name,
    )
    context = browser.new_context(
        user_agent=_USER_AGENT,
        locale="en-US",
        viewport={"width": 1280, "height": 900},
    )
    logger.info(
        "BROWSER_TRACE scraper=Google event=after_new_context elapsed=%.3fs timestamp=%.3f thread=%s",
        time.monotonic() - browser_trace_started_at, time.time(), threading.current_thread().name,
    )
    context.set_default_timeout(8000)
    context.set_default_navigation_timeout(10000)
    if BLOCK_HEAVY_RESOURCES:
        _install_resource_blocking(context)

    handle.browser = browser
    handle.context = context
    handle.uses = 1
    logger.info("Launched a new browser context on %s.", threading.current_thread().name)
    return context


# --- Startup pre-warm --------------------------------------------------
# ROOT CAUSE (the other half of "Google never collects anything", together
# with the deadline-anchoring fix above): production logs show this
# thread's own sync_playwright().start() + attaching to the shared
# Chromium process + new_context() regularly taking 17s+ on a cold/
# contended run (Twitter/Instagram/up to 3 YouTube jobs all doing the same
# at the same instant) - and unlike Twitter/Instagram/YouTube, Google had
# no pre-warm at all, so it paid that full cost inside the real request's
# time budget every single time.
#
# That setup cost is a ONE-TIME, per-worker-thread cost - _ensure_context
# above already reuses a live context on later calls via a cheap
# `ctx.pages` check instead of redoing setup. So instead of a real,
# time-boxed scrape request paying that cost for the first time, pay it
# once here in the background at module import time (mirrors both the
# "Sentiment pipeline preloaded at startup" pattern and the identical
# pre-warm added to twitter_scraper.py/instagram_scraper.py/
# youtube_scraper.py) - well before the first real request in practice.
#
# MAX_BROWSER_WORKERS separate jobs are submitted (not just one) so every
# thread this module's own _EXECUTOR could later hand a real scrape to
# gets warmed, not just whichever one happens to run first.
#
# IMPORTANT - the mistake that silently defeated this same fix on the
# other three scrapers: app.py sets WindowsSelectorEventLoopPolicy
# process-wide at import time (see browser_utils.py), but only the
# Proactor loop can launch subprocesses on Windows - Playwright's sync API
# spawns its driver as a subprocess, so without re-asserting Proactor
# FIRST, this raises NotImplementedError on every warm-up thread on
# Windows (non-fatal, since it's wrapped in try/except below, but it means
# the warm-up silently does nothing and setup stays slow). The real
# request path (_scrape_sync) already re-asserts Proactor for the same
# reason; this warm-up needs its own copy of that same guard since it runs
# on a separate thread before any real request ever calls _scrape_sync.
#
# Best-effort only and never blocks import: each job runs on _EXECUTOR's
# own worker threads, not the importing thread, and any failure (e.g. a
# dev machine without Playwright's browsers installed yet) is caught and
# logged - the first real request then simply falls back to paying the
# setup cost itself, exactly as it did before this change.
def _warm_up_browser_context() -> None:
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    try:
        _ensure_context()
        logger.info(
            "Google: browser context pre-warmed on %s.",
            threading.current_thread().name,
        )
    except Exception:
        logger.exception(
            "Google: background browser context warm-up failed on %s "
            "(non-fatal - the first real request will pay the setup "
            "cost itself instead, same as before this change).",
            threading.current_thread().name,
        )


for _ in range(MAX_BROWSER_WORKERS):
    _EXECUTOR.submit(_warm_up_browser_context)


def _teardown_thread_browser(reason: str = "unknown") -> None:
    logger.info(
        "TEARDOWN_TRACE thread_id=%s reason=%s",
        threading.get_ident(),
        reason,
    )
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

    Runs the actual close on a fresh, throwaway ``threading.Thread`` rather
    than by submitting a job to ``_EXECUTOR``. Submitting to the same
    executor we're trying to shut down is what previously caused
    ``RuntimeError: cannot schedule new futures after shutdown`` — Python's
    own interpreter-shutdown sequence can mark the executor (or the
    process-wide threading state) as shut down before our atexit callback
    runs, and any ``executor.submit()`` after that point raises. A plain,
    unpooled thread has no such shutdown flag to race against.
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
    submitting cleanup jobs back onto ``_EXECUTOR``. This is deliberate:
    see ``_close_handle_with_timeout`` for why submitting to the executor
    from an atexit callback is unsafe. This is still best-effort — if a
    worker thread is mid-scrape when the process exits, its browser
    process may be left for the OS to reap.
    """
    with _worker_handles_lock:
        handles = list(_worker_handles)
    for handle in handles:
        _close_handle_with_timeout(handle)


atexit.register(_shutdown_all_workers)


# Evidence directory for zero-result debug captures. Lives at
# <project_root>/debug (sibling of app.py's DOWNLOADS_DIR), since this file
# is itself at <project_root>/scrapers/google_scraper.py.
_DEBUG_DIR = Path(__file__).resolve().parent.parent / "debug"


def _save_debug_artifacts(page, query: str, reason: str, extra: Dict[str, str] = None) -> None:
    """Save HTML/screenshot/context ONLY on failure/zero-results for
    debugging. Never on the success path.

    Writes three files under ``_DEBUG_DIR``, all sharing one base name
    stamped with date-time + milliseconds + thread id so concurrent or
    rapid-fire failures never overwrite each other:
      * <base>.png - screenshot
      * <base>.html - full page HTML
      * <base>.txt  - final URL, page title, reason, and whatever
        selector/redirect evidence the caller passes in ``extra``
        (e.g. which Reviews selector matched/failed, search-vs-place
        redirect classification).
    """
    try:
        _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        safe_query = re.sub(r'[^a-zA-Z0-9]', '_', query)[:50].strip('_') or "unknown"
        stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}_{threading.get_ident()}"
        base = _DEBUG_DIR / f"google_{safe_query}_{stamp}"

        final_url = _safe_url(page)
        try:
            title = page.title()
        except Exception:
            title = "<unavailable>"

        try:
            page.screenshot(path=f"{base}.png", timeout=3000, full_page=True)
        except Exception:
            logger.warning("[GOOGLE_DEBUG] could not capture screenshot for query=%r", query)

        try:
            with open(f"{base}.html", "w", encoding="utf-8") as f:
                f.write(page.content())
        except Exception:
            logger.warning("[GOOGLE_DEBUG] could not capture HTML for query=%r", query)

        info = {
            "reason": reason,
            "query": query,
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
            "[GOOGLE_DEBUG] saved zero-result evidence for query=%r reason=%s: %s.{png,html,txt}",
            query, reason, base,
        )
    except Exception:
        logger.exception("[GOOGLE_DEBUG] failed to save debug artifacts for query=%r", query)


def _normalize_business_key(name: str) -> str:
    """Strip common product suffixes for stable cache key when no website."""
    if not name:
        return ""
    # Targeted: remove common suffixes like " - Headphones", " Wireless", etc.
    name = re.sub(r"\s*[-–—]\s*(?:headphones?|earphones?|earbuds?|speaker|product|model|wireless|pro|plus|lite|mini|case|charger).*$", "", name, flags=re.IGNORECASE)
    return name.strip().lower()


def _safe_url(page) -> str:
    try:
        return page.url or ""
    except Exception:
        return ""


def _any_selector_present(page, selectors: List[str], timeout: int = 2000) -> bool:
    """True as soon as any selector in the list becomes visible."""
    for sel in selectors:
        try:
            page.wait_for_selector(sel, timeout=timeout)
            return True
        except Exception:
            continue
    return False


# Selectors for the business name heading on a place detail page, tried in
# order. Reused from _BUSINESS_PAGE_INDICATOR_SELECTORS's own name-bearing
# entries rather than duplicated separately.
_BUSINESS_NAME_SELECTORS = ["h1.DUwDvf", "div.TIHn2 h1", "h1"]


def _extract_business_page_name(page, timeout: int = 2500) -> str:
    """Read the business name actually rendered on the current place page.

    Used only to verify a Maps redirect/click landed on the right
    business - see _business_name_matches_target() below.
    """
    for sel in _BUSINESS_NAME_SELECTORS:
        try:
            loc = page.locator(sel).first
            if loc.count() > 0:
                text = loc.inner_text(timeout=timeout).strip()
                if text:
                    return text
        except Exception:
            continue
    return ""


def _business_name_matches_target(candidate_name: str, target_name: str) -> bool:
    """True if a landed/clicked business's displayed name plausibly is the
    target company, not just some other business Maps or a fuzzy text
    search happened to surface.

    ROOT CAUSE this guards against: a single-result Maps redirect
    (layout == "business_page" in _scrape_maps) was previously trusted
    unconditionally - the name-similarity scoring that already existed for
    the search-results-list click-through path never ran for a direct
    redirect. In production this let a query for "boAt Lifestyle" silently
    land on and scrape an unrelated local business called "LIFE STYLE
    TOWN" (Maps' fuzzy match on "Lifestyle"/"Life Style") - a handful of
    real reviews were returned, just for the wrong business entirely, and
    nothing in the pipeline could tell the difference. Low review *volume*
    from that business looked identical to low volume from a genuinely
    quiet Maps listing for the real company.

    Three independent signals, any one of which is enough to pass -
    genuine name variations (extra "Pvt Ltd" suffixes, punctuation,
    word-order differences) shouldn't be rejected just because they don't
    match perfectly:
      1. Direct substring containment either direction.
      2. Character-level similarity (SequenceMatcher) above a modest bar.
      3. Meaningful token overlap (handles reordering / extra boilerplate
         words like "Store", "Official", "India").
    """
    if not candidate_name or not target_name:
        return False
    cand = candidate_name.lower().strip()
    target = target_name.lower().strip()
    if not cand or not target:
        return False

    if target in cand or cand in target:
        return True

    similarity = SequenceMatcher(None, target, cand).ratio()
    if similarity >= 0.75:
        return True

    target_tokens = set(re.findall(r"[a-z0-9]+", target))
    cand_tokens = set(re.findall(r"[a-z0-9]+", cand))
    if target_tokens:
        overlap_count = len(target_tokens & cand_tokens)
        if len(target_tokens) == 1:
            # Single-word target: that one token must actually be present.
            # (Usually already caught by the substring check above - this
            # is mostly a backstop for token-order edge cases.)
            if overlap_count >= 1:
                return True
        else:
            # Multi-word target: require near-full containment, not just
            # >=50% of any tokens. A short brand name like "boAt Lifestyle"
            # has only 2 tokens, and "boat" alone is a common dictionary
            # word - a single shared token isn't enough evidence to trust
            # this is really the same business (e.g. an unrelated "Boat
            # Rental Company" would otherwise pass at a 50% bar). Require
            # at least half AND at least 2 overlapping tokens.
            overlap_ratio = overlap_count / len(target_tokens)
            if overlap_ratio >= 0.5 and overlap_count >= 2:
                return True

    return False


_SETTLE_POLL_SELECTOR = ", ".join(
    _SEARCH_RESULTS_INDICATOR_SELECTORS + _BUSINESS_PAGE_INDICATOR_SELECTORS
)


def _wait_for_layout_ready(page, max_ms: int) -> bool:
    """Poll in short steps for either a search-results or business-page
    indicator instead of always sleeping a fixed settle window.

    Mirrors ``_wait_for_more``'s pattern: cheap ``.count()`` checks (which,
    unlike ``wait_for_selector``, don't themselves block) on a combined
    selector, polled every 100ms until something matches or ``max_ms``
    elapses. Returns True the moment either layout's indicators show up in
    the DOM, so a fast-rendering page moves on immediately instead of
    always paying the full window; a slow one still gets up to ``max_ms``
    before the caller proceeds to the real (visibility-aware) layout
    classification regardless.
    """
    deadline = time.monotonic() + (max_ms / 1000)
    while time.monotonic() < deadline:
        try:
            if page.locator(_SETTLE_POLL_SELECTOR).count() > 0:
                return True
        except Exception:
            return False
        page.wait_for_timeout(100)
    return False


def _detect_maps_layout(page, time_left) -> str:
    """Classify the current Maps page before deciding how to reach the
    business, instead of always assuming a search-results list exists.

    Returns one of:
      * "business_page"  - already on the place detail page. Google skips
        the results list entirely for a sufficiently confident single
        match, which is exactly the case the old code mis-logged as
        "could not click place result" even though nothing was wrong.
      * "search_results"  - a results list/side panel is present and a
        result still needs to be clicked into.
      * "unknown"         - neither indicator set showed up within the
        (small) budget given. Treated like search_results by the caller,
        but failing to click anything here is not itself an error, since
        we may already be sitting on an oddly-marked detail page.
    """
    url = _safe_url(page)
    if "/maps/place/" in url:
        return "business_page"

    # Small, time-budget-aware wait - this check must stay cheap since it
    # runs before we know whether a click loop is even needed.
    budget_ms = min(2500, max(500, int(time_left() * 1000 * 0.15)))

    # Multiple distinct "/maps/place/" links is an unambiguous sign that
    # we're still looking at a multi-result search list (a real business
    # detail page only ever links to itself, if at all). This is checked
    # BEFORE the selector-based layout probes below because business indicators
    # (e.g. div[role='tablist'] for the filter-chip row, or a rating
    # jsaction hook on an individual result card) can also appear on the
    # search-results page itself and were causing a real results list to
    # be misclassified as "business_page" - which then skipped clicking
    # into the actual business entirely and left every downstream
    # Reviews-tab selector looking for a tab that was never on the page.
    try:
        place_link_count = page.locator("a[href*='/maps/place/']").count()
    except Exception:
        place_link_count = 0
    if place_link_count > 1:
        return "search_results"

    if _any_selector_present(page, _SEARCH_RESULTS_INDICATOR_SELECTORS, timeout=budget_ms):
        return "search_results"
    if _any_selector_present(page, _BUSINESS_PAGE_INDICATOR_SELECTORS, timeout=budget_ms):
        return "business_page"
    return "unknown"


# Combined selector string for a single wait_for_selector() call, mirroring
# the Step A fix in _scrape_maps (candidate scan) and _wait_for_layout_ready's
# _SETTLE_POLL_SELECTOR above - the same "one combined wait beats N sequential
# per-selector waits" pattern used twice already elsewhere in this file.
_REVIEW_PANEL_COMBINED_SELECTOR = ", ".join(_REVIEW_PANEL_SELECTORS)


def _wait_for_review_panel(page, time_left) -> bool:
    """One combined wait across every review-feed selector fallback.

    FIX: this used to loop over _REVIEW_PANEL_SELECTORS one at a time, each
    with its own fresh 6000ms page.wait_for_selector() call - the exact same
    anti-pattern already found and fixed for _PLACE_RESULT_SELECTORS in
    _scrape_maps' "Step A" (see that comment for the full writeup).
    wait_for_selector() only returns early on a MATCH, not on a miss, so a
    selector that was never going to appear on this page still burns its
    full 6000ms before the loop tries the next one. With 11 entries in
    _REVIEW_PANEL_SELECTORS, a business whose feed matched late in the list
    (or not at all) could burn most of an 11 x 6s stretch just hunting for
    the panel - and since _scrape_maps calls this up to three times in a row
    (see Step 3), that cost could compound. Whatever it ate came directly
    out of Step 4's scroll loop, which is the part that actually collects
    review text - so this was budget stolen from genuine review collection,
    not spent on it. One combined selector string lets Playwright match
    whichever pattern is present in a single wait, exactly like the Step A
    and _wait_for_layout_ready fixes already applied elsewhere in this file.
    """
    if time_left() <= 10:
        return False
    try:
        page.wait_for_selector(_REVIEW_PANEL_COMBINED_SELECTOR, timeout=6000)
        return True
    except Exception:
        return False


def _js_click_reviews_tab(page) -> bool:
    """Last-resort JS injection: walk the live DOM for any visible,
    interactive element whose text or aria-label is 'Reviews' and click it.

    This bypasses all CSS-selector / ARIA-role matching and reads the
    actual rendered DOM directly, making it resilient to any class-name
    churn or role-attribute changes in Google Maps. Returns True if an
    element was found and clicked, False otherwise.
    """
    try:
        clicked = page.evaluate(
            """
            () => {
                // Candidate selectors that enumerate clickable elements
                const candidates = [
                    ...document.querySelectorAll(
                        "div[role='button'], button, a, [jsaction], [role='tab']"
                    )
                ];
                for (const el of candidates) {
                    const label = (el.getAttribute('aria-label') || '').toLowerCase();
                    const text  = (el.innerText || el.textContent || '').trim().toLowerCase();
                    const jsa   = (el.getAttribute('jsaction') || '').toLowerCase();
                    // Never the "Write a review" control — it opens a
                    // sign-in modal that blocks the page.
                    if (label.includes('write') || text.includes('write')) {
                        continue;
                    }
                    if (
                        label === 'reviews' ||
                        text  === 'reviews' ||
                        jsa.includes('morereview') ||
                        jsa.includes('reviewchart') ||
                        label.startsWith('reviews ')
                    ) {
                        // Only click visible elements (avoid hidden honeypots)
                        const rect = el.getBoundingClientRect();
                        if (rect.width > 0 && rect.height > 0) {
                            el.click();
                            return true;
                        }
                    }
                }
                return false;
            }
            """
        )
        return bool(clicked)
    except Exception:
        return False


def _is_write_review_control(loc) -> bool:
    """True when a Reviews-tab candidate is actually the "Write a review"
    button. Several of the broader selectors above (`[data-value*='Review'
    i]`, `[aria-label*='Review' i]`, the Review XPaths) match it, and
    clicking it opens a "Sign in with your Google Account to write a
    review" modal that blocks the whole page — observed live: that single
    mis-click is what turned an otherwise-successful place-page scrape
    into FEED_MISSING/ZERO_REVIEWS."""
    try:
        label = " ".join(filter(None, [
            loc.get_attribute("aria-label") or "",
            loc.get_attribute("data-value") or "",
        ]))
        if "write" in label.lower():
            return True
        txt = loc.inner_text(timeout=300) or ""
        return "write" in txt.lower()
    except Exception:
        return False


# Review-content probes shared by the post-click verification below: any of
# these appearing means the reviews list actually opened.
_REVIEW_CONTENT_PROBE_SELECTORS = [
    "div[role='feed']",
    "div[data-review-id]",
    "span.wiI7pd",
    "div.jftiEf",
]


def _reviews_content_appeared(page, timeout_ms: int = 2500) -> bool:
    """Quick check that a click on a supposed Reviews control actually
    surfaced review content, instead of assuming any successful click did
    (a click can 'succeed' on the wrong control, e.g. a sign-in modal)."""
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        for sel in _REVIEW_CONTENT_PROBE_SELECTORS:
            try:
                if page.locator(sel).count() > 0:
                    return True
            except Exception:
                continue
        try:
            page.wait_for_timeout(250)
        except Exception:
            return False
    return False


def _dismiss_blocking_modal(page) -> None:
    """Best-effort close of a modal a wrong click may have opened (the
    sign-in-to-write-a-review dialog in particular), so the remaining
    selector candidates get a clean page to act on."""
    try:
        cancel = page.get_by_role("button", name="Cancel", exact=False)
        if cancel.count() > 0 and cancel.first.is_visible():
            cancel.first.click(timeout=1000)
            page.wait_for_timeout(300)
            return
    except Exception:
        pass
    try:
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)
    except Exception:
        pass


def _expand_hidden_review_text(page, time_left, limit: int = 200) -> int:
    """Reveal review text hidden by CSS truncation (max-height / line-clamp
    on the wrapping element) rather than by a clickable "More" control.

    Unlike ``_expand_more_buttons``/``_expand_translated_reviews``, this
    doesn't click anything - some review cards truncate text purely with
    CSS and never render a control to expand it, which would otherwise
    leave ``.inner_text()`` silently returning a clipped string with no
    signal that anything was cut off.
    """
    if time_left() <= 8:
        return 0
    try:
        return page.evaluate(
            """
            () => {
                const nodes = document.querySelectorAll(
                    "span.wiI7pd, div.jftiEf, div[data-review-id]"
                );
                let touched = 0;
                nodes.forEach(el => {
                    const style = el.getAttribute('style') || '';
                    if (/max-height|line-clamp|overflow/i.test(style)) {
                        el.style.maxHeight = 'none';
                        el.style.webkitLineClamp = 'unset';
                        el.style.overflow = 'visible';
                        touched += 1;
                    }
                });
                return touched;
            }
            """
        ) or 0
    except Exception:
        return 0


def _collect_review_texts(page, limit: int) -> List[str]:
    """Extract review text only - not ratings, buttons, owner replies, or
    other metadata/UI labels.

    Tries precise, text-node-only selectors first (most selective — these
    target just the review body span/div, not the whole card). If those
    find nothing, falls back to container-level selectors that may include
    reviewer name / timestamp / owner-reply noise (the post-processing in
    _strip_review_boilerplate cleans that up). A final pass drops any
    node whose entire text is just a UI label.
    """
    # Ordered most-to-least precise. wiI7pd and rsqaWe are the current
    # Maps review-text span classes (2023-2026). review-full-text and
    # review-snippet survive from the legacy Maps / embedded widget builds.
    precise = [
        "span.wiI7pd",
        "span.rsqaWe",          # seen in 2025 Maps builds
        "span.review-full-text",
        "div.review-snippet",
    ]
    # Container-level fallbacks — wider net, higher noise.
    coarse = [
        "div.jftiEf",
        "div[data-review-id]",
        "div.gws-localreviews__google-review",
        # Class-fragment match: Maps sometimes rotates the exact class but
        # keeps a stable infix; this catches variants like MyEned, MyEnedb, etc.
        "div[class*='MyEned']",
    ]

    texts = _collect_texts(page, precise, limit_per_selector=limit)
    if not texts:
        logger.info(
            "Review extraction: precise text selectors found nothing; "
            "falling back to container-level selectors (may include some "
            "reviewer/timestamp metadata that downstream stripping removes)."
        )
        texts = _collect_texts(page, coarse, limit_per_selector=limit)

    return [t for t in texts if t.strip().lower() not in _UI_LABEL_BLACKLIST]


def _disambiguate_maps_query(query: str, company_data: Dict[str, str] = None) -> str:
    """Append a domain hint to a bare company-name query when we have one.

    The log shows both production runs landing on layout="search_results"
    for a plain company-name query ("Headphone Zone", "boAt Lifestyle") -
    a name generic/common enough that Maps returns a results list rather
    than single-result-redirecting to the business page directly. The
    click-through loop that recovers from this already exists and already
    scores candidates by name similarity AND by whether the result's link
    contains the business's own website domain (see best_match_score /
    target_website below) - but every selector it has to try costs real
    time, and in this run there wasn't enough left for even one more.

    Feeding Maps' own search a more specific query up front - the same
    "add a distinguishing word" trick a person doing this search by hand
    would use - measurably increases how often Maps skips the results
    list itself and single-result-redirects straight to the business
    page, which is strictly cheaper than reaching the same page via the
    click-through loop. This changes only the text sent to Maps; the
    existing click-through/scoring logic below is untouched and still
    runs exactly as before if Maps still returns a list.
    """
    website = (company_data or {}).get("website", "") or ""
    if not website:
        return query
    domain = website.lower()
    for prefix in ("https://", "http://", "www."):
        if domain.startswith(prefix):
            domain = domain[len(prefix):]
    domain = domain.split("/")[0].strip()
    if not domain:
        return query

    # FIX: previously returned `query` unchanged whenever the domain's word
    # tokens were already a subset of the query's tokens (e.g. "boAt
    # Lifestyle" already contains "boat"+"lifestyle", the same tokens in
    # "boat-lifestyle.com") on the theory that appending the domain would
    # be redundant. In production this was the exact case that still
    # single-result-redirected to an unrelated business ("LIFE STYLE
    # TOWN") - Maps' fuzzy text matcher works on the literal search
    # string, not a token set, so a shared *word* doesn't mean a shared
    # *string*, and the one extra signal that might have disambiguated it
    # (the literal domain text) was being withheld precisely when the
    # token-overlap heuristic said "not needed". Appending the domain
    # even when it "looks redundant" costs nothing when Maps already had
    # enough to go on, and is exactly the extra specificity needed when it
    # didn't - so always append it when we have one.
    # Always append the domain (see the fix note above for why the old
    # "skip if tokens already overlap" conditional was removed) - the
    # comparison logic that used to live here is gone; this is now a
    # straight, unconditional append.
    return f"{query} {domain}"


def _scrape_maps(page, query: str, time_left, company_data: Dict[str, str] = None) -> Tuple[List[str], bool]:
    texts: List[str] = []
    maps_query = _disambiguate_maps_query(query, company_data)
    # ``hl=en&gl=in`` forces Google to render the Maps UI in English
    # regardless of whatever locale/geolocation it would otherwise infer for
    # the scraping host's IP. The browser context already sets
    # locale="en-US" (navigator.language / Accept-Language), but Maps' own
    # UI text is driven by this URL param, not the browser locale - left
    # unset, a non-US egress IP can get a localized UI whose tab is not
    # literally named "Reviews", which is exactly why the role/name lookup
    # below can find nothing even though the tab is present and clickable.
    maps_url = (
        f"https://www.google.com/maps/search/{maps_query.replace(' ', '+')}"
        "?hl=en&gl=in"
    )

    # This is the one navigation that sits in front of the candidate scan,
    # so it reserves time for the settle/layout-detection step AND the scan
    # itself (_INITIAL_NAV_RESERVE_SECONDS) - without this reservation,
    # adaptive-timeout retries can legally consume down to a few seconds
    # remaining, which is below what the scan needs to even attempt, and
    # BUSINESS_NOT_FOUND follows immediately even though the layout was
    # correctly detected as search_results.
    # DIAGNOSTIC: this specific call is the only place DIAG_NAV_FAILURE /
    # "nav_failure" debug artifacts are produced (see _log_diag call just
    # below), but until now it sat after the last BROWSER_TRACE checkpoint
    # (after_new_page) with no trace of its own — so a "nav_failure" debug
    # artifact could not distinguish "this navigation had barely any time
    # budget left by the time it started" from "it had a full budget and
    # still failed/hung." Added after finding both patterns in the same
    # debug/ folder across companies (some final_url: about:blank, others a
    # real Maps search URL — i.e. navigation partly succeeded before the
    # wait condition timed out) with no way to tell which from existing
    # logs alone. Pure logging; no behavior change.
    logger.info(
        "BROWSER_TRACE scraper=Google event=before_maps_search_nav elapsed=%.3fs "
        "timestamp=%.3f thread=%s time_left=%.1fs",
        time.monotonic() - getattr(_browser_trace, "ensure_context_started_at", time.monotonic()),
        time.time(), threading.current_thread().name, time_left(),
    )
    _maps_nav_ok = _goto_with_retry(
        page, maps_url, timeout=_adaptive_nav_timeout(time_left), time_left=time_left,
        reserve_seconds=_INITIAL_NAV_RESERVE_SECONDS,
    )
    logger.info(
        "BROWSER_TRACE scraper=Google event=after_maps_search_nav elapsed=%.3fs "
        "timestamp=%.3f thread=%s ok=%s time_left=%.1fs",
        time.monotonic() - getattr(_browser_trace, "ensure_context_started_at", time.monotonic()),
        time.time(), threading.current_thread().name, _maps_nav_ok, time_left(),
    )
    if not _maps_nav_ok:
        _log_diag(DIAG_NAV_FAILURE, query, "initial Maps navigation failed after retries")
        _save_debug_artifacts(page, query, "nav_failure")
        return texts, False

    # Budget-aware settle wait: poll for either a search-results or a
    # business-page indicator instead of always sleeping a fixed 1800ms.
    # Exits as soon as one appears; only pays the full window when neither
    # has rendered yet, and never waits longer than what was reserved for
    # this phase (_SETTLE_RESERVE_SECONDS) or than what's actually left.
    settle_budget_ms = max(0, min(int(_SETTLE_RESERVE_SECONDS * 1000), int(time_left() * 1000)))
    _wait_for_layout_ready(page, settle_budget_ms)
    _dismiss_consent(page)
    page.wait_for_timeout(300)

    if _bot_checked(page):
        _log_diag(DIAG_BOT_PAGE, query, "bot-check/captcha page encountered on Maps")
        _save_debug_artifacts(page, query, "bot_check")
        return texts, True

    # --- Step 1: classify the page layout before assuming a results list
    # exists. Maps can land on the exact business detail page directly
    # for a confident single match, on a results list/side panel, or -
    # rarely - on a layout neither indicator set recognizes. ---
    layout = _detect_maps_layout(page, time_left)
    logger.info(
        "Maps layout detected as %r for query=%r (url=%s).",
        layout, query, _safe_url(page),
    )

    clicked_into_place = False
    best_match_score = 0
    target_name = (company_data or {}).get("company_name", query) or query

    if layout == "business_page":
        # Nothing to click - we're already on the place detail page. But
        # don't just trust that: verify the name actually rendered on this
        # page plausibly is the target company before treating it as a
        # real match. See _business_name_matches_target()'s docstring for
        # the production incident (a query for "boAt Lifestyle" silently
        # redirecting to an unrelated business called "LIFE STYLE TOWN")
        # this specifically guards against.
        landed_name = _extract_business_page_name(page)
        if landed_name and not _business_name_matches_target(landed_name, target_name):
            logger.warning(
                "Maps single-result-redirected to %r for query=%r, but that "
                "name doesn't plausibly match the target company - "
                "treating as no match instead of scraping the wrong "
                "business's reviews.",
                landed_name, query,
            )
            _log_diag(
                DIAG_BUSINESS_MISMATCH, query,
                f"single-result redirect landed on {landed_name!r}, which "
                f"does not match target company name {target_name!r}",
            )
            clicked_into_place = False
        else:
            clicked_into_place = True
            if landed_name:
                logger.info(
                    "Already on a business detail page for query=%r "
                    "(single-result redirect); name %r matches target - "
                    "skipping result-list click.",
                    query, landed_name,
                )
            else:
                # Couldn't read a name to verify against (selector churn,
                # slow render) - proceed as before rather than blocking on
                # a check that itself failed to produce a signal either way.
                logger.info(
                    "Already on a business detail page for query=%r "
                    "(single-result redirect); could not read a name to "
                    "verify against, proceeding anyway.",
                    query,
                )
    else:
        target_website = (company_data or {}).get("website", "") or ""
        # Compare by bare domain, not the full URL: target_website is
        # usually "https://www.example.com", which can never appear
        # verbatim inside a Maps place href - so the +100 website score
        # below never fired at all.
        target_domain = target_website.lower()
        for _prefix in ("https://", "http://", "www."):
            if target_domain.startswith(_prefix):
                target_domain = target_domain[len(_prefix):]
        target_domain = target_domain.split("/")[0].strip()

        def _click_place_result(click_target, attempt_query: str) -> bool:
            """Navigate to a place-result element and confirm real
            navigation happened (URL flips from /maps/search/ to
            /maps/place/, or the detail-page title/tablist indicators
            appear) instead of assuming a click always lands on it.

            Google's search-result card anchor (`a.hfpxzc`) already carries
            a fully-resolved "/maps/place/..." href at scrape time, but a
            plain `.click()` hands control to Maps' own jsaction router
            (the anchor's click is *not* a native link follow - Maps calls
            preventDefault() and does an internal SPA transition instead),
            which can race the results list's virtualization or get
            swallowed as a hover/preview instead of a navigation. Since we
            already have the destination URL as a plain string, going
            straight there via goto() is more reliable than depending on
            that client-side transition to complete - so it's tried first,
            with the click-based approach kept as a fallback for the rare
            candidate where no real href was resolvable.
            """
            def _navigated() -> bool:
                url = _safe_url(page)
                # If the URL still contains /maps/search/ we have definitively
                # NOT left the search-results page — return False immediately
                # regardless of what DOM selectors match.  This is the
                # defence-in-depth companion to removing "div[role='main'] h1"
                # from _BUSINESS_PAGE_INDICATOR_SELECTORS: the search-results
                # page hosts role="main" containing <h1>Results</h1>, so any
                # descendant-h1 selector on role=main fires as a false positive
                # while still on the search list.  Anchoring confirmation to
                # URL state first eliminates that class of false positives.
                if "/maps/search/" in url:
                    return False
                if "/maps/place/" in url:
                    return True
                return _any_selector_present(page, _BUSINESS_PAGE_INDICATOR_SELECTORS, timeout=1200)

            try:
                href = click_target.get_attribute("href") or ""
            except Exception as _href_exc:
                logger.warning(
                    "[CPR] get_attribute('href') raised for query=%r: %s",
                    attempt_query, _href_exc,
                )
                href = ""
            logger.info(
                "[CPR] href resolved: query=%r href=%r has_place_path=%s",
                attempt_query, href[:200], "/maps/place/" in href,
            )
            if "/maps/place/" in href:
                logger.info(
                    "[CPR] calling _goto_with_retry: query=%r url=%r remaining=%.1fs",
                    attempt_query, href[:200], time_left(),
                )
                _goto_ok = _goto_with_retry(page, href, timeout=_adaptive_nav_timeout(time_left), time_left=time_left, retries=1)
                logger.info(
                    "[CPR] _goto_with_retry returned %s: query=%r post_goto_url=%r",
                    _goto_ok, attempt_query, _safe_url(page)[:200],
                )
                if _goto_ok:
                    _nav_result = _navigated()
                    logger.info(
                        "[CPR] _navigated() after goto=%s: query=%r url=%r",
                        _nav_result, attempt_query, _safe_url(page)[:200],
                    )
                    if _nav_result:
                        return True
                    logger.warning(
                        "[CPR] goto succeeded but _navigated()=False: query=%r url=%r "
                        "— falling through to click-based attempts",
                        attempt_query, _safe_url(page)[:200],
                    )
                # goto() itself may have "succeeded" (page loaded) without
                # tripping _navigated()'s checks, or may have failed outright;
                # either way fall through to the click-based attempts below
                # rather than giving up on this candidate.

            for attempt in (1, 2):
                logger.info(
                    "[CPR] click attempt %d/2: query=%r pre_click_url=%r remaining=%.1fs",
                    attempt, attempt_query, _safe_url(page)[:200], time_left(),
                )
                try:
                    click_target.click(timeout=3000)
                    logger.info(
                        "[CPR] click() returned without exception: query=%r attempt=%d",
                        attempt_query, attempt,
                    )
                except Exception as _click_exc:
                    logger.warning(
                        "[CPR] click() raised on attempt %d/2: query=%r exc=%s",
                        attempt, attempt_query, _click_exc,
                    )
                    continue

                deadline = time.monotonic() + min(4.0, max(1.5, time_left() * 0.3))
                _poll_n = 0
                while time.monotonic() < deadline:
                    _poll_url = _safe_url(page)
                    _poll_nav = _navigated()
                    logger.info(
                        "[CPR] post-click poll #%d attempt=%d: query=%r navigated=%s url=%r",
                        _poll_n, attempt, attempt_query, _poll_nav, _poll_url[:200],
                    )
                    if _poll_nav:
                        return True
                    _poll_n += 1
                    page.wait_for_timeout(300)

                if attempt == 1:
                    logger.info(
                        "Click on place result did not navigate for "
                        "query=%r; retrying on the same element.",
                        attempt_query,
                    )
            logger.warning(
                "[CPR] _click_place_result returning False: query=%r "
                "all goto+click attempts exhausted final_url=%r",
                attempt_query, _safe_url(page)[:200],
            )
            return False

        def _extract_candidate_name_and_link(node) -> Tuple[str, str]:
            """Resolve a candidate's display name and navigable href.

            Shared by the single-pass scan below and by nothing else now
            that scoring/clicking are decoupled from selector iteration -
            kept as its own function purely so the extraction rules (anchor
            resolution, aria-label-first name lookup, heading fallbacks)
            stay in one place instead of being duplicated per candidate.
            """
            try:
                own_href = node.get_attribute("href") or ""
            except Exception:
                own_href = ""

            # BUGFIX: this used to treat ANY href present on `node` as good
            # enough to use directly ("if node.get_attribute('href')..."),
            # which meant the broad selectors in _PLACE_RESULT_SELECTORS
            # (needed because a card's real place-link doesn't always match
            # the narrower a.hfpxzc / a[href*='/maps/place/'] patterns) also
            # picked up non-place action links inside the same result card -
            # most importantly the "Visit <business>'s website" button,
            # whose href is the business's own external site, not a Maps
            # place page. That external href still contains the target
            # site's domain, so it used to win the +100 domain-match bonus
            # in _score_candidate() below outright, get selected as the
            # "best" candidate, and - since it can never navigate to a
            # "/maps/place/..." URL - burn the entire click-retry budget
            # before failing, while the real place result never got
            # clicked at all (this is exactly the BUSINESS_NOT_FOUND /
            # 0-review failure seen in production for a query where a
            # *different* nearby business's "visit website" link happened
            # to point at our target's domain).
            #
            # Fix: only ever trust `node` itself as the anchor when its own
            # href is an actual place-detail link. For every other node
            # (a container, or an action-button anchor), look for a nested
            # place-detail anchor instead, and treat "no such nested anchor"
            # as "this candidate has no navigable place link" - not as
            # license to fall back to whatever unrelated href happened to
            # be sitting on the node (see the removed fallback below too).
            if "/maps/place/" in own_href:
                anchor = node
            else:
                try:
                    anchor = node.locator("a[href*='/maps/place/']").first
                    if anchor.count() == 0:
                        anchor = None
                except Exception:
                    anchor = None

            name_text = ""
            if anchor is not None:
                try:
                    name_text = (anchor.get_attribute("aria-label") or "").strip()
                except Exception:
                    name_text = ""
            if not name_text:
                for name_sel in (
                    "h1, h2, .fontHeadlineSmall, [role='heading']",
                    "div.qBF1Pd, span.qBF1Pd",
                    "div[class*='fontHeadlineSmall'], span[class*='fontHeadlineSmall']",
                ):
                    try:
                        name_el = node.locator(name_sel).first
                        if name_el.count() > 0:
                            name_text = name_el.inner_text(timeout=1000).strip()
                            if name_text:
                                break
                    except Exception:
                        continue

            # No fallback to `own_href` here on purpose: if `anchor` is
            # None, this candidate has no resolvable place-detail link, and
            # scoring/clicking a non-place href (website/call/directions/
            # share/etc.) as if it were navigable is exactly the bug fixed
            # above. An empty link here just means _score_candidate() below
            # can't award the domain-match bonus for this candidate, which
            # is correct - it also can't be clicked into a place page.
            link = ""
            try:
                link = (anchor.get_attribute("href") if anchor is not None else "") or ""
            except Exception:
                link = ""

            return name_text, link

        def _score_candidate(name_text: str, link: str) -> int:
            """Same scoring rubric as before: name-similarity + website match.

            The domain-match bonus is only ever awarded to a genuine
            "/maps/place/..." link. `link` coming out of
            `_extract_candidate_name_and_link` is already guaranteed to be
            either such a place-detail href or empty (never a "visit
            website"/call/directions action link), but the explicit
            "/maps/place/" check is kept here too as defense-in-depth: a
            future caller of this scoring function should not be able to
            silently reintroduce the "domain-matched non-place link wins
            outright" bug just by passing in a differently-sourced link.
            """
            score = 0
            if name_text:
                name_lower = name_text.lower()
                if target_name.lower() in name_lower:
                    score += 80
                else:
                    similarity = SequenceMatcher(None, target_name.lower(), name_lower).ratio()
                    if similarity >= 0.6:
                        score += int(similarity * 80)
            link_lower = link.lower()
            if target_domain and "/maps/place/" in link_lower and target_domain in link_lower:
                score += 100
            return score

        # --- OPTIMIZATION: candidate collection, scoring, and clicking are
        # now three decoupled steps instead of one tightly-coupled loop.
        #
        # Previously this called page.wait_for_selector(selector,
        # timeout=6000) once per entry in _PLACE_RESULT_SELECTORS,
        # sequentially, and scored+clicked candidates inline as each
        # selector was visited (worst case ~6s x 7 selectors = 42s just to
        # discover candidates, before any scoring/clicking time is spent).
        #
        # Step A - single DOM scan: one combined wait_for_selector() across
        # every known place-result pattern, then one pass collecting every
        # currently-visible candidate each pattern resolves. No further
        # per-selector waiting - by this point in _scrape_maps the page has
        # already had its settle waits, consent dismissal, and layout
        # detection, so the results (if any) are already in the DOM.
        # Step B - score every candidate collected in step A.
        # Step C - click exactly once, on the single highest-scoring
        # candidate.
        logger.debug(
            "[PLACE_RESULT] starting single-pass candidate scan for query=%r: "
            "time_left=%.1fs, min_required=%.1fs, %d selector pattern(s).",
            query, time_left(), _PLACE_RESULT_LOOP_MIN_TIME_SECONDS,
            len(_PLACE_RESULT_SELECTORS),
        )

        scored_candidates = []  # list of (score, name_text, link, element)

        if time_left() <= _PLACE_RESULT_LOOP_MIN_TIME_SECONDS:
            logger.warning(
                "[PLACE_RESULT] skipping candidate scan for query=%r: only "
                "%.1fs left (need >%.1fs).",
                query, time_left(), _PLACE_RESULT_LOOP_MIN_TIME_SECONDS,
            )
        else:
            # Step A: one combined wait replaces the old per-selector waits.
            combined_selector = ", ".join(_PLACE_RESULT_SELECTORS)
            try:
                page.wait_for_selector(combined_selector, timeout=6000)
                logger.debug(
                    "[PLACE_RESULT] combined selector matched for query=%r "
                    "(time_left=%.1fs).", query, time_left(),
                )
            except Exception as _scan_exc:
                logger.info(
                    "PLACE_RESULT_TRACE mode=scan matched_elements=0 "
                    "selector_error=%s", _scan_exc,
                )
                logger.debug(
                    "[PLACE_RESULT] no place-result element matched any "
                    "known selector for query=%r within 6000ms "
                    "(time_left=%.1fs).", query, time_left(),
                )

            # One pass over the now-settled DOM: gather every visible
            # candidate from every selector pattern (locator().all() here
            # does not itself wait). De-duplicate on (name, href) since
            # several patterns above resolve the same underlying cards.
            seen = set()
            results = []
            for selector in _PLACE_RESULT_SELECTORS:
                try:
                    matched = page.locator(selector).all()
                except Exception:
                    continue
                results.extend(matched)
                logger.info(
                    "PLACE_RESULT_TRACE mode=scan selector=%r matched_elements=%d",
                    selector, len(matched),
                )

            for _candidate_idx, node in enumerate(results[:20], start=1):
                if time_left() <= _PLACE_RESULT_LOOP_MIN_TIME_SECONDS:
                    logger.info(
                        "PLACE_RESULT_TRACE mode=scan candidate_index=%d "
                        "candidate_skipped=time_budget", _candidate_idx,
                    )
                    break
                try:
                    if not node.is_visible():
                        continue
                    name_text, link = _extract_candidate_name_and_link(node)
                    key = (name_text, link)
                    if not (name_text or link) or key in seen:
                        continue
                    seen.add(key)
                    score = _score_candidate(name_text, link)
                    scored_candidates.append((score, name_text, link, node))
                    logger.info(
                        "PLACE_RESULT_TRACE mode=scan candidate_index=%d "
                        "name=%r href=%r score=%d",
                        _candidate_idx, name_text, link, score,
                    )
                except Exception as _candidate_exc:
                    logger.info(
                        "PLACE_RESULT_TRACE mode=scan candidate_index=%d "
                        "candidate_error=%s", _candidate_idx, _candidate_exc,
                    )
                    continue

        # DIAGNOSTIC ONLY - no effect on which candidate gets clicked.
        #
        # GAP FOUND (2026-07-31, code-review only): _score_candidate only
        # measures name-similarity-to-target and a website-domain-match
        # bonus - it has no way to tell "this is the one genuine business
        # matching this name" apart from "this is one of many separate
        # physical locations of the same chain/franchise, each with its
        # own Maps listing, none of which represents the company as a
        # whole." A multi-location business (e.g. a restaurant chain) can
        # have every one of its outlets score identically high here, and
        # max() then picks whichever one happens to appear first in DOM
        # order - which Maps' own ranking (often geolocation/IP-dependent)
        # controls, not this code. Confirmed this is a real, not
        # hypothetical, case for at least one company this app has been
        # asked to analyze: India's KFC outlets are run by large multi-
        # city franchise operators (Devyani International / Sapphire
        # Foods) with hundreds of separate physical locations, not a
        # single "KFC" business entity Maps could point at - there is no
        # single Maps listing whose reviews would genuinely represent
        # "KFC" the brand.
        #
        # This doesn't change which candidate gets clicked (that could
        # regress a working case on a heuristic that hasn't been
        # validated against a real run yet) - it only makes the ambiguity
        # visible in the log, so a real run against a chain-style company
        # can be recognized rather than silently treated as a normal
        # single-location result. If a fresh log confirms this actually
        # fires for chain businesses, the next step is deciding what
        # policy to apply (skip Maps for that job, label the result as
        # "not necessarily representative", etc.) - a product decision,
        # not one this code should make unilaterally.
        _strong_matches = [c for c in scored_candidates if c[0] >= 70]
        if len(_strong_matches) >= 3:
            _distinct_names = list({c[1] for c in _strong_matches if c[1]})
            logger.warning(
                "DIAG_MULTI_LOCATION_SUSPECTED query=%r: %d of %d scanned "
                "candidates scored >=70 (strong name match) at once - a "
                "single genuine business rarely has this many separately-"
                "listed strong matches; this usually means a multi-"
                "location chain/franchise, not one business. Names seen: "
                "%r. The candidate that gets clicked below is still just "
                "the single highest scorer among these - treat its "
                "reviews as one location's data, not necessarily "
                "representative of the whole brand.",
                query, len(_strong_matches), len(scored_candidates),
                _distinct_names[:10],
            )

        # Step B/C: pick the single highest-scoring candidate (first one
        # wins any tie) and attempt exactly one click/navigation on it.
        if scored_candidates:
            best_match_score, best_name, best_link, best_node = max(
                scored_candidates, key=lambda c: c[0]
            )
            best_anchor = best_node
            try:
                if not (best_node.get_attribute("href") or ""):
                    _nested = best_node.locator("a[href*='/maps/place/']").first
                    if _nested.count() > 0:
                        best_anchor = _nested
            except Exception:
                best_anchor = best_node

            logger.info(
                "[PLACE_RESULT] %d candidate(s) scanned for query=%r; "
                "selected best candidate name=%r href=%r score=%d for the "
                "single click attempt.",
                len(scored_candidates), query, best_name, best_link, best_match_score,
            )

            click_succeeded = _click_place_result(best_anchor, query)
            click_result = "success" if click_succeeded else "failed"
            logger.info(
                "PLACE_RESULT_TRACE mode=click name=%r href=%r score=%d "
                "click_result=%s",
                best_name, best_link, best_match_score, click_result,
            )
            if click_succeeded:
                clicked_into_place = True
                logger.info(
                    "Google Maps matched business: %r (score=%d) for query=%r",
                    best_name, best_match_score, query,
                )
            else:
                logger.info(
                    "Click did not navigate to a place page for the best "
                    "candidate %r (score=%d, query=%r); no further "
                    "candidates are attempted.",
                    best_name, best_match_score, query,
                )
                best_match_score = 0
        else:
            logger.info(
                "[PLACE_RESULT] no candidates found in the single DOM scan "
                "for query=%r; nothing to click.", query,
            )

        if not clicked_into_place:
            # The layout classifier's first pass ran with a small time
            # budget and may have missed a slow-rendering indicator. One
            # more, slightly more generous check before we conclude the
            # business genuinely wasn't found.
            if time_left() > 10 and _any_selector_present(
                page, _BUSINESS_PAGE_INDICATOR_SELECTORS, timeout=2500
            ):
                clicked_into_place = True
                logger.info(
                    "No result to click for query=%r, but a business-page "
                    "indicator appeared on re-check; continuing as if "
                    "already on the detail page.",
                    query,
                )

        if not clicked_into_place:
            _log_diag(
                DIAG_BUSINESS_NOT_FOUND, query,
                "no place-result selector matched in time and no "
                "business-page indicator was found on re-check",
            )
        elif best_match_score < 60:
            logger.warning(
                "Low confidence business match (score=%d) for %r; "
                "continuing anyway since this is still the best candidate found.",
                best_match_score, query,
            )

    # --- Step 1.5: confirm we're actually on the business detail page
    # (title + tab bar both present) before Step 2 goes looking for the
    # Reviews tab. Both paths above can set clicked_into_place = True
    # without that being true - a "business_page" layout classification
    # can be wrong, and a click can register without the resulting page
    # having actually finished rendering. Proceeding into Step 2 in
    # either case just burns the rest of the time budget on Reviews-tab
    # selectors that were never going to match. ---
    if clicked_into_place and not _any_selector_present(
        page, _BUSINESS_PAGE_INDICATOR_SELECTORS, timeout=2500
    ):
        logger.warning(
            "Expected to be on the business detail page for query=%r but "
            "no business-page indicator (title/tablist) is present; "
            "treating as not found instead of searching for a Reviews "
            "tab that isn't there.",
            query,
        )
        clicked_into_place = False
        _log_diag(
            DIAG_BUSINESS_NOT_FOUND, query,
            "clicked_into_place was set but no business-page indicator "
            "(title/tablist) was present on final check",
        )

    # --- Fail fast: if the business isn't on Maps at all, Steps 2-4 below
    # (Reviews tab hunt, panel wait, scroll loop) cannot possibly succeed -
    # there's no business page for any of them to act on. Previously we ran
    # all of it anyway, which reliably burned 10-15s finding nothing before
    # falling through to DIAG_ZERO_REVIEWS. That time is much better spent
    # in the caller's search/Bing fallbacks, which - unlike Maps reviews -
    # don't require the business to have a Maps listing at all. ---
    if not clicked_into_place:
        _log_diag(
            DIAG_BUSINESS_NOT_FOUND, query,
            "skipping Reviews-tab/panel/scroll steps - no business page to "
            "act on; returning early so search/Bing fallbacks get the "
            "remaining time budget instead",
        )
        return texts, False

    # --- Step 2: switch to the Reviews tab. ---
    #
    # ARCHITECTURE NOTE — why the old approach failed:
    # Google Maps 2024-2026 renders the navigation pills as
    # `div[role="button"]` elements, NOT as `<button>` elements and NOT
    # with role="tab". The previous code tried `get_by_role("tab")` first
    # (5s timeout × 2 names = up to 10s wasted finding nothing) and then
    # `_REVIEWS_TAB_SELECTORS` which were dominated by `button[...]` and
    # `div[role="tab"]` selectors — all wrong element types. Each selector
    # had a 4s wait, so the whole loop burned 30-40s before falling through
    # with nothing found, by which point the time budget was exhausted.
    #
    # Fix: skip `get_by_role("tab")` entirely (Maps doesn't use it).
    # Try each selector with a SHORT 1200ms wait so the wide list finishes
    # fast. On failure, use JS-injection to walk the live DOM directly.
    # After any successful click, immediately verify the feed appeared
    # rather than assuming the click registered.
    #
    reviews_tab_found = False
    reviews_tab_selector_used = None

    # --- Tier 0: the !9m1!1b1 reviews-pane deep-link -----------------------
    # Google serves anonymous/automated sessions a "limited view of Google
    # Maps" (its own banner text) whose place page renders NO Reviews tab
    # at all — the tablist holds only Overview/About, so every selector
    # strategy below is hunting for a control that simply isn't there.
    # But the same limited view still honors the data-URL parameter that
    # deep-links straight into the reviews pane (verified live: 16
    # div[data-review-id] nodes rendered where the tab hunt found
    # nothing). Build that URL from the place's CID token and navigate
    # directly.
    place_url = _safe_url(page)
    if "/maps/place/" not in place_url and time_left() > 8:
        # The Maps SPA can lag flipping the address bar to the
        # /maps/place/ URL even after the detail page itself has rendered
        # (observed live: layout classified as business_page via DOM
        # indicators while page.url still read /maps/search/). Give the
        # URL a moment to settle, then fall back to reading the place
        # link out of the DOM.
        _url_deadline = time.monotonic() + 3.0
        while time.monotonic() < _url_deadline:
            place_url = _safe_url(page)
            if "/maps/place/" in place_url:
                break
            try:
                page.wait_for_timeout(300)
            except Exception:
                break
        if "/maps/place/" not in place_url:
            try:
                _href = page.locator("a[href*='/maps/place/']").first.get_attribute("href") or ""
            except Exception:
                _href = ""
            if "/maps/place/" in _href:
                place_url = _href
    cid_match = re.search(r"!1s(0x[0-9a-fA-F]+:0x[0-9a-fA-F]+)", place_url)
    if cid_match and "/maps/place/" in place_url and time_left() > 8:
        name_part = place_url.split("/maps/place/")[1].split("/")[0]
        deep_url = (
            f"https://www.google.com/maps/place/{name_part}/"
            f"data=!4m8!3m7!1s{cid_match.group(1)}!8m2!3d0!4d0!9m1!1b1!16s"
            "?hl=en&gl=in"
        )
        if _goto_with_retry(page, deep_url, timeout=_adaptive_nav_timeout(time_left), time_left=time_left, retries=1):
            page.wait_for_timeout(1500)
            if _reviews_content_appeared(page, timeout_ms=4000):
                reviews_tab_found = True
                reviews_tab_selector_used = "reviews_deeplink"
                logger.info(
                    "Reviews pane opened via !9m1!1b1 deep-link for query=%r.",
                    query,
                )
            else:
                logger.info(
                    "Reviews deep-link rendered no review content for "
                    "query=%r; falling back to the tab-selector hunt.", query,
                )

    # Short individual wait — the element is either already in the DOM (fast
    # path: ≤100ms) or it isn't going to appear at this selector at all.
    # 1200ms is plenty for a painted element; 4000ms per selector with 30+
    # selectors was the primary budget-killer.
    _TAB_SELECTOR_WAIT_MS = 1200

    for selector in ([] if reviews_tab_found else _REVIEWS_TAB_SELECTORS):
        if time_left() <= 10:
            break
        try:
            page.wait_for_selector(selector, timeout=_TAB_SELECTOR_WAIT_MS)
        except Exception:
            continue
        try:
            loc = page.locator(selector).first
            if loc.count() == 0:
                continue
            # Verify it's actually visible before clicking — some selectors
            # match hidden/offscreen duplicates that don't respond.
            if not loc.is_visible():
                continue
            # Never click the "Write a review" button — several broad
            # selectors match it, and it opens a sign-in modal that
            # blocks the whole page (see _is_write_review_control).
            if _is_write_review_control(loc):
                continue
            loc.click(timeout=2000)
            page.wait_for_timeout(800)
            # Confirm the click actually surfaced review content; a
            # click can land on the wrong control and open a modal
            # instead. If so, dismiss it and keep hunting.
            if not _reviews_content_appeared(page):
                logger.info(
                    "Click via selector %r did not surface review content "
                    "for query=%r; dismissing any modal and trying the "
                    "next selector.", selector, query,
                )
                _dismiss_blocking_modal(page)
                continue
            reviews_tab_found = True
            reviews_tab_selector_used = selector
            logger.info(
                "Reviews tab clicked via selector %r for query=%r.",
                selector, query,
            )
            break
        except Exception:
            continue

    # --- Tier 2 fallback: get_by_role("button") with name matching -------
    # Playwright's `get_by_role` uses the computed accessibility tree, so
    # it matches `div[role='button']` as well as `<button>` — unlike a
    # plain CSS `button[...]` selector which only matches the element type.
    if not reviews_tab_found and time_left() > 8:
        for name_variant in ["Reviews", "Review"]:
            try:
                btn = page.get_by_role("button", name=name_variant, exact=True)
                if btn.count() > 0 and btn.first.is_visible():
                    if _is_write_review_control(btn.first):
                        continue
                    btn.first.click(timeout=2000)
                    page.wait_for_timeout(800)
                    if not _reviews_content_appeared(page):
                        _dismiss_blocking_modal(page)
                        continue
                    reviews_tab_found = True
                    reviews_tab_selector_used = f"get_by_role(button, name={name_variant!r})"
                    logger.info(
                        "Reviews tab clicked via get_by_role(button, name=%r) "
                        "for query=%r.", name_variant, query,
                    )
                    break
            except Exception:
                continue

    # --- Tier 3 fallback: JS-injection DOM walk ---------------------------
    # Bypasses all CSS/ARIA matching entirely. Reads the live rendered DOM
    # and clicks the first visible element whose text or aria-label is
    # "reviews" (case-insensitive). This is the most robust possible
    # fallback because it doesn't depend on any specific element type,
    # class name, or role attribute.
    if not reviews_tab_found and time_left() > 7:
        logger.info(
            "CSS/ARIA selector strategies found no Reviews tab for query=%r; "
            "trying JS-injection DOM walk.", query,
        )
        if _js_click_reviews_tab(page):
            page.wait_for_timeout(1000)
            if _reviews_content_appeared(page):
                reviews_tab_found = True
                reviews_tab_selector_used = "js_injection"
                logger.info(
                    "Reviews tab clicked via JS-injection DOM walk for query=%r.", query,
                )
            else:
                logger.info(
                    "JS-injection click did not surface review content for "
                    "query=%r; dismissing any modal it opened.", query,
                )
                _dismiss_blocking_modal(page)

    # --- Step 3: wait for the reviews feed to actually render. -----------
    # Increased initial wait from 0ms to 1500ms after a tab click — the
    # XHR that loads the feed has its own round-trip time, and starting
    # the panel-selector sweep immediately after the click loses that race
    # on a slow connection. Two retry sweeps instead of one.
    if reviews_tab_found:
        page.wait_for_timeout(1500)

    panel_ready = _wait_for_review_panel(page, time_left)

    if not panel_ready and time_left() > 8:
        # First retry: give the XHR another window.
        page.wait_for_timeout(1500)
        panel_ready = _wait_for_review_panel(page, time_left)

    if not panel_ready and reviews_tab_found and time_left() > 6:
        # Second retry: scroll slightly to trigger lazy-load, then check again.
        try:
            page.mouse.wheel(0, 300)
            page.wait_for_timeout(1000)
        except Exception:
            pass
        panel_ready = _wait_for_review_panel(page, time_left)

    if panel_ready:
        # Extra settle: wait for at least one actual review card to appear
        # inside the feed (the feed wrapper can mount before the card XHR
        # returns its first page of results).
        for review_card_sel in ("div[data-review-id]", "span.wiI7pd", "div.jftiEf"):
            try:
                page.wait_for_selector(review_card_sel, timeout=4000)
                break
            except Exception:
                continue

    if not reviews_tab_found:
        _log_diag(
            DIAG_REVIEWS_TAB_MISSING, query,
            "no Reviews tab/button matched CSS, ARIA, get_by_role(button), or JS-injection strategies",
        )
    else:
        logger.info(
            "Reviews tab detected (selector=%r) for query=%r.",
            reviews_tab_selector_used, query,
        )
    if not panel_ready:
        _log_diag(
            DIAG_FEED_MISSING, query,
            "reviews feed selector never appeared after tab click + 2 retry sweeps; "
            "falling through to the scroll loop anyway",
        )

    # --- Step 4: idle-based scrolling, then a final expansion pass so
    # anything loaded on the very last scroll still gets its "More" /
    # translation / CSS-truncated text revealed before extraction. ---
    # Scroll toward a modestly higher raw-node target than the final cap.
    # MAX_GOOGLE_REVIEWS counts *usable* reviews after boilerplate-stripping,
    # the 5-word minimum, the _looks_like_review filter, and dedupe - all of
    # which run after this scroll loop is done. A raw DOM node count equal to
    # MAX_GOOGLE_REVIEWS routinely yields fewer than MAX_GOOGLE_REVIEWS
    # reviews once those filters run, so stopping the scroll exactly at the
    # cap quietly under-delivers. Padding the *target* by ~20% (small fixed
    # floor of +5) gives the later filters headroom without changing the
    # worst-case runtime: MAX_SCROLL_ITERATIONS, SCROLL_IDLE_LIMIT, and the
    # time budget inside _scroll_reviews_panel still bound it exactly as
    # before. The final returned list is still hard-capped at
    # MAX_GOOGLE_REVIEWS below (see the `[:MAX_GOOGLE_REVIEWS]` slice).
    _scroll_target = int(MAX_GOOGLE_REVIEWS * 1.2) + 5
    found_count = _scroll_reviews_panel(page, time_left, max_results=_scroll_target)
    _expand_more_buttons(page, time_left)
    _expand_translated_reviews(page, time_left)
    _expand_hidden_review_text(page, time_left)

    logger.info("Reviews scroll: settled at approximately %d review node(s).", found_count)

    # --- Step 5: extract review text only - precise text-node selectors
    # first, container-level selectors only as a fallback (see
    # _collect_review_texts for why that ordering matters for keeping
    # ratings/owner-replies/metadata out of the results). ---
    texts.extend(_collect_review_texts(page, limit=MAX_GOOGLE_REVIEWS))

    if not texts:
        _log_diag(
            DIAG_ZERO_REVIEWS, query,
            f"layout={layout} business_found={clicked_into_place} "
            f"reviews_tab_found={reviews_tab_found} panel_ready={panel_ready} "
            f"scroll_node_count={found_count}",
        )
        _save_debug_artifacts(
            page, query, "zero_reviews",
            extra={
                # "business_page" = Google redirected straight to the Place
                # page; "search_results" = still sitting on the Search
                # results list; "unknown" = neither indicator matched.
                "maps_redirect_target": layout,
                "business_matched": str(clicked_into_place),
                "reviews_selector_matched": reviews_tab_selector_used or "<none>",
                "reviews_selectors_tried": ", ".join(_REVIEWS_TAB_SELECTORS),
                "review_panel_rendered": str(panel_ready),
                "scroll_node_count": str(found_count),
            },
        )

    return texts, False


def _scrape_search_fallback(page, query: str, time_left) -> Tuple[List[str], bool]:
    texts: List[str] = []
    search_q = f"{query} reviews"
    url = f"https://www.google.com/search?q={search_q.replace(' ', '+')}&hl=en&gl=in"

    if not _goto_with_retry(page, url, timeout=_adaptive_nav_timeout(time_left), time_left=time_left):
        return texts, False

    page.wait_for_timeout(1200)
    _dismiss_consent(page)
    page.wait_for_timeout(300)

    if _bot_checked(page):
        logger.warning("Bot-check page encountered on Search fallback for query=%r.", query)
        return texts, True

    texts.extend(
        _collect_texts(
            page, _REVIEW_SELECTORS + _SNIPPET_SELECTORS,
            limit_per_selector=min(MAX_GOOGLE_REVIEWS, 50),
        )
    )
    return texts, False


# How much time Bing's navigation must leave behind for the DuckDuckGo
# fallback that always runs after it (in both _scrape_sync and
# _scrape_sync_product - see the "Google search -> Bing -> DuckDuckGo"
# chain description in ARCHITECTURE.md §4.5). Matches _scrape_ddg_fallback's
# own `min(8.0, time_left())` request.
#
# FIXED (2026-07-30, live-log-driven): _scrape_bing_fallback used to call
# _goto_with_retry with no `reserve_seconds`, so its two-attempt retry loop
# (see _goto_with_retry) could legally burn the ENTIRE remaining budget
# down to nothing before DuckDuckGo ever got a turn. Confirmed via a live
# log where two separate product queries each spent ~15-16s across Bing's
# two failed nav attempts, after which DuckDuckGo was never even attempted
# - it was silently skipped by its own `time_left() <= 3` guard, with no
# log line at all, which made it look like DDG "didn't run" rather than
# an obvious starvation failure. This mattered because DuckDuckGo is the
# one tier in this chain that's actually reliable here (plain httpx GET,
# no browser, no bot-wall observed - see _scrape_ddg_fallback's own
# docstring), while Bing is browser-based and was failing 2/2 attempts on
# both queries in that same log - i.e. the least reliable tier was starving
# the most reliable one. This reserve guarantees DuckDuckGo gets a real
# shot regardless of how Bing's navigation goes.
_DDG_RESERVE_SECONDS = 8.0


def _scrape_bing_fallback(page, query: str, time_left) -> List[str]:
    texts: List[str] = []
    # Broad "<query> reviews" query. The previous
    # "site:trustpilot.com OR site:g2.com" restriction returned literally
    # zero results for most consumer brands (verified live: 0 li.b_algo
    # for boAt with the restricted query vs 10 with the broad one), which
    # made this whole fallback a no-op.
    search_q = f"{query} reviews"
    url = f"https://www.bing.com/search?q={search_q.replace(' ', '+')}"

    if not _goto_with_retry(
        page, url, timeout=_adaptive_nav_timeout(time_left), time_left=time_left,
        retries=1, reserve_seconds=_DDG_RESERVE_SECONDS,
    ):
        return texts

    page.wait_for_timeout(1200)
    # div.b_caption p is the current result-snippet container;
    # p.b_algoSlug is kept as a legacy fallback.
    for sel in ["div.b_caption p", "p.b_algoSlug"]:
        try:
            found = page.locator(sel).all_inner_texts()
        except Exception:
            continue
        for txt in found[: min(MAX_GOOGLE_REVIEWS, 50)]:
            txt = (txt or "").strip()
            if txt and len(txt.split()) >= 6:
                texts.append(txt)
        if texts:
            break
    return texts


def _scrape_ddg_fallback(query: str, time_left) -> List[str]:
    """DuckDuckGo's HTML-only endpoint (html.duckduckgo.com/html) —
    server-rendered, works over one plain urllib GET with a browser
    User-Agent, no JS and no captcha wall observed. Kept as the final
    fallback tier because it's the one source here that stays available
    even when Google serves its /sorry/ captcha page to this network
    (observed live) and the business has no Maps listing at all."""
    if time_left() <= 3:
        return []
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query + ' reviews')}"
    request = urllib.request.Request(url, headers={
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })
    try:
        with urllib.request.urlopen(request, timeout=max(3.0, min(8.0, time_left()))) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        logger.info("DuckDuckGo fallback request failed for query=%r.", query)
        return []

    texts: List[str] = []
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for snippet in soup.select("a.result__snippet"):
            txt = snippet.get_text(" ", strip=True)
            if txt and len(txt.split()) >= 6:
                texts.append(txt)
    except Exception:
        logger.warning("DuckDuckGo fallback parse failed for query=%r.", query, exc_info=True)
    if texts:
        logger.info(
            "DuckDuckGo fallback found %d snippet(s) for query=%r.",
            len(texts), query,
        )
    return texts[: min(MAX_GOOGLE_REVIEWS, 50)]


def _scrape_sync(
    query: str, company_data: Dict[str, str], queued_at: float,
) -> List[str]:
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    setup_start = time.monotonic()
    start = queued_at

    try:
        context = _ensure_context()
    except Exception:
        logger.exception("Could not start/obtain a browser context for query=%r.", query)
        return []

    raw_results: List[str] = []
    bot_checked_any = False
    page = None
    try:
        logger.info(
            "BROWSER_TRACE scraper=Google event=before_new_page elapsed=%.3fs timestamp=%.3f thread=%s",
            time.monotonic() - getattr(_browser_trace, "ensure_context_started_at", time.monotonic()), time.time(), threading.current_thread().name,
        )
        page = context.new_page()
        logger.info(
            "BROWSER_TRACE scraper=Google event=after_new_page elapsed=%.3fs timestamp=%.3f thread=%s",
            time.monotonic() - getattr(_browser_trace, "ensure_context_started_at", time.monotonic()), time.time(), threading.current_thread().name,
        )
        page.set_default_timeout(8000)

        # FIX (the real root cause of Google returning 0 comments - see the
        # OUTER_HARD_TIMEOUT_SECONDS block near the top of this file for
        # the full writeup): the previous version of this fix started a
        # FRESH TIME_BUDGET_SECONDS clock here, after setup finished,
        # instead of the queued_at/`start` clock app.py's outer 48s
        # wait_for() has actually been counting against since this job
        # began. That solved the OLDER bug (setup eating the entire budget
        # before a single navigation was attempted) but introduced this
        # one: setup(17.4s) + a fresh 35s totalled ~52.4s, a few seconds
        # OVER app.py's 48s cap - and because that cap is a hard
        # asyncio.wait_for() cliff, not a soft budget, every caller gave up
        # at 48.0s while this scrape kept running unseen and finished at
        # 50.0s with 3 real reviews that never reached the dashboard.
        #
        # Fixed the same way as twitter_scraper.py/instagram_scraper.py/
        # youtube_scraper.py: one real wall-clock deadline anchored to
        # `start` (== queued_at, captured before this job was even
        # submitted to the executor - see scrape_google_reviews()), via
        # OUTER_HARD_TIMEOUT_SECONDS - SAFETY_MARGIN_SECONDS, so this
        # always leaves real margin for the result to make it back through
        # run_in_executor()/asyncio.shield()/wait_for(), no matter how
        # long setup took. TIME_BUDGET_SECONDS is kept only as the
        # "no contention at all" reference value its own comment still
        # describes; it is no longer what deadline/time_left() below
        # actually use.
        setup_elapsed = time.monotonic() - setup_start
        elapsed_since_job_start = time.monotonic() - start
        deadline = start + OUTER_HARD_TIMEOUT_SECONDS - SAFETY_MARGIN_SECONDS
        remaining_budget = deadline - time.monotonic()
        logger.info(
            "Google reviews scrape: browser/context/page setup took %.1fs "
            "(%.1fs elapsed since job start); %.1fs left for "
            "navigation+scrape before the safety-margined internal "
            "deadline (outer cap=%ds, safety margin=%.1fs).",
            setup_elapsed, elapsed_since_job_start, remaining_budget,
            OUTER_HARD_TIMEOUT_SECONDS, SAFETY_MARGIN_SECONDS,
        )

        def time_left() -> float:
            return deadline - time.monotonic()

        if remaining_budget <= MIN_USEFUL_BUDGET_SECONDS:
            logger.warning(
                "Google reviews scrape: only %.1fs left for query=%r after "
                "a %.1fs setup (job start to now: %.1fs) - not enough time "
                "to attempt navigation; returning early instead of risking "
                "the outer hard timeout.",
                remaining_budget, query, setup_elapsed,
                time.monotonic() - start,
            )
        else:
            # STAGE_TRACE: added 2026-08-03 after a live log showed this
            # company-wide ("General") path finishing in 222s-381s for two
            # of four companies tested, against a ~37s internal deadline -
            # while setup and the Maps-navigation step (see the
            # before/after_maps_search_nav BROWSER_TRACE lines added
            # earlier this session) both measured correctly bounded in
            # that same log. Every fallback stage below (_scrape_maps's
            # own nav retries, _scrape_search_fallback,
            # _scrape_bing_fallback, _scrape_ddg_fallback) has an
            # explicit, correctly-computed timeout when read in isolation
            # - nothing here proves which one actually ran long. These
            # entry/exit markers, labeled COMPANY_WIDE specifically (the
            # existing "PRODUCT search" log lines already distinguish
            # themselves, but this path's own lines didn't), exist so the
            # next live run shows exactly which stage the elapsed time
            # actually went into, instead of requiring another guess.
            def _stage_trace(stage: str, event: str) -> None:
                logger.info(
                    "STAGE_TRACE scraper=Google path=COMPANY_WIDE query=%r "
                    "stage=%s event=%s elapsed_since_job_start=%.1fs",
                    query, stage, event, time.monotonic() - start,
                )

            _stage_trace("maps", "before")
            try:
                maps_texts, hit_bot_check = _scrape_maps(page, query, time_left, company_data)
                raw_results.extend(maps_texts)
                bot_checked_any = bot_checked_any or hit_bot_check
            except Exception:
                logger.exception("Maps scrape failed for query=%r.", query)
            _stage_trace("maps", "after")

            if len(raw_results) < MAX_GOOGLE_REVIEWS and time_left() > 8:
                _stage_trace("search_fallback", "before")
                try:
                    search_texts, hit_bot_check = _scrape_search_fallback(page, query, time_left)
                    raw_results.extend(search_texts)
                    bot_checked_any = bot_checked_any or hit_bot_check
                except Exception:
                    logger.exception("Search fallback failed for query=%r.", query)
                _stage_trace("search_fallback", "after")

            # Threshold raised 6 -> _DDG_RESERVE_SECONDS + 2 (10s): Bing now
            # reserves _DDG_RESERVE_SECONDS for DuckDuckGo internally (see
            # that constant's comment), so entering with only 6-10s left
            # meant Bing's own navigable time was clamped to ~0-2s before
            # it even tried - not enough for a real attempt, just wasted
            # overhead ahead of the DDG step it was already starving.
            if len(raw_results) < min(5, MAX_GOOGLE_REVIEWS) and time_left() > _DDG_RESERVE_SECONDS + 2:
                _stage_trace("bing_fallback", "before")
                try:
                    raw_results.extend(_scrape_bing_fallback(page, query, time_left))
                except Exception:
                    logger.exception("Bing fallback failed for query=%r.", query)
                _stage_trace("bing_fallback", "after")

            if len(raw_results) < min(5, MAX_GOOGLE_REVIEWS) and time_left() > 3:
                _stage_trace("ddg_fallback", "before")
                try:
                    raw_results.extend(_scrape_ddg_fallback(query, time_left))
                except Exception:
                    logger.exception("DuckDuckGo fallback failed for query=%r.", query)
                _stage_trace("ddg_fallback", "after")
    finally:
        if page is not None:
            try:
                page.close()
            except Exception:
                pass
        if bot_checked_any:
            # Recycle this worker's identity so the next call on this
            # thread gets a clean context instead of repeatedly hitting
            # the same flagged session.
            logger.warning(
                "Bot-check triggered for query=%r; recycling browser context on %s.",
                query, threading.current_thread().name,
            )
            _teardown_thread_browser(reason="bot_detection")

    elapsed = time.monotonic() - start
    logger.info(
        "Collected %d raw candidate(s) for query=%r in %.1fs.",
        len(raw_results), query, elapsed,
    )

    cleaned: List[str] = []
    for raw in raw_results:
        stripped = _strip_review_boilerplate(raw)
        if stripped and len(stripped.split()) >= 5 and _looks_like_review(stripped):
            cleaned.append(stripped)

    before_dedupe = len(cleaned)
    cleaned = _dedupe(cleaned)
    duplicates_removed = before_dedupe - len(cleaned)

    final = normalize_comments(cleaned)[:MAX_GOOGLE_REVIEWS]
    total_elapsed = time.monotonic() - start
    logger.info(
        "Google reviews for query=%r: raw=%d duplicates_removed=%d final=%d "
        "elapsed=%.1fs (cap=%d).",
        query, before_dedupe, duplicates_removed, len(final), total_elapsed,
        MAX_GOOGLE_REVIEWS,
    )
    if len(final) == 0:
        logger.warning("Zero reviews collected for %r — check logs for genuine absence vs scrape failure.", query)
        # Compare against the anchored working budget (OUTER_HARD_TIMEOUT_
        # SECONDS - SAFETY_MARGIN_SECONDS, counted from job start), not the
        # old fixed TIME_BUDGET_SECONDS - see the fix above for why.
        # `deadline` and `start` are always set by this point since they're
        # assigned right after page setup, before any of the try/except-
        # wrapped scrape calls that could otherwise skip them.
        working_budget_total = OUTER_HARD_TIMEOUT_SECONDS - SAFETY_MARGIN_SECONDS
        scrape_phase_elapsed = working_budget_total - (deadline - time.monotonic())
        if scrape_phase_elapsed >= working_budget_total - 1:
            _log_diag(
                DIAG_TIMEOUT, query,
                f"internal time budget ({working_budget_total:.1f}s, anchored "
                f"to job start with an {SAFETY_MARGIN_SECONDS:.1f}s safety "
                f"margin under the {OUTER_HARD_TIMEOUT_SECONDS}s outer cap) "
                f"nearly/fully exhausted (elapsed={scrape_phase_elapsed:.1f}s "
                f"post-setup, {total_elapsed:.1f}s total incl. setup) before "
                f"any reviews were collected",
            )
    logger.info("Collected %d reviews", len(final))
    return final


def _scrape_sync_product(query: str, queued_at: float) -> List[str]:
    """Genuinely product-specific counterpart to `_scrape_sync`.

    Deliberately SKIPS `_scrape_maps` entirely. Google Maps only ever has a
    listing for the business as a whole (e.g. "boAt Lifestyle" the store),
    never for one SKU ("Airdopes 131") - there is no such thing as a Maps
    listing for a single product. Running `_scrape_maps` here would just
    re-collect the exact same business-wide reviews `scrape_google_reviews`
    already gets for the "General" job, which is the "identical Google
    comments duplicated across every selected product" problem this
    function exists to fix. See `_cache_key_for`'s docstring below for the
    business-level scrape this intentionally does NOT touch.

    Instead this goes straight to the search-snippet chain (Google search
    -> Bing -> DuckDuckGo), using a query that includes the product name,
    so the snippets collected are about that specific product. Not
    cached/shared across products on purpose: every product's query text
    is different, so every product gets its own independent scrape.
    """
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    setup_start = time.monotonic()
    start = queued_at

    try:
        context = _ensure_context()
    except Exception:
        logger.exception("Could not start/obtain a browser context for product query=%r.", query)
        return []

    raw_results: List[str] = []
    bot_checked_any = False
    page = None
    try:
        page = context.new_page()
        page.set_default_timeout(8000)

        setup_elapsed = time.monotonic() - setup_start
        deadline = start + OUTER_HARD_TIMEOUT_SECONDS - SAFETY_MARGIN_SECONDS
        remaining_budget = deadline - time.monotonic()
        logger.info(
            "Google PRODUCT search: browser/context/page setup took %.1fs; "
            "%.1fs left for query=%r before the internal deadline.",
            setup_elapsed, remaining_budget, query,
        )

        def time_left() -> float:
            return deadline - time.monotonic()

        if remaining_budget <= MIN_USEFUL_BUDGET_SECONDS:
            logger.warning(
                "Google PRODUCT search: only %.1fs left for query=%r after "
                "a %.1fs setup - not enough time to attempt navigation; "
                "returning early instead of risking the outer hard timeout.",
                remaining_budget, query, setup_elapsed,
            )
        else:
            try:
                search_texts, hit_bot_check = _scrape_search_fallback(page, query, time_left)
                raw_results.extend(search_texts)
                bot_checked_any = bot_checked_any or hit_bot_check
            except Exception:
                logger.exception("Product search fallback failed for query=%r.", query)

            # See the matching comment in _scrape_sync - same reasoning.
            if len(raw_results) < min(5, MAX_GOOGLE_REVIEWS) and time_left() > _DDG_RESERVE_SECONDS + 2:
                try:
                    raw_results.extend(_scrape_bing_fallback(page, query, time_left))
                except Exception:
                    logger.exception("Product Bing fallback failed for query=%r.", query)

            if len(raw_results) < min(5, MAX_GOOGLE_REVIEWS) and time_left() > 3:
                try:
                    raw_results.extend(_scrape_ddg_fallback(query, time_left))
                except Exception:
                    logger.exception("Product DuckDuckGo fallback failed for query=%r.", query)
    finally:
        if page is not None:
            try:
                page.close()
            except Exception:
                pass
        if bot_checked_any:
            logger.warning(
                "Bot-check triggered for product query=%r; recycling browser context on %s.",
                query, threading.current_thread().name,
            )
            _teardown_thread_browser(reason="bot_detection")

    elapsed = time.monotonic() - start
    logger.info(
        "Collected %d raw product candidate(s) for query=%r in %.1fs.",
        len(raw_results), query, elapsed,
    )

    cleaned: List[str] = []
    for raw in raw_results:
        stripped = _strip_review_boilerplate(raw)
        if stripped and len(stripped.split()) >= 5 and _looks_like_review(stripped):
            cleaned.append(stripped)

    before_dedupe = len(cleaned)
    cleaned = _dedupe(cleaned)
    final = normalize_comments(cleaned)[:MAX_GOOGLE_REVIEWS]
    logger.info(
        "Google product reviews for query=%r: raw=%d duplicates_removed=%d "
        "final=%d elapsed=%.1fs (cap=%d).",
        query, before_dedupe, before_dedupe - len(cleaned), len(final),
        time.monotonic() - start, MAX_GOOGLE_REVIEWS,
    )
    return final


async def scrape_google_product_reviews(product_data: Dict[str, str]) -> List[str]:
    """Genuinely product-specific Google signal.

    Unlike `scrape_google_reviews` (Google Maps, business-wide, cached and
    shared across every product in the run), this searches for
    "<company name> <product name>" review snippets and is never cached
    across products - each product gets its own real scrape.
    """
    company_name = (product_data or {}).get("company_name", "") or ""
    product_name = (product_data or {}).get("product_name", "") or ""
    query = f"{company_name} {product_name}".strip()
    if not query:
        logger.info(
            "scrape_google_product_reviews called with no company/product "
            "name; returning []."
        )
        return []

    queued_at = time.monotonic()
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(_EXECUTOR, _scrape_sync_product, query, queued_at)
    except Exception:
        logger.exception("Unhandled error scraping Google product reviews for query=%r.", query)
        return []


# ---------------------------------------------------------------------------
# Business-level cache + single-flight coalescing.
#
# Google Reviews belong to the business, not to an individual product.
# CURRENT STATE (verified against app.py this session): app.py's "General"
# job (one per analysis run) is the only caller of scrape_google_reviews()
# below - every per-product job instead calls
# scrape_google_product_reviews() (the search-snippet path, defined above),
# which never touches Maps or this cache at all. That split is itself the
# fix for an OLDER bug, kept here as history so it isn't re-litigated: app.py
# used to schedule one job per selected product PLUS one "General" job, with
# EVERY one of those jobs calling scrape_google_reviews() with a query like
# "{company} {product name}" - for 3-5 selected products that was 3-5 full
# Playwright Maps sessions for what is, by construction, the exact same
# business-wide review set, which is exactly why "Google Review Collection"
# used to dominate the runtime in the production log, and why every
# product's "Google" comments used to look suspiciously identical (see
# ARCHITECTURE.md's history section - if that symptom ever reappears, this
# exact bug has regressed).
#
# What this cache still protects against, now that the split above exists:
# it caches the collected reviews per business (keyed by the company's
# website) and, if a scrape for that business is already running when
# another call asks for it, awaits that same in-flight scrape instead of
# starting a second one - defense in depth against scrape_google_reviews()
# ever being invoked more than once in the same run (e.g. a future caller,
# a retry path, or a second analysis of the same company started while the
# first is still running). A later, separate analysis of the same company
# still gets a fresh scrape once GOOGLE_REVIEW_CACHE_TTL_SECONDS has
# elapsed.
# ---------------------------------------------------------------------------
_review_cache: Dict[str, Tuple[float, List[str]]] = {}
_inflight_tasks: Dict[str, "asyncio.Task"] = {}


def _cache_key_for(company_data: Dict[str, str]) -> str:
    website = (company_data or {}).get("website", "") or ""
    website = website.strip().lower().rstrip("/")
    if website:
        return website
    # Improved fallback: strip product suffixes for stable key across products of same business.
    name = (company_data or {}).get("company_name", "") or ""
    return _normalize_business_key(name)


async def scrape_google_reviews(company_data: Dict[str, str]) -> List[str]:
    query = (company_data or {}).get("company_name", "")
    query = query.strip() if isinstance(query, str) else ""
    if not query:
        logger.info("scrape_google_reviews called with no company_name; returning [].")
        return []

    # NOTE: this is a queue timestamp for elapsed-time logging only, not a
    # working deadline. The real TIME_BUDGET_SECONDS nav+scrape deadline is
    # computed inside _scrape_sync, after browser/context/page setup
    # actually finishes - see the fix there for why that matters.
    queued_at = time.monotonic()
    cache_key = _cache_key_for(company_data)
    now = time.monotonic()

    # --- 1) Serve from cache if we already have a fresh result for this
    # business, regardless of which product's query text triggered it. ---
    cached = _review_cache.get(cache_key)
    if cached is not None:
        cached_at, cached_reviews = cached
        if now - cached_at < GOOGLE_REVIEW_CACHE_TTL_SECONDS:
            logger.info(
                "Google reviews cache hit for business=%r (query=%r): "
                "reusing %d review(s) collected %.0fs ago instead of "
                "re-scraping Maps.",
                cache_key, query, len(cached_reviews), now - cached_at,
            )
            return list(cached_reviews)

    # --- 2) If another job for the same business is already scraping,
    # piggyback on that instead of opening a second Maps session. ---
    #
    # IMPORTANT: we `asyncio.shield()` the wait here. Without it, if *this*
    # caller gets cancelled (e.g. app.py's per-job asyncio.wait_for hits its
    # 30s hard timeout), asyncio propagates that cancellation into whatever
    # we're directly awaiting - which would be the SAME shared future every
    # other piggybacking job is also awaiting. One slow job timing out would
    # then kill the scrape for every other product in the batch, and each of
    # those callers would receive a CancelledError instead of a result list
    # (CancelledError isn't an Exception subclass, so it also wasn't being
    # filtered out by the `isinstance(outcome, Exception)` checks upstream).
    # shield() decouples "this caller stopped waiting" from "the shared work
    # stops running".
    existing_task = _inflight_tasks.get(cache_key)
    if existing_task is not None and not existing_task.done():
        logger.info(
            "Google reviews scrape for business=%r already in progress "
            "(started by another selected product's job); reusing that "
            "result for query=%r instead of starting a second Maps scrape.",
            cache_key, query,
        )
        try:
            return list(await asyncio.shield(existing_task))
        except asyncio.CancelledError:
            logger.warning(
                "Caller for query=%r stopped waiting (its own timeout) on "
                "shared Google Reviews scrape for %r; the scrape itself "
                "keeps running in the background for other jobs/cache.",
                query, cache_key,
            )
            return []
        except Exception:
            logger.exception("Shared Google Reviews task for %r failed.", cache_key)
            return []

    # --- 3) Nobody has scraped this business yet - do the real scrape,
    # and let any jobs that arrive while it's running join in via step 2. ---
    loop = asyncio.get_running_loop()
    task = loop.run_in_executor(_EXECUTOR, _scrape_sync, query, company_data, queued_at)
    _inflight_tasks[cache_key] = task

    # Populate the cache from a done-callback rather than from the code
    # right after `await task` below. That matters because this coroutine
    # (the "owner") can itself be cancelled by its own caller's timeout -
    # if caching only happened after a successful `await`, a timed-out
    # owner would never cache anything, and every other job (including
    # ones that arrive *after* this one, like a semaphore-delayed job)
    # would find neither a cache entry nor a live inflight task and would
    # be forced to start yet another full scrape from scratch.
    def _on_scrape_done(t: "asyncio.Future", key: str = cache_key) -> None:
        _inflight_tasks.pop(key, None)
        if t.cancelled():
            logger.warning(
                "Background Google Reviews scrape for business=%r was "
                "cancelled before completion; nothing to cache.", key,
            )
            return
        exc = t.exception()
        if exc is not None:
            logger.error("Background Google Reviews scrape for %r failed: %s", key, exc)
            return
        outcome = list(t.result())
        if not outcome:
            # Zero-review outcomes (TIMEOUT, NAV_FAILURE, BUSINESS_NOT_FOUND,
            # or a genuine zero-review business) are not cached, so the next
            # job for this business retries the scrape instead of being
            # served a stale/empty result for the full cache TTL.
            logger.info(
                "Background Google Reviews scrape for business=%r returned "
                "0 reviews; not caching (will retry on next request).", key,
            )
            return
        _review_cache[key] = (time.monotonic(), outcome)

    task.add_done_callback(_on_scrape_done)

    try:
        result = await asyncio.shield(task)
    except asyncio.CancelledError:
        logger.warning(
            "Caller for query=%r stopped waiting (its own timeout) on the "
            "Google Reviews scrape it started for %r; the scrape itself "
            "keeps running in the background and will populate the cache "
            "for later callers.",
            query, cache_key,
        )
        return []
    except Exception:
        logger.exception("Unhandled error scraping Google reviews for query=%r.", query)
        return []

    return result
