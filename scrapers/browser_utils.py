"""
Module Name
-----------
scrapers/browser_utils.py

Purpose
-------
Shared browser infrastructure used by every Playwright-based scraper
(google_scraper.py, twitter_scraper.py, instagram_scraper.py,
youtube_scraper.py, website_review_scraper.py), plus a few small
platform-agnostic helpers used more broadly (comment normalization,
social-profile URL matching, storage paths).

Responsibilities
-----------------
- `BrowserManager` / the module-level `browser_manager` singleton: launch
  exactly one real Chromium process for the whole application's lifetime,
  and hand every scraper a CDP endpoint to attach to via
  `ensure_shared_browser()` + `playwright.chromium.connect_over_cdp()`,
  instead of each scraper launching (and having to tear down) its own.
  This is the actual fix for a real ~44-46s startup stall previously
  caused by every scraper module launching its own Chromium process, once
  per worker thread — see the class docstring below for the mechanics,
  and ARCHITECTURE.md for the fuller history. A handle obtained via
  connect_over_cdp() can be closed by its own caller without killing the
  shared process out from under every other scraper still using it.
- `browser_launch_slot()` / `run_playwright_async()`: kept under their
  original names/signatures for call-site compatibility; both now
  delegate to the shared manager rather than doing their own launching.
- `get_storage_dir()`: shared on-disk storage path helper.
- `normalize_comments()`: shared comment-list cleanup used by multiple
  scrapers before their results reach app.py's merge step.
- `find_social_profile_url()`: matches a company name against a
  discovered social-platform URL.

Dependencies
------------
`playwright` (sync and async APIs, depending on caller), plus standard
library `asyncio`, `atexit`, `concurrent.futures`, `threading`, `time`,
`pathlib`.
"""

import asyncio
import atexit
import concurrent.futures
import logging
import os
import re
import socket
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Awaitable, Callable, Iterator, List, Optional, TypeVar
from urllib.parse import quote_plus, unquote

import httpx

# Generic type variable used for preserving return types of async functions
_T = TypeVar("_T")

logger = logging.getLogger(__name__)


# ===========================================================================
# Browser Lifecycle Management
# ===========================================================================
#
# Two distinct concerns are bounded here, deliberately kept separate because
# they protect against two different failure modes:
#
#   1. Simultaneous *launches*. Starting a Chromium process is a short but
#      CPU-heavy burst (~1-2s). Too many launches at the exact same instant
#      is what the original semaphore protected against, and still does -
#      bounded by MAX_CONCURRENT_BROWSER_LAUNCHES.
#
#   2. Simultaneous *sessions*. A browser that is open - even sitting idle
#      between actions - holds real memory and a live OS process for its
#      whole lifetime, which is much longer than the launch burst. The old
#      code didn't bound this at all: the launch semaphore was released
#      "once the browser is launched" (see the original docstring this file
#      used to carry), so nothing stopped an unbounded number of browsers
#      from being open at once. This is bounded now by
#      MAX_CONCURRENT_BROWSER_SESSIONS, via a fixed-size, application-scoped
#      ThreadPoolExecutor: a worker thread is only returned to the pool once
#      the Playwright coroutine running in it has fully finished - and that
#      coroutine necessarily includes the browser's shutdown. See the
#      docstring on run_playwright_async() below for why that's structurally
#      guaranteed rather than merely a convention callers have to follow.
#
# BrowserManager owns both controls plus the shared executor. One instance
# (`browser_manager`, below) is shared by the whole application.
#
# browser_launch_slot() and run_playwright_async() are kept with their
# original names and call signatures, so any existing code that already
# imports and calls them keeps working unchanged. They now simply delegate
# to the shared manager instead of each rolling its own throwaway
# semaphore/executor.
#
#   3. Simultaneous *processes*. This is the one that was actually causing
#      the 44-46s stall in production: google_scraper.py, twitter_scraper.py,
#      instagram_scraper.py, and youtube_scraper.py each maintain their own
#      per-worker-thread Playwright/browser/context pool (see each module's
#      own "Browser pool" section) - which is good for reuse *within* a
#      module, but every module still calls `pw.chromium.launch(...)`
#      independently the first time each of its own worker threads needs a
#      browser. Across four scraper modules x up to 3 workers each, that's
#      up to 12 independent Chromium *process* launches for a single
#      analysis run, all funneling through the one
#      MAX_CONCURRENT_BROWSER_LAUNCHES=2 semaphore below - so most of them
#      simply queue behind each other before any of them can navigate
#      anywhere. ensure_shared_browser() (below) fixes this at its root: it
#      launches exactly ONE Chromium process for the lifetime of the
#      application, and every worker thread in every scraper module attaches
#      to that one process over CDP instead of launching its own. See that
#      method's docstring for the mechanics and why CDP (rather than sharing
#      one Python Browser object) is what makes this safe across threads.

# Max Chromium processes allowed to be *launching* at the same instant.
# Unchanged from the original value and meaning. Largely moot for the four
# Playwright scrapers now that they share one already-launched process via
# ensure_shared_browser() - retained for any other caller still using
# browser_launch_slot() directly around its own launch() call.
MAX_CONCURRENT_BROWSER_LAUNCHES = 2

# Max Playwright *sessions* (launch -> automation -> close) allowed to be
# in flight across the whole application at once. Deliberately looser than
# MAX_CONCURRENT_BROWSER_LAUNCHES: once past the launch burst, an already-
# open browser is comparatively cheap to leave running, so more of them are
# allowed to be alive concurrently than are allowed to be *launching* at any
# one instant. Chosen as a conservative starting point - retune against real
# host resources and actual scraper fan-out per job.
MAX_CONCURRENT_BROWSER_SESSIONS = 8

# Extra Chromium flags used only for the one real, shared browser launch
# (see ensure_shared_browser()). Union of what google_scraper.py,
# twitter_scraper.py, instagram_scraper.py, and youtube_scraper.py each
# used to pass to their own separate launch() call - now that there's only
# one launch for all of them to share, they share one flag set too.
# --disable-blink-features=AutomationControlled was previously only set by
# two of the four modules (google, instagram); including it for all four
# is strictly a stealth improvement for twitter/youtube, not a behavior
# change any scraper relies on.
_SHARED_LAUNCH_ARGS = [
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-dev-shm-usage",
    "--disable-blink-features=AutomationControlled",
]

# Diagnostic-only override: set SOCIAL_ANALYZER_HEADLESS=false to launch the
# one shared Chromium process with a visible window instead of headless.
# Default (unset, or anything other than "false"/"0"/"no") keeps the
# existing headless=True behavior unchanged - this exists purely so
# headless-vs-headful can be A/B tested for a single run (e.g. X/Instagram
# navigation timing out with wait_until="commit" - the loosest possible
# wait condition - while a plain httpx GET to the identical URL from the
# same host gets 200 OK almost instantly is a pattern consistent with
# automation detection stalling the connection rather than a real network
# or server-side slowness, and headless Chromium is one of the more common
# signals such detection keys on) without editing this file again.
_HEADLESS = os.environ.get("SOCIAL_ANALYZER_HEADLESS", "true").strip().lower() not in (
    "false", "0", "no",
)


def _pick_free_port() -> int:
    """Ask the OS for an unused local TCP port, for Chromium's
    --remote-debugging-port. Binding to port 0 and immediately closing the
    socket is the standard way to reserve a free port without a race: the
    OS won't hand that exact port back out to anyone else until it's
    reused, and Chromium binds it again immediately after we close it here.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class BrowserManager:
    """
    Application-wide owner of Playwright browser concurrency controls.

    Replaces two things this module used to do ad hoc:
      - a bare module-level threading.Semaphore that only ever bounded the
        launch call itself (callers released it right after launching, so
        a browser sitting open afterwards no longer counted against
        anything)
      - a brand new concurrent.futures.ThreadPoolExecutor created and torn
        down on every single run_playwright_async() call

    Both are now owned by one object with a single, explicit shutdown path.
    """

    def __init__(
        self,
        max_concurrent_launches: int = MAX_CONCURRENT_BROWSER_LAUNCHES,
        max_concurrent_sessions: int = MAX_CONCURRENT_BROWSER_SESSIONS,
    ) -> None:
        self._launch_semaphore = threading.Semaphore(max_concurrent_launches)

        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrent_sessions,
            thread_name_prefix="playwright-session",
        )

        # Guards _is_shutdown / _active_sessions. A plain threading.Lock
        # (not asyncio.Lock) on purpose: shutdown() must be safely callable
        # from a synchronous context too (e.g. an atexit handler, where
        # there's no guarantee an event loop is running), and the critical
        # sections here are just integer/bool updates - short enough that
        # briefly blocking an event loop thread is a non-issue.
        self._state_lock = threading.Lock()
        self._is_shutdown = False
        self._active_sessions = 0

        # -- shared, application-wide Chromium process -----------------------
        # See ensure_shared_browser() below. Kept on its own lock rather than
        # _state_lock: launching Chromium can take a second or more, and
        # that shouldn't block unrelated, near-instant reads/writes of
        # _active_sessions/_is_shutdown for the duration of a launch.
        self._shared_browser_lock = threading.Lock()
        self._shared_playwright = None
        self._shared_browser = None
        self._shared_cdp_endpoint: Optional[str] = None
        self._shared_debug_port = _pick_free_port()

        # Dedicated, single-purpose worker thread that owns the shared
        # Chromium's sync_playwright() session for its entire life (launch
        # here, close() here too - see ensure_shared_browser() and
        # shutdown()). It must never run anything other than that: every
        # caller of ensure_shared_browser() is a scraper worker thread that
        # has typically ALREADY opened its own, separate sync_playwright()
        # session on the calling thread (see e.g. google_scraper.py's
        # _ensure_context(), which starts its own local `pw` before ever
        # reaching this call) - see ensure_shared_browser()'s docstring for
        # why launching inline on such a thread is unsafe.
        self._shared_browser_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="shared-browser-owner"
        )

    # -- launch pacing -------------------------------------------------------

    @contextmanager
    def launch_slot(self) -> Iterator[None]:
        """
        Reserves a launch slot for the duration of the `with` block.

        Intended to wrap only the actual `browser_type.launch(...)` call,
        same as the original browser_launch_slot() - see that function's
        docstring below for why the slot is released immediately after
        launch rather than held for the browser's whole life.
        """
        self._launch_semaphore.acquire()
        try:
            yield
        finally:
            self._launch_semaphore.release()

    # -- shared browser process -----------------------------------------------

    def ensure_shared_browser(self) -> str:
        """
        Launches the single, application-wide Chromium process the first
        time any scraper calls this, and returns its CDP endpoint (e.g.
        "http://127.0.0.1:54231"). Every later call - from any thread, in
        any scraper module - returns that same endpoint immediately;
        nothing further is launched.

        This is the actual fix for the ~44-46s startup stall: previously
        every scraper module launched its OWN Chromium process, once per
        worker thread, and all of those launches funneled through the
        same MAX_CONCURRENT_BROWSER_LAUNCHES=2 semaphore - so most of
        them simply queued behind each other before any of them could
        navigate anywhere. Now there is exactly one real
        `chromium.launch()` call for the whole process's lifetime;
        everyone else attaches to that one process.

        Why CDP instead of just handing out the same Python Browser
        object: Playwright's sync API is not thread-safe - a
        Browser/Context/Page created inside one thread's
        `sync_playwright()` instance cannot be driven from another
        thread. Every scraper module still runs its Playwright calls on
        its own worker threads (that part is deliberately unchanged), so
        each thread still needs its own lightweight Playwright client.
        Chromium's CDP endpoint lets many independent client connections
        attach to the SAME underlying browser *process*, each creating
        its own isolated BrowserContext via
        `pw.chromium.connect_over_cdp(endpoint)`. So the expensive part -
        spawning the Chromium process - happens once, while each
        worker thread's own Playwright driver/context lifecycle (and the
        reuse/recycle logic built around it in each scraper module) stays
        exactly as it was.

        Contexts obtained this way are still fully isolated from each
        other (browser.new_context() never shares cookies/storage across
        contexts, regardless of how many client connections are attached
        to the browser), so scrapers running concurrently against
        different platforms - or the same platform on different worker
        threads - never see each other's cookies, login state, or
        consent choices.

        Thread-safe via double-checked locking: the common case (browser
        already launched) never touches the lock at all, and concurrent
        first-callers (e.g. all four scrapers starting at once at the
        beginning of an analysis run) block briefly on
        `_shared_browser_lock` rather than racing to launch Chromium
        multiple times.

        IMPORTANT - why the actual launch happens on a dedicated thread:
        Playwright's sync API (`sync_playwright()`) pumps its event loop
        through a greenlet dispatcher for as long as that session stays
        open, which leaves the OS thread that opened it looking to
        asyncio as if it has a *running* event loop the whole time - not
        just while `.start()` itself is executing. Every scraper worker
        thread that reaches this method has, by that point, already
        opened its OWN local sync_playwright() session on that same
        thread (see e.g. google_scraper.py's `_ensure_context()`, which
        starts its own local `pw` before ever calling this method). So a
        second, nested `sync_playwright().start()` call - i.e. this
        method's own launch - running inline on that same thread would
        always be detected as "Sync API used inside the asyncio loop"
        and fail, regardless of which caller happens to win the race
        below; it's not a timing fluke, that thread is never safe to
        launch on. Running the actual launch on `_shared_browser_executor`
        - a dedicated worker thread that never runs anything except this
        one launch (and, later, shutdown()'s close()) - guarantees a
        clean thread with no pre-existing Playwright/asyncio session, no
        matter which thread called ensure_shared_browser() or what that
        caller thread was already doing.
        """
        endpoint = self._shared_cdp_endpoint
        if endpoint is not None:
            return endpoint

        with self._shared_browser_lock:
            if self._shared_cdp_endpoint is not None:
                return self._shared_cdp_endpoint

            with self._state_lock:
                if self._is_shutdown:
                    raise RuntimeError(
                        "BrowserManager has been shut down; the shared "
                        "browser cannot be (re)launched."
                    )

            # Block the calling thread (whatever it is - the FastAPI
            # startup thread, or any scraper's worker thread) until the
            # dedicated owner thread has actually finished launching.
            # .result() is a plain OS-level wait (no asyncio involved), so
            # this is safe to call even from a thread that already has its
            # own sync_playwright() session open.
            self._shared_cdp_endpoint = self._shared_browser_executor.submit(
                self._launch_shared_browser
            ).result()
            return self._shared_cdp_endpoint

    def _launch_shared_browser(self) -> str:
        """
        Does the actual `sync_playwright().start()` + `chromium.launch()`.

        Must only ever run on `_shared_browser_executor`'s single
        dedicated worker thread - see ensure_shared_browser()'s docstring
        for why launching from an arbitrary caller thread is unsafe. This
        method itself doesn't need to guard against concurrent execution:
        ensure_shared_browser() already ensures it's submitted at most
        once, under `_shared_browser_lock`.
        """
        # Same guard as _run_playwright_in_fresh_loop() below: app.py sets
        # WindowsSelectorEventLoopPolicy process-wide at import time, and
        # that policy - not thread history - is what sync_playwright()
        # consults via asyncio.new_event_loop() when it finds no running
        # loop on this thread. SelectorEventLoop can't back a subprocess on
        # Windows, which is exactly what the Playwright driver needs, so
        # this thread must re-assert Proactor for itself before starting.
        if sys.platform.startswith("win"):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

        from playwright.sync_api import sync_playwright

        pw = sync_playwright().start()
        browser = pw.chromium.launch(
            headless=_HEADLESS,
            args=[
                *_SHARED_LAUNCH_ARGS,
                f"--remote-debugging-port={self._shared_debug_port}",
            ],
        )
        self._shared_playwright = pw
        self._shared_browser = browser
        logger.info(
            "Launched the single application-wide Chromium process "
            "(CDP port %d, headless=%s). All scraper worker threads now "
            "attach to this one process instead of launching their own.",
            self._shared_debug_port, _HEADLESS,
        )
        return f"http://127.0.0.1:{self._shared_debug_port}"

    # -- session execution -----------------------------------------------------

    async def run_session(self, coro_factory: Callable[[], Awaitable[_T]]) -> _T:
        """
        Runs one full Playwright session (coro_factory) to completion on
        the shared, bounded worker pool.

        The worker thread picked up here is not returned to the pool until
        coro_factory() itself returns - which is the *whole* session, launch
        through close, because _run_playwright_in_fresh_loop closes its
        event loop the moment coro_factory() finishes, and any Playwright
        object (like an unclosed browser) created on that loop cannot
        survive past it. So a session slot is, structurally, held until the
        browser actually closes - it isn't just a convention callers need
        to remember to follow.

        If more sessions are requested than max_concurrent_sessions allows,
        this call queues (inside the executor) until a worker frees up,
        rather than letting an unbounded number of concurrent Chromium
        processes pile up.
        """
        with self._state_lock:
            if self._is_shutdown:
                raise RuntimeError(
                    "BrowserManager has been shut down; no new Playwright "
                    "sessions can be started"
                )

        def _tracked_run() -> _T:
            # Runs inside the worker thread itself, so the counter only
            # goes up once a worker has actually picked up the job (i.e.
            # a browser is genuinely about to launch), not the moment the
            # job is merely requested/queued - queued-but-not-yet-running
            # requests must not count as "active".
            with self._state_lock:
                self._active_sessions += 1
            try:
                return _run_playwright_in_fresh_loop(coro_factory)
            finally:
                with self._state_lock:
                    self._active_sessions -= 1

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _tracked_run)

    @property
    def active_sessions(self) -> int:
        """Number of Playwright sessions currently in flight."""
        with self._state_lock:
            return self._active_sessions

    # -- cleanup ---------------------------------------------------------------

    def shutdown(self, wait: bool = True) -> None:
        """
        Shuts down the shared executor. Safe to call more than once, and
        safe to call from a non-async context (e.g. atexit).

        wait=True (default): let in-flight sessions finish, then shut down.
        wait=False: shut down immediately without waiting; only appropriate
        when waiting genuinely isn't an option (e.g. a forced/urgent exit).
        """
        with self._state_lock:
            if self._is_shutdown:
                return
            self._is_shutdown = True
            in_flight = self._active_sessions

        if not wait and in_flight:
            logger.warning(
                "BrowserManager.shutdown(wait=False): %d Playwright "
                "session(s) still in flight and will be abandoned.",
                in_flight,
            )

        self._executor.shutdown(wait=wait)

        # Close the one real Chromium process last, after in-flight
        # sessions above have had the chance to finish. This is the ONLY
        # close() call that actually terminates the shared browser -
        # every scraper's own per-thread handle.browser.close() (e.g.
        # google_scraper._teardown_thread_browser) just disconnects that
        # thread's CDP client, because those handles were obtained via
        # connect_over_cdp(), not launch(). That's what requirement #6
        # ("shutdown only when the application exits") means concretely:
        # no scraper, no matter how it tears down its own handle, can
        # accidentally kill the shared process out from under every other
        # scraper still using it.
        #
        # The close() itself is run on _shared_browser_executor - the same
        # dedicated thread that launched the browser - rather than
        # whichever thread happens to be calling shutdown(). Playwright's
        # sync objects are tied to the greenlet dispatcher of the thread
        # that created them, so closing them from that same thread is both
        # the safe option and symmetric with how they were launched.
        def _close_shared() -> None:
            with self._shared_browser_lock:
                if self._shared_browser is not None:
                    try:
                        self._shared_browser.close()
                    except Exception:
                        logger.exception("Error closing the shared Chromium browser.")
                if self._shared_playwright is not None:
                    try:
                        self._shared_playwright.stop()
                    except Exception:
                        logger.exception("Error stopping the shared Playwright driver.")
                self._shared_browser = None
                self._shared_playwright = None
                self._shared_cdp_endpoint = None

        try:
            self._shared_browser_executor.submit(_close_shared).result()
        except Exception:
            logger.exception("Error tearing down the shared Chromium browser.")
        self._shared_browser_executor.shutdown(wait=wait)


# Single shared instance used by the whole application. Import this
# directly if you need lifetime introspection (e.g. active_sessions) or
# want to trigger shutdown explicitly; existing scraper code doesn't need
# to change and can keep using browser_launch_slot() / run_playwright_async()
# below exactly as before.
browser_manager = BrowserManager()

# Best-effort safety net: if the process exits without an explicit shutdown
# call, make sure the executor's worker threads are still released instead
# of leaking until garbage collection.
atexit.register(browser_manager.shutdown)


@contextmanager
def browser_launch_slot() -> Iterator[None]:
    """
    Context manager that reserves a browser *launch* slot.

    Only limits the launch process itself. Once the browser is launched,
    the slot is released immediately - overall concurrent-session
    throttling now happens one layer up, inside run_playwright_async(), via
    BrowserManager's bounded executor, which holds its slot until the
    session (including the browser's close) actually completes.

    Kept as a standalone function with the same name and signature as
    before purely for compatibility with existing call sites; it now just
    delegates to the shared browser_manager.
    """
    with browser_manager.launch_slot():
        yield


# ===========================================================================
# Storage Directory
# ===========================================================================

def get_storage_dir() -> Path:
    """
    Creates (if necessary) and returns the Playwright storage directory.
    """
    path = Path("downloads") / ".playwright"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ===========================================================================
# Comment Cleaning
# ===========================================================================

def normalize_comments(items: List[str]) -> List[str]:
    """
    Cleans a list of comments by:
    - Removing empty comments
    - Ignoring comments shorter than 3 characters
    - Removing comments containing URLs
    - Removing duplicate comments (case-insensitive)
    """
    cleaned = []
    seen = set()

    for item in items:
        text = (item or "").strip()

        # Skip empty or very short comments
        if not text or len(text) < 3:
            continue

        # Skip comments containing links
        if "http" in text.lower():
            continue

        lowered = text.lower()

        # Skip duplicate comments
        if lowered in seen:
            continue

        seen.add(lowered)
        cleaned.append(text)

    return cleaned


# ===========================================================================
# Social Profile Discovery Fallback
# ===========================================================================
# company_discovery.py finds real handles by scanning the company's own
# homepage for links to twitter.com/instagram.com/youtube.com. When that
# scan comes up empty, twitter_scraper.py / instagram_scraper.py /
# youtube_scraper.py previously fell back to slugifying the company name
# into a guessed handle (e.g. "Headphone Zone" -> "@headphone_zone"). That
# guess is frequently wrong and fails *silently* - the scraper just
# returns zero results with no indication the handle itself was bad. This
# helper replaces that guess with an actual (lightweight) search, so a
# missing homepage link no longer means "give up and guess" - it means
# "go find the real one first, and only return [] if that also comes up
# empty."
#
# Uses DuckDuckGo's plain-HTML search endpoint rather than Google: Google
# blocks bare httpx requests without JS, while DuckDuckGo's /html/ endpoint
# is designed to be scraped without a browser. This keeps the fallback
# cheap (one plain GET) instead of needing a full Playwright page just to
# resolve a handle.

_SEARCH_TIMEOUT_SECONDS = 4.0

_PLATFORM_SEARCH_DOMAIN = {
    "twitter": "x.com",
    "instagram": "instagram.com",
    "youtube": "youtube.com",
}

_SEARCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}


def find_social_profile_url(company_name: str, platform: str) -> str:
    """Best-effort discovery of a company's real profile URL on
    ``platform`` via a plain-HTML web search. Intended as a fallback ONLY
    when the company's own homepage didn't link to that platform.

    Returns "" on any failure (network error, no match, unsupported
    platform) - never raises. Callers should treat "" exactly like "not
    found", not like an error worth logging loudly.
    """
    company_name = (company_name or "").strip()
    domain = _PLATFORM_SEARCH_DOMAIN.get(platform)
    if not company_name or not domain:
        return ""

    query = f'site:{domain} "{company_name}"'
    search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

    try:
        with httpx.Client(follow_redirects=True, headers=_SEARCH_HEADERS) as client:
            resp = client.get(search_url, timeout=_SEARCH_TIMEOUT_SECONDS)
        if resp.status_code >= 400:
            return ""

        # DuckDuckGo's HTML results wrap outbound links in a redirect
        # (//duckduckgo.com/l/?uddg=<url-encoded target>&...); pull the
        # first one that actually points at the target platform.
        for raw in re.findall(
            r'href="(https?://duckduckgo\.com/l/\?uddg=[^"]+)"', resp.text
        ):
            target = unquote(raw.split("uddg=", 1)[1].split("&", 1)[0])
            if domain in target.lower():
                return target

        # Fallback: some result rows link straight to the platform
        # without going through the redirect wrapper.
        for raw in re.findall(
            r'href="(https?://[^"]*' + re.escape(domain) + r'[^"]*)"', resp.text
        ):
            return unquote(raw)
    except Exception:
        return ""

    return ""


# ===========================================================================
# Playwright Execution
# ===========================================================================

def _run_playwright_in_fresh_loop(coro_factory: Callable[[], Awaitable[_T]]) -> _T:
    """
    Runs a Playwright coroutine inside a brand-new event loop.

    This avoids conflicts when running Playwright from worker threads,
    especially on Windows.
    """

    # Use Windows-compatible event loop policy if required
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # Create and set a fresh event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        return loop.run_until_complete(coro_factory())
    finally:
        loop.close()


async def run_playwright_async(coro_factory: Callable[[], Awaitable[_T]]) -> _T:
    """
    Executes a Playwright coroutine inside a separate worker thread.

    This keeps the main asyncio event loop responsive while Playwright
    performs browser automation, and now also bounds how many such
    sessions may run concurrently application-wide (see BrowserManager).

    Same name and signature as before, and same observable contract from a
    caller's point of view (runs coro_factory to completion in a dedicated
    thread + fresh event loop, returns its result, propagates its
    exceptions). The only internal difference: this now runs on a shared,
    bounded, reused thread pool instead of creating and tearing down a
    brand new one-worker executor on every call.
    """
    return await browser_manager.run_session(coro_factory)
