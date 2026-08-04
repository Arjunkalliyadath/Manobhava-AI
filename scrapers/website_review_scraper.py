"""Website product-review scraper. **New module — Priority 1 source.**

============================================================================
Why this module exists
----------------------------------------------------------------------------
Every zero-comment run in the logs shares one trait: the four existing
scrapers (Google Maps, YouTube, Reddit, Twitter/Instagram) all go looking
for reviews on *someone else's* platform, which means fighting that
platform's anti-bot defenses, login walls, and DOM churn. But the single
most reliable place to find opinions about "boAt Airdopes 181 Pro" is the
boAt Airdopes 181 Pro product page itself — which app.py has *already*
fetched successfully, via plain httpx, with a 200 OK, during product
discovery (see the `INFO:httpx:HTTP Request: GET .../products/... 200 OK`
lines in the log). No browser, no anti-bot fight, no auth wall: that page
is public HTML served to any GET request.

The overwhelming majority of e-commerce sites (Shopify in particular, which
`boat-lifestyle.com`'s `/products/...` URL structure indicates) render
reviews through one of a handful of well-known apps, each of which either
(a) embeds the reviews as structured data directly in the page HTML, or
(b) fetches them client-side from that app's own public, unauthenticated
JSON API. This module checks for both, in order of reliability, using the
exact same plain-HTTP approach reddit_scraper.py already uses.

----------------------------------------------------------------------------
Anti-bot resilience (added — see root-cause note below)
----------------------------------------------------------------------------
In practice, Shopify storefronts sometimes answer a cold, cookie-less,
sparsely-headered `httpx` GET with `503` even though the exact same URL
returned `200` minutes earlier during product discovery. That is not a
broken URL — it's the storefront's edge/WAF scoring this client's request
fingerprint (missing session cookies, missing modern browser headers,
high request velocity with no referer chain) as synthetic and starting to
throttle it. Two layers of defense are used, in order, before giving up:

  1. Header/session hardening: rotate through several realistic, complete
     modern-browser header profiles (Chrome/Windows, Safari/macOS,
     Firefox/Windows — including sec-ch-ua / sec-fetch-* / accept-language),
     warm up a real session by visiting the site root first (so Shopify
     session cookies are picked up and a referer chain exists), and retry
     with a fresh header profile + short jittered backoff specifically on
     403/429/503 (bot-mitigation-shaped statuses), not on 404/410/etc.
  2. Playwright fallback: if hardened HTTP still comes back 503 for a
     specific product page, that single page (and only that page — no
     other scraper's behavior changes) is re-fetched with a real headless
     browser, which doesn't trip the same fingerprint heuristic. This runs
     through the shared BrowserManager in `scrapers.browser_utils`
     (`ensure_shared_browser`/`connect_over_cdp` to attach to the one
     shared Chromium process rather than launching a second one,
     `run_playwright_async` to run the session on the shared, bounded
     executor) — the same mechanism google_scraper.py / youtube_scraper.py
     / instagram_scraper.py / twitter_scraper.py use — imported lazily and
     defensively so a missing/renamed utility can never break this
     module's import; it just logs and skips the fallback instead.

Whichever way the HTML was obtained, it is fed through the exact same
extraction tiers below — nothing about the extraction logic itself changes
based on how the page was fetched.

----------------------------------------------------------------------------
Detection order (first hit wins, cheapest/most-structured first):
  1. schema.org JSON-LD (`Review` / `AggregateRating`) embedded in
     `<script type="application/ld+json">` — works regardless of which
     review app is installed, since most apps emit this for SEO.
  2. Shopify's own public `{product_url}.json` endpoint — rarely carries
     review text itself, but reliably yields the numeric Shopify product
     id, which is what Judge.me/Loox/Yotpo widgets are usually keyed off
     of. Used as a fallback id source when it can't be scraped out of the
     HTML (e.g. because the HTML we got was a bot-check interstitial, or
     the theme's markup doesn't match the regexes below).
  3. Known review-app JSON endpoints, called directly once the app's
     product/shop id is found on the page (or via the Shopify JSON id
     above):
       - Judge.me   (`judge.me/api/v1/widgets/...` — very common on Shopify)
       - Yotpo      (`api.yotpo.com/v1/widget/{app_key}/products/...json`)
       - Loox       (`loox.io/widget/reviews/product/{shop_id}/{product_id}`)
  4. Generic HTML fallback: heuristically-identified repeating review
     blocks (rating + reviewer + body) anywhere in the rendered page HTML,
     for custom/unrecognized review widgets.

Each of these is a plain GET/POST against a public API — nothing here
needs a browser, a login, or an API key, so (outside of the Playwright
fallback above, used only when hardened HTTP is blocked) it doesn't
touch the shared BrowserManager at all.

NOTE on tier 1 vs. Judge.me widget-click pagination: tiers 1-4 above are
"first hit wins" — whichever tier finds reviews first supplies the initial
review list, and the rest aren't tried. But JSON-LD (tier 1) is usually a
small SEO sample, not a review app's full corpus, and Judge.me is known to
emit JSON-LD itself — so on a genuinely Judge.me-powered page, JSON-LD can
win tier 1 with only a handful of reviews while Judge.me's own widget (with
many more behind pagination) never even gets tried at tiers 2-4. Whether
the widget-click pagination below (the only path past page 1 once the
tokenless API 401s) gets attempted is therefore driven by a Judge.me
fingerprint check on the raw HTML (`has_judgeme` / _SyncResult), NOT by
which tier supplied the current review list — so it still fires even when
JSON-LD (or Yotpo/Loox) won tier 1-3. See scrape_website_reviews().
----------------------------------------------------------------------------

Public function (same contract as every other scraper module, UNCHANGED):

    async def scrape_website_reviews(company_data: Dict[str, str]) -> List[str]

Reads from ``company_data``:
    product_url   - preferred; the specific product page discovered for
                     this job (product_extraction.py already puts this in
                     each product record's "url" field — app.py's job
                     builder just needs to forward it into company_data
                     under this key, exactly as it already forwards
                     product_name/product_brand).
    website        - fallback root domain if product_url is absent (used
                      only for the "General" company-wide job).
    product_name   - used to keep only reviews that plausibly discuss this
                      product when a website-wide fallback page is scraped.
============================================================================
"""

import asyncio
import concurrent.futures
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from scrapers.browser_utils import normalize_comments

logger = logging.getLogger(__name__)

MAX_WORKERS = 3
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=MAX_WORKERS, thread_name_prefix="website_review_scraper"
)

# Plain HTTP against the merchant's own site (and, at most, a couple of
# well-known review-app APIs) — no browser, so this stays a fairly tight
# budget; a handful of httpx calls (including header-profile retries)
# finishes in well under this even on a slow site. The Playwright fallback
# below has its own, separate, larger budget that only gets spent when
# hardened HTTP is actually blocked.
TIME_BUDGET_SECONDS = 30
PLAYWRIGHT_TIME_BUDGET_SECONDS = 20
MAX_REVIEWS = 30  # cut from 3000 - priority shifted to 3-min total time; genuine over volume
# (see WIDGET_PAGINATE_TIME_BUDGET_SECONDS / WIDGET_PAGINATE_MAX_CLICKS below and the
# nav-retry hardening in _run_playwright_widget_paginate_session), not this ceiling.
_MIN_WORDS = 4

# Pagination tuning for review-widget APIs (Judge.me, Yotpo) that only
# return one page per request. This is a slice carved out of the overall
# TIME_BUDGET_SECONDS above — kept a bit below it so there's still room
# left for the root-warmup GET, the main page GET, and the Shopify
# product-id lookup that all happen before pagination starts.
# NOTE: this API-pagination path is dead weight on shops where Judge.me's
# public widget API 401s (see _extract_judgeme_dom's docstring) — it's
# kept as a fallback for shops where it does still work, but the real
# fix for those (the common case) is the Playwright widget-click
# pagination below.
_JUDGEME_MAX_PAGES = 10
_JUDGEME_PAGE_TIME_BUDGET_SECONDS = 18.0
_YOTPO_MAX_PAGES = 10
_YOTPO_PAGE_TIME_BUDGET_SECONDS = 18.0

# Real fix for "only 4-6 reviews when the product has hundreds": Judge.me
# (and most review widgets) only server-render their FIRST page of
# reviews into the product HTML. Getting the rest requires an actual
# browser clicking through the widget's own pagination/"load more"
# control — there's no working unauthenticated API for it on shops where
# the token-less endpoint 401s. This is a generous budget on purpose,
# per explicit request to scrape as many genuine reviews as a product
# actually has rather than stopping early; see WEBSITE_REVIEW_TIMEOUT_SECONDS
# in app.py, which was raised in lockstep to give this room to run.
# Raised again (90s->400s, 150->400 clicks): a product with 1000+ reviews
# needs several hundred click+wait round trips at ~10-20 reviews per click,
# which simply doesn't fit in 90s regardless of how reliable each click is.
WIDGET_PAGINATE_TIME_BUDGET_SECONDS = 30.0  # cut from 400 - priority shifted to 3-min total time; MAX_REVIEWS is now 30, so pagination should rarely need this long.
WIDGET_PAGINATE_MAX_CLICKS = 15  # cut from 400 - matches the lower MAX_REVIEWS cap.

# Outer safety-net allowance for everything in _run_playwright_widget_paginate_session
# BEFORE the pagination clock starts: up to two nav attempts (12000ms + 15000ms
# = 27s worst case — see the comment on that retry loop), up to 5s waiting for
# the review selector, plus ensure_shared_browser()/connect_over_cdp()/
# new_context()/new_page() overhead (previously chromium.launch() instead of
# connect_over_cdp() — see the fix note on _run_playwright_widget_paginate_session
# above. connect_over_cdp() to an already-running shared browser is normally
# much cheaper than a fresh launch, but new_context()/new_page() still cost
# real time under contention from the other scrapers also attached to the
# same shared browser, so this constant is left unchanged rather than
# shrunk without a live log to confirm the new margin is still safe).
# Added when the pagination clock was moved to start AFTER setup instead of
# before it (previously setup silently ate into WIDGET_PAGINATE_TIME_BUDGET_SECONDS
# itself, which is what let pagination add "0 additional reviews" even when
# nothing was actually wrong with the click-through logic). This is a ceiling
# on setup time only, kept separate from WIDGET_PAGINATE_TIME_BUDGET_SECONDS so
# the two can be tuned independently against live-log evidence, same reasoning
# as MAX_COMMENTS_PER_PRODUCT vs the per-platform scrape caps in config.py.
WIDGET_PAGINATE_SETUP_ALLOWANCE_SECONDS = 35.0
WIDGET_PAGINATE_IDLE_LIMIT = 5  # consecutive clicks with zero new reviews = assume done
# (raised from 3 -> 5: a couple of transient re-render misses shouldn't look
# identical to "genuinely out of pages" when the goal is exhausting a large
# widget, not stopping at the first hint of a stall)

# Statuses shaped like bot-mitigation (as opposed to "this page genuinely
# doesn't exist") — worth retrying with a different browser fingerprint.
_RETRYABLE_STATUSES = {403, 429, 503}

# Several complete, realistic modern-browser header profiles. Rotated on
# 403/429/503 so a single incomplete fingerprint doesn't sink every
# attempt — a plain User-Agent with no sec-ch-ua/sec-fetch-*/accept-language
# is itself a signal Shopify's edge can key off of.
_HEADER_PROFILES: List[Dict[str, str]] = [
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        # NOTE: no Accept-Encoding here (or in the other profiles). Hardcoding
        # "gzip, deflate, br" made servers respond with brotli, which this
        # venv's httpx cannot decode (no brotli package installed) — so every
        # extraction tier ran against raw compressed bytes and found nothing.
        # Leaving the header out lets httpx advertise exactly what it can
        # actually decode.
        "Sec-Ch-Ua": '"Chromium";v="126", "Google Chrome";v="126", "Not-A.Brand";v="99"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
    },
    {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
            "(KHTML, like Gecko) Version/17.4 Safari/605.1.15"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Upgrade-Insecure-Requests": "1",
    },
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) "
            "Gecko/20100101 Firefox/127.0"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.5",
        "Upgrade-Insecure-Requests": "1",
    },
]

# Simple, single header set used for calls to review-app APIs (Judge.me /
# Yotpo / Loox / Shopify's own .json endpoint) — those aren't fronted by
# the same anti-bot heuristics as the merchant's rendered storefront pages,
# so they don't need the full rotation/retry treatment.
_API_HEADERS = dict(_HEADER_PROFILES[0])


@dataclass
class _Review:
    text: str
    source: str  # e.g. "jsonld", "judgeme", "yotpo", "loox", "html_heuristic"


@dataclass
class _SyncResult:
    reviews: List[_Review] = field(default_factory=list)
    needs_playwright: bool = False
    target: str = ""
    shop_domain: str = ""
    shopify_product_id: Optional[str] = None
    # Whether the fetched HTML fingerprints as Judge.me ("jdgm"/"judge.me"
    # anywhere in the page), independent of which extraction tier actually
    # supplied `reviews`. See scrape_website_reviews()'s widget-pagination
    # trigger for why this has to be tracked separately from reviews[0].source
    # — a JSON-LD tier win must not hide the fact that Judge.me is also
    # installed and click-through pagination is worth attempting.
    has_judgeme: bool = False


# --- HTTP helpers --------------------------------------------------------
def _get_simple(client: httpx.Client, url: str, **kwargs) -> Optional[httpx.Response]:
    """Single-attempt GET with a plain realistic header set — used for the
    unauthenticated review-app / Shopify JSON APIs, which aren't subject to
    the storefront's anti-bot fingerprinting."""
    try:
        resp = client.get(url, headers=_API_HEADERS, timeout=8.0, **kwargs)
        if resp.status_code >= 400:
            logger.info("Website review scrape: GET %s -> HTTP %d", url, resp.status_code)
            return None
        return resp
    except Exception as exc:
        logger.info("Website review scrape: GET %s failed: %s", url, exc)
        return None


def _get_with_profiles(
    client: httpx.Client, url: str, referer: Optional[str] = None
) -> "tuple[Optional[httpx.Response], Optional[int]]":
    """GET a merchant storefront page, rotating through realistic browser
    header profiles and retrying with a fresh one specifically on
    403/429/503 (bot-mitigation-shaped statuses). Returns (response, status)
    — response is None on failure, status carries the last HTTP status seen
    (so the caller can tell a 503 apart from e.g. a 404, and decide whether
    a Playwright fallback is worth attempting)."""
    last_status: Optional[int] = None
    for i, profile in enumerate(_HEADER_PROFILES):
        headers = dict(profile)
        if referer:
            headers["Referer"] = referer
        try:
            resp = client.get(url, headers=headers, timeout=8.0)
        except Exception as exc:
            logger.info(
                "Website review scrape: GET %s failed (header profile %d/%d): %s",
                url, i + 1, len(_HEADER_PROFILES), exc,
            )
            last_status = None
            continue

        if resp.status_code < 400:
            return resp, resp.status_code

        last_status = resp.status_code
        logger.info(
            "Website review scrape: GET %s -> HTTP %d (header profile %d/%d)",
            url, resp.status_code, i + 1, len(_HEADER_PROFILES),
        )
        if resp.status_code not in _RETRYABLE_STATUSES:
            # A 404/410/etc. won't be fixed by a different fingerprint —
            # no point burning the remaining profiles on it.
            break
        if i < len(_HEADER_PROFILES) - 1:
            time.sleep(0.35 + random.uniform(0.15, 0.55))

    return None, last_status


# --- 1. JSON-LD review schema -------------------------------------------
def _extract_jsonld_reviews(html: str) -> List[_Review]:
    out: List[_Review] = []
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = tag.string or tag.get_text() or ""
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        candidates = data if isinstance(data, list) else [data]
        for node in candidates:
            if not isinstance(node, dict):
                continue
            graph = node.get("@graph")
            nodes = graph if isinstance(graph, list) else [node]
            for n in nodes:
                if not isinstance(n, dict):
                    continue
                reviews = n.get("review") or n.get("reviews") or []
                if isinstance(reviews, dict):
                    reviews = [reviews]
                for r in reviews:
                    if not isinstance(r, dict):
                        continue
                    body = (
                        r.get("reviewBody")
                        or r.get("description")
                        or ""
                    )
                    body = body.strip() if isinstance(body, str) else ""
                    if body and len(body.split()) >= _MIN_WORDS:
                        out.append(_Review(text=body, source="jsonld"))
    return out


# --- 2. Shopify's own public product JSON (id-resolution helper) --------
def _shopify_json_url(target: str) -> str:
    parsed = urlparse(target)
    path = parsed.path.rstrip("/")
    if path.endswith(".json"):
        return target
    return f"{parsed.scheme}://{parsed.netloc}{path}.json"


def _extract_shopify_product_json_id(client: httpx.Client, target: str) -> Optional[str]:
    """Every Shopify product page has a public, unauthenticated JSON
    twin at the same path with `.json` appended. It rarely contains review
    text (that lives in the review app's own storage), but it reliably
    gives us the numeric Shopify product id — which Judge.me and Loox key
    their widgets off of directly. Used as a fallback id source when the
    id can't be scraped out of the HTML (unfamiliar theme markup, or the
    HTML we have is a bot-check interstitial rather than the real page)."""
    url = _shopify_json_url(target)
    resp = _get_simple(client, url)
    if resp is None:
        return None
    try:
        data = resp.json()
        product_id = (data.get("product") or {}).get("id")
        return str(product_id) if product_id else None
    except Exception:
        return None


# --- 3a. Judge.me --------------------------------------------------------
_JUDGEME_ID_RE = re.compile(r"data-product-id=[\"'](\d+)[\"']|jdgm-widget[^>]*data-id=[\"'](\d+)[\"']")

# Judge.me's public widget API is keyed to the store's *.myshopify.com
# domain, not whatever custom domain the storefront is actually served
# from (e.g. headphonezone.in) — a mismatch here silently returns an
# empty review list even when Judge.me is genuinely installed with real
# reviews (confirmed by jdgm-* classes in the DOM). Shopify themes almost
# always leak this domain somewhere in the page (Shopify.shop, a
# checkout/asset host, etc.), so this recovers it without needing a
# browser.
_MYSHOPIFY_DOMAIN_RE = re.compile(r"([a-z0-9][a-z0-9\-]*\.myshopify\.com)", re.IGNORECASE)


def _extract_myshopify_domain(html: str) -> Optional[str]:
    m = _MYSHOPIFY_DOMAIN_RE.search(html)
    return m.group(1).lower() if m else None


def _extract_judgeme_dom(html: str) -> List[_Review]:
    """Server-rendered Judge.me reviews, straight out of the product page
    HTML. Judge.me's tokenless public widget API now answers 401
    ("Failed to authenticate. Shop domain or Api Token is wrong" —
    verified live against boat-lifestyle.com), so the API tier below can
    no longer be the primary path. Fortunately most storefronts render
    the first page of reviews directly into the product HTML as
    .jdgm-rev__body elements, which this reads without any extra request."""
    if "jdgm" not in html and "judge.me" not in html:
        return []
    out: List[_Review] = []
    try:
        soup = BeautifulSoup(html, "html.parser")
        for body in soup.select(".jdgm-rev__body, .jdgm-rev__text"):
            text = body.get_text(" ", strip=True)
            if text and len(text.split()) >= _MIN_WORDS:
                out.append(_Review(text=text, source="judgeme"))
    except Exception:
        pass
    return out


def _extract_judgeme(
    client: httpx.Client,
    html: str,
    shop_domains: List[str],
    fallback_product_id: Optional[str] = None,
) -> List[_Review]:
    dom_reviews = _extract_judgeme_dom(html)
    if dom_reviews:
        return dom_reviews

    m = _JUDGEME_ID_RE.search(html)
    product_id = (m.group(1) or m.group(2)) if m else None
    has_fingerprint = "judge.me" in html or "jdgm" in html
    if not product_id:
        if not has_fingerprint and not fallback_product_id:
            return []
        product_id = fallback_product_id
    if not product_id:
        return []

    # FIX: this used to fetch a single hardcoded &page=1 and stop — so a
    # product with hundreds of real Judge.me reviews only ever yielded
    # whatever fit on page 1 (per_page=50, but the widget endpoint often
    # returns far fewer usable review-text nodes per page after filtering
    # ratings-only/empty entries). Now paginate up to _JUDGEME_MAX_PAGES or
    # MAX_REVIEWS, whichever comes first, stopping early on an empty page
    # (no more reviews) or once a per-domain time slice runs out — so a
    # popular product's review count is actually reflected instead of
    # being capped at "whatever page 1 happened to contain."
    for shop_domain in shop_domains:
        if not shop_domain:
            continue
        out: List[_Review] = []
        seen_texts: set = set()
        deadline = time.monotonic() + _JUDGEME_PAGE_TIME_BUDGET_SECONDS
        page = 1
        while (
            page <= _JUDGEME_MAX_PAGES
            and len(out) < MAX_REVIEWS
            and time.monotonic() < deadline
        ):
            url = (
                "https://judge.me/api/v1/widgets/product_review"
                f"?shop_domain={shop_domain}&platform=shopify&product_id={product_id}"
                f"&per_page=50&page={page}"
            )
            resp = _get_simple(client, url)
            if resp is None:
                break
            page_reviews: List[_Review] = []
            try:
                soup = BeautifulSoup(resp.text, "html.parser")
                for body in soup.select(".jdgm-rev__body, .jdgm-rev__text"):
                    text = body.get_text(" ", strip=True)
                    if text and len(text.split()) >= _MIN_WORDS and text not in seen_texts:
                        seen_texts.add(text)
                        page_reviews.append(_Review(text=text, source="judgeme"))
            except Exception:
                pass
            if not page_reviews:
                # Empty page = this shop_domain's widget has no more
                # reviews to give (or doesn't work at all) — stop
                # paginating it and try the next candidate domain, if any.
                break
            out.extend(page_reviews)
            page += 1
        if out:
            logger.info(
                "Website review scrape: judgeme paginated %d page(s), %d review(s) for shop_domain=%s.",
                page - 1, len(out), shop_domain,
            )
            return out[:MAX_REVIEWS]
    return []

# --- 3b. Yotpo ------------------------------------------------------------
_YOTPO_APPKEY_RE = re.compile(r"yotpo[_-]?app[_-]?key[\"'\s:=]+[\"']?([A-Za-z0-9]{6,})", re.IGNORECASE)
_YOTPO_PRODUCT_ID_RE = re.compile(r"data-product-id=[\"'](\w[\w-]*)[\"']")


def _extract_yotpo(
    client: httpx.Client, html: str, fallback_product_id: Optional[str] = None
) -> List[_Review]:
    if "yotpo" not in html.lower():
        return []
    key_match = _YOTPO_APPKEY_RE.search(html)
    if not key_match:
        # No app key visible anywhere = no way to call the API; a fallback
        # product id can't compensate for a missing key.
        return []
    pid_match = _YOTPO_PRODUCT_ID_RE.search(html)
    product_id = (pid_match.group(1) if pid_match else None) or fallback_product_id
    if not product_id:
        return []
    app_key = key_match.group(1)

    # FIX: this used to fetch a single unpaginated request and stop,
    # capping output at whatever Yotpo's default single-page response
    # contained. Paginate the same way as Judge.me above, using Yotpo's
    # documented `page`/`per_page` widget-API query params.
    out: List[_Review] = []
    seen_texts: set = set()
    deadline = time.monotonic() + _YOTPO_PAGE_TIME_BUDGET_SECONDS
    page = 1
    while (
        page <= _YOTPO_MAX_PAGES
        and len(out) < MAX_REVIEWS
        and time.monotonic() < deadline
    ):
        url = (
            f"https://api.yotpo.com/v1/widget/{app_key}/products/{product_id}/reviews.json"
            f"?page={page}&per_page=50"
        )
        resp = _get_simple(client, url)
        if resp is None:
            break
        page_reviews: List[_Review] = []
        try:
            data = resp.json()
            reviews = (((data or {}).get("response") or {}).get("reviews")) or []
            for r in reviews:
                text = (r.get("content") or "").strip()
                if text and len(text.split()) >= _MIN_WORDS and text not in seen_texts:
                    seen_texts.add(text)
                    page_reviews.append(_Review(text=text, source="yotpo"))
        except Exception:
            break
        if not page_reviews:
            break
        out.extend(page_reviews)
        page += 1
    if out:
        logger.info(
            "Website review scrape: yotpo paginated %d page(s), %d review(s).",
            page - 1, len(out),
        )
    return out


# --- 3c. Loox --------------------------------------------------------------
_LOOX_SHOP_RE = re.compile(r"loox[_-]?shop[_-]?id[\"'\s:=]+[\"']?(\d+)")


def _extract_loox(
    client: httpx.Client, html: str, url_slug: str, fallback_product_id: Optional[str] = None
) -> List[_Review]:
    if "loox" not in html.lower():
        return []
    shop_match = _LOOX_SHOP_RE.search(html)
    if not shop_match:
        return []
    shop_id = shop_match.group(1)
    # Loox's widget API expects the numeric Shopify product id, not the URL
    # slug — prefer the Shopify-JSON-resolved id when we have it and only
    # fall back to the slug (which will usually just 404) as a last resort.
    product_ref = fallback_product_id or url_slug
    url = f"https://loox.io/widget/reviews/product/{shop_id}/{product_ref}?fields=all"
    resp = _get_simple(client, url)
    if resp is None:
        return []
    out: List[_Review] = []
    try:
        data = resp.json()
        reviews = data.get("reviews") or []
        for r in reviews:
            text = (r.get("content") or r.get("text") or "").strip()
            if text and len(text.split()) >= _MIN_WORDS:
                out.append(_Review(text=text, source="loox"))
    except Exception:
        pass
    return out


# --- 4. Generic HTML heuristic fallback ------------------------------------
# NOTE: _REVIEW_BLOCK_HINTS is kept only because it is a public-ish module
# constant that other code may still import; Tier 4 candidate discovery
# below no longer reads it (see _find_repeated_sibling_blocks / _tag_shape).
_REVIEW_BLOCK_HINTS = [
    "review", "rating", "testimonial", "feedback", "comment",
    "jdgm", "yotpo", "loox", "oke-", "okendo", "stamped", "rivyo",
    "reviewsio", "trustpilot", "junip", "ali-reviews",
]
_NOISE_RE = re.compile(
    r"^(add to cart|buy now|write a review|sort by|filter|load more|"
    r"was this helpful|verified purchase|share|helpful)\W*$",
    re.IGNORECASE,
)

# Structural candidate discovery (replaces class/id keyword matching below).
# Modern React/Next.js builds ship hashed, meaningless CSS classes, so class
# or id text can no longer be trusted to mean anything. What survives that
# is HTML *shape*: a list of reviews is rendered as one parent element with
# several children that share (nearly) the same tag hierarchy, e.g.
# <ul><li><div><span/><p/></div></li> x N</ul>. We detect that shape
# directly instead of guessing at naming conventions.
_MIN_SIBLING_REPEATS = 3  # how many structurally-identical siblings = "repeated"
_STRUCTURE_MAX_DEPTH = 2  # how deep into each child's own children we compare
# Non-content tags that can legitimately repeat (e.g. several <meta> or
# <script> tags) but are never review blocks; excluded so they can't be
# mistaken for a repeated review list purely by tag-hierarchy coincidence.
_STRUCTURAL_SKIP_TAGS = {"script", "style", "noscript", "template", "head", "meta", "link"}


def _tag_shape(tag, max_depth: int = _STRUCTURE_MAX_DEPTH):
    """Describe a tag purely by its HTML tag-name hierarchy (own tag name
    plus its children's shapes, recursed a couple of levels deep) — no
    class, id, or attribute of any kind is inspected. Two elements with the
    same shape have a (nearly) identical tag hierarchy even when their CSS
    classes are hashed/auto-generated and share nothing in common."""
    if tag is None or not getattr(tag, "name", None):
        return None
    if max_depth <= 0:
        return tag.name
    children = tag.find_all(True, recursive=False)
    return (tag.name, tuple(_tag_shape(c, max_depth - 1) for c in children))


def _find_repeated_sibling_blocks(soup):
    """Tier 4 candidate discovery: find parent elements that contain
    several direct children with the same structural shape (_tag_shape),
    and return those children as review-block candidates. This is the
    entire replacement for the old class/id keyword search — no CSS class
    name, id, or framework-specific selector is used anywhere in here."""
    candidates = []
    seen_ids = set()
    for parent in soup.find_all(True):
        children = [
            c for c in parent.find_all(True, recursive=False)
            if c.name not in _STRUCTURAL_SKIP_TAGS
        ]
        if len(children) < _MIN_SIBLING_REPEATS:
            continue
        groups = {}
        for child in children:
            groups.setdefault(_tag_shape(child), []).append(child)
        for shape, group in groups.items():
            if shape is None or len(group) < _MIN_SIBLING_REPEATS:
                continue
            for el in group:
                if id(el) not in seen_ids:
                    seen_ids.add(id(el))
                    candidates.append(el)
    return candidates


def _extract_html_heuristic(html: str) -> List[_Review]:
    """Last resort when no known review app is detected: instead of
    matching class/id keywords (unreliable on hashed-class React/Next.js
    sites), find repeated sibling blocks with matching tag structure via
    _find_repeated_sibling_blocks(), then keep the ones that read like
    actual prose rather than UI chrome. Deliberately conservative (min
    word count, noise-phrase filter) since this has no structured signal
    to lean on."""
    out: List[_Review] = []
    soup = BeautifulSoup(html, "html.parser")
    seen = set()
    candidates = _find_repeated_sibling_blocks(soup)
    for tag in candidates[:300]:
        text = tag.get_text(" ", strip=True)
        if not text or len(text.split()) < 6 or len(text) > 1200:
            continue
        if _NOISE_RE.match(text):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(_Review(text=text, source="html_heuristic"))
        if len(out) >= MAX_REVIEWS:
            break
    return out


# --- shared extraction pipeline -------------------------------------------
def _shop_domain_from_url(url: str) -> str:
    netloc = urlparse(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def _extract_all_tiers(
    html: str, target: str, shop_domain: str, shopify_product_id: Optional[str]
) -> List[_Review]:
    """Runs the full detection-tier pipeline against a page's HTML. Used
    identically whether that HTML came from plain httpx or from the
    Playwright fallback — extraction logic never needs to know or care
    which one fetched it."""
    collected = _extract_jsonld_reviews(html)
    if collected:
        logger.info(
            "Website review scrape: %d review(s) via JSON-LD for %s.",
            len(collected), target,
        )
        return collected[:MAX_REVIEWS]

    with httpx.Client(follow_redirects=True) as client:
        url_slug = urlparse(target).path.rstrip("/").rsplit("/", 1)[-1]
        myshopify_domain = _extract_myshopify_domain(html)
        judgeme_domains = [d for d in (shop_domain, myshopify_domain) if d]
        for extractor, args in (
            (_extract_judgeme, (client, html, judgeme_domains, shopify_product_id)),
            (_extract_yotpo, (client, html, shopify_product_id)),
            (_extract_loox, (client, html, url_slug, shopify_product_id)),
        ):
            try:
                found = extractor(*args)
            except Exception:
                logger.exception(
                    "Website review scrape: extractor %s failed for %s.",
                    extractor.__name__, target,
                )
                found = []
            if found:
                logger.info(
                    "Website review scrape: %d review(s) via %s for %s.",
                    len(found), found[0].source, target,
                )
                return found[:MAX_REVIEWS]

    collected = _extract_html_heuristic(html)
    if collected:
        logger.info(
            "Website review scrape: %d review(s) via HTML heuristic for %s.",
            len(collected), target,
        )
        return collected[:MAX_REVIEWS]

    logger.info(
        "Website review scrape: no reviews found on %s (no known review app detected).",
        target,
    )
    return []


# --- sync (httpx) phase — runs in the thread pool, unchanged threading model
def _scrape_sync(product_url: str, website: str, product_name: str) -> _SyncResult:
    target = product_url or website
    if not target:
        return _SyncResult(target=target)

    shop_domain = _shop_domain_from_url(target)
    parsed = urlparse(target)
    root = f"{parsed.scheme}://{parsed.netloc}/"

    with httpx.Client(follow_redirects=True) as client:
        # Session warm-up: a real navigation to this product page would have
        # picked up Shopify session cookies and carried a Referer from an
        # earlier page view. A bare cookie-less direct hit to /products/...
        # looks synthetic; visiting the root first (best-effort — failure
        # here is not fatal) closes that gap cheaply.
        if root != target:
            _get_with_profiles(client, root)

        resp, status = _get_with_profiles(client, target, referer=root)
        shopify_product_id = _extract_shopify_product_json_id(client, target)

        if resp is not None:
            html = resp.text
            reviews = _extract_all_tiers(html, target, shop_domain, shopify_product_id)
            return _SyncResult(
                reviews=reviews, needs_playwright=False, target=target,
                shop_domain=shop_domain, shopify_product_id=shopify_product_id,
                has_judgeme=("jdgm" in html or "judge.me" in html),
            )

        if status is None:
            logger.info("Website review scrape: could not fetch %r.", target)
        needs_playwright = status == 503
        return _SyncResult(
            reviews=[], needs_playwright=needs_playwright, target=target,
            shop_domain=shop_domain, shopify_product_id=shopify_product_id,
        )


# --- Playwright fallback (async — the session itself runs on the shared,
# bounded BrowserManager executor via run_playwright_async(), exactly like
# google_scraper.py / youtube_scraper.py / instagram_scraper.py /
# twitter_scraper.py; only launch pacing + submission happen here) -------
async def _goto_and_extract_html(page, url: str, nav_timeout_ms: int = 9000) -> Optional[str]:
    for attempt in (1, 2):
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=nav_timeout_ms)
            try:
                await page.wait_for_load_state("networkidle", timeout=4000)
            except Exception:
                pass  # best-effort settle; domcontentloaded content is enough if this times out
            return await page.content()
        except Exception as exc:
            logger.warning(
                "Website review scrape: Playwright nav attempt %d/2 to %s failed: %s",
                attempt, url, exc,
            )
            if attempt == 2:
                return None
            await asyncio.sleep(1.0)
    return None


async def _connect_shared_browser_or_fallback(pw) -> Tuple[Any, bool]:
    """Attach to the shared Chromium process via CDP; if that specific
    call fails, fall back to launching a private, non-shared browser for
    just this one session instead of giving up entirely.

    Added after a production log (2026-08-03) showed this exact failure:
    BrowserType.connect_over_cdp raising "Connection closed while reading
    from the driver", immediately preceded by the underlying Playwright
    Node.js driver process crashing on an internal assertion
    (CRBrowser._onAttachedToTarget) while processing an already-open
    shared_worker target from a concurrently-scraped x.com page. That's a
    real bug in Playwright's own CDP handling for that target type, not
    something fixable from this side - before this file shared a browser
    with anything else, it could never encounter another scraper's targets
    at all, so this crash was never reachable here. Retrying the same
    connect_over_cdp call once is cheap and occasionally enough (the
    crash's timing looked race-y, not deterministic, in the one log seen
    so far - this hasn't been observed enough times yet to be certain it
    always reproduces). If it fails twice, fall back to a private launch()
    so pagination still gets a browser to work with, accepting the
    resource-contention cost this file's shared-browser fix was written to
    avoid, only for this one already-failing session rather than always.

    Returns (browser, was_shared) - was_shared is only for logging; the
    existing `browser.close()` cleanup already used by both callers is
    correct either way (it safely disconnects a CDP-obtained handle without
    killing the shared process, and correctly terminates a privately
    launched one).
    """
    from scrapers.browser_utils import browser_manager

    cdp_endpoint = browser_manager.ensure_shared_browser()
    try:
        return await pw.chromium.connect_over_cdp(cdp_endpoint), True
    except Exception as e1:
        logger.warning(
            "Website review scrape: shared-browser connect_over_cdp failed "
            "(%s); retrying once before falling back.", e1,
        )
        try:
            return await pw.chromium.connect_over_cdp(cdp_endpoint), True
        except Exception as e2:
            logger.warning(
                "Website review scrape: shared-browser connect_over_cdp "
                "failed twice (%s); falling back to a private browser for "
                "this session only.", e2,
            )
            return await pw.chromium.launch(headless=True), False


async def _run_playwright_fetch_session(url: str) -> Optional[str]:
    """One full Playwright session — connect, new context/page, fetch,
    close — handed to browser_utils.run_playwright_async() as the
    coro_factory. That function runs this on the shared BrowserManager's
    bounded executor (the same pool the other scrapers' Playwright work
    runs on) for session pacing. As of this fix, the session also attaches
    to the single shared Chromium process via
    browser_manager.ensure_shared_browser() + connect_over_cdp() instead of
    launching a second, separate Chromium process — the same pattern
    google_scraper.py / twitter_scraper.py / instagram_scraper.py /
    youtube_scraper.py already use (adapted here for the async Playwright
    API, since those four drive the sync API internally). This was the
    last of the six scrapers still spawning its own browser process; on
    this project's 8GB-RAM, no-GPU hardware, a second Chromium process
    competing with the shared one is exactly the kind of contention the
    2026-07-30 log below traced navigation failures to.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        logger.warning(
            "Website review scrape: Playwright is not installed; cannot fall back for %s.", url
        )
        return None

    async with async_playwright() as pw:
        browser, _ = await _connect_shared_browser_or_fallback(pw)
        try:
            context = await browser.new_context(
                user_agent=_HEADER_PROFILES[0]["User-Agent"],
                locale="en-US",
            )
            try:
                page = await context.new_page()
                return await _goto_and_extract_html(page, url)
            finally:
                await context.close()
        finally:
            await browser.close()


async def _fetch_html_via_playwright(url: str) -> Optional[str]:
    """Best-effort Playwright fallback, dispatched through the shared
    BrowserManager (scrapers.browser_utils.run_playwright_async) so its
    launch pacing and concurrent-session cap are shared with every other
    scraper instead of this module rolling its own throwaway browser +
    semaphore. Imported lazily (never at module load) and wrapped
    defensively so a missing/renamed browser_utils utility can never break
    this module's import — it just logs and skips the fallback instead.
    """
    try:
        from scrapers.browser_utils import run_playwright_async
    except ImportError:
        logger.warning(
            "Website review scrape: shared browser_utils.run_playwright_async "
            "is unavailable; skipping Playwright fallback for %s.", url,
        )
        return None

    try:
        return await run_playwright_async(lambda: _run_playwright_fetch_session(url))
    except Exception:
        logger.exception(
            "Website review scrape: Playwright fallback failed for %s.", url
        )
        return None


# --- Judge.me widget click-through pagination ------------------------------
# The reason a product with hundreds of real reviews only ever yielded 4-6:
# Judge.me (like most review widgets) only server-renders its FIRST page of
# reviews into the product HTML — that's all _extract_judgeme_dom can ever
# see from a plain GET, no matter how the request is made. The widget's
# own public API is meant to serve the rest, but on shops where that API's
# token-less endpoint 401s (documented on _extract_judgeme_dom above),
# there is no unauthenticated way to ask for "page 2" over HTTP. The only
# remaining path is a real browser clicking through the widget's own
# pagination / "load more" control and re-reading the DOM after each
# click, which is what this function does.
_JUDGEME_REVIEW_TEXT_SELECTORS = ".jdgm-rev__body, .jdgm-rev__text"
_JUDGEME_NEXT_PAGE_SELECTORS = [
    "a.jdgm-paginate__next-page:not(.jdgm-paginate__page--disabled)",
    "a.jdgm-paginate__page.jdgm-paginate__page--next:not(.jdgm-paginate__page--disabled)",
    "button.jdgm-btn--load-more",
    ".jdgm-rev-widg__load-more-btn",
]


async def _paginate_judgeme_widget(page, time_left) -> List[_Review]:
    """Assumes `page` is already navigated to the product URL. Repeatedly
    clicks whatever "next page" / "load more" control the Judge.me widget
    exposes, re-reading review text out of the DOM after each click, until
    one of: MAX_REVIEWS reached, WIDGET_PAGINATE_MAX_CLICKS used up,
    WIDGET_PAGINATE_IDLE_LIMIT consecutive clicks add nothing new (the
    widget has no more pages), no clickable next/load-more control can be
    found at all, or time_left() runs out."""
    seen_texts: set = set()
    out: List[_Review] = []

    async def _extract_current() -> int:
        added = 0
        try:
            bodies = await page.query_selector_all(_JUDGEME_REVIEW_TEXT_SELECTORS)
        except Exception:
            return 0
        for body in bodies:
            try:
                text = (await body.inner_text() or "").strip()
            except Exception:
                continue
            if text and len(text.split()) >= _MIN_WORDS and text not in seen_texts:
                seen_texts.add(text)
                out.append(_Review(text=text, source="judgeme"))
                added += 1
        return added

    await _extract_current()  # whatever's already rendered on page 1

    idle_streak = 0
    clicks = 0
    while (
        clicks < WIDGET_PAGINATE_MAX_CLICKS
        and len(out) < MAX_REVIEWS
        and idle_streak < WIDGET_PAGINATE_IDLE_LIMIT
        and time_left() > 2
    ):
        clicked = False
        for selector in _JUDGEME_NEXT_PAGE_SELECTORS:
            try:
                control = await page.query_selector(selector)
                if control is None:
                    continue
                if not await control.is_visible():
                    continue
                await control.scroll_into_view_if_needed()
                await control.click(timeout=3000)
                clicked = True
                break
            except Exception:
                continue
        if not clicked:
            # One grace re-poll before concluding pagination is genuinely
            # exhausted — the widget's own JS sometimes re-renders the
            # next/load-more control a beat after the page settles, and a
            # single transient miss here shouldn't look identical to
            # "no more pages" when the whole point is exhausting a large
            # widget rather than stopping at the first stumble.
            try:
                await page.wait_for_timeout(1200)
            except Exception:
                pass
            for selector in _JUDGEME_NEXT_PAGE_SELECTORS:
                try:
                    control = await page.query_selector(selector)
                    if control is None:
                        continue
                    if not await control.is_visible():
                        continue
                    await control.scroll_into_view_if_needed()
                    await control.click(timeout=3000)
                    clicked = True
                    break
                except Exception:
                    continue
        if not clicked:
            logger.info(
                "Website review scrape: Judge.me widget pagination stopped — "
                "no next-page/load-more control found after %d click(s).",
                clicks,
            )
            break
        clicks += 1
        try:
            await page.wait_for_timeout(1300)
        except Exception:
            pass
        added = await _extract_current()
        if added == 0:
            idle_streak += 1
        else:
            idle_streak = 0

    logger.info(
        "Website review scrape: Judge.me widget pagination made %d click(s), "
        "collected %d total review(s).",
        clicks, len(out),
    )
    return out[:MAX_REVIEWS]


async def _run_playwright_widget_paginate_session(url: str) -> List[_Review]:
    """Same launch/context/page pattern as _run_playwright_fetch_session,
    but keeps the page open to click through pagination instead of just
    grabbing HTML once."""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        logger.warning(
            "Website review scrape: Playwright is not installed; cannot "
            "paginate the review widget for %s.", url,
        )
        return []

    # DIAGNOSTIC clock (separate from the pagination-budget start/time_left
    # below) — added because the 2026-07-30 log showed navigation failing on
    # BOTH attempts for all 6 Judge.me products tested (3 boAt, 3 Headphone
    # Zone), and there was no visibility at the time into whether that was
    # browser-launch contention, context/page setup overhead, or the
    # navigation calls themselves — all were lumped into one eventual
    # "Playwright nav ... failed after retry" warning. As of this session,
    # this function no longer launches its own Chromium process (see the
    # module-level fix note above _run_playwright_fetch_session) — it
    # attaches to the shared one via connect_over_cdp(), which is normally
    # near-instant since the shared browser is almost always already
    # running by the time this fires. These BROWSER_TRACE lines are kept
    # (renamed below) so a future run can still confirm that: if
    # after_shared_browser_connect now lands quickly and navigation still
    # fails, that points at something other than process contention (e.g.
    # bot detection); if it's still slow, the shared browser itself is
    # under load. Format matches google_scraper/twitter_scraper/
    # instagram_scraper/youtube_scraper's own BROWSER_TRACE lines, so all
    # five are directly comparable within the same run.
    _diag_start = time.monotonic()

    def _diag(event: str) -> None:
        logger.info(
            "BROWSER_TRACE scraper=WebsiteReview event=%s elapsed=%.3fs url=%s",
            event, time.monotonic() - _diag_start, url,
        )

    _diag("before_shared_browser_connect")

    async with async_playwright() as pw:
        browser, was_shared = await _connect_shared_browser_or_fallback(pw)
        _diag("after_shared_browser_connect" if was_shared else "after_fallback_private_launch")
        try:
            context = await browser.new_context(
                user_agent=_HEADER_PROFILES[0]["User-Agent"],
                locale="en-US",
            )
            _diag("after_new_context")
            try:
                page = await context.new_page()
                _diag("after_new_page")
                nav_ok = False
                last_nav_error: Optional[Exception] = None
                # Two attempts, second one more generous. A single 12s try
                # under heavy shared-Chromium contention (this session's own
                # log showed 4 other browser-based scrapers launching at the
                # same time) is exactly what dropped Airdopes-131's entire
                # pagination contribution in production - it fell back to
                # only its 4 page-1 reviews instead of the 11+ pagination
                # would have found.
                for attempt, nav_timeout_ms in enumerate((12000, 15000)):
                    _diag(f"before_nav_attempt_{attempt+1}")
                    try:
                        await page.goto(url, wait_until="domcontentloaded", timeout=nav_timeout_ms)
                        nav_ok = True
                        _diag(f"after_nav_attempt_{attempt+1}_ok")
                        break
                    except Exception as e:
                        last_nav_error = e
                        _diag(f"after_nav_attempt_{attempt+1}_failed")
                        continue
                if not nav_ok:
                    logger.warning(
                        "Website review scrape: Playwright nav for widget "
                        "pagination failed for %s after retry (%s).", url, last_nav_error,
                    )
                if not nav_ok:
                    return []
                try:
                    await page.wait_for_selector(
                        _JUDGEME_REVIEW_TEXT_SELECTORS, timeout=5000,
                    )
                    _diag("after_selector_wait_ok")
                except Exception:
                    _diag("after_selector_wait_timed_out")
                    pass  # best-effort — pagination loop still tries even if this times out

                # FIX: this clock used to start at function entry, before
                # browser launch and before navigation — meaning setup alone
                # (nav retries up to 27s, selector wait up to 5s, launch/
                # context overhead) could consume the entire budget before a
                # single pagination click happened. That's confirmed as part
                # of why pagination was adding "0 additional reviews": the
                # 2026-07-30 log showed navigation itself failing outright in
                # all 6 tested cases, so this specific clock-start fix hasn't
                # been exercised yet by real data (nav never got far enough to
                # reach the click loop) — but it's still correct to keep, since
                # a still-open question is whether some navigation attempts
                # were ALSO failing before due to a too-early clock silently
                # cutting them off rather than a genuine nav problem. The
                # _diag() trace above should make that distinguishable next run.
                start = time.monotonic()

                def time_left() -> float:
                    return WIDGET_PAGINATE_TIME_BUDGET_SECONDS - (time.monotonic() - start)

                return await _paginate_judgeme_widget(page, time_left)
            finally:
                await context.close()
        finally:
            await browser.close()


async def _paginate_judgeme_reviews_via_browser(url: str) -> List[_Review]:
    """Public-ish entry point for the widget-pagination fallback, dispatched
    through the shared BrowserManager exactly like _fetch_html_via_playwright."""
    try:
        from scrapers.browser_utils import run_playwright_async
    except ImportError:
        logger.warning(
            "Website review scrape: shared browser_utils.run_playwright_async "
            "is unavailable; skipping widget pagination for %s.", url,
        )
        return []
    try:
        return await run_playwright_async(lambda: _run_playwright_widget_paginate_session(url))
    except Exception:
        logger.exception(
            "Website review scrape: widget pagination failed for %s.", url
        )
        return []


async def scrape_website_reviews(company_data: Dict[str, str]) -> List[str]:
    """Public entry point — same contract as every other scrape_x()
    function: takes the per-job company_data dict, returns a flat
    List[str] of clean review text. Signature unchanged."""
    product_url = (company_data.get("product_url") or "").strip()
    website = (company_data.get("website") or "").strip()
    product_name = (company_data.get("product_name") or "").strip()

    if not product_url and not website:
        return []

    loop = asyncio.get_event_loop()
    try:
        result: _SyncResult = await asyncio.wait_for(
            loop.run_in_executor(_EXECUTOR, _scrape_sync, product_url, website, product_name),
            timeout=TIME_BUDGET_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Website review scrape exceeded its %ds HTTP budget for %r.",
            TIME_BUDGET_SECONDS, product_url or website,
        )
        return []
    except Exception:
        logger.exception("Unhandled error scraping website reviews for %r.", product_url or website)
        return []

    reviews = result.reviews
    has_judgeme = result.has_judgeme

    if not reviews and result.needs_playwright:
        logger.info(
            "Website review scrape: hardened HTTP was blocked (503) for %s; "
            "falling back to Playwright.", result.target,
        )
        html: Optional[str] = None
        try:
            html = await asyncio.wait_for(
                _fetch_html_via_playwright(result.target),
                timeout=PLAYWRIGHT_TIME_BUDGET_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Website review scrape: Playwright fallback exceeded its %ds budget for %s.",
                PLAYWRIGHT_TIME_BUDGET_SECONDS, result.target,
            )
        except Exception:
            logger.exception(
                "Website review scrape: Playwright fallback failed for %s.", result.target
            )

        if html:
            has_judgeme = has_judgeme or ("jdgm" in html or "judge.me" in html)
            try:
                reviews = await asyncio.wait_for(
                    loop.run_in_executor(
                        _EXECUTOR, _extract_all_tiers, html, result.target,
                        result.shop_domain, result.shopify_product_id,
                    ),
                    timeout=10.0,
                )
            except Exception:
                logger.exception(
                    "Website review scrape: extraction on Playwright-fetched HTML failed for %s.",
                    result.target,
                )
                reviews = []
            if reviews:
                logger.info(
                    "Website review scrape: %d review(s) recovered via Playwright fallback for %s.",
                    len(reviews), result.target,
                )
            else:
                logger.info(
                    "Website review scrape: Playwright fetch succeeded but no reviews were "
                    "found on %s.", result.target,
                )

    # FIX (the real "only 4-6 reviews when the product has hundreds" bug):
    # a plain GET (with or without the 503-triggered Playwright fallback
    # above) only ever sees whatever Judge.me server-rendered into page 1
    # of the widget. If Judge.me is genuinely installed on this page and
    # there's a real chance more pages exist (we're still under
    # MAX_REVIEWS), spend a browser session actually clicking through the
    # widget's own pagination/"load more" control and merge in whatever
    # new, unique reviews that surfaces.
    #
    # PREVIOUSLY this checked `reviews[0].source == "judgeme"` instead of
    # `has_judgeme`, which meant: whenever the JSON-LD tier in
    # _extract_all_tiers() found ANY reviews at all (tier 1, tried before
    # judgeme/yotpo/loox), it short-circuited the whole tier pipeline —
    # judgeme's own DOM/API extraction never ran, reviews[0].source came
    # back "jsonld" instead of "judgeme", and this pagination block was
    # skipped entirely. Since JSON-LD is typically a small SEO sample (not
    # the widget's full review set) and Judge.me is known to auto-inject
    # exactly this kind of JSON-LD, this was very likely the dominant
    # reason genuinely Judge.me-powered pages (confirmed live on
    # boat-lifestyle.com and headphonezone.in product pages, both of which
    # show a small initial batch out of a much larger total, e.g.
    # "Based on 147 reviews" with ~10 rendered) still only ever produced a
    # handful of reviews: the fix for exactly that gap existed in this file
    # but the JSON-LD short-circuit meant it could never fire on the
    # affected sites. `has_judgeme` is computed from the raw HTML/fingerprint
    # directly (see _SyncResult), independent of which tier supplied
    # `reviews`, so this now fires whenever Judge.me is actually present —
    # including when JSON-LD, Yotpo, or Loox happened to win the tier race.
    if has_judgeme and len(reviews) < MAX_REVIEWS:
        logger.info(
            "Website review scrape: Judge.me detected with %d review(s) so far "
            "for %s; clicking through the widget's own pagination for more.",
            len(reviews), result.target,
        )
        _widget_paginate_outer_timeout = (
            WIDGET_PAGINATE_TIME_BUDGET_SECONDS + WIDGET_PAGINATE_SETUP_ALLOWANCE_SECONDS
        )
        try:
            more_reviews = await asyncio.wait_for(
                _paginate_judgeme_reviews_via_browser(result.target),
                timeout=_widget_paginate_outer_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Website review scrape: widget pagination exceeded its %.0fs budget "
                "(%.0fs setup allowance + %.0fs click budget) for %s.",
                _widget_paginate_outer_timeout, WIDGET_PAGINATE_SETUP_ALLOWANCE_SECONDS,
                WIDGET_PAGINATE_TIME_BUDGET_SECONDS, result.target,
            )
            more_reviews = []
        except Exception:
            logger.exception(
                "Website review scrape: widget pagination failed for %s.", result.target
            )
            more_reviews = []

        if more_reviews:
            seen = {r.text for r in reviews}
            added = 0
            for r in more_reviews:
                if r.text not in seen:
                    seen.add(r.text)
                    reviews.append(r)
                    added += 1
            logger.info(
                "Website review scrape: widget pagination added %d new review(s) "
                "for %s (total now %d).",
                added, result.target, len(reviews),
            )

    texts = [r.text for r in reviews]

    # FIX: product_name was documented (top of file) and threaded all the way
    # through as a parameter, but never actually used to filter anything —
    # dead promise, not dead code exactly, but the same practical effect.
    # This only matters on the website-fallback path (product_url empty,
    # target came from `website` instead): that page isn't this specific
    # product's own page, so without a check here, reviews for a DIFFERENT
    # product on that fallback page would be returned and labeled as this
    # product's — confirmed live 2026-07-30 (Samsung's "Galaxy S26 Ultra"
    # job fell back to the Z Fold8's page and returned Z Fold8 reviews under
    # the S26 Ultra label). That specific case is now caught upstream in
    # app.py (the job is skipped before this function is ever called), but
    # this is the scraper's own documented contract and a second line of
    # defense for any other caller (e.g. product_intelligence.py, if it ever
    # calls this with only `website` set) that reaches the fallback path.
    # Product's own page (the normal case) is exempt — every review there is
    # inherently about that product, so filtering would only risk discarding
    # genuine ones for no reason.
    if not product_url and product_name:
        keywords = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", product_name) if len(w) >= 3]
        if keywords:
            before = len(texts)
            texts = [t for t in texts if any(kw in t.lower() for kw in keywords)]
            dropped = before - len(texts)
            if dropped:
                logger.info(
                    "Website review scrape: dropped %d/%d review(s) on the website-"
                    "fallback page for %r that didn't mention any keyword from the "
                    "product name — not genuinely product-specific.",
                    dropped, before, product_name,
                )

    final = normalize_comments(texts)[:MAX_REVIEWS]
    logger.info("Website reviews returned %d comments", len(final))
    return final
