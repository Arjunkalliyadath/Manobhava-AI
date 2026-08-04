"""
Module Name
    product_discovery.py

Purpose
    Discovers a company's product and service catalogue from its public
    website, returning a ranked, deduplicated list of products/services
    with per-item metadata for downstream review scraping and reporting.

Responsibilities
    - Reject anything that isn't a real website URL (a bare keyword or
      company name like "boat audio") before any network activity starts -
      see the is_url() guard at the top of discover_products(). This is a
      second line of defense; the input step (app.py, before
      company_discovery.py even runs) is meant to gate on the same check.
    - Fetch a target site over plain HTTP where possible, and fall back to
      a single, shared Playwright browser session when the site is blocked
      or renders content via JavaScript.
    - Discover category/collection pages via on-page navigation, mega-menu
      / footer widgets, and sitemap.xml, then crawl each for product data.
    - Parse JSON-LD structured data and HTML product-card markup into
      Product records, merging duplicates discovered by different stages.
    - Score, filter, and classify each candidate (product vs. service)
      before ranking and truncating to the public output shape.

Session notes (2026-07-31)
    Verified against samsung.com/in, apple.com/in, mamaearth.in,
    lenskart.com, vguard.com, boat-lifestyle.com, headphonezone.in,
    sony.co.in, ajio.com, and online.kfc.co.in. Fixes made this pass:
    - Added the is_url() input guard described above.
    - Tried "--disable-http2" on the Playwright browser launch as an
      untested attempt at online.kfc.co.in's ERR_HTTP2_PROTOCOL_ERROR,
      which happens before any parsing code runs.
    - ajio.com remains hard-blocked (its anti-bot interstitial is
      correctly detected and rejected rather than mined for garbage) -
      that's working as intended, not a bug; getting past it would need
      residential proxies or a paid unblocking service, out of scope here.
    - Removed a duplicate copy of _root_url() (now imported from
      product_extraction.py, which already had its own copy).
    Most of the empty-product-URL fixes for this session live in
    product_extraction.py (see its own docstring), since that's where the
    URL was actually being lost.

Architecture
    A single plain-HTTP probe of the homepage decides which of two paths
    the rest of the analysis takes. If it succeeds, the static pipeline
    (stages 1-3b below) runs entirely over httpx/BeautifulSoup, and
    Playwright is only invoked afterwards if results are still thin. If it
    fails (blocked status code or anti-bot interstitial), the entire
    analysis is handed to a single Playwright session
    (_run_playwright_driven_discovery / _PlaywrightSession) that owns one
    browser, one context, and a small pool of reused pages for every page
    it needs to render - never more than one browser launch per analysis.
    A shared `visited_urls` set is threaded through the static pipeline so
    a URL discovered by more than one stage is only ever fetched once.

Discovery Pipeline
    1. JSON-LD / schema.org structured data (Product, Offer, ItemList,
       BreadcrumbList) - highest-confidence source.
    2. Static HTML product-card parsing (product-card/grid containers,
       product-shaped anchors).
    3. Category/collection page crawl: on-page nav/header links plus
       mega-menu, dropdown, flyout, and footer widget links, each walked
       through its own ?page=N pagination until a page returns nothing
       new.
    3b. robots.txt -> Sitemap: -> sitemap.xml category discovery, used to
       widen the category list when stage 3 is still thin.
    4. Playwright fallback for JS-rendered or bot-protected sites: scroll,
       click "Load more", walk ?page=N pagination, and fall back to
       clicking a stateful "Next" control when a site doesn't paginate via
       a predictable query parameter.
    5. De-duplication, confidence scoring, noise filtering, and ranking
       into the final catalogue.

Inputs
    company_data: Dict[str, str] with at least a "website" key (and
    typically "company_name") describing the site to analyze.

Outputs
    A dict with the backward-compatible keys products, services,
    scrape_targets, products_found, services_found, discovery_method,
    discovery_version, plus the additive keys site_type, categories, and
    catalogue (the full structured product list).

Dependencies
    httpx and BeautifulSoup for the static pipeline; Playwright
    (scrapers.browser_utils.run_playwright_async) for the JS-rendered
    fallback; product_extraction for parsing/classification primitives;
    config for shared, cross-module tunables.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import uuid
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

import config
from url_utils import is_url
from scrapers.browser_utils import run_playwright_async
from product_extraction import (
    Product,
    SiteType,
    classify_as_service,
    detect_site_type,
    discover_category_links,
    extract_meta_image,
    extract_sitemap_locs,
    filter_category_urls_from_sitemap,
    is_sitemap_index,
    is_valid_candidate_name,
    parse_breadcrumb_category,
    parse_html_product_cards,
    parse_jsonld_products,
    parse_robots_sitemaps,
    _looks_like_product_url,
    _root_url,
    _CATEGORY_HINT_WORDS,
    _CATEGORY_EXCLUDE_WORDS,
)

logger = logging.getLogger(__name__)

DIAGNOSTICS_ENABLED = True
DEBUG_HTML_DIR = "debug_html"

_DIAG_EXTRA_BOT_MARKERS = (
    "captcha", "are you a robot", "verify you are human", "/sorry/",
    "unusual traffic", "bot detection", "automated access",
    "checking if the site connection is secure", "ray id",
)

_ADD_TO_CART_MARKERS = (
    "add to cart", "add to bag", "add to basket", "buy now",
)

DISCOVERY_VERSION = "product-discovery-v7"

MAX_CATALOGUE_SIZE = 200  # raised from 60 - was silently truncating real catalogs (boat-lifestyle.com has 166 real products; this crawl already visits every discovered category page regardless of this cap, so raising it costs close to nothing in extra time - it's not visiting more pages, just not throwing away products it already found on pages it was fetching anyway). Product discovery is a separate, untimed step before the person picks products and starts the actual 1-3 min analysis clock, so this isn't part of that budget at all.

MIN_CONFIDENCE = 0.35

MAX_CATEGORY_PAGES = 12
MAX_PLAYWRIGHT_PAGES = 6
MAX_SCROLL_ROUNDS = 6
MAX_PAGINATION_PAGES = 4

PLAYWRIGHT_POOL_SIZE = 3

MAX_SITEMAPS_TO_FOLLOW = 3
MAX_SITEMAP_INDEX_CHILDREN = 3

THIN_RESULT_THRESHOLD = 8

# Cap on how many products we'll pay an extra per-product-page fetch for to
# backfill a missing image (see _enrich_missing_images). Capped to the same
# size as what's actually shown in select_products.html - no point spending
# a request on a product that will be truncated out of the final list.
MAX_IMAGE_ENRICH_TARGETS = MAX_CATALOGUE_SIZE

CANDIDATE_PATH_SUFFIXES = (
    "/products", "/collections/all", "/shop", "/shop-all", "/collections",
    "/store", "/catalog", "/services", "/solutions",
)

EXTRA_CATEGORY_SCOPES = (
    "[class*=mega i]", "[class*=megamenu i]", "[class*=dropdown i]",
    "[class*=submenu i]", "[class*=flyout i]", "[class*=nav-panel i]",
    "footer",
)

_NEXT_PAGE_SELECTORS = (
    "a[rel=next]",
    "link[rel=next]",
    "a[aria-label='Next']",
    "a[aria-label*=next i]",
    "button[aria-label*=next i]",
    "[class*=pagination i] a:has-text('Next')",
    "a:has-text('Next')",
    "button:has-text('Next')",
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


class ProductDiscoveryResult:
    def __init__(
        self,
        catalogue: Optional[List[Product]] = None,
        discovery_method: str = "none",
        site_type: SiteType = SiteType.UNKNOWN,
        categories: Optional[List[str]] = None,
    ) -> None:
        self.catalogue = catalogue or []
        self.discovery_method = discovery_method
        self.site_type = site_type
        self.categories = categories or []

    def as_dict(self) -> Dict:
        ranked = sorted(self.catalogue, key=lambda p: p.confidence, reverse=True)
        product_items = [p for p in ranked if not p.is_service]
        service_items = [p for p in ranked if p.is_service]

        product_names = _unique_names(product_items)[: config.MAX_PRODUCTS]
        service_names = _unique_names(service_items)[: config.MAX_PRODUCTS]

        return {
            "products": product_names,
            "services": service_names,
            "scrape_targets": product_names,
            "products_found": len(product_items),
            "services_found": len(service_items),
            "discovery_method": self.discovery_method,
            "discovery_version": DISCOVERY_VERSION,
            "site_type": self.site_type.value,
            "categories": self.categories,
            "catalogue": [p.as_dict() for p in ranked[:MAX_CATALOGUE_SIZE]],
        }


def _unique_names(products: List[Product]) -> List[str]:
    seen: Set[str] = set()
    names: List[str] = []
    for p in products:
        lower = p.name.lower()
        if lower in seen:
            continue
        seen.add(lower)
        names.append(p.name)
    return names


async def discover_products(company_data: Dict[str, str]) -> Dict:
    website = (company_data.get("website") or "").strip()
    company_name = (company_data.get("company_name") or "").strip()

    if not website:
        return ProductDiscoveryResult(discovery_method="skipped-no-website").as_dict()

    if not is_url(website):
        # Second line of defense against a bare keyword/company name (e.g.
        # "boat audio") reaching this far and triggering a live crawl - the
        # input step is meant to gate on this same is_url() check before
        # company_discovery.py even runs, but checking it again here means
        # this module can never go crawl-happy on a non-URL string even if
        # that upstream gate is ever bypassed or changed.
        logger.info(
            "Product discovery skipped: %r does not look like a website "
            "URL.", website,
        )
        return ProductDiscoveryResult(discovery_method="skipped-invalid-url").as_dict()

    root = _root_url(website)
    stages_used: List[str] = []
    registry: Dict[str, Product] = {}
    categories_seen: List[str] = []
    visited_urls: Set[str] = set()

    homepage_html = await _fetch_static(root)
    visited_urls.add(root)
    site_type = detect_site_type(homepage_html or "")

    if not homepage_html:
        logger.info(
            "Using Playwright fallback for %s: plain HTTP homepage fetch "
            "did not return usable HTML (blocked status code or anti-bot "
            "interstitial).", root,
        )
        seed_links = await _safe_fetch_sitemap_links(root, MAX_CATEGORY_PAGES)
        if seed_links:
            stages_used.append("sitemap")

        pw_registry, pw_categories, pw_site_type, _ = await _run_playwright_driven_discovery(
            root, seed_links, rediscover_categories=True,
        )
        if _merge_playwright_results(registry, categories_seen, pw_registry, pw_categories):
            stages_used.append("playwright-rendered")
        if pw_site_type is not SiteType.UNKNOWN:
            site_type = pw_site_type

    else:
        _ingest(registry, parse_jsonld_products(homepage_html, root))
        _ingest(registry, parse_html_product_cards(homepage_html, root))
        if registry:
            stages_used.append("jsonld" if any("jsonld" in p.source for p in registry.values()) else "html")

        category_links = discover_category_links(homepage_html, root, limit=MAX_CATEGORY_PAGES)

        extra_links = _discover_extra_category_links(
            homepage_html, root, exclude=set(category_links), limit=MAX_CATEGORY_PAGES,
        )
        if extra_links:
            logger.info(
                "Mega-menu/footer discovery found %d additional category "
                "link(s) for %s (nav crawl alone found %d).",
                len(extra_links), root, len(category_links),
            )
            category_links = list(dict.fromkeys([*category_links, *extra_links]))
            stages_used.append("mega-menu")

        category_links = _merge_default_paths(root, category_links)

        if category_links:
            crawl_targets = [u for u in category_links[:MAX_CATEGORY_PAGES] if u not in visited_urls]
            if await _crawl_urls(crawl_targets, registry, categories_seen, visited_urls):
                stages_used.append("category-crawl")

        if _accepted_count(registry) < THIN_RESULT_THRESHOLD:
            sitemap_links = await _safe_fetch_sitemap_links(root, MAX_CATEGORY_PAGES)

            new_links = [
                link for link in sitemap_links
                if link not in category_links and link not in visited_urls
            ]
            if new_links:
                if await _crawl_urls(new_links[:MAX_CATEGORY_PAGES], registry, categories_seen, visited_urls):
                    stages_used.append("sitemap")
                category_links.extend(new_links)

        if _accepted_count(registry) < THIN_RESULT_THRESHOLD:
            logger.info(
                "Using Playwright fallback for %s: static pipeline found "
                "only %d accepted product(s), below THIN_RESULT_THRESHOLD "
                "(%d).", root, _accepted_count(registry), THIN_RESULT_THRESHOLD,
            )
            try:
                pw_registry, pw_categories, _, _ = await _run_playwright_driven_discovery(
                    root, category_links, rediscover_categories=False,
                )
                if _merge_playwright_results(registry, categories_seen, pw_registry, pw_categories):
                    stages_used.append("playwright")
            except Exception as exc:
                logger.info("Playwright product discovery failed for %s: %s", root, exc)

    final_catalogue: List[Product] = []
    for product in registry.values():
        product.score()
        if product.confidence < MIN_CONFIDENCE:
            continue
        if not is_valid_candidate_name(product.name):
            continue
        product.is_service = classify_as_service(product)
        final_catalogue.append(product)

    try:
        await _enrich_missing_images(final_catalogue)
    except Exception as exc:
        logger.info("Image enrichment pass failed for %s: %s", root, exc)

    method = "+".join(stages_used) if stages_used else "none"
    result = ProductDiscoveryResult(
        catalogue=final_catalogue,
        discovery_method=method,
        site_type=site_type,
        categories=categories_seen,
    )
    payload = result.as_dict()

    accepted = len(final_catalogue) - sum(1 for p in final_catalogue if p.is_service)
    if accepted >= MAX_CATALOGUE_SIZE:
        stop_reason = f"reached MAX_CATALOGUE_SIZE ({MAX_CATALOGUE_SIZE})"
    elif stages_used:
        stop_reason = f"all applicable stages exhausted (stages_used={stages_used})"
    else:
        stop_reason = "no stage produced any accepted candidate"
    logger.info("Discovery for %s stopped because: %s", root, stop_reason)

    logger.info(
        "Product discovery for %s: %d products, %d services (method=%s, site_type=%s, "
        "categories=%d, urls_fetched=%d)",
        company_name, payload["products_found"], payload["services_found"],
        method, site_type.value, len(categories_seen), len(visited_urls),
    )
    return payload


ANTI_BOT_STATUS_CODES = {401, 403, 429, 500, 502, 503, 504}

ANTI_BOT_MARKERS = (
    "checking your browser before accessing",
    "just a moment...",
    "cf-browser-verification",
    "attention required! | cloudflare",
    "enable javascript and cookies to continue",
    "please verify you are a human",
    "request unsuccessful. incapsula",
    "perimeterx",
    "you have been blocked",
    "access denied",
)
ANTI_BOT_MAX_LEN = 6000


def _looks_like_anti_bot_page(html: str) -> bool:
    if not html or len(html) > ANTI_BOT_MAX_LEN:
        return False
    lowered = html.lower()
    return any(marker in lowered for marker in ANTI_BOT_MARKERS)


async def _fetch_static(url: str) -> str:
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=6.0),
            follow_redirects=True,
            headers=HEADERS,
            verify=False,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            text = resp.text
            if _looks_like_anti_bot_page(text):
                logger.info("Anti-bot challenge page detected (HTTP 200): %s", url)
                return ""
            return text
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status in ANTI_BOT_STATUS_CODES:
            logger.info(
                "HTTP blocked (%d) for %s - skipping HTTP entirely (no "
                "retries) and switching to Playwright.", status, url,
            )
        else:
            logger.info("Static fetch failed for %s: HTTP %d", url, status)
        return ""
    except Exception as exc:
        logger.info("Static fetch failed for %s: %s", url, exc)
        return ""


async def _fetch_many(urls: List[str]) -> Dict[str, str]:
    semaphore = asyncio.Semaphore(config.MAX_PARALLEL_TASKS)

    async def _one(url: str) -> tuple:
        async with semaphore:
            html = await _fetch_static(url)
            return url, html

    results = await asyncio.gather(*(_one(u) for u in urls), return_exceptions=False)
    return {url: html for url, html in results}


async def _crawl_category_static_with_pagination(
    url: str,
    semaphore: asyncio.Semaphore,
    registry: Dict[str, Product],
    categories_seen: List[str],
    visited: Set[str],
) -> bool:
    if url in visited:
        return False
    visited.add(url)

    async with semaphore:
        html = await _fetch_static(url)
    if not html:
        return False

    category_label = parse_breadcrumb_category(html) or _guess_category_from_url(url)
    if category_label and category_label not in categories_seen:
        categories_seen.append(category_label)
    _ingest(registry, _extract_products(html, url, category=category_label))

    for page_num in range(2, MAX_PAGINATION_PAGES + 1):
        if _accepted_count(registry) >= MAX_CATALOGUE_SIZE:
            break
        sep = "&" if "?" in url else "?"
        paged_url = f"{url}{sep}page={page_num}"
        if paged_url in visited:
            break
        visited.add(paged_url)

        async with semaphore:
            paged_html = await _fetch_static(paged_url)
        if not paged_html:
            break

        new_products = _extract_products(paged_html, paged_url, category=category_label)
        if not new_products:
            break
        _ingest(registry, new_products)

    return True


async def _fetch_sitemap_category_links(root: str, limit: int = 15) -> List[str]:
    robots_txt = await _fetch_static(f"{root}/robots.txt")
    sitemap_urls = parse_robots_sitemaps(robots_txt)
    if not sitemap_urls:
        sitemap_urls = [f"{root}/sitemap.xml"]

    all_locs: List[str] = []
    for sitemap_url in sitemap_urls[:MAX_SITEMAPS_TO_FOLLOW]:
        xml_text = await _fetch_static(sitemap_url)
        if not xml_text:
            continue
        locs = extract_sitemap_locs(xml_text)
        if is_sitemap_index(xml_text):
            child_htmls = await _fetch_many(locs[:MAX_SITEMAP_INDEX_CHILDREN])
            for child_xml in child_htmls.values():
                all_locs.extend(extract_sitemap_locs(child_xml))
        else:
            all_locs.extend(locs)

    return filter_category_urls_from_sitemap(all_locs, root, limit=limit)


async def _safe_fetch_sitemap_links(root: str, limit: int) -> List[str]:
    try:
        return await _fetch_sitemap_category_links(root, limit=limit)
    except Exception as exc:
        logger.info("Sitemap discovery failed for %s: %s", root, exc)
        return []


async def _crawl_urls(
    urls: List[str],
    registry: Dict[str, Product],
    categories_seen: List[str],
    visited_urls: Set[str],
) -> bool:
    semaphore = asyncio.Semaphore(config.MAX_PARALLEL_TASKS)
    results = await asyncio.gather(
        *(
            _crawl_category_static_with_pagination(
                url, semaphore, registry, categories_seen, visited_urls,
            )
            for url in urls
        )
    )
    return any(results)


async def _enrich_missing_images(products: List[Product]) -> None:
    """Backfill product.image for products whose listing-card had no usable
    photo, by fetching that product's own detail page and reading its
    JSON-LD `image` field or its og:image/twitter:image meta tags.

    This exists because of a real, confirmed gap: some storefronts (boAt's
    "Gifting with boAt" promotional carousel is the case that surfaced it)
    inject a card's real photo entirely client-side, with no data-src/
    srcset fallback for a static-HTML scrape to fall back on - so no amount
    of listing-card parsing can recover it. A product's own detail page,
    by contrast, reliably renders its OpenGraph/Twitter image server-side
    on essentially every ecommerce platform, so it's a much sturdier
    fallback source than continuing to fight listing-page card markup.

    Only products that already passed confidence/name filtering and are
    missing an image are targeted, and the count is capped
    (MAX_IMAGE_ENRICH_TARGETS) so this can't blow up total request volume
    on a large catalogue - it's a backfill for the common case of a
    handful of stragglers, not a second full crawl.
    """
    targets = [p for p in products if p.url and not p.image]
    if not targets:
        return
    # Prioritise the higher-confidence products first, since a large
    # catalogue may have more gaps than the cap allows.
    targets = sorted(targets, key=lambda p: p.confidence, reverse=True)[:MAX_IMAGE_ENRICH_TARGETS]

    semaphore = asyncio.Semaphore(config.MAX_PARALLEL_TASKS)

    async def _one(product: Product) -> None:
        async with semaphore:
            html = await _fetch_static(product.url)
        if not html:
            return
        for jsonld_product in parse_jsonld_products(html, product.url):
            if jsonld_product.image:
                product.image = jsonld_product.image
                product.source.add("image-enriched")
                return
        meta_image = extract_meta_image(html)
        if meta_image:
            product.image = meta_image
            product.source.add("image-enriched")

    await asyncio.gather(*(_one(p) for p in targets))


def _merge_playwright_results(
    registry: Dict[str, Product],
    categories_seen: List[str],
    pw_registry: Dict[str, Product],
    pw_categories: List[str],
) -> bool:
    if pw_registry:
        _ingest(registry, list(pw_registry.values()))
    for c in pw_categories:
        if c not in categories_seen:
            categories_seen.append(c)
    return bool(pw_registry)


def _merge_default_paths(root: str, discovered: List[str]) -> List[str]:
    merged = list(dict.fromkeys(discovered))
    for suffix in CANDIDATE_PATH_SUFFIXES:
        candidate = f"{root}{suffix}"
        if candidate not in merged:
            merged.append(candidate)
    return merged


def _discover_extra_category_links(
    html: str,
    page_url: str,
    exclude: Optional[Set[str]] = None,
    limit: int = MAX_CATEGORY_PAGES,
) -> List[str]:
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    root = _root_url(page_url)
    seen: Set[str] = set(exclude or ())

    found: List[str] = []
    try:
        scopes = soup.select(", ".join(EXTRA_CATEGORY_SCOPES))
    except Exception:
        return []

    for scope in scopes:
        for a in scope.find_all("a", href=True):
            text = (a.get_text(" ", strip=True) or "").lower()
            href = a["href"]
            if not href or href.startswith("#") or href.startswith("javascript:"):
                continue
            full = urljoin(page_url, href)
            if not full.startswith(root) or full in seen:
                continue
            if any(w in text for w in _CATEGORY_EXCLUDE_WORDS):
                continue
            if any(w in text for w in _CATEGORY_HINT_WORDS) or any(
                w in full.lower() for w in _CATEGORY_HINT_WORDS
            ):
                seen.add(full)
                found.append(full)
                if len(found) >= limit:
                    return found

    return found


def _guess_category_from_url(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if not path:
        return ""
    segment = path.split("/")[-1]
    return segment.replace("-", " ").replace("_", " ").title()[: config.MAX_CANDIDATE_LEN]


def _diag_is_bot_check_page(html: str) -> bool:
    if not html:
        return False
    lowered = html.lower()
    return _looks_like_anti_bot_page(html) or any(marker in lowered for marker in _DIAG_EXTRA_BOT_MARKERS)


def _safe_filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    raw = f"{parsed.netloc}{parsed.path}"
    if parsed.query:
        raw += f"_{parsed.query}"
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw).strip("_")
    return (safe or "page")[:150]


def _save_debug_html_sync(url: str, html: str) -> Optional[str]:
    try:
        os.makedirs(DEBUG_HTML_DIR, exist_ok=True)
        base = _safe_filename_from_url(url)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{base}__{stamp}__{uuid.uuid4().hex[:6]}.html"
        path = os.path.join(DEBUG_HTML_DIR, filename)
        with open(path, "w", encoding="utf-8", errors="ignore") as f:
            f.write(html or "")
        return path
    except Exception as exc:
        logger.info("[DIAG] Failed to save debug HTML for %s: %s", url, exc)
        return None


async def _log_page_diagnostics(page, requested_url: str, html: str) -> None:
    try:
        final_url = page.url or requested_url
    except Exception:
        final_url = requested_url

    status = getattr(page, "_diag_last_status", None)

    try:
        title = await page.title()
    except Exception:
        title = ""

    html_len = len(html or "")

    soup = BeautifulSoup(html or "", "html.parser")
    all_links = soup.find_all("a", href=True)
    num_links = len(all_links)

    num_candidate_product_links = 0
    for a in all_links:
        try:
            href = urljoin(final_url, a["href"])
        except Exception:
            continue
        if _looks_like_product_url(href):
            num_candidate_product_links += 1

    try:
        jsonld_products = parse_jsonld_products(html or "", final_url)
    except Exception:
        jsonld_products = []
    try:
        card_products = parse_html_product_cards(html or "", final_url)
    except Exception:
        card_products = []

    has_jsonld_product = bool(jsonld_products)
    num_candidate_cards = len(card_products)

    lowered = (html or "").lower()
    has_cart_cta = any(marker in lowered for marker in _ADD_TO_CART_MARKERS)
    is_bot_check = _diag_is_bot_check_page(html or "")

    logger.info(
        "[DIAG] requested_url=%s final_url=%s http_status=%s title=%r "
        "html_len=%d links=%d candidate_product_links=%d "
        "candidate_product_cards=%d jsonld_product=%s add_to_cart_text=%s "
        "bot_check_page=%s",
        requested_url, final_url, status, title, html_len, num_links,
        num_candidate_product_links, num_candidate_cards, has_jsonld_product,
        has_cart_cta, is_bot_check,
    )

    total_found = num_candidate_cards + len(jsonld_products)
    if total_found == 0:
        saved_path = await asyncio.to_thread(_save_debug_html_sync, final_url, html or "")
        if saved_path:
            logger.info(
                "[DIAG] Zero products extracted from %s - saved HTML to %s (%d bytes) for inspection.",
                final_url, saved_path, html_len,
            )


class _PlaywrightSession:
    def __init__(self, playwright, browser, context, pages: List) -> None:
        self._playwright = playwright
        self._browser = browser
        self._context = context
        self._all_pages = pages
        self._pool: asyncio.Queue = asyncio.Queue()
        for page in pages:
            self._pool.put_nowait(page)

    @classmethod
    async def launch(cls, pool_size: int = PLAYWRIGHT_POOL_SIZE) -> "_PlaywrightSession":
        from playwright.async_api import async_playwright

        logger.info("Launching Playwright (one browser for this analysis)...")
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
                # online.kfc.co.in fails page.goto() with
                # ERR_HTTP2_PROTOCOL_ERROR before any of our own parsing
                # code ever runs - a broken/incompatible HTTP/2 response
                # from that site's edge for automated clients specifically.
                # Forcing HTTP/1.1 is a cheap, universally-safe thing to
                # try (every server that speaks HTTP/2 also speaks
                # HTTP/1.1), but this hasn't been verified live yet - it
                # needs a real run against that site to confirm it helps.
                "--disable-http2",
            ],
        )
        context = await browser.new_context(
            viewport={"width": 1440, "height": 900},
            user_agent=HEADERS["User-Agent"],
            locale="en-US",
            timezone_id="Asia/Kolkata",
            device_scale_factor=1,
            is_mobile=False,
            has_touch=False,
            extra_http_headers={
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,"
                    "image/webp,*/*;q=0.8"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            },
        )

        stealth_script = """
        (() => {
          // 1. navigator.webdriver - the single most common automation check.
          Object.defineProperty(navigator, 'webdriver', { get: () => undefined });

          // 2. window.chrome - missing/incomplete in headless Chromium by default.
          window.chrome = window.chrome || { runtime: {} };

          // 3. Permissions API - headless Chromium answers a Notification
          //    permission query synchronously in a way that differs from a
          //    real profile; align it with the real Notification.permission.
          if (window.navigator.permissions && window.navigator.permissions.query) {
            const originalQuery = window.navigator.permissions.query.bind(window.navigator.permissions);
            window.navigator.permissions.query = (parameters) => (
              parameters && parameters.name === 'notifications'
                ? Promise.resolve({ state: Notification.permission })
                : originalQuery(parameters)
            );
          }

          // 4. navigator.plugins / mimeTypes - empty arrays in headless by default.
          Object.defineProperty(navigator, 'plugins', {
            get: () => [1, 2, 3, 4, 5].map(() => ({ name: 'Chrome PDF Plugin' })),
          });
          Object.defineProperty(navigator, 'mimeTypes', {
            get: () => [1, 2].map(() => ({ type: 'application/pdf' })),
          });

          // 5. navigator.languages - keep consistent with the context locale.
          Object.defineProperty(navigator, 'languages', {
            get: () => ['en-US', 'en'],
          });

          // 6. WebGL vendor/renderer - default headless value ("Google
          //    SwiftShader" / software rendering) is a well-known headless
          //    tell. Report a plausible real-GPU vendor/renderer instead.
          const patchWebGL = (proto) => {
            if (!proto) return;
            const originalGetParameter = proto.getParameter;
            proto.getParameter = function (parameter) {
              if (parameter === 37445) return 'Intel Inc.';               // UNMASKED_VENDOR_WEBGL
              if (parameter === 37446) return 'Intel Iris OpenGL Engine'; // UNMASKED_RENDERER_WEBGL
              return originalGetParameter.apply(this, arguments);
            };
          };
          patchWebGL(window.WebGLRenderingContext && window.WebGLRenderingContext.prototype);
          patchWebGL(window.WebGL2RenderingContext && window.WebGL2RenderingContext.prototype);
        })();
        """
        await context.add_init_script(stealth_script)

        pages = [await context.new_page() for _ in range(pool_size)]
        logger.info("Playwright session ready: 1 browser, 1 context, %d pages.", pool_size)
        return cls(playwright, browser, context, pages)

    async def acquire_page(self):
        return await self._pool.get()

    async def release_page(self, page) -> None:
        await self._pool.put(page)

    async def render(self, url: str) -> str:
        page = await self.acquire_page()
        try:
            return await self.goto_and_extract(page, url)
        finally:
            await self.release_page(page)

    async def goto_and_extract(self, page, url: str) -> str:
        try:
            response = await page.goto(url, wait_until="domcontentloaded", timeout=20000)
            page._diag_last_status = response.status if response else None
            try:
                await page.wait_for_load_state("networkidle", timeout=6000)
            except Exception:
                pass
            await _auto_scroll_and_expand(page)
            html = await page.content()
            if DIAGNOSTICS_ENABLED:
                try:
                    await _log_page_diagnostics(page, url, html)
                except Exception as diag_exc:
                    logger.info("[DIAG] instrumentation failed for %s: %s", url, diag_exc)
            return html
        except Exception as exc:
            logger.info("Playwright render failed for %s: %s", url, exc)
            return ""

    async def aclose(self) -> None:
        for page in self._all_pages:
            try:
                await page.close()
            except Exception:
                pass
        try:
            await self._context.close()
        except Exception:
            pass
        try:
            await self._browser.close()
        except Exception:
            pass
        try:
            await self._playwright.stop()
        except Exception:
            pass
        logger.info("Playwright session closed.")


async def _click_next_page(page) -> bool:
    for sel in _NEXT_PAGE_SELECTORS:
        try:
            locator = page.locator(sel).first
            if await locator.is_visible(timeout=500):
                await locator.click(timeout=1500)
                try:
                    await page.wait_for_load_state("networkidle", timeout=6000)
                except Exception:
                    pass
                return True
        except Exception:
            continue
    return False


async def _render_category_with_pagination(
    session: _PlaywrightSession,
    url: str,
    registry: Dict[str, Product],
    categories_seen: List[str],
) -> None:
    page = await session.acquire_page()
    try:
        html = await session.goto_and_extract(page, url)
        if not html:
            return
        category_label = parse_breadcrumb_category(html) or _guess_category_from_url(url)
        if category_label and category_label not in categories_seen:
            categories_seen.append(category_label)
        _ingest(registry, _extract_products(html, url, category=category_label))

        used_query_pagination = False
        for page_num in range(2, MAX_PAGINATION_PAGES + 1):
            if _accepted_count(registry) >= MAX_CATALOGUE_SIZE:
                return
            sep = "&" if "?" in url else "?"
            paged_url = f"{url}{sep}page={page_num}"
            paged_html = await session.goto_and_extract(page, paged_url)
            if not paged_html:
                break
            new_products = _extract_products(paged_html, paged_url, category=category_label)
            if not new_products:
                break
            _ingest(registry, new_products)
            used_query_pagination = True

        if not used_query_pagination and _accepted_count(registry) < MAX_CATALOGUE_SIZE:
            html = await session.goto_and_extract(page, url)
            if not html:
                return
            for _ in range(MAX_PAGINATION_PAGES - 1):
                if _accepted_count(registry) >= MAX_CATALOGUE_SIZE:
                    break
                clicked = await _click_next_page(page)
                if not clicked:
                    break
                await _auto_scroll_and_expand(page)
                try:
                    next_html = await page.content()
                except Exception:
                    break
                new_products = _extract_products(next_html, page.url, category=category_label)
                if not new_products:
                    break
                _ingest(registry, new_products)
    except Exception:
        # A single bad category page (a selector that doesn't parse on this
        # site's markup, a timeout, a transient nav failure, etc.) must not
        # wipe out everything already found - not just on other category
        # pages, but on the homepage too, since this used to propagate all
        # the way out of _run_playwright_driven_discovery's own try/except
        # and get replaced with a flat {} (see there). registry already
        # has whatever was ingested before the failure point (dicts are
        # shared/mutated in place), so just log and move on.
        logger.warning(
            "Category page %s failed during Playwright-driven discovery; "
            "keeping whatever products were already found on it and "
            "elsewhere.", url, exc_info=True,
        )
    finally:
        await session.release_page(page)


async def _run_playwright_driven_discovery(
    root: str,
    category_urls: List[str],
    rediscover_categories: bool = False,
) -> tuple:

    async def _coro() -> tuple:
        session: Optional[_PlaywrightSession] = None
        try:
            session = await _PlaywrightSession.launch()

            registry: Dict[str, Product] = {}
            categories_seen: List[str] = []

            homepage_html = await session.render(root)
            site_type = detect_site_type(homepage_html or "")
            if homepage_html:
                try:
                    _ingest(registry, parse_jsonld_products(homepage_html, root))
                except Exception:
                    logger.warning("parse_jsonld_products failed for homepage %s", root, exc_info=True)
                try:
                    _ingest(registry, parse_html_product_cards(homepage_html, root))
                except Exception:
                    logger.warning("parse_html_product_cards failed for homepage %s", root, exc_info=True)

            pages_to_visit = list(dict.fromkeys(category_urls))
            if rediscover_categories:
                try:
                    nav_links = discover_category_links(homepage_html or "", root, limit=MAX_CATEGORY_PAGES)
                except Exception:
                    logger.warning("discover_category_links failed for %s", root, exc_info=True)
                    nav_links = []
                try:
                    extra_links = _discover_extra_category_links(
                        homepage_html or "", root, exclude=set(nav_links), limit=MAX_CATEGORY_PAGES,
                    )
                except Exception:
                    logger.warning("_discover_extra_category_links failed for %s", root, exc_info=True)
                    extra_links = []
                if extra_links:
                    logger.info(
                        "Mega-menu/footer discovery (rendered DOM) found %d "
                        "additional category link(s) for %s.", len(extra_links), root,
                    )
                pages_to_visit = list(dict.fromkeys([*nav_links, *extra_links, *pages_to_visit]))
                if not pages_to_visit:
                    pages_to_visit = _merge_default_paths(root, [])
            pages_to_visit = pages_to_visit[:MAX_PLAYWRIGHT_PAGES]

            if pages_to_visit:
                _page_results = await asyncio.gather(
                    *(
                        _render_category_with_pagination(session, url, registry, categories_seen)
                        for url in pages_to_visit
                    ),
                    return_exceptions=True,
                )
                for _url, _outcome in zip(pages_to_visit, _page_results):
                    if isinstance(_outcome, BaseException):
                        logger.warning(
                            "Category page %s raised past its own handler "
                            "during Playwright-driven discovery (%s: %s) - "
                            "continuing with the other pages.",
                            _url, type(_outcome).__name__, _outcome,
                        )

            return registry, categories_seen, site_type, homepage_html
        finally:
            if session is not None:
                await session.aclose()

    try:
        return await run_playwright_async(_coro)
    except Exception as exc:
        logger.info("Playwright-driven discovery failed for %s: %s", root, exc)
        return {}, [], SiteType.UNKNOWN, ""


async def _auto_scroll_and_expand(page) -> None:
    load_more_selectors = (
        "button:has-text('Load more')",
        "button:has-text('Show more')",
        "a:has-text('Load more')",
        "a:has-text('Show more')",
        "[class*=load-more i]",
        "[class*=show-more i]",
    )

    last_height = 0
    for _ in range(MAX_SCROLL_ROUNDS):
        try:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(700)
            clicked = False
            for sel in load_more_selectors:
                try:
                    locator = page.locator(sel).first
                    if await locator.is_visible(timeout=500):
                        await locator.click(timeout=1500)
                        await page.wait_for_timeout(900)
                        clicked = True
                        break
                except Exception:
                    continue
            height = await page.evaluate("document.body.scrollHeight")
            if height == last_height and not clicked:
                break
            last_height = height
        except Exception:
            break


def _extract_products(html: str, url: str, category: Optional[str] = None) -> List[Product]:
    return parse_jsonld_products(html, url) + parse_html_product_cards(html, url, category=category)


def _ingest(registry: Dict[str, Product], products: List[Product]) -> None:
    for product in products:
        if not is_valid_candidate_name(product.name):
            continue
        key = product.key()
        if key in registry:
            registry[key].merge(product)
        else:
            registry[key] = product


def _accepted_count(registry: Dict[str, Product]) -> int:
    count = 0
    for product in registry.values():
        if product.score() >= MIN_CONFIDENCE:
            count += 1
    return count
