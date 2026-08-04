"""
company_discovery.py

Given a company's website URL, fetches the homepage and extracts basic
brand/company metadata: company name, logo, and social profile links
(Twitter/X, Instagram, YouTube, Facebook, LinkedIn).

Scope: this module is URL-only by design. The caller supplies an already
validated/normalized URL (see url_utils.is_url / normalize_url); this
module never searches for, guesses, or resolves a URL from a brand name
or keyword (e.g. a query like "boat audio" is out of scope here - that
would need a separate search/lookup step upstream, not this file).

Fetch strategy: plain HTTP first (httpx), with a single Playwright
browser fallback when the page is blocked (403/429/5xx), unreachable,
looks like an unrendered JS shell - a 200 response with almost no links
and almost no text (confirmed live on ajio.com and online.kfc.co.in,
which both return just a bare <title> and a "you need JavaScript"
message over plain HTTP, so there's nothing to extract without letting
the client-side app render) - or looks like an anti-bot challenge page
served with a 200 instead of a 4xx/5xx (confirmed live on ajio.com,
whose Playwright-rendered fallback can itself land on an "Access Denied"
interstitial rather than the real homepage; see _looks_like_anti_bot_page
/ _pw_render_is_usable). Every Playwright-fallback acceptance point
re-checks the anti-bot signal before trusting the render, since
"the request didn't raise an exception" and "this is the real page" are
not the same thing on a protected site.

Social handle sources, checked in priority order per platform: the
page's own twitter:site/twitter:creator meta tag (Twitter only), then
schema.org Organization/WebSite JSON-LD "sameAs" - a more authoritative
signal than a footer icon link, and one that still works on large
corporate/multi-region sites (Apple/Samsung/Sony-shaped) whose rendered
footer social icons may be sparse, region-specific, or built without a
plain <a href> to scan - then generic anchor-tag scanning, then
og:/social/same_as meta tags as a last resort.

Returns a fixed-shape dict compatible with the legacy discover_company()
output, so downstream consumers (product discovery, scrapers, dashboard)
require no changes.
"""

import json
import logging
import re
from typing import Dict, List, Tuple
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from url_utils import derive_company_name
from scrapers.browser_utils import run_playwright_async

logger = logging.getLogger(__name__)

DISCOVERY_VERSION = "website-metadata-extractor-v2"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_FETCH_TIMEOUT = httpx.Timeout(12.0, connect=6.0)

_SOCIAL_DOMAIN_MAP = {
    "twitter.com": "twitter", "x.com": "twitter",
    "instagram.com": "instagram",
    "youtube.com": "youtube", "youtu.be": "youtube",
    "facebook.com": "facebook", "fb.com": "facebook",
    "linkedin.com": "linkedin",
}

# Per-platform first-path-segment blocklist: share/intent/widget/utility
# URLs that live on the platform's own domain but are never a company's
# profile. twitter/instagram are unchanged from before. facebook is new -
# there was no facebook blocklist at all previously, which is
# inconsistent with twitter/instagram and a real gap (Facebook share/
# like-button URLs on the platform's own domain are common boilerplate).
# youtube is new too, and is now load-bearing: a bare custom-channel URL
# is treated as a handle by default (see _extract_handle), so the known
# non-channel paths must be excluded explicitly or they'd be misread as
# channel handles.
_BLOCKED_HANDLES = {
    "twitter": {
        "home", "intent", "i", "share", "search", "hashtag", "explore",
        "settings", "login", "signup", "messages", "compose", "notifications",
    },
    "instagram": {
        "p", "reel", "reels", "stories", "explore", "accounts", "direct", "tv",
    },
    "facebook": {
        "sharer", "share", "dialog", "plugins", "tr", "l.php", "login",
        "checkpoint", "policies", "legal", "privacy", "terms", "help",
        "business", "ads", "watch", "marketplace", "events", "groups", "pages",
    },
    "youtube": {
        "channel", "user", "c", "watch", "playlist", "results", "feed",
        "embed", "shorts", "hashtag", "trending", "premium", "gaming",
        "kids", "live", "clip", "post", "subscription_manager", "account",
        "upload", "attribution_link", "redirect", "oembed", "subscribe_embed",
        "about", "creators", "ads",
    },
}

_ICON_RELS = (
    "icon", "shortcut icon", "apple-touch-icon",
    "apple-touch-icon-precomposed", "mask-icon",
)

_BLOCKED_STATUS_CODES = {403, 429, 500, 502, 503, 504}

_PW_LAUNCH_ARGS = [
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-dev-shm-usage",
    "--disable-blink-features=AutomationControlled",
]

_PW_NAV_TIMEOUT_MS = 15000

# Bumped from 1000ms: this fallback now also carries unrendered SPA
# shells (see _looks_like_js_shell below), which need a little more time
# after domcontentloaded to finish their own fetch-then-render cycle
# before title/meta tags are meaningful.
_PW_POST_LOAD_WAIT_MS = 1500

# A "200 OK" response with almost no links and almost no text is an
# unrendered JS shell, not a real homepage - confirmed live on ajio.com
# and online.kfc.co.in (0 links, well under 100 characters of text).
# Every real homepage checked in the same pass (Samsung, V-Guard, Apple,
# Sony, Mamaearth, Lenskart) had 50+ links and thousands of characters,
# so there's a wide safety margin between "empty shell" and "real page."
_THIN_HTML_MIN_LINKS = 5
_THIN_HTML_MIN_TEXT_CHARS = 500

_SOCIAL_URL_TEMPLATES = {
    "linkedin": "https://www.linkedin.com/{}",
    "twitter": "https://x.com/{}",
    "instagram": "https://www.instagram.com/{}",
    "youtube": "https://www.youtube.com/@{}",
    "facebook": "https://www.facebook.com/{}",
}

# YouTube channel IDs are always exactly 24 characters starting with "UC"
# (confirmed live on Mamaearth's own footer link, July 31, 2026:
# youtube.com/channel/UC5qhqSIcahBKKMEdLPFqeVA -> _extract_handle()'s
# "channel"/"user"/"c" branch was taking parts[1] verbatim regardless of
# whether it's a real handle or a raw ID, so this ID was landing straight
# in the "/@{}" handle template below, producing the broken
# youtube.com/@UC5qhqSIcahBKKMEdLPFqeVA instead of the correct
# youtube.com/channel/UC5qhqSIcahBKKMEdLPFqeVA. _build_social_url() checks
# for this shape and routes it to the /channel/ URL instead.
_YT_CHANNEL_ID_RE = re.compile(r"^UC[\w-]{22}$")


async def extract_website_metadata(url: str) -> Dict[str, object]:
    result = _empty_result(url)
    if not url:
        return result

    html, final_url = await _fetch_homepage(url)
    if not html:
        return result

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception as exc:
        logger.warning("Website metadata: failed to parse HTML for %s: %s", url, exc)
        return result

    base_url = final_url or url
    # "website" reports back the caller's own input URL, not wherever a
    # redirect chain landed - confirmed via live log that Samsung's
    # homepage 301/302-redirects through a seasonal campaign page
    # (.../smartphones/galaxy-z-fold8/?page=home), which is not what this
    # field should show. base_url (the actual, possibly redirected, final
    # location) is still used below for resolving relative links (favicon,
    # social hrefs, domain-token scoring), since that needs to be correct
    # relative to wherever the HTML was actually fetched from.
    result["website"] = url
    result["company_name"] = _extract_company_name(soup, base_url)
    result["logo"] = _extract_logo(soup, base_url)

    socials = _extract_social_handles(soup, base_url)
    for platform, handle in socials.items():
        result[platform] = handle
        result[f"{platform}_url"] = _build_social_url(handle, platform)

    logger.info(
        "Website metadata extracted for %s: name=%r logo=%r twitter=%r "
        "instagram=%r youtube=%r facebook=%r linkedin=%r",
        url, result["company_name"], result["logo"], result["twitter"],
        result["instagram"], result["youtube"], result["facebook"], result["linkedin"],
    )
    return result


async def _fetch_homepage(url: str) -> "Tuple[str, str]":
    try:
        async with httpx.AsyncClient(
            timeout=_FETCH_TIMEOUT,
            follow_redirects=True,
            verify=False,
            headers=_HEADERS,
        ) as client:
            response = await client.get(url)
            final_url = str(response.url) or url

            if response.status_code >= 400:
                if response.status_code in _BLOCKED_STATUS_CODES:
                    logger.info(
                        "Website metadata: %s returned HTTP %s (bot-protected) "
                        "- skipping HTTP retries. Using Playwright fallback.",
                        url, response.status_code,
                    )
                    pw_html, pw_url = await _fetch_homepage_playwright(url)
                    # Confirmed live on ajio.com, July 31, 2026: the
                    # Playwright fallback can "succeed" (no exception) while
                    # still only rendering the site's own anti-bot
                    # interstitial ("Access Denied", 403 again on every
                    # request) rather than the real homepage - a thin page
                    # by the same measure _looks_like_js_shell() already
                    # uses for the plain-HTTP case. Without this check that
                    # interstitial was extracted as if it were real content,
                    # producing company_name="Access Denied". Unlike the
                    # JS-shell branch below - where even a thin *rendered*
                    # result is deliberately kept because it's still more
                    # than the original empty shell - here the plain-HTTP
                    # response was already a hard block, so a still-thin
                    # Playwright result means the block held, not that we
                    # gained anything by rendering it. _pw_render_is_usable
                    # also rejects an anti-bot-marker match specifically
                    # (see its docstring) even on a render that isn't thin
                    # by link/text count - a challenge page can carry enough
                    # boilerplate to clear that bar on its own.
                    if pw_html and _pw_render_is_usable(pw_html) and not _looks_like_js_shell(pw_html):
                        return pw_html, pw_url
                    if pw_html:
                        logger.info(
                            "Website metadata: Playwright fallback for %s "
                            "rendered a page, but it still looks like a "
                            "thin anti-bot interstitial rather than the "
                            "real homepage - treating this as blocked "
                            "rather than extracting metadata from it.", url,
                        )
                else:
                    logger.info(
                        "Website metadata: %s returned HTTP %s", url, response.status_code
                    )
                return "", final_url

            html = response.text

            # A 200 status doesn't mean the body is real: some anti-bot
            # layers serve a challenge/interstitial page with a 200 rather
            # than a 4xx/5xx (ported from product_discovery.py's
            # ANTI_BOT_MARKERS, the same signal used there for exactly
            # this case). This is checked separately from
            # _looks_like_js_shell below because a challenge page often
            # carries enough boilerplate script/style/footer content to
            # clear that check's link/text-count bar on its own, even
            # though it's just as unusable as an empty shell.
            if _looks_like_anti_bot_page(html):
                logger.info(
                    "Website metadata: %s returned HTTP 200 but looks like "
                    "an anti-bot challenge page - using Playwright fallback.",
                    url,
                )
                pw_html, pw_url = await _fetch_homepage_playwright(url)
                if pw_html and _pw_render_is_usable(pw_html):
                    return pw_html, pw_url
                return "", final_url

            if _looks_like_js_shell(html):
                logger.info(
                    "Website metadata: %s returned HTTP 200 but looks like an "
                    "unrendered JS shell - using Playwright fallback.", url,
                )
                pw_html, pw_url = await _fetch_homepage_playwright(url)
                # Prefer the rendered version if we got one, even if it's
                # still thin (still strictly more than the plain-HTTP
                # shell) - _pw_render_is_usable only rejects an anti-bot
                # match, not thinness on its own, so a genuinely thin but
                # real render is still accepted here. If Playwright is
                # unavailable/failed/still blocked, fall back to the thin
                # HTML we already have - a generic <title> beats an
                # entirely empty result.
                if pw_html and _pw_render_is_usable(pw_html):
                    return pw_html, pw_url
                return html, final_url

            return html, final_url
    except Exception as exc:
        logger.warning("Website metadata: fetch failed for %s: %s", url, exc)
        logger.info("Website metadata: %s - using Playwright fallback.", url)
        pw_html, pw_url = await _fetch_homepage_playwright(url)
        if pw_html and _pw_render_is_usable(pw_html) and not _looks_like_js_shell(pw_html):
            return pw_html, pw_url
        return "", url


def _looks_like_js_shell(html: str) -> bool:
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return False
    links = len(soup.find_all("a", href=True))
    text_len = len(soup.get_text(strip=True))
    return links < _THIN_HTML_MIN_LINKS and text_len < _THIN_HTML_MIN_TEXT_CHARS


# Ported from product_discovery.py's ANTI_BOT_MARKERS/_looks_like_anti_bot_page
# (same codebase, already proven there) rather than reinventing an
# equivalent list - keeping the two in sync is easier than maintaining
# parallel near-duplicates. Bounded to short pages only: a real product
# or article page that happens to mention "access denied" once somewhere
# in thousands of characters of real content shouldn't be flagged, only
# a page that's essentially JUST the challenge screen.
_ANTI_BOT_MARKERS = (
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
_ANTI_BOT_MAX_LEN = 6000


def _looks_like_anti_bot_page(html: str) -> bool:
    if not html or len(html) > _ANTI_BOT_MAX_LEN:
        return False
    lowered = html.lower()
    return any(marker in lowered for marker in _ANTI_BOT_MARKERS)


def _pw_render_is_usable(html: str) -> bool:
    """Guards every acceptance point above against a Playwright render
    that "succeeded" (no exception) while still only showing an anti-bot
    challenge page rather than the real homepage - confirmed live on
    ajio.com, where the fallback rendered an "Access Denied" interstitial
    on every attempt. Deliberately does NOT also require the render to
    pass _looks_like_js_shell's thinness bar: a genuinely still-thin-but-
    real render is fine to use (see the JS-shell branch above, which
    keeps its own explicit thinness tolerance) - this specifically
    catches an anti-bot page dressed up as content, not thinness itself.
    """
    return bool(html) and not _looks_like_anti_bot_page(html)


async def _fetch_homepage_playwright(url: str) -> "Tuple[str, str]":
    try:
        from playwright.async_api import async_playwright
    except Exception as exc:
        logger.warning(
            "Website metadata: Playwright unavailable (%s); cannot use "
            "the browser fallback for %s.", exc, url,
        )
        return "", ""

    async def _coro() -> "Tuple[str, str]":
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True, args=_PW_LAUNCH_ARGS)
            try:
                context = await browser.new_context(
                    user_agent=_HEADERS["User-Agent"],
                    locale="en-US",
                    viewport={"width": 1280, "height": 900},
                )
                page = await context.new_page()
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=_PW_NAV_TIMEOUT_MS)
                    await page.wait_for_timeout(_PW_POST_LOAD_WAIT_MS)
                    html = await page.content()
                    final_url = page.url or url
                    logger.info(
                        "Website metadata: Playwright fallback succeeded for %s.", url
                    )
                    return html, final_url
                finally:
                    await page.close()
                    await context.close()
            finally:
                await browser.close()

    try:
        # Run on the same dedicated-thread-with-Proactor-loop path
        # product_discovery.py already uses successfully (see
        # browser_utils.run_playwright_async / BrowserManager.run_session).
        # Calling async_playwright() directly on FastAPI's own request-
        # handling event loop crashes on Windows: Playwright's driver needs
        # asyncio.create_subprocess_exec, which raises NotImplementedError
        # on a SelectorEventLoop (confirmed via live log - this fallback
        # was silently returning nothing on every Windows run before this
        # fix, which is also why company/product discovery never found any
        # social media links: the plain-HTTP fetch alone can't render
        # JS-injected footer/header links, and this fallback was the only
        # other path that could).
        return await run_playwright_async(_coro)
    except Exception as exc:
        logger.warning("Website metadata: Playwright fallback failed for %s: %s", url, exc)
        return "", ""


# Separators a marketing <title> tag commonly uses between the brand name
# and a description/tagline, IN EITHER ORDER - see _best_title_candidate
# for why order can't be assumed. ": " (bare colon) was added after
# confirming live on boat-lifestyle.com that a real title can use one:
# "Buy Earbuds, Headphones, Earphones at India's No.1 Earwear Brand: boAt"
# has no " | "/" - "/etc., only a bare colon, with the brand name LAST,
# not first.
_TITLE_SEPARATORS = (" | ", " – ", " — ", " - ", " :: ", ": ")

# Known two-label ccTLD suffixes where the registrable domain is actually
# THREE labels (e.g. "kfc.co.in", "sony.co.in" - not "co.in"). Without
# this, a naive "second-to-last label" guess at the brand token would
# grab "co" instead of "kfc"/"sony" for hosts like "online.kfc.co.in" or
# "www.sony.co.in". Not a full public-suffix-list implementation, just
# the patterns this app is realistically going to see (India-focused
# company analysis, per ARCHITECTURE.md) plus a few common lookalikes -
# extend this set if a new site needs it.
_MULTI_LABEL_SUFFIXES = {
    "co.in", "org.in", "net.in", "gov.in", "ac.in", "res.in", "gen.in",
    "firm.in", "ind.in", "co.uk", "com.au", "co.nz", "com.sg",
}


def _domain_tokens(base_url: str) -> set:
    """Rough brand-name tokens straight from the domain, used only to
    score which segment of a noisy <title> is most likely the real brand
    name (see _best_title_candidate). Deliberately self-contained rather
    than reusing derive_company_name (url_utils) - this only needs a
    bag of tokens to score against, not a display-ready name, so it
    doesn't need to match that function's exact output shape.
    """
    host = urlparse(base_url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    labels = host.split(".")
    if len(labels) >= 3 and ".".join(labels[-2:]) in _MULTI_LABEL_SUFFIXES:
        main_label = labels[-3]
    elif len(labels) >= 2:
        main_label = labels[-2]
    elif labels:
        main_label = labels[0]
    else:
        return set()
    return {t for t in re.split(r"[-_]+", main_label) if t}


def _best_title_candidate(title: str, base_url: str) -> str:
    """Pick the most plausible brand-name segment out of a <title> tag.

    FIX: the previous version only ever took the segment BEFORE the
    first separator found, on a "Brand | Tagline" assumption. That
    assumption fails when the title is shaped "Tagline: Brand" instead -
    confirmed live on boat-lifestyle.com, whose real <title> is "Buy
    Earbuds, Headphones, Earphones at India's No.1 Earwear Brand: boAt"
    (long SEO sentence, THEN a bare colon, THEN the actual brand name
    last). Blindly taking the first segment on ": " would return the
    entire SEO sentence as company_name, not "boAt".

    Rather than hard-coding which side of a given separator is the brand,
    this collects every segment produced by splitting on ANY known
    separator (both the part before and the part after), plus the whole
    title as a last-resort candidate, and scores each by token overlap
    against the site's own domain (see _domain_tokens) - preferring
    higher overlap, then shorter length (a short, high-overlap segment is
    more likely to BE the brand name than a long sentence that happens to
    also contain those words). This mirrors the "score every candidate,
    take the best" shape already used for Google Maps business matching
    in google_scraper.py, applied here to the same underlying problem:
    picking the real name out of noisy text.

    Falls back to the old first-segment behavior when there's no domain
    signal to score against, or when no candidate actually overlaps the
    domain at all - so untested titles keep their previous behavior
    rather than being reshuffled by a guess with zero evidence behind it.
    """

    def _first_segment_fallback() -> str:
        for sep in _TITLE_SEPARATORS:
            if sep in title:
                head = title.split(sep)[0].strip()
                if head:
                    return head
        return title

    domain_tokens = _domain_tokens(base_url)
    if not domain_tokens:
        return _first_segment_fallback()

    candidates = {title}
    for sep in _TITLE_SEPARATORS:
        if sep in title:
            head, _, tail = title.partition(sep)
            if head.strip():
                candidates.add(head.strip())
            if tail.strip():
                candidates.add(tail.strip())

    def _score(candidate: str) -> Tuple[int, int]:
        cand_tokens = {t for t in re.findall(r"[a-z0-9]+", candidate.lower())}
        overlap = len(cand_tokens & domain_tokens)
        return (overlap, -len(candidate))

    best = max(candidates, key=_score)
    if _score(best)[0] > 0:
        return best
    return _first_segment_fallback()


def _extract_company_name(soup: BeautifulSoup, base_url: str) -> str:
    og_site = soup.find("meta", attrs={"property": "og:site_name"})
    if og_site and og_site.get("content", "").strip():
        return og_site["content"].strip()

    app_name = soup.find("meta", attrs={"name": "application-name"})
    if app_name and app_name.get("content", "").strip():
        return app_name["content"].strip()

    if soup.title:
        # get_text(), not .string: .string is None (not "") whenever the
        # <title> tag has more than one child node - e.g. a stray inline
        # tag or comment inside it - which silently skipped a perfectly
        # usable title and fell straight to the URL-derived fallback.
        # separator=" " + the final whitespace collapse avoid gluing
        # adjacent text nodes together with no space between them.
        title = " ".join(soup.title.get_text(separator=" ", strip=True).split())
        if title:
            return _best_title_candidate(title, base_url)

    return derive_company_name(base_url)


def _extract_logo(soup: BeautifulSoup, base_url: str) -> str:
    for link in soup.find_all("link", href=True):
        rel_attr = link.get("rel", "")
        rel = " ".join(rel_attr).lower() if isinstance(rel_attr, list) else str(rel_attr).lower()
        if any(icon_rel in rel for icon_rel in _ICON_RELS):
            resolved = _resolve_url(link["href"], base_url)
            if resolved:
                return resolved

    og_image = soup.find("meta", attrs={"property": "og:image"})
    if og_image and og_image.get("content", "").strip():
        resolved = _resolve_url(og_image["content"].strip(), base_url)
        if resolved:
            return resolved

    return _resolve_url("/favicon.ico", base_url)


def _extract_jsonld_social_candidates(soup: BeautifulSoup) -> List[str]:
    """Pull social profile URLs out of schema.org Organization/WebSite
    JSON-LD ("sameAs"), when present.

    Large corporate/multi-region sites - Apple, Samsung, Sony, exactly
    the shape ARCHITECTURE.md flags as untested for footer/header link
    scanning ("these sites' footer/header social links may be structured
    very differently... not tested live") - commonly declare their
    official accounts this way even when the rendered footer's social
    icons are sparse, region-specific, or built from SVG sprites with no
    plain <a href="instagram.com/..."> to scan. It's also a more
    authoritative signal than an arbitrary footer icon link in general:
    sameAs is meant to name the brand's canonical accounts, not whatever
    share-widget or campaign-specific page happens to sit nearby in the
    DOM (see the Sony "Tweet this page" note above). Best-effort - a
    missing or malformed JSON-LD block just yields no candidates, never
    an error."""
    urls: List[str] = []
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            raw = tag.string or tag.get_text() or ""
            if not raw.strip():
                continue
            data = json.loads(raw)
        except Exception:
            continue

        # A JSON-LD block can be a single object, a list of objects, or
        # an object with an "@graph" array holding several - normalize
        # to a flat list of dicts to scan for a "sameAs" key. Not
        # restricted to "@type": "Organization" specifically since sites
        # vary (Corporation, LocalBusiness, WebSite all show up with a
        # sameAs list in the wild) and a missing/absent key elsewhere is
        # harmless to check for.
        if isinstance(data, dict) and isinstance(data.get("@graph"), list):
            entities = data["@graph"]
        elif isinstance(data, list):
            entities = data
        else:
            entities = [data]

        for entity in entities:
            if not isinstance(entity, dict):
                continue
            same_as = entity.get("sameAs")
            if isinstance(same_as, str):
                urls.append(same_as)
            elif isinstance(same_as, list):
                urls.extend(u for u in same_as if isinstance(u, str))
    return urls


def _extract_social_handles(soup: BeautifulSoup, base_url: str) -> Dict[str, str]:
    found: Dict[str, str] = {}

    raw_candidates: List[str] = []

    # The page's own declared Twitter handle, checked first: confirmed
    # accurate on every real site checked (Samsung, Sony, V-Guard,
    # Mamaearth, Lenskart) and immune to the DOM-order mixups a "Tweet
    # this page" widget link can cause (Sony's own footer has one right
    # next to the real profile link).
    twitter_meta = soup.find("meta", attrs={"name": "twitter:site"}) or soup.find(
        "meta", attrs={"name": "twitter:creator"}
    )
    if twitter_meta and twitter_meta.get("content", "").strip():
        handle = twitter_meta["content"].strip().lstrip("@").split("?")[0].strip()
        if handle:
            raw_candidates.append(f"https://twitter.com/{handle}")

    # Structured JSON-LD "sameAs" next - see
    # _extract_jsonld_social_candidates for why this ranks above a
    # generic footer/header link scan.
    raw_candidates.extend(_extract_jsonld_social_candidates(soup))

    raw_candidates.extend(tag["href"] for tag in soup.find_all("a", href=True))

    for tag in soup.find_all("meta", content=True):
        prop = (tag.get("property") or tag.get("name") or "").lower()
        if prop.startswith("og:") or "social" in prop or "same_as" in prop:
            raw_candidates.append(tag["content"])

    for raw in raw_candidates:
        resolved = _resolve_url(raw, base_url)
        if not resolved:
            continue
        host = urlparse(resolved.lower()).netloc
        if host.startswith("www."):
            host = host[4:]
        # Exact-or-subdomain match, not a bare substring check (a bare
        # `domain in host` would also match a lookalike host such as
        # "notfacebook.com.evil.com").
        platform = next(
            (
                p for domain, p in _SOCIAL_DOMAIN_MAP.items()
                if host == domain or host.endswith(f".{domain}")
            ),
            None,
        )
        if not platform or platform in found:
            continue
        handle = _extract_handle(resolved, platform)
        if handle:
            found[platform] = handle

    return found


def _extract_handle(url: str, platform: str) -> str:
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if not parts:
        return ""

    blocked = _BLOCKED_HANDLES.get(platform, set())

    if platform == "youtube":
        if "youtu.be" in parsed.netloc.lower():
            return ""
        first = parts[0]
        if first.startswith("@"):
            handle = first.lstrip("@")
        elif first.lower() in {"channel", "user", "c"} and len(parts) > 1:
            handle = parts[1].lstrip("@")
        elif first.lower() not in blocked:
            # Bare/legacy channel URL, e.g. youtube.com/SamsungMobileIndia
            # - confirmed live in Samsung's own footer link. Previously
            # this returned "" even though the link was right there.
            handle = first
        else:
            return ""
        return "" if handle.lower() in blocked else handle

    if platform == "linkedin":
        if parts[0].lower() == "company" and len(parts) > 1:
            return f"company/{parts[1]}"
        if parts[0].lower() == "school" and len(parts) > 1:
            return f"school/{parts[1]}"
        return ""

    handle = parts[0].lstrip("@")
    return "" if handle.lower() in blocked else handle


def _build_social_url(handle: str, platform: str) -> str:
    if not handle:
        return ""
    if platform == "youtube" and _YT_CHANNEL_ID_RE.match(handle):
        return f"https://www.youtube.com/channel/{handle}"
    template = _SOCIAL_URL_TEMPLATES.get(platform)
    return template.format(handle) if template else ""


def _resolve_url(raw: str, base_url: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    if raw.startswith("//"):
        raw = f"https:{raw}"
    elif raw.startswith(("/", "./", "../")):
        raw = urljoin(base_url, raw)
    if not raw.startswith(("http://", "https://")):
        return ""
    return raw


def _empty_result(url: str) -> Dict[str, object]:
    return {
        "company_name": "",
        "website": url or "",
        "logo": "",
        "twitter": "",
        "instagram": "",
        "youtube": "",
        "facebook": "",
        "linkedin": "",
        "twitter_url": "",
        "instagram_url": "",
        "youtube_url": "",
        "facebook_url": "",
        "linkedin_url": "",
        "google_business": "",
        "discovery_version": DISCOVERY_VERSION,
        "website_type": "unknown",
        "website_verified": False,
        "website_confidence": "low",
        "discovery_notes": [],
    }
