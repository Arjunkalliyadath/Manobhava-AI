"""
Module Name
-----------
config.py

Purpose
-------
Single source of truth for every tunable limit and threshold used across
the pipeline: how many products/comments to collect per platform, how long
to wait on slow navigations, and the keyword/scoring tables Product
Discovery uses to rank and filter candidate products.

Responsibilities
----------------
- Hold per-platform comment/review volume caps (MAX_GOOGLE_REVIEWS,
  MAX_YOUTUBE_COMMENTS, MAX_REDDIT_COMMENTS, MAX_INSTAGRAM_COMMENTS,
  MAX_TWITTER_POSTS) and the product-selection ceiling
  (MAX_SELECTABLE_PRODUCTS) — the values that most directly trade comment
  volume against the 1-3 minute total execution time budget.
- Hold navigation/retry tuning (NAV_TIMEOUT_MS and its MIN/MAX adaptive
  bounds, NAVIGATION_RETRIES, MAX_SCROLL_ITERATIONS, SCROLL_IDLE_LIMIT)
  used by every Playwright-based scraper.
- Hold Product Discovery's text-filtering tables (BLACKLIST_WORDS,
  PROMINENCE_TIERS, EXACT_BADGE_TERMS, FEATURE_KEYWORDS) and candidate
  length bounds (MIN_CANDIDATE_LEN / MAX_CANDIDATE_LEN).
- Hold GOOGLE_REVIEW_CACHE_TTL_SECONDS, the TTL for caching business-level
  Google reviews across products in one analysis run (Google reviews are
  per-business, not per-product, so this avoids re-scraping Maps once per
  selected product).

On the inline comments in this file
------------------------------------
Several constants below carry comments citing a specific date, a live-run
measurement, or a person's own stated priority. These are load-bearing
history, not clutter: most of the volume caps here were deliberately cut
from earlier, larger values specifically to fit the 3-minute execution
budget, and the comments record the evidence that drove each cut (e.g. a
live log showing Sentiment Analysis alone taking 110s of a 188.56s run).
Re-raising a cap without accounting for that evidence risks reintroducing
the exact timing problem it was cut to fix — see sentiment.py and
ARCHITECTURE.md for the fuller picture.

Dependencies
------------
Standard library only: `typing.List` (for the keyword/tier table type
hints). No I/O, no imports from elsewhere in this project — every other
module imports FROM this one, never the other way around.
"""

from typing import List

# Number of products returned by Product Discovery.
# We only analyze 5 later, but we want the user to be able
# to choose from the COMPLETE catalogue.

MAX_PRODUCTS: int = 500
MAX_SELECTABLE_PRODUCTS: int = 3  # how many products the person can pick for one analysis run (was 5) - each one multiplies scraper work across every platform, so this is the main lever keeping even the "extreme case" under the 3-minute ceiling. Enforced in both the frontend (templates/select_products.html) and the backend (app.py, both /analyze and /analyze_selected) so it holds even for requests that bypass the UI.

MAX_PARALLEL_TASKS: int = 5

MAX_COMMENTS_PER_PRODUCT: int = 20

# ---------------------------------------------------------------------------
# Per-platform scrape volume caps.
#
# These bound how many raw reviews/comments/posts each scraper will try to
# collect before giving up. They are intentionally separate from
# MAX_COMMENTS_PER_PRODUCT above (which controls how many of the collected
# comments are later used for analysis) so that raising the analysis sample
# size and raising how much raw data is scraped can be tuned independently.
#
# Raising these lets a scraper keep going past the old ~20-item cutoff and
# collect as much as a product actually has, up to the number below. Each
# scraper still respects its own internal TIME_BUDGET_SECONDS and the
# scroll-idle detection below, so it will return early (with whatever
# partial results it has) if the platform simply has fewer items, if
# nothing new is loading anymore, or if time runs out.
MAX_GOOGLE_REVIEWS: int = 30  # cut from 200 - priority shifted to 3-min total time; genuine over volume
MAX_YOUTUBE_COMMENTS: int = 70  # cut again from 100 - see the note this replaces for the 200->100 cut and its reasoning. Live log with 100/product still showed Sentiment Analysis alone taking 110s for 742 total comments (58.5% of a 188.56s run) once genuine per-product data was actually flowing in. Combined with GENERAL_YOUTUBE_TIMEOUT_SECONDS (app.py) giving the brand-wide "General" job a much shorter budget than product-specific jobs, this is aimed squarely at the person's own framing: "no need of 300+ ... minimum of 100 genuine comments ... depending on time taking" - time is the overriding constraint when it conflicts with volume.
MAX_TWITTER_POSTS: int = 100
MAX_INSTAGRAM_COMMENTS: int = 25  # cut from 100 - priority shifted to 3-min total time; genuine over volume

# Added - reddit_scraper.py previously hardcoded its own MAX_TOTAL_COMMENTS = 60
# locally instead of reading a config.py cap like every other platform does,
# which is well under the 100-200-per-product target. Raised well past that
# target for the same reason MAX_YOUTUBE_COMMENTS was: the real ceiling for
# Reddit is how many relevant discussion threads/comments actually exist and
# how much of the scraper's own internal TIME_BUDGET_SECONDS is left, not this
# number - see reddit_scraper.py's _scrape_sync for the matching tier-
# accumulation fix (it used to stop at the first search tier that returned
# ANY comment at all, even a thin one, instead of trying broader tiers too).
MAX_REDDIT_COMMENTS: int = 70  # cut again from 100, same live-log evidence and reasoning as MAX_YOUTUBE_COMMENTS just above.

# ---------------------------------------------------------------------------
# Scroll / navigation tuning shared by all scrapers.
#
# MAX_SCROLL_ITERATIONS: hard ceiling on how many scroll steps a scraper will
#   perform on any single page, regardless of whether new content is still
#   appearing. This is a safety valve, not the primary stopping condition.
# SCROLL_IDLE_LIMIT: number of consecutive scrolls that must produce zero new
#   items before a scraper concludes "nothing more is loading" and stops.
# NAVIGATION_RETRIES: number of times a scraper will retry a failed page
#   navigation (timeout, transient network error) before giving up on that
#   URL and moving on / falling back.
MAX_SCROLL_ITERATIONS: int = 100
SCROLL_IDLE_LIMIT: int = 9

# Reduced from 3 -> 1. A blocked/dead site (login wall, bot-check,
# rate-limit) essentially never recovers on retry #2/#3 - it just burns
# the scraper's whole internal time budget waiting on a page that was
# never going to load, and it gets cut off cold by app.py's outer hard
# timeout with 0 results. One retry gives transient network blips a
# chance to clear while keeping the fail-fast guarantee below.
NAVIGATION_RETRIES: int = 1

# Hard per-navigation-attempt timeout. Previously each scraper hardcoded
# its own value (8000-10000ms), and with NAVIGATION_RETRIES=3 that meant
# up to ~40s could be spent just waiting on a single blocked URL before
# any scroll/collect logic ever ran. Centralized here so every scraper
# fails fast and consistently: worst case for one URL is now
# NAV_TIMEOUT_MS * (NAVIGATION_RETRIES + 1) plus a small backoff, i.e.
# ~10-11s instead of ~40s.
NAV_TIMEOUT_MS: int = 5000

# Ceiling for the adaptive navigation timeout used by the scrapers
# (google/twitter/instagram/youtube). NAV_TIMEOUT_MS above is now treated
# as the fast-page floor rather than a fixed value: a scraper scales the
# timeout it hands to page.goto() up to this ceiling based on how much of
# its own internal time budget is still left, so a merely slow (not dead)
# page isn't cut off after a token 5s, while a navigation call is still
# never allowed to ask for more time than the scraper actually has left.
NAV_TIMEOUT_MS_MAX: int = 12000

# Floor guaranteed to the FIRST navigation attempt of any scrape, once the
# browser/context/page for that attempt already exist. The internal time
# budget clock for each scraper starts only after browser launch + context
# creation + page creation have finished (see each scraper's _run/_run_sync
# entry point), specifically so this floor is meaningful and isn't silently
# eaten by Chromium startup time.
NAV_TIMEOUT_MS_MIN: int = 8000

# TTL for the per-business Google Reviews cache (google_scraper.py).
# Google Reviews belong to the company/business, not to an individual
# product, so once we've scraped a business's Maps listing we reuse that
# same result for every product job in the same analysis run instead of
# opening Maps again per product. A single analysis run completes in well
# under this window, so every job in one run shares one scrape; a fresh
# analysis of the same company later still gets a fresh scrape once the
# TTL has passed.
GOOGLE_REVIEW_CACHE_TTL_SECONDS: int = 900

BLACKLIST_WORDS: List[str] = [

    "page not found", "404", "not found", "page",

    "all collections", "collection", "collections",

    "community", "deal", "deals", "budget",

    "under", "above", "best headphones", "best earphones",

    "shop all", "shop", "accessories", "accessory", "brands", "brand",

    "about", "about us", "contact", "contact us", "wishlist",

    "support", "help", "helpdesk",

    "replacement cable",

    "filters", "filter", "sort", "sort by", "view all", "see all",
    "explore", "load more", "show more",

    "menu", "navigation", "nav", "footer", "sidebar", "breadcrumb",
    "search", "cart", "checkout", "login", "log in", "sign in",
    "sign up", "register", "account", "my account",
    "blog", "news", "faq", "faqs", "terms", "privacy", "careers",
    "jobs", "press", "media", "investors", "sitemap", "cookie",
    "sign out", "logout", "close", "skip to content", "subscribe",
    "newsletter", "language", "currency", "back", "next", "previous",
    "read more", "learn more",
]

PROMINENCE_TIERS: List[List[str]] = [
    ["best seller", "bestseller", "best-seller", "top rated", "top seller"],
    ["featured", "editor's pick", "editors pick", "staff pick"],
    ["trending", "most popular", "popular"],
    ["latest", "new arrival", "just launched", "newly added", "new"],
]

EXACT_BADGE_TERMS: List[str] = [
    "new", "sale", "sold out", "best seller", "bestseller", "best-seller",
    "featured", "trending", "hot", "limited edition", "exclusive",
    "top rated", "popular", "most popular", "staff pick", "editor's pick",
    "editors pick", "top seller", "new arrival", "just launched",
    "newly added", "in stock", "out of stock", "low stock", "coming soon",
    "just in", "back in stock",
]

MIN_CANDIDATE_LEN: int = 2
MAX_CANDIDATE_LEN: int = 45

FEATURE_KEYWORDS: List[str] = [
    "price", "quality", "delivery", "shipping", "packaging", "sound",
    "battery", "comfort", "design", "durability", "customer service",
    "service", "support", "warranty", "size", "fit", "material",
    "performance", "value", "build quality", "noise cancellation", "app",
]