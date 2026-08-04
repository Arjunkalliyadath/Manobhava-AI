"""
Module Name
-----------
utils.py

Purpose
-------
Small, dependency-free text-cleaning helpers shared by every scraper and
by the merge/dedup step in app.py, ahead of sentiment analysis.

Responsibilities
-----------------
- `clean_comment`: strip boilerplate/noise characters from a raw scraped
  comment.
- `normalize_text`: collapse whitespace for consistent downstream
  comparison and display.
- `remove_links`: strip URLs out of comment text before sentiment
  analysis and dedup, so a shared link doesn't affect either.
- `unique_comments`: order-preserving de-duplication of (platform, text)
  pairs, used when merging comments collected across all six sources.

Inputs / Outputs
-----------------
All four functions take and return plain strings or lists of strings —
no external objects, no I/O.

Dependencies
------------
Standard library only: `re`, `typing`.
"""

import re
from typing import List, Tuple


# Cleans a comment by removing extra whitespace and trimming leading/trailing spaces
def clean_comment(text: str) -> str:
    if not text:
        return ""
    
    # Replace multiple spaces, tabs, or newlines with a single space
    text = re.sub(r"\s+", " ", text)
    
    # Remove leading and trailing whitespace
    return text.strip()


# Normalizes text by standardizing whitespace and removing invisible characters
def normalize_text(text: str) -> str:
    
    # Replace multiple whitespace characters with a single space
    text = re.sub(r"\s+", " ", text or "")
    
    # Remove zero-width space characters
    text = re.sub(r"\u200b", "", text)
    
    # Remove leading and trailing whitespace
    return text.strip()


# Removes URLs from a text string
def remove_links(text: str) -> str:
    
    # Remove links starting with http, https, or www
    return re.sub(r"https?://\S+|www\.\S+", "", text or "")


# Removes duplicate comments while preserving original order
def unique_comments(items: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    
    # Store already-seen (platform, comment) pairs
    seen = set()
    result = []

    for platform, comment in items:
        
        # Convert comment to lowercase for case-insensitive comparison
        key = (platform, comment.lower())

        # Skip if already processed
        if key in seen:
            continue

        # Add unique comment to tracking set and result list
        seen.add(key)
        result.append((platform, comment))

    return result