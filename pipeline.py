"""
pipeline.py
-----------
The main orchestrator. For every configured RSS feed, this:
  1. Fetches the feed's entries (title, link, published date)
  2. Skips articles already stored (dedup by URL)
  3. Scrapes each article's full body text from its HTML page
  4. Cleans the text and extracts top keywords via NLTK
  5. Tags a topic using simple keyword-overlap rules
  6. Stores everything (article + keywords) in SQLite

Run with:
    python pipeline.py
"""

import logging
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

import db
import nlp_processing
from sources.rss_feeds import RSS_FEEDS, fetch_feed_entries

LOG_PATH = Path(__file__).parent / "logs" / "pipeline.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; AfsanaNewsPipeline/1.0; educational project)"}
REQUEST_DELAY_SECONDS = 2
MAX_ARTICLES_PER_FEED = 5  # keep runs short and polite while testing


def fetch_article_body(url: str) -> str | None:
    """
    Downloads an article's HTML page and extracts visible paragraph text.
    Returns None on any request failure or bad status code, logged
    rather than raised, so one broken link doesn't stop the whole run.

    Note: this uses a generic <p> tag extraction strategy rather than a
    site-specific adapter, since news sites vary widely in markup. It's
    a reasonable general-purpose approach, though a production system
    would likely add per-site adapters for cleaner extraction (similar
    to the site-adapter pattern used in the price-tracker project).
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
    except requests.exceptions.RequestException as exc:
        logger.warning(f"Request failed for {url}: {exc}")
        return None

    if not response.ok:
        logger.warning(f"Bad status code {response.status_code} for {url}")
        return None

    soup = BeautifulSoup(response.text, "lxml")
    paragraphs = soup.find_all("p")

    # Heuristic: real article sentences are usually longer than nav links
    # or footer text ("Home | News | Sports", "Copyright 2026"). Filtering
    # out very short paragraphs is a simple, imperfect way to reduce that
    # boilerplate noise without needing a site-specific adapter for every
    # news outlet. It won't be perfect on every site, but it meaningfully
    # improves signal-to-noise for the keyword/topic analysis downstream.
    candidate_paragraphs = [
        p.get_text(strip=True) for p in paragraphs
        if len(p.get_text(strip=True)) > 40
    ]
    body_text = " ".join(candidate_paragraphs)

    return body_text if body_text else None


def process_feed(source_name: str, feed_url: str):
    """Fetches, scrapes, analyses, and stores all new articles from one feed."""
    logger.info(f"Fetching feed: {source_name} ({feed_url})")
    entries = fetch_feed_entries(feed_url, max_entries=MAX_ARTICLES_PER_FEED)
    logger.info(f"  Found {len(entries)} entries")

    for entry in entries:
        url = entry["url"]

        if not url:
            continue

        if db.article_exists(url):
            logger.info(f"  Skipping (already stored): {entry['title']}")
            continue

        raw_text = fetch_article_body(url)
        if raw_text is None:
            # Fall back to the RSS summary if the full article couldn't
            # be scraped (e.g. paywall, JS-rendered page) — better to
            # store something useful than to skip the article entirely.
            raw_text = entry["summary"]
            logger.info(f"  Using RSS summary as fallback for: {entry['title']}")

        clean = nlp_processing.clean_text(raw_text)
        keywords = nlp_processing.extract_top_keywords(clean, top_n=8)
        topic = nlp_processing.tag_topic(keywords)

        article_id = db.insert_article(
            source=source_name,
            url=url,
            title=entry["title"],
            published=entry["published"],
            raw_text=raw_text,
            clean_text=clean,
            topic=topic,
        )
        db.insert_keywords(article_id, keywords)

        logger.info(f"  Stored: '{entry['title']}' -> topic: {topic}, "
                    f"top keywords: {[w for w, _ in keywords[:4]]}")

        time.sleep(REQUEST_DELAY_SECONDS)


def run_once():
    db.init_db()
    logger.info(f"Starting pipeline run across {len(RSS_FEEDS)} feed(s)")

    for source_name, feed_url in RSS_FEEDS:
        process_feed(source_name, feed_url)

    logger.info("Pipeline run complete.\n")
    print_summary()


def print_summary():
    """Prints a quick topic breakdown after each run."""
    counts = db.get_topic_counts()
    if not counts:
        logger.info("No articles stored yet.")
        return

    logger.info("--- Topic summary (all-time) ---")
    for row in counts:
        logger.info(f"  {row['topic']}: {row['count']} article(s)")


if __name__ == "__main__":
    run_once()
