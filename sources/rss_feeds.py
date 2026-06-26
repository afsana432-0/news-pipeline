"""
sources/rss_feeds.py
----------------------
Fetches article metadata from multiple RSS feeds using `feedparser`.
RSS is the easiest, most stable way to get structured article lists
(title, link, published date, summary) without scraping fragile HTML
on the listing/homepage level.

Each feed entry still needs its full article text scraped separately
via scraper.fetch_article_body() — RSS only gives us the summary.
"""

import feedparser

# A small, deliberately varied set of stable, public RSS feeds.
# Each tuple is (source_name, feed_url).
RSS_FEEDS = [
    ("bbc_world", "http://feeds.bbci.co.uk/news/world/rss.xml"),
    ("npr_news", "https://feeds.npr.org/1001/rss.xml"),
    ("hacker_news", "https://hnrss.org/frontpage"),
]


def fetch_feed_entries(feed_url: str, max_entries: int = 10) -> list[dict]:
    """
    Parses an RSS feed and returns a list of dicts:
        {"title": str, "url": str, "published": str, "summary": str}

    feedparser handles malformed/irregular feeds gracefully (it sets
    .bozo = 1 instead of raising), so we check that flag and log a
    warning rather than crashing on a slightly broken feed.
    """
    parsed = feedparser.parse(feed_url)

    if parsed.bozo:
        # bozo_exception holds the parse error, if any. Still try to use
        # whatever entries were salvaged, since most "bozo" feeds are
        # still mostly readable (e.g. minor encoding quirks).
        print(f"Warning: feed at {feed_url} had parsing issues: {parsed.bozo_exception}")

    entries = []
    for entry in parsed.entries[:max_entries]:
        entries.append({
            "title": entry.get("title", "Untitled"),
            "url": entry.get("link", ""),
            "published": entry.get("published", ""),
            "summary": entry.get("summary", ""),
        })

    return entries
