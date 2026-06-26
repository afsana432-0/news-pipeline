# Multi-Source News Data Pipeline

A pipeline that aggregates news articles from multiple RSS feeds, scrapes
their full text, runs NLP pre-processing, extracts keywords, and tags each
article with a topic — turning scattered web content into a clean,
queryable dataset.

## What it does

1. **Fetches** article metadata (title, link, date, summary) from multiple RSS feeds
2. **Scrapes** the full article text from each article's HTML page
3. **Deduplicates** — skips articles already stored from a previous run
4. **Cleans** the text and tokenizes it using NLTK
5. **Removes stop words** and extracts the top keywords by frequency
6. **Tags a topic** (politics, technology, business, sports, health, or general)
   using a transparent, rule-based keyword-overlap method
7. **Stores** everything — article text, keywords, and topic — in SQLite
8. **Summarizes** topic distribution after each run

## Why this design

- **RSS first, HTML second** — RSS feeds give stable, structured entry points
  (title/link/date) without needing a custom scraper for every site's homepage.
  Full article text is then scraped from the linked page itself.
- **Rule-based topic tagging, not a black box** — a simple keyword-overlap
  classifier is fully explainable: you can see exactly why an article got
  tagged "politics" (which words matched). This is a deliberate first step
  before reaching for an ML/embedding-based classifier.
- **Graceful degradation everywhere** — a broken RSS feed, a failed article
  fetch, or a paywalled page never crashes the run. Each falls back sensibly
  (e.g. using the RSS summary if full-text scraping fails) and logs what happened.
- **Dedup by URL** — running the pipeline repeatedly (e.g. on a schedule)
  won't create duplicate rows for articles already seen.

## Project structure

```
news-pipeline/
├── pipeline.py          # main orchestrator: fetch -> scrape -> NLP -> store
├── nlp_processing.py     # tokenization, stop-word removal, keywords, topic tagging
├── db.py                 # SQLite schema + read/write helpers
├── sources/
│   └── rss_feeds.py      # RSS feed list + feedparser wrapper
├── data/
│   └── news_pipeline.db  # SQLite database (created on first run)
└── logs/
    └── pipeline.log       # run history
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

First run will also download NLTK's `punkt` and `stopwords` data automatically.

## Usage

```bash
python pipeline.py
```

This fetches a small batch from each configured RSS feed (5 articles per
feed by default, set in `pipeline.py`'s `MAX_ARTICLES_PER_FEED`), scrapes
and analyzes each new one, and prints a topic summary at the end.

## RSS feeds used

- BBC World News
- NPR News
- Hacker News (front page)

All three are public, well-formed RSS feeds with no authentication required.
Add more by editing `RSS_FEEDS` in `sources/rss_feeds.py`.

## Extending the topic tagger

`nlp_processing.TOPIC_KEYWORDS` is a plain dictionary — add a new topic by
adding a new key with a list of representative words:

```python
TOPIC_KEYWORDS["entertainment"] = ["movie", "celebrity", "music", "film", "actor"]
```

## Known limitation

Full-article scraping uses a generic `<p>` tag extraction strategy (filtered
to skip very short paragraphs, to reduce nav/footer noise) rather than a
site-specific adapter per news outlet. This works reasonably well across
most sites but isn't perfect — a production version would add per-site
adapters for cleaner extraction, the same pattern used in the companion
price-tracker project.

## Possible extensions

- Add per-site article-body adapters for cleaner text extraction
- Swap the rule-based topic tagger for a proper ML classifier (e.g. scikit-learn
  Naive Bayes trained on a labeled dataset)
- Add sentiment analysis per article
- Build a small dashboard (Streamlit) to browse articles by topic/keyword
