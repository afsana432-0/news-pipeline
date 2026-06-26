"""
db.py
-----
Storage layer for the news pipeline. Two tables:

    articles      -> one row per unique article (dedup by URL)
    keywords      -> top keywords extracted per article (many rows per article)

Design choice: keywords live in their own table (not a single comma-joined
column) so we can later query "which articles mention 'inflation'?" with a
simple WHERE clause instead of string matching inside a blob.
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).parent / "data" / "news_pipeline.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            url TEXT NOT NULL UNIQUE,
            title TEXT,
            published TEXT,
            raw_text TEXT,
            clean_text TEXT,
            topic TEXT,
            scraped_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS keywords (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER NOT NULL,
            keyword TEXT NOT NULL,
            frequency INTEGER NOT NULL,
            FOREIGN KEY (article_id) REFERENCES articles(id)
        )
    """)

    conn.commit()
    conn.close()


def article_exists(url: str) -> bool:
    """Used to skip articles we've already stored (dedup across runs)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM articles WHERE url = ?", (url,))
    exists = cur.fetchone() is not None
    conn.close()
    return exists


def insert_article(source: str, url: str, title: str, published: str,
                    raw_text: str, clean_text: str, topic: str) -> int:
    """Inserts a new article and returns its id."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO articles
           (source, url, title, published, raw_text, clean_text, topic, scraped_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (source, url, title, published, raw_text, clean_text, topic,
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    article_id = cur.lastrowid
    conn.close()
    return article_id


def insert_keywords(article_id: int, keyword_freq_pairs: list[tuple[str, int]]):
    """Stores the top keywords for an article. Expects [(word, count), ...]."""
    conn = get_connection()
    cur = conn.cursor()
    cur.executemany(
        "INSERT INTO keywords (article_id, keyword, frequency) VALUES (?, ?, ?)",
        [(article_id, word, count) for word, count in keyword_freq_pairs],
    )
    conn.commit()
    conn.close()


def get_articles_by_topic(topic: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM articles WHERE topic = ? ORDER BY scraped_at DESC", (topic,))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_topic_counts():
    """Returns how many articles fall under each topic — useful for a quick summary."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT topic, COUNT(*) as count FROM articles GROUP BY topic ORDER BY count DESC")
    rows = cur.fetchall()
    conn.close()
    return rows
