"""
nlp_processing.py
-------------------
Text pre-processing and lightweight analysis, building directly on
standard NLTK techniques: tokenisation, stop-word removal, and
frequency-based keyword extraction. Also includes a simple rule-based
topic tagger (keyword matching against category word-lists) — a
practical first step before reaching for a full ML classifier.
"""

import string
import re
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data quietly on first run. Subsequent runs are
# instant since nltk caches these locally.
# Download required NLTK data quietly on first run. Subsequent runs are
# instant since nltk caches these locally. We just call download() every
# time with quiet=True — nltk internally skips re-downloading if the data
# already exists, which is simpler and more reliable than manually
# checking each resource's internal path structure.
for resource in ["punkt", "punkt_tab", "stopwords"]:
    nltk.download(resource, quiet=True)

STOPWORDS = set(stopwords.words("english"))


def clean_text(raw_text: str) -> str:
    """
    Basic cleanup before tokenising: collapses whitespace and strips
    characters that aren't useful for keyword/topic analysis.
    Keeps the cleaned text human-readable (not a token list) so it can
    still be stored and re-processed differently later if needed.
    """
    text = re.sub(r"\s+", " ", raw_text)  # collapse newlines/multiple spaces
    text = text.strip()
    return text


def tokenize_and_filter(text: str) -> list[str]:
    """
    Tokenises text, lowercases, strips punctuation tokens, and removes
    stop words. Returns a list of meaningful word tokens.
    """
    tokens = word_tokenize(text.lower())
    filtered = [
        tok for tok in tokens
        if tok not in STOPWORDS
        and tok not in string.punctuation
        and len(tok) > 2          # drop very short tokens ("a", "is", "an" already gone, but also stray fragments)
        and tok.isalpha()         # drop pure numbers/symbols
    ]
    return filtered


def extract_top_keywords(text: str, top_n: int = 8) -> list[tuple[str, int]]:
    """
    Returns the top_n most frequent meaningful words in the text as
    (word, count) tuples. This is the same FreqDist-style approach as
    standard NLTK frequency analysis, applied to full article text
    instead of a toy sentence.
    """
    tokens = tokenize_and_filter(text)
    freq = Counter(tokens)
    return freq.most_common(top_n)


# Simple rule-based topic categories: each topic has a list of keywords
# that, if found in the article text, count as a vote for that topic.
# This is intentionally simple (no ML model) — a transparent, explainable
# first pass that's easy to extend by just adding more words to a list.
TOPIC_KEYWORDS = {
    "politics": ["election", "government", "minister", "president", "parliament",
                 "senate", "policy", "vote", "campaign", "political", "congress",
                 "lawmakers", "law", "bill", "administration"],
    "technology": ["technology", "software", "startup", "app", "data", "ai",
                   "artificial", "internet", "tech", "computer", "google", "apple",
                   "model", "models", "anthropic", "open", "source", "meta",
                   "platform", "prediction"],
    "business": ["market", "stock", "economy", "company", "trade", "investment",
                 "business", "financial", "bank", "economic", "service", "postal",
                 "usps", "mail", "cash", "workers", "union", "labor"],
    "sports": ["match", "team", "player", "tournament", "championship", "league",
               "sport", "coach", "score", "world cup", "round", "turkey"],
    "health": ["health", "hospital", "disease", "vaccine", "medical", "doctor",
               "patient", "treatment", "virus", "pandemic", "research", "studies",
               "papers"],
    "world_news": ["earthquake", "war", "conflict", "killed", "injured", "disaster",
                   "evacuate", "crisis", "heatwave", "climate", "temperatures",
                   "europe", "venezuela", "caracas", "iran"],
}


def tag_topic(keywords: list[tuple[str, int]]) -> str:
    """
    Assigns a single topic label by scoring each topic's keyword overlap
    against the article's extracted top keywords, then picking the
    highest-scoring topic. Falls back to "general" if nothing matches.

    Note: "game" was deliberately removed from the sports keyword list.
    In real-world testing it caused a video-game-related shooting story
    to be mistagged as "sports" — a good example of why simple keyword
    overlap is a transparent but imperfect first-pass classifier, and why
    a production system would eventually want a trained ML model instead.
    """
    keyword_words = {word for word, _count in keywords}

    scores = {}
    for topic, topic_words in TOPIC_KEYWORDS.items():
        overlap = keyword_words.intersection(topic_words)
        if overlap:
            scores[topic] = len(overlap)

    if not scores:
        return "general"

    # Pick the topic with the highest overlap score.
    return max(scores, key=scores.get)
