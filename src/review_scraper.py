"""Step 3b - Scrape critic scores (Rotten Tomatoes % / Metacritic) from Wikipedia film articles.

Why Wikipedia and not rottentomatoes.com: Rotten Tomatoes' terms of use prohibit scraping. Wikipedia
film articles quote the Tomatometer and the Metacritic score in the "Critical response" section, the
text is CC BY-SA, and automated reading of /wiki/ article pages is allowed by its robots.txt when you
identify yourself with a descriptive User-Agent. This is real HTML scraping (requests + BeautifulSoup)
and complements the OMDb API in scraper.py.

Politeness rules enforced here:
  * robots.txt is fetched and checked for EVERY URL (a disallowed URL is never requested),
  * a descriptive User-Agent with a contact address (--contact),
  * >= 1 s between requests (+ jitter), exponential back-off on 429/5xx, Retry-After honoured,
  * resumable: every film is written to the CSV as soon as it is done and is never fetched twice.

Caveats (say them in your report):
  * scores are the snapshot an editor typed into the article, usually shortly after release, and are
    measured AFTER release -> "post-screening" feature, never a greenlight feature;
  * not every article quotes both scores; coverage is partial and biased toward well-known films
    (check the coverage report printed by critic_study.py).

Run:
    python review_scraper.py --contact you@example.com --limit 300      # most recent films first
    python review_scraper.py --contact you@example.com --order random --limit 500
    python review_scraper.py --merge-only                               # combine with OMDb scores
Out:
    data/critic_scores_scraped.csv   (id, rt_critic, metacritic, status, ...)
    --merge fills gaps of data/critic_scores.csv (OMDb) with scraped values (original is backed up).
"""
import argparse
import csv
import random
import shutil
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import robotparser
from urllib.parse import quote, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

import config

BASE = "https://en.wikipedia.org/wiki/"
SCRAPED_CSV = Path(config.CRITIC_SCORES).with_name("critic_scores_scraped.csv")
FIELDS = ["id", "title", "year", "wiki_title", "rt_critic", "metacritic", "source", "status", "scraped_at"]
DONE_STATUSES = {"ok", "no_scores", "not_found", "blocked_by_robots"}     # "error" rows are retried


# ------------------------------------------------------------------ polite HTTP
class PoliteFetcher:
    """requests wrapper: robots.txt check, rate limit with jitter, retries with back-off."""

    def __init__(self, user_agent, delay=1.0, session=None, max_retries=3, backoff=5.0):
        self.user_agent, self.delay, self.max_retries, self.backoff = user_agent, delay, max_retries, backoff
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self._robots = {}
        self._last = 0.0

    def _wait(self):
        gap = self.delay + random.uniform(0, 0.3 * self.delay)
        remaining = self._last + gap - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        self._last = time.monotonic()

    def allowed(self, url):
        parts = urlparse(url)
        host = f"{parts.scheme}://{parts.netloc}"
        if host not in self._robots:
            rp = robotparser.RobotFileParser()
            try:
                self._wait()
                r = self.session.get(f"{host}/robots.txt", timeout=20)
                if r.status_code == 200:
                    rp.parse(r.text.splitlines())
                else:                                   # no robots.txt (4xx) -> nothing disallowed
                    rp.parse([] if r.status_code < 500 else ["User-agent: *", "Disallow: /"])
            except requests.RequestException:
                rp.parse(["User-agent: *", "Disallow: /"])   # cannot verify -> stay out
            self._robots[host] = rp
        return self._robots[host].can_fetch(self.user_agent, url)

    def get(self, url):
        """Return (status_code, html), or None if robots.txt forbids the URL."""
        if not self.allowed(url):
            return None
        status, text = 0, ""
        for attempt in range(self.max_retries):
            self._wait()
            try:
                r = self.session.get(url, timeout=20)
            except requests.RequestException:
                time.sleep(min(2 ** attempt * self.backoff, 60))
                continue
            status, text = r.status_code, r.text
            if status in (429, 500, 502, 503, 504):
                retry_after = r.headers.get("Retry-After", "")
                time.sleep(float(retry_after) if retry_after.isdigit() else min(2 ** attempt * self.backoff, 60))
                continue
            break
        return status, text


# ------------------------------------------------------------------ HTML parsing
_CITATION = re.compile(r"\[(?:\d+|[a-z]|note \d+|citation needed)\]", re.I)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\u201c'])")
_YEAR = re.compile(r"\b(?:18|19|20)\d{2}\b")
_RT_PERCENT = re.compile(r"(\d{1,3})\s?(?:%|percent\b)")
_MC_PATTERNS = [re.compile(r"\b(\d{1,3})\s?(?:out of|/)\s?100\b"),
                re.compile(r"\bscore of (\d{1,3})\b"),
                re.compile(r"\bscored (\d{1,3})\b")]


def page_sentences(soup):
    """Sentences of the article body (paragraphs and list items), citations removed."""
    body = soup.select_one("div.mw-parser-output") or soup
    sentences = []
    for el in body.find_all(["p", "li"]):
        text = re.sub(r"\s+", " ", _CITATION.sub("", el.get_text(" ", strip=True)))
        sentences += [s for s in _SENTENCE_SPLIT.split(text) if s]
    return sentences


def extract_rt(sentences):
    """First Tomatometer-style percentage that is not the audience score."""
    for s in sentences:
        if "rotten tomatoes" not in s.lower():
            continue
        for m in _RT_PERCENT.finditer(s):
            before = s[max(0, m.start() - 45):m.start()].lower()
            if "audience" in before or "popcornmeter" in before:
                continue
            if 0 <= int(m.group(1)) <= 100:
                return int(m.group(1))
    return None


def extract_metacritic(sentences):
    for s in sentences:
        if "metacritic" not in s.lower():
            continue
        for pattern in _MC_PATTERNS:
            m = pattern.search(s)
            if m and 0 <= int(m.group(1)) <= 100:
                return int(m.group(1))
    return None


def infobox_years(soup):
    box = soup.select_one("table.infobox")
    years = []
    if box is not None:
        for tr in box.select("tr"):
            th = tr.find("th")
            if th is not None and "release" in th.get_text(" ", strip=True).lower():
                years += [int(y) for y in _YEAR.findall(tr.get_text(" ", strip=True))]
    return years


def looks_like_film(soup, year=None):
    """Accept only film articles (infobox with 'Directed by') whose release year matches +-1."""
    lead = soup.get_text(" ", strip=True)[:600].lower()
    if "may refer to" in lead or "disambiguation" in lead[:200]:
        return False
    box = soup.select_one("table.infobox")
    if box is None:
        return False
    if not any(th.get_text(" ", strip=True).lower().startswith("directed by") for th in box.select("th")):
        return False
    years = infobox_years(soup)
    if year and years and not any(abs(y - int(year)) <= 1 for y in years):
        return False
    return True


def candidate_urls(title, year):
    """Most specific Wikipedia article names first: 'T (1993 film)', 'T (film)', 'T'."""
    t = str(title).strip().replace(" ", "_")
    names = [f"{t}_({int(year)}_film)", f"{t}_(film)", t] if year else [f"{t}_(film)", t]
    return [BASE + quote(n, safe="()_,'!:.-") for n in names]


def scrape_film(fetcher, title, year):
    """Try the candidate articles; return a result dict (never raises on missing pages)."""
    result = {"wiki_title": None, "rt_critic": None, "metacritic": None, "status": "not_found"}
    errors = blocked = 0
    for url in candidate_urls(title, year):
        resp = fetcher.get(url)
        if resp is None:
            blocked += 1
            continue
        status, html = resp
        if status == 404:
            continue
        if status != 200:
            errors += 1
            continue
        soup = BeautifulSoup(html, "html.parser")
        if not looks_like_film(soup, year):
            continue
        sentences = page_sentences(soup)
        rt, mc = extract_rt(sentences), extract_metacritic(sentences)
        h1 = soup.select_one("h1#firstHeading")
        result.update(wiki_title=h1.get_text(strip=True) if h1 else url.rsplit("/", 1)[-1],
                      rt_critic=rt, metacritic=mc,
                      status="ok" if (rt is not None or mc is not None) else "no_scores")
        return result
    if errors:
        result["status"] = "error"                      # retried on the next run
    elif blocked:
        result["status"] = "blocked_by_robots"
    return result


# ------------------------------------------------------------------ bookkeeping
def load_films():
    """Films worth scraping: the modelling frame (modelable films only)."""
    path = Path(config.MODELING_FRAME)
    if path.exists():
        df = pd.read_pickle(path)
    else:
        from features import build_modeling_frame
        df = build_modeling_frame()
    films = df[["id", "title", "release_date"]].copy()
    films["year"] = pd.to_datetime(films["release_date"]).dt.year
    return films


def already_done(path):
    if not path.exists():
        return set()
    prev = pd.read_csv(path)
    return set(prev.loc[prev["status"].isin(DONE_STATUSES), "id"])


def append_row(path, row):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in FIELDS})


def merge_into_critic_scores(scraped_path=SCRAPED_CSV, omdb_path=None):
    """Fill gaps of the OMDb table with scraped values (OMDb wins when both exist). Backs up the original."""
    omdb_path = Path(omdb_path or config.CRITIC_SCORES)
    scraped = pd.read_csv(scraped_path)
    scraped = scraped[scraped["status"] == "ok"].drop_duplicates("id", keep="last").set_index("id")
    if omdb_path.exists():
        backup = omdb_path.with_name(omdb_path.stem + ".omdb_backup.csv")
        if not backup.exists():                         # keep the pristine OMDb table once
            shutil.copy2(omdb_path, backup)
        base = pd.read_csv(omdb_path).drop_duplicates("id").set_index("id")
    else:
        base = pd.DataFrame(index=pd.Index([], name="id"), columns=["rt_critic", "metacritic"])
    merged = base.reindex(base.index.union(scraped.index))
    for col in ("rt_critic", "metacritic"):
        if col not in merged:
            merged[col] = float("nan")
        filled = merged[col].isna() & scraped[col].reindex(merged.index).notna()
        merged[col] = merged[col].combine_first(scraped[col].reindex(merged.index))
        print(f"{col}: filled {int(filled.sum())} missing values from the scrape")
    merged.reset_index().to_csv(omdb_path, index=False)
    print(f"merged table -> {omdb_path} ({len(merged)} films)")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--contact", default="", help="your e-mail/URL, put in the User-Agent (please set it)")
    ap.add_argument("--limit", type=int, default=300, help="max NEW films to scrape in this run")
    ap.add_argument("--order", choices=["recent", "random", "oldest"], default="recent",
                    help="recent = test period first (default)")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests (min 1.0)")
    ap.add_argument("--merge", action="store_true", help="merge into critic_scores.csv after scraping")
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    if args.merge_only:
        return merge_into_critic_scores()

    ua = f"MovieSuccessStudentProject/1.0 (educational; contact: {args.contact or 'not provided'}) python-requests"
    fetcher = PoliteFetcher(ua, delay=max(args.delay, 1.0))
    films = load_films()
    films = films[~films["id"].isin(already_done(SCRAPED_CSV))]
    if args.order == "recent":
        films = films.sort_values("release_date", ascending=False)
    elif args.order == "oldest":
        films = films.sort_values("release_date")
    else:
        films = films.sample(frac=1, random_state=config.SEED)
    films = films.head(args.limit)
    print(f"scraping {len(films)} films -> {SCRAPED_CSV}")

    counts = {}
    for i, f in enumerate(films.itertuples(index=False), 1):
        res = scrape_film(fetcher, f.title, f.year)
        counts[res["status"]] = counts.get(res["status"], 0) + 1
        append_row(SCRAPED_CSV, {"id": f.id, "title": f.title, "year": f.year, "source": "wikipedia",
                                 "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **res})
        if i % 25 == 0 or i == len(films):
            print(f"  {i}/{len(films)}  {counts}")
    print("done:", counts)
    if args.merge:
        merge_into_critic_scores()


if __name__ == "__main__":
    main()
