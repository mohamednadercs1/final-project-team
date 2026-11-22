"""Offline tests for review_scraper.py (no network: a fake requests session serves canned HTML).

NOTE: the HTML below is hand-written in the style of Wikipedia film articles. It tests the parsing
logic, not the live site. After the first real run, open data/critic_scores_scraped.csv and spot-check
a few rows against the actual articles (see "Validating on the real site" in the README).

Run:  pytest -q tests/test_review_scraper.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import review_scraper as rs  # noqa: E402


# ------------------------------------------------------------------ fixtures
def film_page(title="Test Film", year=1999, body="", director=True, extra_lead=""):
    director_row = "<tr><th>Directed by</th><td>Someone</td></tr>" if director else ""
    return f"""<html><body><h1 id="firstHeading">{title}</h1>
    <div class="mw-parser-output">
      <table class="infobox vevent"><tr><th>{title}</th></tr>{director_row}
        <tr><th>Release date</th><td>June 11, {year} (United States)</td></tr></table>
      <p>{extra_lead}{title} is a {year} American film.</p>
      {body}
    </div></body></html>"""


CRITICAL = ("<p>On Rotten Tomatoes, the film holds an approval rating of 87% based on 250 reviews, "
            "with an average rating of 7.9/10.[12] On Metacritic, the film has a weighted average score "
            "of 72 out of 100, based on 40 critics, indicating generally favorable reviews.[13]</p>")


def sentences(html):
    return rs.page_sentences(BeautifulSoup(html, "html.parser"))


class FakeResponse:
    def __init__(self, status_code=200, text="", headers=None):
        self.status_code, self.text, self.headers = status_code, text, headers or {}


class FakeSession:
    """Serves canned responses by URL suffix; records every URL requested."""

    def __init__(self, pages, robots="User-agent: *\nAllow: /\n", robots_status=200):
        self.pages, self.robots, self.robots_status = pages, robots, robots_status
        self.headers, self.calls = {}, []

    def get(self, url, timeout=None):
        self.calls.append(url)
        if url.endswith("/robots.txt"):
            return FakeResponse(self.robots_status, self.robots)
        for suffix, resp in self.pages.items():
            if url.endswith(suffix):
                return resp if isinstance(resp, FakeResponse) else FakeResponse(200, resp)
        return FakeResponse(404, "")


def fetcher(pages, **kw):
    robots = kw.pop("robots", "User-agent: *\nAllow: /\n")
    return rs.PoliteFetcher("test-agent", delay=0, backoff=0, max_retries=kw.pop("max_retries", 2),
                            session=FakeSession(pages, robots=robots, **kw))


# ------------------------------------------------------------------ extraction
def test_extracts_rt_and_metacritic_and_ignores_citations():
    s = sentences(film_page(body=CRITICAL))
    assert rs.extract_rt(s) == 87
    assert rs.extract_metacritic(s) == 72


def test_audience_percentage_is_skipped():
    body = ("<p>The audience score on Rotten Tomatoes is 91%. "
            "Critics gave the film an approval rating of 64% on Rotten Tomatoes.</p>")
    assert rs.extract_rt(sentences(film_page(body=body))) == 64


def test_rt_without_percentage_returns_none():
    body = "<p>On Rotten Tomatoes the film has an average rating of 7.9/10 based on 80 reviews.</p>"
    assert rs.extract_rt(sentences(film_page(body=body))) is None


@pytest.mark.parametrize("text, expected", [
    ("Metacritic assigned the film a score of 59 out of 100 based on 36 critics.", 59),
    ("On Metacritic the film scored 81 based on reviews from 30 critics.", 81),
    ("Metacritic gave it 45/100.", 45),
    ("Metacritic is a review aggregator.", None),
])
def test_metacritic_variants(text, expected):
    assert rs.extract_metacritic(sentences(film_page(body=f"<p>{text}</p>"))) == expected


def test_percent_word_and_out_of_range_values():
    assert rs.extract_rt(sentences(film_page(body="<p>Rotten Tomatoes gave it 93 percent.</p>"))) == 93
    assert rs.extract_rt(sentences(film_page(body="<p>Rotten Tomatoes lists 250% growth.</p>"))) is None


# ------------------------------------------------------------------ film-page validation
def soup(html):
    return BeautifulSoup(html, "html.parser")


def test_looks_like_film_checks_infobox_and_year():
    assert rs.looks_like_film(soup(film_page(year=1999)), 1999)
    assert rs.looks_like_film(soup(film_page(year=1999)), 2000)            # +-1 tolerance
    assert not rs.looks_like_film(soup(film_page(year=1999)), 2005)        # remake with same title
    assert not rs.looks_like_film(soup(film_page(director=False)), 1999)   # not a film infobox
    assert not rs.looks_like_film(soup("<html><body><p>Just text</p></body></html>"), 1999)


def test_disambiguation_page_is_rejected():
    page = film_page(extra_lead="Test Film may refer to: ")
    assert not rs.looks_like_film(soup(page), 1999)


def test_candidate_urls_most_specific_first():
    urls = rs.candidate_urls("Jurassic Park", 1993)
    assert urls[0].endswith("Jurassic_Park_(1993_film)")
    assert urls[1].endswith("Jurassic_Park_(film)")
    assert urls[2].endswith("/Jurassic_Park")
    assert "%26" in rs.candidate_urls("Mr. & Mrs. Smith", 2005)[0]


# ------------------------------------------------------------------ scrape_film end-to-end (fake network)
def test_scrape_film_falls_back_to_second_candidate():
    f = fetcher({"Test_Film_(film)": film_page(year=1999, body=CRITICAL)})
    r = rs.scrape_film(f, "Test Film", 1999)
    assert (r["status"], r["rt_critic"], r["metacritic"], r["wiki_title"]) == ("ok", 87, 72, "Test Film")


def test_scrape_film_wrong_year_is_not_accepted():
    f = fetcher({"Test_Film": film_page(year=1960, body=CRITICAL)})
    assert rs.scrape_film(f, "Test Film", 1999)["status"] == "not_found"


def test_scrape_film_no_scores_status():
    f = fetcher({"Test_Film_(1999_film)": film_page(year=1999, body="<p>Nothing about reviews.</p>")})
    assert rs.scrape_film(f, "Test Film", 1999)["status"] == "no_scores"


def test_server_errors_are_marked_for_retry():
    f = fetcher({"Test_Film_(1999_film)": FakeResponse(503, "")}, max_retries=2)
    assert rs.scrape_film(f, "Test Film", 1999)["status"] == "error"


def test_robots_txt_is_obeyed_and_page_never_requested():
    robots = "User-agent: *\nDisallow: /wiki/"
    f = fetcher({"Test_Film": film_page(body=CRITICAL)}, robots=robots)
    r = rs.scrape_film(f, "Test Film", 1999)
    assert r["status"] == "blocked_by_robots"
    assert not [u for u in f.session.calls if "/wiki/" in u], "a disallowed URL was requested"


def test_unverifiable_robots_means_no_scraping():
    f = fetcher({"Test_Film": film_page(body=CRITICAL)}, robots_status=503)
    assert rs.scrape_film(f, "Test Film", 1999)["status"] == "blocked_by_robots"


def test_missing_robots_file_means_allowed():
    f = fetcher({"Test_Film_(film)": film_page(body=CRITICAL)}, robots_status=404, robots="")
    assert rs.scrape_film(f, "Test Film", 1999)["status"] == "ok"


# ------------------------------------------------------------------ bookkeeping
def test_resume_skips_done_but_retries_errors(tmp_path):
    path = tmp_path / "scraped.csv"
    for i, status in enumerate(["ok", "not_found", "error", "no_scores"]):
        rs.append_row(path, {"id": i, "title": "t", "year": 2000, "status": status})
    assert rs.already_done(path) == {0, 1, 3}


def test_merge_fills_gaps_and_keeps_omdb_values(tmp_path):
    omdb, scraped = tmp_path / "critic_scores.csv", tmp_path / "scraped.csv"
    pd.DataFrame({"id": [1, 2], "rt_critic": [90.0, np.nan], "metacritic": [np.nan, 50.0],
                  "imdb_rating": [8.0, 6.0]}).to_csv(omdb, index=False)
    pd.DataFrame({"id": [1, 2, 3], "rt_critic": [10, 77, 55], "metacritic": [66, 11, 44],
                  "status": ["ok", "ok", "ok"]}).to_csv(scraped, index=False)
    rs.merge_into_critic_scores(scraped, omdb)
    out = pd.read_csv(omdb).set_index("id")
    assert out.loc[1, "rt_critic"] == 90 and out.loc[1, "metacritic"] == 66      # OMDb wins, gap filled
    assert out.loc[2, "rt_critic"] == 77 and out.loc[2, "metacritic"] == 50
    assert out.loc[3, "rt_critic"] == 55                                          # new film added
    assert out.loc[1, "imdb_rating"] == 8.0                                       # other columns kept
    assert (tmp_path / "critic_scores.omdb_backup.csv").exists()
    rs.merge_into_critic_scores(scraped, omdb)                                    # idempotent
    assert pd.read_csv(omdb).set_index("id").loc[1, "rt_critic"] == 90
