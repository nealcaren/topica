"""Rebuild the U.S. party-platforms corpus for examples/ectm_platforms.py by
scraping the American Presidency Project (presidency.ucsb.edu).

Writes examples/platforms_data/platforms.json.gz: a list of
{year, party, text} paragraph records for every Democratic and Republican
national platform from 1948 to the present. The repo already ships this file;
run this only to refresh or extend it.

    python examples/prep_platforms.py
"""
import gzip
import html
import json
import os
import re
import time
import urllib.request

OUT = os.path.join(os.path.dirname(__file__), "platforms_data")
BASE = "https://www.presidency.ucsb.edu"
LISTING = BASE + "/documents/app-categories/elections-and-transitions/party-platforms"


def platform_urls(min_year=1948):
    """Collect every Dem/Rep platform document path across both URL slug formats."""
    pages = []
    for p in range(13):
        try:
            req = urllib.request.Request(f"{LISTING}?page={p}", headers={"User-Agent": "Mozilla/5.0"})
            pages.append(urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore"))
        except Exception:
            break
        time.sleep(0.3)
    h = "\n".join(pages)
    urls = set(re.findall(r"/documents/\d{4}-(?:democratic|republican)-party-platform", h))
    urls |= set(re.findall(r"/documents/(?:democratic|republican)-party-platform-\d{4}", h))
    best = {}
    for u in urls:
        year = int(re.search(r"(\d{4})", u).group(1))
        party = "D" if "democratic" in u else "R"
        if year >= min_year:
            best.setdefault((year, party), u)  # one URL per (year, party)
    return best


def scrape(url):
    req = urllib.request.Request(BASE + url, headers={"User-Agent": "Mozilla/5.0"})
    h = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
    m = (re.search(r"field-docs-content[^>]*>(.*?)</div>\s*</div>", h, re.S)
         or re.search(r"field-docs-content[^>]*>(.*?)</div>", h, re.S))
    body = re.sub(r"(?i)</p>", "\n", m.group(1) if m else "")
    paras = [re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", p))).strip()
             for p in body.split("\n")]
    return [p for p in paras if len(p.split()) >= 25]


def main():
    os.makedirs(OUT, exist_ok=True)
    urls = platform_urls()
    print(f"scraping {len(urls)} platforms...")
    rows = []
    for (year, party), u in sorted(urls.items()):
        paras = scrape(u)
        rows.extend({"year": year, "party": party, "text": p} for p in paras)
        print(f"  {year} {party}: {len(paras)} paragraphs")
        time.sleep(0.4)
    with gzip.open(os.path.join(OUT, "platforms.json.gz"), "wt") as f:
        json.dump(rows, f)
    print(f"wrote {len(rows)} paragraph records across {len({r['year'] for r in rows})} elections")


if __name__ == "__main__":
    main()
