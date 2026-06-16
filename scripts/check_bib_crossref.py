"""Verify BibTeX entries against CrossRef.

For each target entry: if it has a DOI, look it up directly
(api.crossref.org/works/<doi>); otherwise run a bibliographic title search and
take the top hit. Then compare title, first-author surname, year, container
(journal/booktitle), and pages, and print MATCH / DIFF / (missing DOI) so the
canonical metadata can be checked by eye before submission.

Stdlib only. Polite-pool usage via a mailto.

    python scripts/check_bib_crossref.py                 # last 8 entries of paper/topica.bib
    python scripts/check_bib_crossref.py --bib paper/topica.bib --last 8
    python scripts/check_bib_crossref.py --key wu2023infoctm pham2024topicgpt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request

MAILTO = "neal.caren@unc.edu"
API = "https://api.crossref.org/works"


# --- minimal brace-aware .bib parser ---------------------------------------

def parse_bib(text):
    """Return [(key, type, {field: value})] in file order."""
    entries = []
    i = 0
    while True:
        at = text.find("@", i)
        if at == -1:
            break
        open_brace = text.find("{", at)
        if open_brace == -1:
            break
        etype = text[at + 1:open_brace].strip().lower()
        depth, j = 0, open_brace
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        body = text[open_brace + 1:j]
        i = j + 1
        if etype in ("comment", "string", "preamble"):
            continue
        key, _, rest = body.partition(",")
        entries.append((key.strip(), etype, _parse_fields(rest)))
    return entries


def _parse_fields(rest):
    fields = {}
    pos = 0
    for m in re.finditer(r"(\w+)\s*=\s*", rest):
        name = m.group(1).lower()
        start = m.end()
        if start < len(rest) and rest[start] == "{":
            depth, k = 0, start
            while k < len(rest):
                if rest[k] == "{":
                    depth += 1
                elif rest[k] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            val = rest[start + 1:k]
        elif start < len(rest) and rest[start] == '"':
            k = rest.find('"', start + 1)
            val = rest[start + 1:k]
        else:
            k = rest.find(",", start)
            val = rest[start:(k if k != -1 else len(rest))]
        fields[name] = _clean(val)
    return fields


def _clean(s):
    s = re.sub(r"\\pkg\{([^}]*)\}|\\code\{([^}]*)\}|\\proglang\{([^}]*)\}",
               lambda m: next(g for g in m.groups() if g), s)
    s = s.replace("{", "").replace("}", "").replace("\\&", "&")
    return re.sub(r"\s+", " ", s).strip()


# --- CrossRef ---------------------------------------------------------------

def _get(url):
    req = urllib.request.Request(url, headers={
        "User-Agent": f"topica-bibcheck/1.0 (mailto:{MAILTO})"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def crossref_by_doi(doi):
    try:
        return _get(f"{API}/{urllib.parse.quote(doi)}?mailto={MAILTO}")["message"]
    except Exception as e:
        return {"_error": str(e)}


def crossref_by_title(title, author=""):
    q = urllib.parse.urlencode({
        "query.bibliographic": f"{title} {author}".strip(),
        "rows": "3", "mailto": MAILTO})
    try:
        items = _get(f"{API}?{q}")["message"]["items"]
        return items[0] if items else {"_error": "no results"}
    except Exception as e:
        return {"_error": str(e)}


# --- comparison -------------------------------------------------------------

def _norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def cr_field(msg, name):
    if name == "title":
        return (msg.get("title") or [""])[0]
    if name == "container":
        return (msg.get("container-title") or [""])[0]
    if name == "year":
        dp = (msg.get("published") or msg.get("issued") or {}).get("date-parts") or [[None]]
        return str(dp[0][0]) if dp[0] and dp[0][0] else ""
    if name == "first_author":
        a = msg.get("author") or []
        return a[0].get("family", "") if a else ""
    if name == "pages":
        return msg.get("page", "")
    return ""


def check(entry):
    key, etype, f = entry
    bib_title = f.get("title", "")
    bib_author = (f.get("author", "").split(" and ")[0].split(",")[0]).strip()
    bib_year = f.get("year", "")
    bib_container = f.get("journal") or f.get("booktitle") or ""
    bib_pages = f.get("pages", "")
    doi = f.get("doi", "")

    if doi:
        msg, how = crossref_by_doi(doi), f"DOI {doi}"
    else:
        msg, how = crossref_by_title(bib_title, bib_author), "title search"

    print(f"\n=== {key}  ({how}) ===")
    if "_error" in msg:
        print(f"  CrossRef error: {msg['_error']}")
        return
    rows = [
        ("title", bib_title, cr_field(msg, "title")),
        ("first author", bib_author, cr_field(msg, "first_author")),
        ("year", bib_year, cr_field(msg, "year")),
        ("container", bib_container, cr_field(msg, "container")),
        ("pages", bib_pages, cr_field(msg, "pages")),
    ]
    for name, bib, cr in rows:
        if name in ("title", "container", "first author"):
            ok = _norm(bib) == _norm(cr) or (_norm(cr) and _norm(cr) in _norm(bib)) \
                 or (_norm(bib) and _norm(bib) in _norm(cr))
        else:
            ok = _norm(bib) == _norm(cr)
        flag = "ok " if ok else "DIFF"
        print(f"  [{flag}] {name:13} bib: {bib!r}")
        if not ok:
            print(f"         {'':13} crossref: {cr!r}")
    if not doi and msg.get("DOI"):
        print(f"  >> CrossRef DOI (consider adding): {msg['DOI']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bib", default="paper/topica.bib")
    ap.add_argument("--last", type=int, default=8)
    ap.add_argument("--key", nargs="*", help="check specific keys instead of --last")
    args = ap.parse_args()

    text = open(args.bib, encoding="utf-8").read()
    entries = parse_bib(text)
    if args.key:
        targets = [e for e in entries if e[0] in args.key]
    else:
        targets = entries[-args.last:]

    print(f"Checking {len(targets)} entr(y/ies) from {args.bib} against CrossRef")
    for e in targets:
        check(e)
        time.sleep(0.5)  # be gentle on the API


if __name__ == "__main__":
    main()
