"""Build examples/reddit_threads.csv: the ThreadTM two-subreddit vignette corpus.

Source: ConvoKit ``reddit-corpus-small`` (Chang et al. 2020), a 100-subreddit
Reddit sample. We keep two subreddits chosen to make ThreadTM's point honestly:

  - askscience    -- technical Q&A; replies genuinely answer their parent, so the
                     reply tree carries topic structure (persistence is identified).
  - pokemontrades -- has the DEEPEST reply trees in the whole corpus, yet its
                     replies coordinate trades ("added you on DS") rather than
                     respond on-topic, so persistence is NOT identifiable. The
                     honest counter-example: tree depth is not persistence.

Output schema (one row per comment/post, ordered by subreddit, then thread, then
within-thread breadth-first so every parent precedes its children):

    doc_id, thread_root, parent, subreddit, timestamp, text

``parent`` is the 0-based row index (into this file's order) of the comment this
one replies to, or -1 for a thread root. Text is raw and untokenized.

Deterministic: same source, same bytes. Regenerate with::

    pip install convokit
    python examples/build_reddit_threads.py

The committed CSV is what ``topica.datasets.load_threads()`` fetches; its sha256
is pinned in the datasets registry.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

KEEP = ["askscience", "pokemontrades"]
OUT = Path(__file__).resolve().parent / "reddit_threads.csv"


def build_from_convokit() -> list[dict]:
    from convokit import Corpus, download

    corpus = Corpus(filename=download("reddit-corpus-small"))
    # Group utterances by (subreddit, thread root). In ConvoKit a Reddit thread is
    # one Conversation; the root utterance has reply_to == None.
    rows: list[dict] = []
    for sub in KEEP:
        convos = [c for c in corpus.iter_conversations()
                  if c.meta.get("subreddit") == sub]
        # stable order: by conversation id
        for convo in sorted(convos, key=lambda c: c.id):
            utts = {u.id: u for u in convo.iter_utterances()}
            root = next((u for u in utts.values() if u.reply_to is None), None)
            if root is None:
                continue
            # breadth-first from the root so parents precede children
            children: dict[str, list[str]] = {}
            for u in utts.values():
                if u.reply_to is not None:
                    children.setdefault(u.reply_to, []).append(u.id)
            order, queue = [], [root.id]
            while queue:
                uid = queue.pop(0)
                order.append(uid)
                queue.extend(sorted(children.get(uid, [])))
            local = {uid: i for i, uid in enumerate(order)}
            base = len(rows)
            for uid in order:
                u = utts[uid]
                text = (u.text or "").strip()
                if not text or text in ("[deleted]", "[removed]"):
                    continue  # deleted content; roots are always non-deleted here
                parent = -1 if u.reply_to is None else base + local[u.reply_to]
                rows.append({
                    "doc_id": u.id,
                    "thread_root": root.id,
                    "parent": parent,
                    "subreddit": sub,
                    "timestamp": u.timestamp,
                    "text": text,
                })
    return rows


def write_csv(rows: list[dict]) -> None:
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["doc_id", "thread_root", "parent",
                                           "subreddit", "timestamp", "text"])
        w.writeheader()
        w.writerows(rows)
    # integrity: every non-root parent precedes its child and shares its thread
    for i, r in enumerate(rows):
        p = r["parent"]
        if p >= 0:
            assert p < i, f"row {i}: parent {p} not before child"
            assert rows[p]["thread_root"] == r["thread_root"], f"row {i}: cross-thread parent"
    print(f"wrote {len(rows)} rows, "
          f"{sum(1 for r in rows if r['parent'] < 0)} threads -> {OUT}")


if __name__ == "__main__":
    try:
        rows = build_from_convokit()
    except ImportError:
        sys.exit("needs convokit: pip install convokit")
    write_csv(rows)
