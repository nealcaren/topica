"""Build the congressional-speeches corpus for examples/ectm_speeches.py.

Downloads the Eugleo/us-congressional-speeches-subset parquet shards from the
Hugging Face Hub and joins speaker party from Voteview member data, then writes a
compact, subsampled corpus to examples/speech_data/speeches.parquet
(columns: period, party, text).

The raw shards total ~2.9 GB, so this is a one-time, bandwidth-heavy step; the
output parquet is small. Requires pandas + pyarrow. Run from the repo root:

    python examples/prep_speeches.py

Group = speaker party (D / R, via Voteview).  Time = 4-year period, 1948-2008.
"""
import os
import urllib.request

import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "speech_data")
HF = "https://huggingface.co/api/datasets/Eugleo/us-congressional-speeches-subset/parquet/default/train"
VOTEVIEW = "https://voteview.com/static/data/out/members/HSall_members.csv"
PER_CELL = 1500  # speeches kept per (period, party)


def fetch(url, dest):
    if not os.path.exists(dest):
        print(f"  downloading {os.path.basename(dest)} ...")
        urllib.request.urlretrieve(url, dest)


def party_lookup():
    """(congress, last_name) -> 'D'/'R', keeping only unambiguous surnames."""
    fetch(VOTEVIEW, os.path.join(OUT, "members.csv"))
    mem = pd.read_csv(os.path.join(OUT, "members.csv"), usecols=["congress", "party_code", "bioname"])
    mem = mem[mem.party_code.isin([100, 200])].copy()
    mem["last"] = mem.bioname.str.split(",").str[0].str.upper().str.strip()
    mem["P"] = mem.party_code.map({100: "D", 200: "R"})
    agg = mem.groupby(["congress", "last"]).P.agg(lambda s: s.iloc[0] if s.nunique() == 1 else "X")
    return {k: v for k, v in agg.to_dict().items() if v in ("D", "R")}


def main():
    os.makedirs(OUT, exist_ok=True)
    k2p = party_lookup()
    parts = []
    for i in range(8):
        shard = os.path.join(OUT, f"shard{i}.parquet")
        fetch(f"{HF}/{i}.parquet", shard)
        d = pd.read_parquet(shard, columns=["date", "chamber", "last_name", "word_count", "text"])
        d = d[d.chamber.isin(["H", "S"]) & d.word_count.between(60, 500) & (d.date.dt.year >= 1948)].copy()
        d["year"] = d.date.dt.year
        d["congress"] = ((d.year - 1789) // 2) + 1
        d["last"] = d.last_name.str.upper().str.strip()
        d["party"] = [k2p.get((c, l)) for c, l in zip(d.congress, d["last"])]
        d = d[d.party.isin(["D", "R"])]
        d["period"] = (d.year // 4 * 4).astype(int)
        parts.append(d[["period", "party", "text"]])
        print(f"  shard {i}: {len(d)} D/R speeches")
    df = pd.concat(parts, ignore_index=True)
    counts = df.groupby(["period", "party"]).size().unstack().fillna(0).astype(int)
    keep = [p for p in counts.index if counts.loc[p].min() >= 300]
    df = df[df.period.isin(keep)]
    df = pd.concat([g.sample(min(len(g), PER_CELL), random_state=0)
                    for _, g in df.groupby(["period", "party"])]).reset_index(drop=True)
    df.to_parquet(os.path.join(OUT, "speeches.parquet"))
    print(f"wrote {len(df)} speeches across periods {keep}")


if __name__ == "__main__":
    main()
