# Scaling up: all 20 newsgroups

The 5-group experiment (`report.md`) is here rerun on the **full 20 Newsgroups**
train split — same recipe, same pipeline, just 20 groups and K = 20 instead of
5. Build the data with `prepare_full_ng20.py` (downloads 20NG, applies the
bundled preprocessing recipe, computes `all-MiniLM-L6-v2` embeddings) and run
with `run_full.py`. After filtering (≥ 20 words, ≥ 5 in-vocab tokens):
**10,115 documents, 20 groups, 9,144-term vocabulary.**

## Headline numbers

| quantity | 5 groups | **20 groups** |
|---|---:|---:|
| embeddings → LDA cluster (CV) | 0.850 | **0.729** |
| embeddings → true group (ceiling) | 0.899 | **0.725** |
| embeddings → shuffled cluster (floor) | 0.434 | **0.258** |
| base rate (largest cluster) | 0.453 | 0.266 |
| LDA vs. true — ARI / NMI / best-map | 0.32 / 0.47 / 0.66 | **0.16 / 0.43 / 0.39** |
| disagreements | 15.0% | **27.1%** |

The clusters stay strongly learnable: **73%** cross-validated recovery against a
**26%** base rate (shuffled floor 26%). Two independent views of a 20-way
partition still agree on nearly three-quarters of documents.

## The interesting flip

At 5 groups the embeddings recovered the *true* newsgroups (90%) better than
the *LDA clusters* (85%) — the real labels were the easier target. **At 20
groups that reverses: embeddings recover LDA's clusters (72.9%) as well as, or
marginally better than, the official newsgroups (72.5%).**

Why: several of the 20 official groups overlap heavily —
`comp.sys.ibm.pc.hardware` vs. `comp.sys.mac.hardware`, the three `talk.politics.*`,
`soc.religion.christian` vs. `talk.religion.misc` vs. `alt.atheism`. LDA doesn't
carve along those administrative seams; it groups by content, and the embedding
geometry agrees with content grouping at least as much as with the newsgroup
taxonomy. The LDA topics themselves are clean and readable — space, `sci.crypt`
(key/encryption/clipper), hockey, guns, C source code (`file/entry/output/char`),
Mideast politics — even though they only best-map to the official labels at
0.39 accuracy. The disagreement between LDA and the taxonomy is real, and the
embeddings don't clearly take the taxonomy's side.

## Are the disagreements still a meaningful set? Mostly — with two caveats.

27.1% of documents (2,746) land in a different cluster. As before they skew
toward LDA's lower-confidence assignments (mean topic entropy 0.52 vs. 0.45;
max-θ 0.41 vs. 0.53), and many are genuine semantic corrections:

- A hockey **world-championships** post — *"world championships germany group
  standings munich … canada czech republic russia finland"* — that LDA filed
  under **talk.politics.mideast** on the flood of country names; the embedding
  reads it correctly as **rec.sport.hockey**.
- An **alt.atheism** FAQ that LDA dropped in the generic email/admin cluster
  (mapped to mac-hardware); the embedding recognizes **alt.atheism**.
- An **Armenian human-rights** post filed under `talk.religion.misc`; embedding
  → **talk.politics.mideast**.
- An edge-detection **image** routine filed under the C-code cluster; embedding
  → **comp.graphics**.

But two things change at 20-way:

1. **The junk-post confound.** 20NG's notorious garbage posts (uuencode/ASCII
   spew — *"max max bxn bxn giz qax"*) form their own high-confidence LDA cluster
   (topic 3, θ ≈ 0.99). The embedding classifier can't place gibberish and dumps
   it in the generic bucket, so these dominate the *most-confident* disagreements
   without being semantically meaningful. Filtering the junk and the two generic
   "discussion" clusters leaves **737** content-vs-content disagreements, of which
   **233** are cases where the embedding matches the true group and LDA didn't.

2. **The referee goes quiet.** With the true label as external referee, the
   embedding pick and the LDA cluster now match the true group about equally on
   the disagreement set (24.1% vs. 25.5%; overall 38.2% vs. 38.5%). The clean
   "embedding corrects the topic model" win from the 5-group run flattens to a
   tie — not because the disagreements are meaningless, but because at 20-way the
   *ground truth itself is ambiguous*: when `talk.politics.misc`,
   `talk.politics.guns`, and `talk.religion.misc` all bleed together, "the right
   newsgroup" is barely defined, so neither view can look decisively righter
   against it (both sit near the 0.39 best-map ceiling).

## K sweep (full corpus)

| K | recovery | disagree | ARI to true |
|--:|---:|---:|---:|
| 10 | 0.774 | 22.6% | 0.157 |
| 20 | 0.729 | 27.1% | 0.164 |
| 30 | 0.653 | 34.7% | 0.211 |
| 40 | 0.636 | 36.4% | 0.241 |

Same shape as before — finer slicing lowers recovery and raises disagreement —
with a twist: ARI to the true labels *rises* with K (0.16 → 0.24). Twenty
content topics under-describe the corpus; letting LDA use 30–40 lets it
approximate the fine, overlapping newsgroup structure more closely.

## Answer at full scale

The different-cluster documents remain a meaningful set — genuinely ambiguous,
boundary-straddling, and full of legible cases where sentence meaning corrects a
bag-of-words slip. What the full corpus adds is honesty about the limits: at
20-way the set is **diluted** by degenerate junk posts and by the fact that the
official newsgroups themselves overlap, so you can no longer claim the embedding
is cleanly "more right." The technique still does its main job well — surfacing
the hard, confusable documents — you just have to read past the junk cluster,
and accept that some disagreements are two defensible answers to a genuinely
ambiguous document rather than a right and a wrong one.
