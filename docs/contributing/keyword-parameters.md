# Keyword and seed parameters across models

Several models let a user steer topics with words: seed words, keywords, or
anchor words. These models come from different literatures and reference
packages, so their parameters drifted apart. This page records how each one
treats keywords as parameters, and the conventions we hold new work to so the
family reads as one library. It pairs with the general
[API conventions](conventions.md); this page is the keyword-specific extension.

## The models that take user keywords

Five models accept user-supplied keyword input. Two look like they do but do
not, and are worth naming so nobody mistakes them for a seeding surface:

- `AnchorLDA` selects its anchor words *algorithmically* (Arora et al.'s
  FastAnchorWords on the co-occurrence matrix). There is no user keyword
  argument; `anchor` here is the spectral sense, not a supplied word.
- `KeyNMF` *extracts* keywords per document from embeddings (`top_n`). The user
  supplies embeddings, not seeds.

The five genuine keyword models:

| Model | Keyword arg | Key type | Required | Strength knob(s) | Mechanism |
|---|---|---|---|---|---|
| `SeededLDA` | `seed_words` | `dict[str, list]` | yes | `weight` (0.01), `seed_prior` | asymmetric β prior pseudocount |
| `KeyATM` | `keywords` | `dict[str, list]` | yes | `beta_keyword` (0.1), `gamma1`/`gamma2` (1.0) | switch variable π + separate keyword distribution |
| `GuidedNMF` | `seed_words` | `dict[str, list]` | yes | `guidance` (3.0), `seed_weight` (1.0) | supervision penalty term `λ‖Y−BS‖²` |
| `CorEx` | `anchor_words` | `dict[str, list]` \| None | no | `anchor_strength` (1.0) | α membership override in the total-correlation objective |
| `ThreadTM` | `seed_words` | `dict[str, list]` \| `dict[int, list]` \| None | no | `weight` (0.01), `seed_prior`, `seed_strength` | asymmetric β prior pseudocount |

## Four mechanisms, not one

Keywords do not enter the math the same way, and this is intentional: each model
mirrors a different reference. The strength knob therefore cannot be unified
into a single scalar, because "how hard do I push" means a different quantity in
each family.

1. **Prior pseudocount** (`SeededLDA`, `ThreadTM`). The seed adds an asymmetric
   Dirichlet pseudocount to the seeded topic's β row, `β_kw = β + m_kw`. It is a
   soft prior: the topic keeps learning the rest of its vocabulary. This is the
   only mechanism where two models share the same math, so it is the one place we
   require the knobs to agree (see below).
2. **Switch variable** (`KeyATM`). Each keyword topic has its own restricted
   keyword distribution and a per-topic Bernoulli switch (Beta prior `gamma1`,
   `gamma2`) that decides, per token, whether to draw from the keywords or the
   full vocabulary. Not a prior on β.
3. **Penalty term** (`GuidedNMF`). The seed matrix enters the objective as a
   second Frobenius term `λ‖Y−BS‖²` (`guidance` is `λ`); a learned mixing matrix
   maps seed groups to topics.
4. **Membership override** (`CorEx`). Anchoring writes `anchor_strength` directly
   into the word-to-topic weight `alpha` used by the total-correlation objective.

## The shared machinery

Four of the five (`SeededLDA`, `KeyATM`, `GuidedNMF`, `CorEx`) parse their
keyword dict through the same Rust helpers, and `ThreadTM` reuses the matcher.
New keyword models should use them rather than re-parse a dict:

- `parse_seed_dict` / `seed_word_ids` (`src/python/mod.rs`) turn a
  `dict[name → words]` into per-topic word-id lists, with the name preserved as
  the topic label.
- `SeedMatch` gives every keyword model the same matching vocabulary:
  `seed_match` in `{"fixed", "glob", "regex"}` plus `case_insensitive`. `"fixed"`
  is exact literal equality; `"glob"` reads `*`/`?` wildcards anchored to the
  whole token; `"regex"` uses Rust's linear-time `regex` crate (no
  backreferences or lookaround). `KeyATM` is exact-literal only.

## Conventions for new keyword models

1. **Keyword input is a name-keyed dict**, `dict[str, list[str]]`, where the key
   names the seeded topic and takes a leading topic slot in insertion order. This
   is the majority form and matches quanteda's dictionary spirit.
   `ThreadTM` additionally accepts `dict[int, list]` for positional slots because
   its topics pre-exist independent of the seeds; where a model's seed groups
   *define* the topics (as in `SeededLDA`/`KeyATM`), use only the name-keyed form.
2. **Keep the reference-faithful argument name** where the reference community has
   one: `keywords` (keyATM), `seed_words` (seededlda), `anchor_words` (Anchored
   CorEx). We accept these three names for the same concept as a deliberate cost
   of matching the source literatures; do not invent a fourth.
3. **For the prior-pseudocount mechanism, use `weight`.** A `[0, 1]` fraction,
   default `0.01`, scaled to a corpus-count-magnitude pseudocount internally
   (`corpus_count(word) * weight * 100` under `seed_prior="frequency"`, a flat
   `weight * 100` under `"uniform"`). This is `SeededLDA`'s scheme, and any other
   prior-based seeder must express strength the same way so a value is portable.
   `seed_strength` is the escape hatch for a raw, unscaled per-word pseudocount.
   Mechanisms that are not prior pseudocounts keep their reference-tied strength
   name (`beta_keyword`, `guidance`, `anchor_strength`); those numbers are not
   comparable across mechanisms and should not be forced into one name.
4. **`anchor` refers to word-level anchoring only.** `anchor_words` /
   `anchor_strength` mean pinning words to a topic (CorEx's sense). Steering topic
   *prevalence* toward a target is a different axis and uses `prevalence_*`
   (`ThreadTM`'s `prevalence_anchor` / `prevalence_strength`). Do not reuse
   `anchor_strength` for a prevalence quantity.
5. **Seeding is a soft prior, and says so.** A seeded topic must keep mass on
   unseeded words rather than collapsing onto the seeds. Expose an audit of what
   the patterns actually matched (`ThreadTM.seed_matches`), and warn on the
   natural mistakes: an orphaned strength knob with no keyword dict, a keyword
   group that matched nothing, an index-looking string key.

## Divergences we accept

- **Three names for one concept** (`keywords` / `seed_words` / `anchor_words`).
  Each is faithful to its reference package. The cost is that the concept is not
  greppable under a single name; we judge reference fidelity the larger good.
- **keyATM's switch mechanism** is the odd one out and its strength knobs
  (`beta_keyword`, `gamma1`, `gamma2`) do not translate to the prior-based
  models. That is inherent to the model, not drift.
- **GuidedNMF's `guidance`** is a relative penalty weight (default `3.0`, tuned
  down from the reference's `20` to preserve document prevalence), not a
  pseudocount. It is not comparable to `weight`.

## History

`ThreadTM` was realigned to these conventions in
[#862](https://github.com/nealcaren/topica/pull/862): `seed_weight` became
`weight` on SeededLDA's scale, `seed_words` gained the name-keyed form and a
`topic_names` property, and `anchor_strength` became `prevalence_strength`.
