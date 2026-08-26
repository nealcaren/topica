# Task recipes

Small, self-contained scripts organized by the **task** you were handed, not by
the corpus they happen to use. Each is ~60 lines, runs on data that ships with
topica (so it works with no download beyond the one noted), and is written to be
transplanted: swap the load + column names at the top for your own DataFrame and
the rest carries over. Each is smoke-tested in `tests/test_recipes.py`.

| If your task is... | Recipe | Model |
|---|---|---|
| Explore themes with no prior structure, and justify K | [`lda_explore.py`](lda_explore.py) | LDA + search_k |
| Measure concepts you can name in advance (seed words) | [`keyatm_seeded.py`](keyatm_seeded.py) | keyATM |
| Topics for very short documents (tweets, survey answers) | [`gsdmm_short_text.py`](gsdmm_short_text.py) | GSDMM |
| Cluster documents by meaning using embeddings | [`bertopic_embeddings.py`](bertopic_embeddings.py) | BERTopic |
| Do two groups talk about different things? (treatment vs control, party A vs B) | [`stm_prevalence_groups.py`](stm_prevalence_groups.py) | STM prevalence covariate |
| Do two groups word the same topics differently? | [`stm_content_groups.py`](stm_content_groups.py) | STM content covariate |
| How does a topic's wording drift over time? | [`dtm_over_time.py`](dtm_over_time.py) | Dynamic Topic Model |
| Are my topics real, or seed/sample noise? | [`robustness.py`](robustness.py) | bootstrap_stability |
| Record provenance so the analysis reproduces | [`provenance.py`](provenance.py) | record_fit → AnalysisManifest |

Run any of them directly:

```bash
python examples/recipes/stm_prevalence_groups.py
```

For the canonical end-to-end STM walkthrough (labelling, FREX, topic correlation,
representative documents), see [`../stm_vignette.py`](../stm_vignette.py). For the
one-screen API cheat sheet, run `topica.guide()`.
