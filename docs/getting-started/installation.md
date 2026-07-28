# Installation

topica ships as a compiled wheel. No Rust toolchain or JVM required.

```bash
pip install topica
```

or, with [uv](https://github.com/astral-sh/uv):

```bash
uv pip install topica
```

The runtime dependencies are **NumPy** and **pandas** (the DataFrame workflow,
`from_dataframe`, and the bundled datasets all build on pandas). Everything else
is an optional extra, so the core stays light. Install the ones you need:

| Install | Enables |
|---------|---------|
| `pip install "topica[viz]"` | matplotlib figures: [`plot_report`](../api/diagnostics.md), `quality_frontier(plot=True)`, the search-K and discovery plots |
| `pip install "topica[formula]"` | The R-style formula interface ([`design_matrix`](../api/keywords.md), `estimate_effect(formula=...)`); pulls in `formulaic` and `pandas` |
| `pip install "topica[polars]"` | Pass Polars DataFrames/Series to [`from_dataframe`](../api/keywords.md), `align`, and `design_matrix` |
| `pip install "topica[llm]"` | LLM topic labels and embeddings ([`llm_topic_labels`](../api/diagnostics.md), [`llm_embed`](../api/keywords.md)); installs `llm` plus the ollama plugin, so OpenAI works with `OPENAI_API_KEY` and a fully local path runs through ollama |

Combine extras in one install, e.g. `pip install "topica[llm,viz,formula]"`. For
local *sentence-transformer* embeddings add `llm-sentence-transformers` (which
pulls in PyTorch); ollama's own embedding models need nothing extra. Two more
packages also light up if already present: `pyLDAvis` (interactive
intertopic-distance charts) and `pandas` (tabular handling of effect/diagnostic
tables).

## Requirements

- Python 3.9+
- A platform with a prebuilt wheel (Linux, macOS, Windows on x86-64 / arm64).

## Building from source

If you want to build from the repository (e.g. to hack on the Rust core) you'll
need a Rust toolchain and [maturin](https://github.com/PyO3/maturin):

```bash
git clone https://github.com/nealcaren/topica
cd topica
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy pytest
maturin develop --release --features python
pytest
```
