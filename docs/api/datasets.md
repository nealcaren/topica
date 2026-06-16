# Datasets

Bundled example datasets for quickstarts and worked examples. Small datasets ship
inside the wheel and load offline; larger ones are downloaded once from GitHub on
first use and cached locally (under `~/.cache/topica/datasets`, or
`TOPICA_DATA_HOME`). Each loader returns a `pandas` DataFrame ready for
[`from_dataframe`](keywords.md); pass `return_path=True` for the cached CSV path
without pandas.

```python
import topica

df = topica.datasets.load_gadarian()
corpus = topica.from_dataframe(
    df, text_col="open.ended.response", stopwords=topica.ENGLISH_STOPWORDS
)
```

::: topica.datasets.load_gadarian

::: topica.datasets.load_poliblog

::: topica.datasets.load_dubois

::: topica.datasets.get_data_home

::: topica.datasets.clear_cache
