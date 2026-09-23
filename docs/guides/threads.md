# Threaded conversations: thread-context shrinkage

Forum comments are written in reply to something, and deep replies are often too short to
place on their own words. `topica.threads.ThreadSmoother` keeps an ordinary topic model as the
base (LDA, STM, CTM, or any fitted model with `doc_topic` and `topic_word`) and lets each
document borrow topic mass from its reply context:

$$
\tilde\theta_d = \frac{n_d\,\theta_d + a_p\,\theta_{\mathrm{par}(d)} + a_t\,\theta_{\mathrm{thr}(d)}}
                      {n_d + a_p + a_t}
$$

Here $n_d$ is the document's in-vocabulary token count, $\theta_{\mathrm{par}(d)}$ its parent's
base topic mix, and $\theta_{\mathrm{thr}(d)}$ the token-weighted mean mix of the rest of its
thread. This is the posterior mean of $\theta_d$ under a Dirichlet prior centered on the reply
context, with pseudo-counts $a_p$ and $a_t$. A three-token reply is dominated by its context; a
three-hundred-token reply barely moves.

!!! warning "Experimental"
    `ThreadSmoother` is an original construction validated by planted recovery (see
    `tests/test_threads.py`), with no published reference yet. Call
    `topica.enable_experimental()` first. It may change without a deprecation cycle.

## Fit

```python
import topica
from topica import threads

topica.enable_experimental()
data = topica.datasets.load_threads()
docs, parents = data.documents, data.parents

sm = threads.ThreadSmoother()                   # contexts=("parent", "thread")
sm.fit(docs, parents,
       base=lambda corpus: topica.LDA(20, seed=13).fit(corpus, iters=1000),
       corpus_kwargs={"min_cf": 5}, seed=13)
print(sm)
theta = sm.theta_tilde                          # (D, K), one row per input document
```

`base` is a factory, `corpus -> fitted model`, because the pseudo-counts are estimated on a
masked refit: half the tokens of eligible leaf replies are held out, the base is fit on the
rest, the pseudo-counts are chosen by held-out log likelihood on validation threads, and every
reported number comes from separate test threads. The factory is then called once more on the
full corpus, and `theta_tilde` smooths that fit. For an STM base, index the prevalence design by
`corpus.kept_indices` inside the factory so its rows follow the documents the corpus kept.

To smooth a model you fit yourself on the same documents, use
`sm.transform(model, corpus, parents)`.

## Read

| Attribute | Meaning |
|---|---|
| `alpha`, `alpha_ci` | pseudo-counts per context, with 95% thread-bootstrap intervals |
| `parent_share`, `parent_share_ci` | $a_p / (a_p + a_t)$: how dyadic the conversation is |
| `edge_effect` | held-out gain of the true parent over a shuffled parent, nats per token |
| `completion` | held-out gain over the base, overall and by reply-length tercile |
| `alpha_by_group` | per-community pseudo-counts when `groups=` is passed |

`edge_effect` is the placebo-netted measure of what the *specific* parent adds. The placebo
permutes parent assignments within (thread, depth), which keeps every reply's depth and thread
and every parent's number of children, and changes only which comment each reply answers.

## Strip quotes first

Quoted text makes a reply look like its parent for reasons that have nothing to do with topical
uptake. Strip it before fitting:

```python
texts = [threads.strip_quotes(t) for t in raw_texts]        # ">" and "&gt;" lines, <i> spans
docs = [topica.tokenize(t, stopwords="english") for t in texts]
docs = threads.strip_copied_runs(docs, parents, n=5)        # unmarked verbatim copying
```

## Limits

- One pseudo-count per context borrows for every reply alike. When only some replies take up
  their parent's topics and the others turn to sharply different ones, borrowing can cost the
  second group as much as it helps the first, and the fit returns zero. Read a zero
  pseudo-count as "borrowing does not help on average", not "no reply follows its parent".
  The gain by reply length (`completion["by_length"]`) shows where borrowing helps and hurts.
- Pseudo-counts are calibrated on leaf replies and applied to every document.
- `theta_tilde` is a topic measure. Lexical reuse of the parent's own words is a separate
  construct and is not folded in.
