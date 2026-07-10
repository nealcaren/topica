# arXiv submission

Build the self-contained tarball:

```bash
VIRTUAL_ENV="$PWD/.venv-dev" .venv-dev/bin/python paper/replication.py --quick  # poliblog figures
VIRTUAL_ENV="$PWD/.venv-dev" .venv-dev/bin/python benchmarks/bench.py            # fig_thread_scaling
bash paper/make_arxiv.sh                                                         # the tarball
```

This produces `paper/arxiv-submission.tar.gz` containing exactly:

- `topica.tex` — the manuscript (already `\documentclass[article,nojss]{jss}`:
  preprint mode, no JSS masthead or logo, so no `jsslogo.jpg` is needed)
- `topica.bbl` — the compiled bibliography, so arXiv does not run BibTeX
- `jss.cls`, `jss.bst` — bundled so the build does not depend on arXiv carrying
  the `jss` package (it does, but bundling is belt-and-suspenders)
- `fig_poliblog_report.pdf`, `fig_poliblog_effect.pdf`, `fig_scaling.pdf`,
  `fig_thread_scaling.pdf` — the four figures the manuscript includes

The script compiles the assembled tarball in isolation before packing it, so what
ships is what was verified. Upload the tarball directly to arXiv (do not unpack).

## Submission form metadata

**Title:** topica: A Unified Python Framework for Topic Modeling in the Social
Sciences  (must match the `\title` in topica.tex)

**Authors:** Neal Caren (University of North Carolina at Chapel Hill)

**Primary category:** cs.CL (Computation and Language)
**Cross-list:** stat.CO (Computation), stat.ME (Methodology)

  Rationale: the audience is computational social science and text-as-data, which
  reads cs.CL; stat.CO is the natural home for a statistical-software paper aimed at
  JSS, and stat.ME catches the methods-minded readers. If you would rather lead with
  the software-paper framing, make stat.CO primary and cs.CL the cross-list.

**License:** CC BY 4.0 (recommended; permissive, matches the Apache-2.0 software).
The arXiv non-exclusive license is the fallback if a target journal forbids CC BY.

**Comments field:** 26 pages, 4 figures. Software: https://github.com/nealcaren/topica

**Abstract (plain text for the web form):**

Topic modeling is now a standard tool for social scientists who work with text, but
the models they need are scattered across incompatible software: the structural
topic model lives in R, the fastest collapsed-Gibbs sampler lives in Java, the
keyword-assisted and embedding-based models live in separate Python repositories,
and none of them share a data format, a diagnostic suite, or a guarantee that a fit
can be reproduced. We present topica, a Python library that brings more than a dozen
topic models behind one NumPy-native interface, running on a parallel Rust core.
Every model exposes the same shape, so the same coherence, exclusivity, stability,
and covariate-effect tools apply across all of them, and classical and
embedding-based models can be compared on one corpus without leaving the session.
The variational models are deterministic to the bit, including when multithreaded,
and the sampling models are reproducible from a fixed seed, which makes "refit and
check that the result holds" a real test rather than a hope. Each model is validated
against its reference implementation: the collapsed-Gibbs sampler reproduces a
Rust MALLET port byte-for-byte and recovers Java MALLET's topics, the
keyword-assisted model matches the R keyATM
package, and the structural topic model recovers the substantive conclusions of the
stm vignettes with honest uncertainty from the method of composition. On matched
iterations topica fits these models from three to twenty-two times faster than the
reference implementations. We describe the design, the model family, and a
principled analysis workflow built around the premise that the researcher, not the
software, owns the theoretical decisions that make a topic-model study credible.

## Notes

- The committed `topica.tex` is the arXiv-first preprint (`[article,nojss]{jss}`).
  For an eventual JSS journal submission, remove the `nojss` option, which restores
  the masthead and then needs the real `jsslogo.jpg` from
  https://www.jstatsoft.org/style.
- Before posting: re-verify Table 1 against current tool releases and consider a
  final read-through (the two open TODOs in `README.md`).
