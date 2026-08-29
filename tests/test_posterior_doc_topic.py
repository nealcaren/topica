"""STM/CTM posterior_doc_topic — the hedged E[softmax(eta)] readout, symmetric with
ThreadTM.posterior_doc_topic (issue #840)."""
import numpy as np
import pytest

import topica


def _corpus(seed=0, n=120):
    rng = np.random.default_rng(seed)
    A = [f"a{i}" for i in range(6)]
    B = [f"b{i}" for i in range(6)]
    docs, grp = [], []
    for t in range(n):
        g = t % 2
        pool = A if g == 0 else B
        docs.append([pool[rng.integers(6)] for _ in range(8)])  # short docs -> non-trivial nu
        grp.append(g)
    prevalence = np.array(grp, dtype=float).reshape(-1, 1)  # STM needs a covariate; CTM ignores it
    return docs, prevalence


def _fit(Model, docs, prevalence, **kw):
    # STM requires prevalence/content; CTM takes neither.
    if Model is topica.STM:
        return Model(3, seed=13).fit(docs, prevalence=prevalence, **kw)
    return Model(3, seed=13).fit(docs, **kw)


@pytest.mark.parametrize("Model", [topica.CTM, topica.STM])
def test_posterior_doc_topic(Model):
    docs, prevalence = _corpus()
    m = _fit(Model, docs, prevalence)
    plug = np.asarray(m.doc_topic)
    pdt = np.asarray(m.posterior_doc_topic(n_samples=400, seed=13))
    assert pdt.shape == plug.shape
    assert np.allclose(pdt.sum(axis=1), 1.0) and (pdt >= 0).all()
    # deterministic given seed; integrates nu, so it is not the plug-in softmax(mean eta)
    assert np.array_equal(pdt, np.asarray(m.posterior_doc_topic(n_samples=400, seed=13)))
    assert not np.allclose(pdt, plug)
    # posterior-predictive hedges: on average flatter (higher entropy) than the plug-in
    ent = lambda p: -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1)
    assert ent(pdt).mean() > ent(plug).mean()
    with pytest.raises(ValueError):
        m.posterior_doc_topic(n_samples=0)


def test_posterior_doc_topic_needs_eta_cov():
    docs, prevalence = _corpus(n=60)
    m = topica.STM(3, seed=13).fit(docs, prevalence=prevalence, keep_eta_cov=False)
    with pytest.raises(ValueError, match="keep_eta_cov"):
        m.posterior_doc_topic()
