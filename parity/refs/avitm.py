"""Shared PyTorch primitives for the AVITM / ProdLDA reference encoders used by
topica's parity scripts (``prodlda_compare.py``, ``infoctm_compare.py``).

Both scripts build the *same* autoencoding-variational LDA encoder backbone --
softplus MLP, separate mean/logvar heads, affine-free batch normalization on the
heads, the Laplace approximation to the Dirichlet prior (Srivastava & Sutton 2017,
eq. 6), the reparameterization trick, and the Dirichlet-Laplace KL. Only those
pieces live here; each script keeps its own decoder and training loop (ProdLDA: a
``Linear(k, v, bias=False)`` + decoder batchnorm + log-softmax reconstruction with
dropout on theta; InfoCTM: a ``phi`` matmul + decoder batchnorm + softmax recon,
two languages, plus the TAMI alignment term).

torch is imported lazily inside each function so the scripts still import (and skip
cleanly) when torch is unavailable.

**Layer construction order is load-bearing.** torch consumes its RNG stream in the
order ``nn.Linear`` layers are constructed, so to keep each reference's output
byte-identical to its pre-refactor form the backbone builder must construct
``fc1, fc2, mu, lv`` in exactly that order. ``BatchNorm1d(affine=False)`` and
``Dropout`` have no learnable parameters and consume no RNG, so they do not affect
the stream -- matching both original scripts.
"""

from __future__ import annotations

import numpy as np


def laplace_prior(k: int, alpha: float):
    """Diagonal logistic-normal Laplace approximation to a symmetric Dirichlet
    prior in the softmax basis (Srivastava & Sutton 2017, eq. 6).

    Returns ``(mu, var)`` as float32 tensors of shape ``(k,)``. They broadcast
    against batched ``(batch, k)`` mean/logvar exactly as both originals relied on.
    """
    import torch

    a = np.full(k, alpha)
    mu1 = np.log(a) - np.mean(np.log(a))
    var1 = (1.0 / a) * (1.0 - 2.0 / k) + (1.0 / (k * k)) * np.sum(1.0 / a)
    return (
        torch.tensor(mu1, dtype=torch.float32),
        torch.tensor(var1, dtype=torch.float32),
    )


def build_encoder(v_in: int, k: int, hidden: int, dropout: float = 0.0):
    """Construct an AVITM encoder backbone module.

    Layers are built in the order ``fc1, fc2, mu, lv`` (RNG-stream critical), with
    affine-free ``BatchNorm1d(k)`` on each head and an optional ``Dropout`` between
    the second hidden layer and the heads. The returned module exposes
    ``encode(x) -> (mu, lv)`` applying ``softplus(fc1) -> softplus(fc2) -> dropout
    -> (bn_mu(mu), bn_lv(lv))``.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(v_in, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.mu = nn.Linear(hidden, k)
            self.lv = nn.Linear(hidden, k)
            self.drop = nn.Dropout(dropout)
            self.bn_mu = nn.BatchNorm1d(k, affine=False, eps=1e-5, momentum=0.1)
            self.bn_lv = nn.BatchNorm1d(k, affine=False, eps=1e-5, momentum=0.1)

        def encode(self, x):
            h = F.softplus(self.fc1(x))
            h = F.softplus(self.fc2(h))
            h = self.drop(h)
            return self.bn_mu(self.mu(h)), self.bn_lv(self.lv(h))

        def forward(self, x):
            return self.encode(x)

    return _Encoder()


def reparameterize(mu, lv):
    """Gaussian reparameterization trick: ``mu + exp(0.5*lv) * randn_like(mu)``."""
    import torch

    return mu + torch.exp(0.5 * lv) * torch.randn_like(mu)


def dirichlet_laplace_kl(mu, lv, prior_mu, prior_var, k: int):
    """Per-doc KL between the diagonal-Gaussian posterior and the Laplace-Dirichlet
    prior:

        0.5 * ( (var/prior_var).sum + ((prior_mu-mu)^2/prior_var).sum
                - k + prior_var.log().sum() - lv.sum() )

    where ``var = lv.exp()``. Returns a per-doc tensor (shape ``(batch,)``).
    """
    var = lv.exp()
    return 0.5 * (
        (var / prior_var).sum(1)
        + ((prior_mu - mu) ** 2 / prior_var).sum(1)
        - k
        + prior_var.log().sum()
        - lv.sum(1)
    )
