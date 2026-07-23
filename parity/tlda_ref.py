"""Portable locator for the upstream TensorLy TLDA reference implementation.

topica's ``TensorLDA`` is validated against
[TensorLy TLDA](https://github.com/tensorly/tlda), the reference NumPy
implementation from Kangaslahti et al. (2026). That package is not on PyPI and
is not a topica dependency, so the parity scripts locate it at run time and skip
cleanly when it is absent.

Obtaining the reference (one-time)::

    git clone https://github.com/tensorly/tlda /path/to/tlda
    export TOPICA_TLDA_REF=/path/to/tlda

With ``TOPICA_TLDA_REF`` set, ``parity/tlda_compare.py`` runs the upstream
comparison; without it (the CI default) the comparison reports that the
reference is unavailable and exits 0. This module never mutates ``sys.path`` for
a machine-specific hard-coded location -- the reference path is always taken from
the environment.
"""

from __future__ import annotations

import os
import sys
from typing import Optional


def reference_dir() -> Optional[str]:
    """Return the reference checkout directory from ``TOPICA_TLDA_REF``, if set."""
    path = os.environ.get("TOPICA_TLDA_REF")
    if path and os.path.isdir(path):
        return path
    return None


def load_reference_tlda():
    """Return the upstream ``TLDA`` class, or ``None`` if the reference is absent.

    The reference checkout is placed on ``sys.path`` only when
    ``TOPICA_TLDA_REF`` points at a real directory. Several import shapes are
    tried so a flat repo checkout, a packaged install, or a local convenience
    wrapper all resolve.
    """
    ref = reference_dir()
    if ref and ref not in sys.path:
        sys.path.insert(0, ref)

    for module, attr in (
        ("tlda_final", "TLDA"),   # flat tensorly/tlda checkout
        ("tlda", "TLDA"),         # packaged install
        ("tlda_wrapper", "TLDA"),  # local convenience wrapper
    ):
        try:
            mod = __import__(module, fromlist=[attr])
            return getattr(mod, attr)
        except (ImportError, AttributeError):
            continue
    return None
