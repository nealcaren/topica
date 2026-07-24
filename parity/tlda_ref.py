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

    The env-var-only contract is strict: with ``TOPICA_TLDA_REF`` unset we return
    ``None`` immediately and never probe the ambient path, so a stray installed
    ``tlda`` package or a leftover local ``tlda_wrapper.py`` cannot silently stand
    in for the configured reference. When the var is set, the checkout is placed
    on ``sys.path`` and several import shapes are tried (flat repo checkout,
    packaged install, local wrapper) -- but a resolved module is accepted only if
    it actually originates under the configured checkout, not from an ambient
    install shadowing the same name.

    If the var is set but nothing usable resolves (e.g. the reference's own deps
    are broken), a warning is emitted to stderr so a misconfigured integration
    job does not look like a clean skip.
    """
    ref = reference_dir()
    if ref is None:
        return None
    if ref not in sys.path:
        sys.path.insert(0, ref)

    ref_real = os.path.realpath(ref)
    for module, attr in (
        ("tlda_final", "TLDA"),   # flat tensorly/tlda checkout
        ("tlda", "TLDA"),         # packaged install
        ("tlda_wrapper", "TLDA"),  # local convenience wrapper
    ):
        try:
            mod = __import__(module, fromlist=[attr])
            cls = getattr(mod, attr)
        except (ImportError, AttributeError):
            continue
        mod_file = getattr(mod, "__file__", None)
        if mod_file and os.path.realpath(mod_file).startswith(ref_real + os.sep):
            return cls

    print(
        f"[tlda_ref] TOPICA_TLDA_REF={ref!r} is set but no usable TLDA reference "
        f"resolved under it (broken deps or unexpected layout); treating as absent.",
        file=sys.stderr,
    )
    return None
