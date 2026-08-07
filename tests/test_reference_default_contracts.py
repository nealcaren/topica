"""Default-contract checks against the original reference implementations.

The slow parity tests compare fitted models statistically. These tests cover a
different question: when topica says a default mirrors R ``stm``, R ``keyATM``,
or Java MALLET, does the public API default actually match the reference
program's own default?

Where exact default parity is not currently true, the mismatch is pinned as a
visible contract. If the library later decides to adopt the upstream default,
these tests should be updated with that decision rather than silently drifting.
"""

from __future__ import annotations

import ast
import math
import re
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PYI = REPO_ROOT / "python" / "topica" / "_topica.pyi"

pytestmark = pytest.mark.parity


def _topica_defaults(class_name: str, method_name: str) -> dict[str, object]:
    """Return default values from the public stub for a class method."""

    tree = ast.parse(PYI.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    out: dict[str, object] = {}
                    kw_defaults = item.args.kw_defaults
                    for arg, default in zip(item.args.kwonlyargs, kw_defaults):
                        if default is not None:
                            out[arg.arg] = ast.literal_eval(default)
                    positional = item.args.args
                    defaults = item.args.defaults
                    for arg, default in zip(positional[-len(defaults) :], defaults):
                        out[arg.arg] = ast.literal_eval(default)
                    return out
    raise AssertionError(f"{class_name}.{method_name} not found in {PYI}")


def _r_key_values(script: str) -> dict[str, str]:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not installed")
    proc = subprocess.run(
        ["Rscript", "-e", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        pytest.skip(f"R reference script failed:\n{proc.stderr[-2000:]}")
    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def _mallet_help_defaults() -> dict[str, str]:
    mallet = shutil.which("mallet")
    if mallet is None:
        pytest.skip("mallet CLI not installed")
    proc = subprocess.run(
        [mallet, "train-topics", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    text = proc.stdout + proc.stderr
    defaults: dict[str, str] = {}
    current: str | None = None
    for line in text.splitlines():
        opt = re.match(r"^--([a-z0-9-]+)\b", line)
        if opt:
            current = opt.group(1)
            continue
        default = re.match(r"\s*Default is (.+)$", line)
        if current and default:
            defaults[current] = default.group(1).strip()
    if not defaults:
        pytest.skip("could not parse mallet train-topics defaults")
    return defaults


def test_stm_fit_defaults_match_r_stm() -> None:
    r = _r_key_values(
        """
        if (!requireNamespace("stm", quietly=TRUE)) quit(status=2)
        f <- formals(stm::stm)
        cat("init=", eval(f$init.type)[[1]], "\\n", sep="")
        cat("max_em_its=", f$max.em.its, "\\n", sep="")
        cat("emtol=", f$emtol, "\\n", sep="")
        """
    )
    init_defaults = _topica_defaults("STM", "__init__")
    fit_defaults = _topica_defaults("STM", "fit")

    assert init_defaults["init"] == r["init"].lower()
    assert fit_defaults["iters"] == int(r["max_em_its"])
    assert math.isclose(fit_defaults["convergence_tol"], float(r["emtol"]), rel_tol=0, abs_tol=0)


def test_keyatm_base_defaults_match_r_keyatm_for_shared_controls() -> None:
    r = _r_key_values(
        r"""
        if (!requireNamespace("keyATM", quietly=TRUE) ||
            !requireNamespace("quanteda", quietly=TRUE)) quit(status=2)
        suppressMessages(library(keyATM))
        suppressMessages(library(quanteda))
        texts <- c("tax market tax budget", "war troop iraq war",
                   "tax budget market", "iraq troop war")
        docs <- keyATM_read(texts = quanteda::dfm(quanteda::tokens(texts)))
        keywords <- list(econ=c("tax","market"), war=c("war","iraq"))
        out <- keyATM(docs = docs, model = "base", no_keyword_topics = 0,
                      keywords = keywords,
                      options = list(seed = 1, iterations = 1, verbose = FALSE))
        cat("beta=", out$priors$beta, "\n", sep="")
        cat("beta_keyword=", out$priors$beta_s, "\n", sep="")
        cat("gamma1=", out$priors$gamma[1,1], "\n", sep="")
        cat("gamma2=", out$priors$gamma[1,2], "\n", sep="")
        cat("alpha=", out$priors$alpha[[1]], "\n", sep="")
        cat("estimate_alpha=", out$options$estimate_alpha, "\n", sep="")
        cat("weights=", out$options$weights_type, "\n", sep="")
        cat("parallel_init=", out$options$parallel_init, "\n", sep="")
        """
    )
    init_defaults = _topica_defaults("KeyATM", "__init__")
    fit_defaults = _topica_defaults("KeyATM", "fit")

    assert init_defaults["beta"] == float(r["beta"])
    assert init_defaults["beta_keyword"] == float(r["beta_keyword"])
    assert init_defaults["gamma1"] == float(r["gamma1"])
    assert init_defaults["gamma2"] == float(r["gamma2"])
    assert init_defaults["estimate_alpha"] is (r["estimate_alpha"] == "1")
    assert fit_defaults["weights"] == r["weights"]
    # num_threads defaults to 1 in the constructor (matching R keyATM's single-threaded
    # default when parallel_init=FALSE). In fit(), num_threads=None means "use the
    # constructor value", so the effective default is still 1.
    assert init_defaults["num_threads"] == (1 if r["parallel_init"] == "FALSE" else 4)
    assert fit_defaults["num_threads"] is None  # fit() overrides constructor; None = use constructor


def test_keyatm_alpha_default_matches_r_keyatm() -> None:
    # topica's KeyATM `alpha` defaults to None, which resolves to 1/num_topics,
    # matching R keyATM's base prior.
    r = _r_key_values(
        r"""
        if (!requireNamespace("keyATM", quietly=TRUE) ||
            !requireNamespace("quanteda", quietly=TRUE)) quit(status=2)
        suppressMessages(library(keyATM))
        suppressMessages(library(quanteda))
        texts <- c("tax market tax budget", "war troop iraq war",
                   "tax budget market", "iraq troop war")
        docs <- keyATM_read(texts = quanteda::dfm(quanteda::tokens(texts)))
        keywords <- list(econ=c("tax","market"), war=c("war","iraq"))
        out <- keyATM(docs = docs, model = "base", no_keyword_topics = 0,
                      keywords = keywords,
                      options = list(seed = 1, iterations = 1, verbose = FALSE))
        cat("alpha=", out$priors$alpha[[1]], "\n", sep="")
        cat("num_topics=", ncol(out$theta), "\n", sep="")
        """
    )
    init_defaults = _topica_defaults("KeyATM", "__init__")

    # R keyATM's base alpha is 1/K; topica defaults to None -> 1/K.
    assert float(r["alpha"]) == 1.0 / int(r["num_topics"])
    assert init_defaults["alpha"] is None


def test_mallet_defaults_match_topica_for_shared_lda_controls() -> None:
    mallet = _mallet_help_defaults()
    init_defaults = _topica_defaults("LDA", "__init__")
    fit_defaults = _topica_defaults("LDA", "fit")

    assert init_defaults["beta"] == float(mallet["beta"])
    assert init_defaults["burn_in"] == int(mallet["optimize-burn-in"])
    assert init_defaults["num_threads"] == int(mallet["num-threads"])
    assert init_defaults["use_symmetric_alpha"] is (mallet["use-symmetric-alpha"] == "true")
    assert fit_defaults["iters"] == int(mallet["num-iterations"])


def test_mallet_default_mismatches_are_explicit_contracts() -> None:
    """These are intentional topica defaults, not Java MALLET defaults."""

    mallet = _mallet_help_defaults()
    init_defaults = _topica_defaults("LDA", "__init__")

    # Java MALLET does not optimize hyperparameters unless asked; topica does.
    assert int(mallet["optimize-interval"]) == 0
    assert init_defaults["optimize_interval"] == 50

    # Java MALLET's seed=0 means clock seed. topica's default is reproducible.
    assert int(mallet["random-seed"]) == 0
    assert init_defaults["seed"] == 13

    # Java MALLET has a fixed alpha sum default. topica defaults to K, so it is
    # only equal to MALLET when the caller chooses K equal to MALLET's alpha sum.
    assert float(mallet["alpha"]) == 5.0
    assert init_defaults["alpha_sum"] is None
