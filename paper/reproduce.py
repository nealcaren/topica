"""One-command reproduction of every number in the topica paper.

Runs the three empirical parts of the paper end to end and writes a single
consolidated report that maps each paper claim to its freshly computed value:

  - Section 5  Validation against reference implementations (parity/*)
  - Section 6  Performance benchmarks (benchmarks/*)   [--no-benchmarks to skip]
  - Section 7  Worked example (paper/replication.py --quick)

Each step runs as an isolated subprocess with its own timeout and unbuffered
output, so one stalled reference toolchain cannot hang the whole run silently
(it is marked TIMEOUT and the rest proceeds). Steps whose toolchain is absent
(Rscript+stm/keyATM, the mallet CLI) skip themselves cleanly and are reported as
SKIP rather than failure -- so a contributor without those tools still gets a
useful run. For the archival run that backs the paper's numbers, pass --strict:
then SKIP and TIMEOUT also count as failures (every toolchain must be present
and every step must actually reproduce), so a green --strict run is the evidence
the manuscript's Sections 5-6 cite.

Usage (from the repo root):

    VIRTUAL_ENV="$PWD/.venv-dev" .venv-dev/bin/python paper/reproduce.py
    ... --no-benchmarks    # Sections 5 + 7 only (deterministic, machine-independent)
    ... --only 6           # one section
    ... --stamp "2026-06-16"   # provenance date written into the report
    ... --strict           # SKIP/TIMEOUT are failures (archival provenance run)

Performance numbers (Section 6) are hardware-dependent. The report records the
machine so the absolute timings are interpretable; relative speedups are the
portable claim. Run on a quiet machine.

Output: paper/generated/replication_report.md  (+ per-step logs in
paper/generated/logs/, and the benchmark JSONs the scripts write under
benchmarks/).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "paper" / "generated"
LOGS = GEN / "logs"

# (key, section, argv, timeout_seconds, extra_env). Timeouts are generous;
# benchmarks at 5k docs against R/MALLET are minutes each. extra_env lets a step
# pin thread count (the §6 STM table reports both single- and multi-thread topica,
# so bench_stm runs twice).
STEPS = [
    ("worked_example", 7, ["paper/replication.py", "--quick"], 1200, None),
    ("stm_prevalence", 5, ["parity/stm_poliblog_compare.py"], 1800, None),
    ("stm_content", 5, ["parity/stm_content_r_compare.py"], 900, None),
    ("keyatm", 5, ["parity/keyatm_r_compare.py"], 1800, None),
    ("sts", 5, ["parity/sts_r_compare.py"], 1800, None),
    ("mallet_lda", 5, ["parity/mallet_parity.py"], 1800, None),
    ("bench", 6, ["benchmarks/bench.py"], 3600, None),  # fig_thread_scaling, fig_memory
    ("bench_stm_st", 6, ["benchmarks/bench_stm.py"], 1800, {"RAYON_NUM_THREADS": "1"}),  # STM table, single-thread column
    ("bench_stm_mt", 6, ["benchmarks/bench_stm.py"], 1800, None),  # STM table, all-cores column
    ("speed_vs_r", 6, ["benchmarks/speed_vs_r.py"], 3600, None),
    ("speed_vs_size", 6, ["benchmarks/speed_vs_size.py"], 3600, None),
    ("bench_scaling", 6, ["benchmarks/bench_scaling.py"], 3600, None),  # fig_scaling
    ("k_crossover", 6, ["benchmarks/k_crossover.py"], 3600, None),
]

# Benchmark JSONs the Section-6 scripts emit, surfaced verbatim in the report.
BENCH_JSON = [
    "bench_results.json",
    "speed_vs_size.json",
    "scaling_results.json",
    "k_crossover_results.json",
    "matrix_results.json",
]

# Paper claims to check against the fresh run. Each points at the step/log or
# JSON that now carries the authoritative value. (Driver+report mode: we surface
# the numbers; the .tex is updated by hand from this checklist.)
CLAIMS = [
    ("5", "STM content (SAGE) per-group cosine = 1.000 (de, en)", "log: stm_content"),
    ("5", "STM prevalence Poliblog aligned cosine vs R", "log: stm_prevalence"),
    ("5", "keyATM agreement with R keyATM", "log: keyatm"),
    ("5", "STS benchmarks vs the sts package", "log: sts"),
    ("5", "LDA vs Java MALLET (cosine / Jaccard)", "log: mallet_lda"),
    ("6", "STM 3-22x faster than R stm (single / multi-thread)", "json: bench_results.json + speed_vs_r"),
    ("6", "LDA at parity with MALLET; multithread speedup grows with size", "json: bench_results.json, speed_vs_size.json"),
    ("6", "keyATM ~2x multithreaded vs R keyATM", "json: bench_results.json"),
    ("6", "K-scaling / crossover curves", "json: scaling_results.json, k_crossover_results.json"),
    ("7", "Spanning comparison: per-model coherence/exclusivity on one corpus", "log: worked_example"),
    ("7", "STM Poliblog covariate-effect z-values (fig_poliblog_effect)", "log: worked_example"),
]


def toolchains() -> dict:
    def r_pkg(pkg):
        if not shutil.which("Rscript"):
            return False
        try:
            out = subprocess.run(
                ["Rscript", "-e", f'cat(requireNamespace("{pkg}",quietly=TRUE))'],
                capture_output=True, text=True, timeout=60,
            )
            return out.stdout.strip().upper() == "TRUE"
        except Exception:
            return False

    return {
        "Rscript": bool(shutil.which("Rscript")),
        "R:stm": r_pkg("stm"),
        "R:keyATM": r_pkg("keyATM"),
        "java": bool(shutil.which("java")),
        "mallet": bool(shutil.which("mallet")),
    }


def run_step(key, argv, timeout, extra_env=None):
    log = LOGS / f"{key}.log"
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.setdefault("VIRTUAL_ENV", str(ROOT / ".venv-dev"))
    if extra_env:
        env.update(extra_env)
    t0 = time.time()
    try:
        with open(log, "w", encoding="utf-8") as fh:
            proc = subprocess.run(
                [sys.executable, *argv], cwd=ROOT, env=env,
                stdout=fh, stderr=subprocess.STDOUT, timeout=timeout,
            )
        dt = time.time() - t0
        text = log.read_text(encoding="utf-8", errors="replace")
        if "SKIP" in text and proc.returncode == 0:
            status = "SKIP"
        elif proc.returncode == 0:
            status = "OK"
        else:
            status = f"FAIL({proc.returncode})"
        return status, dt, text
    except subprocess.TimeoutExpired:
        return "TIMEOUT", time.time() - t0, log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""


def topica_version():
    try:
        import topica  # noqa
        return topica.__version__
    except Exception:
        return "unknown"


def tail(text, n=40):
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[-n:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-benchmarks", action="store_true", help="skip Section 6")
    ap.add_argument("--only", type=int, choices=[5, 6, 7], help="run one section")
    ap.add_argument("--stamp", default="", help="provenance date for the report")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="treat SKIP and TIMEOUT as failures too (for an archival "
             "full-provenance run where every toolchain must be present)",
    )
    args = ap.parse_args()

    GEN.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    steps = STEPS
    if args.only:
        steps = [s for s in steps if s[1] == args.only]
    if args.no_benchmarks:
        steps = [s for s in steps if s[1] != 6]

    tc = toolchains()
    print(f"topica {topica_version()} | {platform.platform()} | {os.cpu_count()} cores")
    print("toolchains:", ", ".join(f"{k}={'yes' if v else 'NO'}" for k, v in tc.items()))
    print(f"running {len(steps)} step(s)\n")

    results = []
    for key, section, argv, timeout, extra_env in steps:
        print(f"[§{section}] {key} ... ", end="", flush=True)
        status, dt, text = run_step(key, argv, timeout, extra_env)
        print(f"{status} ({dt:.0f}s)")
        results.append((key, section, status, dt, text))

    # --- write the consolidated report ---
    lines = ["# topica paper: reproduction report", ""]
    if args.stamp:
        lines.append(f"- Date: {args.stamp}")
    lines += [
        f"- topica: {topica_version()}",
        f"- Machine: {platform.platform()}, {os.cpu_count()} cores "
        f"({platform.processor() or 'cpu n/a'})",
        f"- Toolchains: " + ", ".join(f"{k}={'yes' if v else 'NO'}" for k, v in tc.items()),
        "",
        "> Section 6 (performance) is hardware-dependent; absolute timings reflect "
        "the machine above. Relative speedups are the portable claim.",
        "",
        "## Step status",
        "",
        "| Step | Section | Status | Time |",
        "|---|---|---|---|",
    ]
    for key, section, status, dt, _ in results:
        lines.append(f"| {key} | {section} | {status} | {dt:.0f}s |")

    lines += ["", "## Paper claims to update", "",
              "| § | Claim | Fresh value in |", "|---|---|---|"]
    for sec, claim, where in CLAIMS:
        lines.append(f"| {sec} | {claim} | {where} |")

    # Benchmark JSONs verbatim
    lines += ["", "## Benchmark outputs (Section 6)", ""]
    for name in BENCH_JSON:
        p = ROOT / "benchmarks" / name
        if p.exists():
            lines += [f"### {name}", "", "```json",
                      p.read_text(encoding="utf-8").strip(), "```", ""]

    # Per-step captured tails (the numbers)
    lines += ["## Step output (tails)", ""]
    for key, section, status, dt, text in results:
        lines += [f"### §{section} {key} — {status}", "", "```",
                  tail(text) or "(no output)", "```", ""]

    report = GEN / "replication_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {report.relative_to(ROOT)}")

    # Default: nonzero exit only on hard FAIL (SKIP/TIMEOUT are tolerated, since
    # a contributor may not have R/MALLET installed). With --strict, a SKIP or
    # TIMEOUT is also a failure — use it for the archival run that backs the
    # paper's numbers, where every toolchain must be present and every step must
    # actually reproduce.
    def _bad(status):
        if status.startswith("FAIL"):
            return True
        if args.strict and status in ("SKIP", "TIMEOUT"):
            return True
        return False

    bad = [r for r in results if _bad(r[2])]
    if bad:
        kinds = ", ".join(sorted({r[2].split("(")[0] for r in bad}))
        print(f"\n{len(bad)} step(s) not reproduced ({kinds})"
              + (" — strict mode" if args.strict else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
