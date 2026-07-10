"""Run the full synthetic-recovery benchmark suite.

Executes each score_*.py in turn as a subprocess and prints its output under a
per-corpus header. The frozen corpora + answer keys live in data/; the scorers
fit topica models and check recovery against the planted ground truth.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# (header, script) in the order they should run
SUITE = [
    ("ADMIXTURE (multi-axis recovery scorecard)", "score_admixture.py"),
    ("THETA (doc-topic mixture recovery)",         "score_theta.py"),
    ("MIXEDNESS (admixture/mixture crossover)",    "score_mixedness.py"),
    ("K-SWEEP (misspecified-K robustness)",        "score_ksweep.py"),
    ("NARRATIVE (trajectory order recovery)",      "score_narrative.py"),
    ("HIERARCHICAL (HLDA tree recovery)",          "score_hierarchical.py"),
]


def main():
    for header, script in SUITE:
        print("\n" + "#" * 70)
        print(f"# {header}")
        print(f"# {script}")
        print("#" * 70, flush=True)
        subprocess.run([sys.executable, os.path.join(HERE, script)], check=False)


if __name__ == "__main__":
    main()
