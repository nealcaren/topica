# Reproducing topica's paper §6 benchmarks on Longleaf

The paper's Section 6 timing/memory numbers are hardware-dependent, and a JSS
referee asked for a documented environment plus run-to-run variance instead of a
single laptop run. A named, exclusive-ish HPC node with reported specs and ≥3
repeats is the credible fix. (Section 5 parity and Section 7 are
machine-INDEPENDENT — a cosine is the same on any CPU — so those run on the
laptop via `paper/reproduce.py --no-benchmarks` and are not repeated here.)

## Quick start
```bash
# on a Longleaf login node:
bash longleaf/setup_env.sh           # one-time: clone, build, install deps (~20 min)
sbatch longleaf/bench.sl             # §6 ×3 for variance; writes $WORK/bench_rep*.md
```
Setup defaults to checking out `main` (it must include the temp-dir portability
fix; `v0.23.1` predates it). Pass a ref to override: `bash longleaf/setup_env.sh v0.24.0`.

## What `setup_env.sh` installs
A fresh, dedicated `topica-bench` conda env (Python 3.11) with: rustup + `maturin`
build of topica from source; `numpy pandas scipy scikit-learn gensim tomotopy
matplotlib`; and R `stm` + `keyATM` (+`quanteda`) in a personal library. It never
touches a shared env.

## Hardware-setup lessons baked into these scripts
Running this cross-platform shook out several issues; the scripts now encode the fixes:

- **`--mem=0` is a trap on this cluster** — it silently caps at 1 GB (not "all node
  memory") → OOM. `bench.sl` requests an explicit `--mem=340g`.
- **`--exclusive` is cleanest but queues for days** here. `bench.sl` instead asks for
  44 cores + 340 GB on `general`, which starves co-tenants (near-exclusive) while
  scheduling at normal priority.
- **No `rust` module** on Longleaf → rustup (user space).
- **R has no writable default library** → create `R_LIBS_USER` before
  `install.packages`, else it fails with "unable to install packages".
- **§6 needs R `stm` AND `keyATM`** (the `bench` and `speed_vs_size` steps include a
  keyATM-vs-R leg) plus **matplotlib** (figure steps). All installed by setup.
- **Benchmarks honor `$TMPDIR`** (the old hardcoded `/private/tmp` crashed on Linux;
  fixed in the benchmark scripts). `bench.sl` sets `$TMPDIR` to `$WORK/tmp`.

## Smoke-test before batch (do not skip)
```bash
srun -p general --cpus-per-task=8 --mem=32g -t 0:30:00 --pty bash
conda activate $WORK/envs/topica-bench && cd $WORK/topica
python paper/reproduce.py --only 6 --stamp smoke
```

## Output
`$WORK/bench_rep{1,2,3}.md` (+ `bench_results_rep*.json`), each stamped with the
node hostname, CPU model, core count, RAM, BLAS, and R/stm/keyATM versions from
the job's provenance header. Take the median + range into Section 6 and the
paper's environment line.
