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

### The to-convergence STM headline (poliblog5k + Congress)

§6's STM speedup is reported as **wall-clock to convergence** (the number a user
waits for), with per-iteration cost as the mechanism. `reproduce.py --only 6` runs
`bench_stm_convergence.py` on poliblog5k automatically. The medium **Congress**
point (~25k speeches, `~party + s(congress)`) needs its corpus, which lives in the
separate ECTM project and is **not** on Longleaf — ship the prepped CSV from your
laptop first (it's small, ~40 MB):

```bash
# on the laptop (needs the ECTM congress_prepped.pkl):
python benchmarks/export_congress.py                       # -> benchmarks/congress_prepped.csv
scp benchmarks/congress_prepped.csv \
    longleaf:/work/users/n/c/ncaren/topica/benchmarks/
```

If the CSV is absent the Congress leg skips cleanly (the poliblog5k headline still
runs). Note R `stm` to convergence on 25k docs is the long pole — `bench.sl`'s
12 h walltime covers the ×3 repeats, but watch the first repeat's timing.

## What `setup_env.sh` installs
A fresh, dedicated `topica-bench` conda env (Python 3.11) with: rustup + `maturin`
build of topica from source; `numpy pandas scipy scikit-learn gensim tomotopy
matplotlib`; R `stm` + `keyATM` (+`quanteda`) in a personal library; and **MALLET**
(v202108, under `$WORK/mallet`) for the LDA-vs-MALLET comparison. It never touches
a shared env. (Java for MALLET is loaded as a module at runtime by `bench.sl`,
not installed into the env.)

## Hardware-setup lessons baked into these scripts
Running this cross-platform shook out several issues; the scripts now encode the fixes:

- **`--mem=0` is a trap on this cluster** — it silently caps at 1 GB (not "all node
  memory") → OOM. Request an explicit `--mem`.
- **But a *large* `--mem` is the opposite trap: it steers you onto the bad node.**
  Only the 3 TB 4-socket Xeon E7 boxes (`t0601`–`t0605`) have hundreds of GB free,
  so asking for `--mem=340g` pins you there (it cost two wasted runs). `bench.sl`
  asks for `--mem=96g` (enough for the K=200 scaling covariance) and
  `--exclude=t0601..t0605`, so it lands on a modern 2-socket EPYC node.
- **`--exclusive` queues for days** here, and a big core/mem ask queues for hours
  under normal fairshare. `bench.sl` asks for 32 cores + 96 GB on `general`, enough
  for the 16-core STM cap and 8-thread Gibbs scaling, which schedules far sooner.
- **No `rust` module** on Longleaf → rustup (user space).
- **R has no writable default library** → create `R_LIBS_USER` before
  `install.packages`, else it fails with "unable to install packages".
- **§6 needs R `stm` AND `keyATM`** (the `bench` and `speed_vs_size` steps include a
  keyATM-vs-R leg) plus **matplotlib** (figure steps) and **MALLET** (the
  LDA-vs-MALLET leg; without it that column is null but the step still runs). All
  installed/wired by setup + `bench.sl`.
- **Pick a sane node.** The `general` pool is heterogeneous; an unlucky draw can be
  a slow 4-socket Xeon E7 (NUMA, unrepresentative thread-scaling). Check the
  provenance header's CPU/socket line in the `.out`; if it's a 4-socket box,
  cancel and resubmit to re-roll for a 1–2-socket node.
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
