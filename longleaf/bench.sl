#!/bin/bash
#SBATCH -J topica_bench
#SBATCH -n 1
#SBATCH --cpus-per-task=44          # near-whole-node; with --mem below it starves
#SBATCH --mem=340g                  # co-tenants -> near-exclusive but normal priority.
#SBATCH -t 12:00:00                 # NOTE: do NOT use --mem=0 (it silently caps at
#SBATCH -p general                  # 1G here -> OOM). --exclusive is cleanest but can
#SBATCH -o /work/users/n/c/ncaren/topica_bench_%j.out   # queue for days; 44c/340g
#SBATCH -e /work/users/n/c/ncaren/topica_bench_%j.err   # schedules in normal priority.
#
# Section 6 (machine-dependent) benchmarks on a documented node, repeated for
# variance. Section 5 parity + Section 7 are machine-INDEPENDENT and run on the
# laptop (paper/reproduce.py --no-benchmarks). Requires the env from
# longleaf/setup_env.sh (topica built + R stm/keyATM + matplotlib).
#
# Smoke-test before batch (lesson: never batch an untested script):
#   srun -p general --cpus-per-task=8 --mem=32g -t 0:30:00 --pty bash
#   conda activate $WORK/envs/topica-bench
#   python paper/reproduce.py --only 6 --stamp smoke   # one quick pass
set -euo pipefail

WORK=/work/users/n/c/ncaren
REPO=$WORK/topica

module purge
module load anaconda/2024.02
module load r/4.5.0
eval "$(conda shell.bash hook)"
conda activate $WORK/envs/topica-bench
export PYTHONNOUSERSITE=1
export TMPDIR=$WORK/tmp; mkdir -p "$TMPDIR"   # benchmarks honor $TMPDIR (no hardcoded path)

cd "$REPO"
echo "=== provenance ==="
hostname
lscpu | grep -E "Model name|^CPU\(s\)|Thread\(s\) per core|Socket\(s\)"
free -h | head -2
python -c "import numpy,sys; print('py', sys.version.split()[0]); numpy.show_config()" 2>/dev/null \
  | grep -iE "openblas|mkl|blas|lapack" | head
Rscript -e 'cat(R.version.string, "| stm", as.character(packageVersion("stm")),
  "| keyATM", as.character(packageVersion("keyATM")), "\n")' 2>/dev/null || true
echo "=================="

STAMP="$(date +%F) longleaf $(hostname) topica-$(python -c 'import topica;print(topica.__version__)')"
# Repeat the §6 suite so the table reports a range, not a point.
for rep in 1 2 3; do
  echo "##### repeat $rep #####"
  python paper/reproduce.py --only 6 --stamp "$STAMP rep=$rep" || echo "rep $rep had non-OK steps"
  cp paper/generated/replication_report.md "$WORK/bench_rep${rep}.md" 2>/dev/null || true
  cp benchmarks/bench_results.json        "$WORK/bench_results_rep${rep}.json" 2>/dev/null || true
done
echo "BENCH_DONE"
