#!/bin/bash
# Set up a DEDICATED conda env on Longleaf (UNC HPC) to reproduce topica's
# paper Section 6 benchmarks. Run once on a login node, then submit bench.sl.
#
# This script is the validated result of an actual cluster bring-up; the inline
# notes record the things that bit us so they do not bite again:
#   - build a FRESH env; never pip-install into a shared/hand-tuned one.
#   - Longleaf has NO `rust` module -> install via rustup (user space).
#   - R has no writable default library -> create R_LIBS_USER first, else
#     install.packages fails with "unable to install packages".
#   - the §6 benchmarks need R stm AND keyATM (+quanteda), plus matplotlib for
#     the figure steps -- all installed here.
#   - check out a ref that INCLUDES the temp-dir portability fix (main / a
#     release > 0.23.1); v0.23.1 itself hardcodes /private/tmp and crashes on
#     Linux.
set -euo pipefail

WORK=/work/users/n/c/ncaren                 # your /work scratch
ENV_PREFIX=$WORK/envs/topica-bench          # fresh, dedicated env
REPO=$WORK/topica
REF=${1:-main}                              # ref with the /private/tmp fix

module purge
module load anaconda/2024.02
eval "$(conda shell.bash hook)"

# 1. Clone (or update) the repo at REF.
if [ ! -d "$REPO/.git" ]; then git clone https://github.com/nealcaren/topica.git "$REPO"; fi
cd "$REPO"; git fetch --all --tags -q; git checkout -q "$REF"; git pull -q --ff-only 2>/dev/null || true
echo "repo at $(git describe --tags --always)"

# 2. Fresh Python env (everything else pinned deliberately).
[ -d "$ENV_PREFIX" ] || conda create -y -q -p "$ENV_PREFIX" python=3.11
conda activate "$ENV_PREFIX"
export PYTHONNOUSERSITE=1

# 3. Rust toolchain via rustup (no `rust` module on Longleaf).
if ! command -v cargo >/dev/null 2>&1; then
  [ -f "$HOME/.cargo/env" ] || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  source "$HOME/.cargo/env"
fi
cargo --version

# 4. Python deps (incl. matplotlib for the figure steps) + build topica.
pip install -q maturin numpy pandas scipy scikit-learn gensim tomotopy matplotlib
VIRTUAL_ENV="$ENV_PREFIX" maturin develop --release --features python

# 5. R reference packages (stm + keyATM/quanteda) into a writable personal lib.
module load r/4.5.0
Rscript -e '
  lib <- Sys.getenv("R_LIBS_USER"); dir.create(lib, recursive=TRUE, showWarnings=FALSE)
  pkgs <- c("stm","keyATM","quanteda")
  need <- pkgs[!sapply(pkgs, requireNamespace, quietly=TRUE)]
  if (length(need)) install.packages(need, lib=lib, repos="https://cloud.r-project.org")
  for (p in pkgs) cat(p, as.character(packageVersion(p)), "\n")
'

# 6. MALLET (Java) for the LDA-vs-MALLET §6 leg. The v202108 binary release
#    ships bin/mallet + dist/ jars + class/, so it serves both §6 (CLI on PATH)
#    and §5 parity (classpath). Java comes from a module at runtime (see bench.sl).
if [ ! -x "$WORK/mallet/bin/mallet" ]; then
  curl -fsSL https://github.com/mimno/Mallet/releases/download/v202108/Mallet-202108-bin.tar.gz \
    -o "$WORK/mallet.tgz"
  tar xzf "$WORK/mallet.tgz" -C "$WORK" && rm -f "$WORK/mallet.tgz"
  mv "$WORK"/Mallet-202108 "$WORK/mallet"
fi
echo "mallet: $WORK/mallet/bin/mallet"

# 7. Smoke-test (catches build/import/data issues before any batch job).
python -c "import topica,numpy,pandas,sklearn,gensim,tomotopy,matplotlib; print('topica', topica.__version__, '- env OK')"
echo "Setup done. Activate: conda activate $ENV_PREFIX ; then: sbatch longleaf/bench.sl"
