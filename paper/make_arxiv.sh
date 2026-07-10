#!/usr/bin/env bash
# Build a self-contained arXiv submission tarball for the topica paper.
# topica.tex is already the [article,nojss] preprint (no masthead/logo), so this
# just:
#   1. Generates the .bbl and ships it, so arXiv does not need to run bibtex.
#   2. Bundles jss.cls and jss.bst (arXiv's TeXLive has the jss package, but
#      bundling guarantees the build) and the worked-example figure.
#   3. Compiles the assembled submission in isolation to prove it is
#      self-contained, then tars it.
#
# Prereq: generate the figures first --
#   python paper/replication.py --quick   # fig_poliblog_effect.pdf, fig_poliblog_report.pdf
#   python benchmarks/bench.py            # fig_thread_scaling.pdf, fig_memory.pdf (needs R/MALLET; quiet machine)
#   python benchmarks/bench_scaling.py    # fig_scaling.pdf (K-scaling memory/speed; quiet machine)
# Usage:  bash paper/make_arxiv.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/arxiv-submission.tar.gz"
STAGE="$(mktemp -d /private/tmp/topica-arxiv.XXXXXX)"
trap 'rm -rf "$STAGE"' EXIT

# --- locate jss.cls / jss.bst (kpsewhich, else R's bundled texmf) -----------
find_tex() {
  # Prefer a copy bundled in paper/ (the README tells users to put jss.cls here).
  [ -f "$HERE/$1" ] && { echo "$HERE/$1"; return; }
  local f; f="$(kpsewhich "$1" 2>/dev/null || true)"
  [ -n "$f" ] || f="$(find /Library/Frameworks/R.framework /usr/local/texlive \
                        -name "$1" 2>/dev/null | head -1)"
  [ -n "$f" ] || { echo "ERROR: $1 not found (install the jss package)"; exit 1; }
  echo "$f"
}
JSS_CLS="$(find_tex jss.cls)"
JSS_BST="$(find_tex jss.bst)"

for fig in fig_poliblog_effect.pdf fig_poliblog_report.pdf fig_scaling.pdf fig_thread_scaling.pdf; do
  [ -f "$HERE/$fig" ] || {
    echo "ERROR: $fig missing. Generate it (replication.py --quick / benchmarks/bench.py)"; exit 1; }
done
# The side-by-side validation appendix now lives in the supplement
# (supplementary.tex), which \input-s generated/validation_appendix.tex (committed).
APP="$HERE/generated/validation_appendix.tex"
[ -f "$APP" ] || {
  echo "ERROR: $APP missing. Generate it (python paper/gen_validation_appendix.py)"; exit 1; }

# --- build the .bbl ---------------------------------------------------------
BUILD="$STAGE/build"; mkdir -p "$BUILD/generated"
cp "$HERE/topica.tex" "$HERE/supplementary.tex" "$HERE/topica.bib" "$JSS_CLS" "$JSS_BST" \
   "$HERE/fig_poliblog_effect.pdf" "$HERE/fig_poliblog_report.pdf" "$HERE/fig_scaling.pdf" \
   "$HERE/fig_thread_scaling.pdf" "$BUILD/"
cp "$APP" "$BUILD/generated/"
( cd "$BUILD"
  export TEXINPUTS=".:" BSTINPUTS=".:" BIBINPUTS=".:"
  pdflatex -interaction=nonstopmode topica.tex >build.log 2>&1
  bibtex topica >>build.log 2>&1
  pdflatex -interaction=nonstopmode topica.tex >>build.log 2>&1
  pdflatex -interaction=nonstopmode topica.tex >>build.log 2>&1
  # The supplement has no citations, so it just needs two pdflatex passes.
  pdflatex -interaction=nonstopmode supplementary.tex >>build.log 2>&1
  pdflatex -interaction=nonstopmode supplementary.tex >>build.log 2>&1 ) || {
    echo "ERROR: .bbl build failed; tail of $BUILD/build.log:"; tail -30 "$BUILD/build.log"; exit 1; }

# --- assemble the submission (tex + bbl + class/style + figure) -------------
SUB="$STAGE/submission"; mkdir -p "$SUB/generated"
cp "$BUILD/topica.tex" "$BUILD/topica.bbl" "$HERE/supplementary.tex" "$JSS_CLS" "$JSS_BST" \
   "$HERE/fig_poliblog_effect.pdf" "$HERE/fig_poliblog_report.pdf" "$HERE/fig_scaling.pdf" \
   "$HERE/fig_thread_scaling.pdf" "$SUB/"
cp "$APP" "$SUB/generated/"

# --- prove it compiles in isolation (no .bib, bibtex not run) ---------------
( cd "$SUB"
  export TEXINPUTS=".:" BSTINPUTS=".:"
  pdflatex -interaction=nonstopmode topica.tex >/dev/null
  pdflatex -interaction=nonstopmode topica.tex > /tmp/arxiv_compile.log 2>&1
  pdflatex -interaction=nonstopmode supplementary.tex >> /tmp/arxiv_compile.log 2>&1
  pdflatex -interaction=nonstopmode supplementary.tex >> /tmp/arxiv_compile.log 2>&1 )
if grep -qiE "Citation .* undefined|LaTeX Error|Undefined control" /tmp/arxiv_compile.log; then
  echo "ERROR: isolated compile had problems; see /tmp/arxiv_compile.log"; exit 1
fi

tar czf "$OUT" -C "$SUB" topica.tex topica.bbl supplementary.tex jss.cls jss.bst \
  fig_poliblog_effect.pdf fig_poliblog_report.pdf fig_scaling.pdf fig_thread_scaling.pdf \
  generated/validation_appendix.tex
echo "wrote $OUT"
echo "contents:"; tar tzf "$OUT" | sed 's/^/  /'
echo "pages: $(pdfinfo "$SUB/topica.pdf" 2>/dev/null | awk '/Pages/{print $2}')"
