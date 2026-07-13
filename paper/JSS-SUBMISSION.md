# JSS submission checklist

Generate the submission archive with:

    bash paper/make_jss_submission.sh

It includes:

- the JSS-formatted manuscript PDF and LaTeX sources;
- the full package source at the committed revision;
- the standalone reproduction driver;
- the generated replication report and captured step logs; and
- the JSS class, bibliography style, logo, figures, and generated validation
  appendix required to compile the manuscript.

Before creating the archive, run the relevant reproduction command and inspect
its report:

    VIRTUAL_ENV="$PWD/.venv-dev" .venv-dev/bin/python paper/reproduce.py --strict

The performance section is hardware-dependent. Its report records the machine
and toolchain versions; all other manuscript results are reproduced through the
same driver with fixed seeds. The archive is checked against JSS's 50 MB upload
limit.
