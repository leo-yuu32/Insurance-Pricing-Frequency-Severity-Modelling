# Phase 0 audit scripts

These scripts produced every figure in `docs/AUDIT.md`. They only read the repo: they import the
legacy pipeline in `src/`, run it unmodified, and analyse its output. Large intermediate files are
cached in the system temp folder (`%TEMP%\mtpl_audit_cache`), not in the repo.

Run from the repo root, in this order (PowerShell):

```powershell
$env:MPLBACKEND = "Agg"
python docs\audit_scripts\audit_run.py        # sections A to G: reproduce scalars.json, data counts, Gini, A/E, caps, Excel
python docs\audit_scripts\gini_sanity.py      # Gini with the largest test-set policy excluded
python docs\audit_scripts\notebook_check.py   # NumPy notebook update rule against statsmodels
```

`output/` holds the printed output from the run on 11 November 2026.
