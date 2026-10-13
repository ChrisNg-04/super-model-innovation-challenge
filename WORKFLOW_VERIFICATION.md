# Final combined workflow verification

Verified on 12 October 2026 on Windows, using Python 3.12 and the supplied pinned dependencies. Installation used an existing isolated verification environment and local package caches; these timings are not a clean-machine speed guarantee.

| Required command | Result | Seconds |
|---|---|---:|
| just install | Passed | 12.35 |
| just predict | Passed | 13.72 |
| just suggest | Passed | 0.42 |
| just replicate | Passed | 113.29 |

Additional checks passed: all Python sources compile; four supplied candidate tests; all-six-model agreement between the direct fitting solver and the supplied simulator; censored likelihood; unknown-run exclusion rejection; 384 finite predictions and ordered nonnegative interval endpoints; campaign constraints; all ten training-only fold likelihoods reproduce; complete 100-per-model/fold saved bootstrap samples; regenerated coverage matches the incoming testing report.

The M6 expanded-bound review used six starts: three converged, one within 0.05 NLL of the best, Ea3 at its lower bound. Its improved AIC remains far worse than M2/M4. This is documented numerical sensitivity, not a claim of global optimality.

Final outputs: 384 prediction rows, 14 regenerated figures, three campaign experiments with 18 samples consuming 1.500 g A. The original M2/M4 fits, AIC weights and saved bootstrap distributions are retained.

The packaged ZIP was extracted into a different folder and its prediction workflow passed there, generating all 384 rows in 12.5 seconds. ZIP integrity and every packaged file checksum were also verified.

## Windows terminal fix — 13 October 2026

The entry point now detects native-library folders from the active Windows interpreter and registers/prepends them before importing scientific libraries or launching child processes. This fixes the native process failure encountered with unactivated Spyder Python; it contains no machine-specific path. The complete workflow passed from the extracted project using Spyder Python 3.12.11 in 122.4 seconds, producing 384 predictions, the 1.500 g/18-sample campaign and all 14 figures.

## Fresh-environment contract check — 13 October 2026

Created a new .judge-test-env and ran all four Just recipes in sequence from the extracted project. The recipes selected that environment through PATH; no PowerShell Python alias was required by their child shells. Dependency downloads used local caches where available.

| Command | Result | Seconds |
|---|---|---:|
| just install | Passed | 175.29 |
| just predict | Passed | 44.20 |
| just suggest | Passed | 0.61 |
| just replicate | Passed | 103.60 |

Total: 323.70 seconds (under ten minutes on this machine). Generated 384 predictions, the 1.500 g/18-sample campaign and 14 figures. The local test environment is ignored by Git and excluded from the shareable ZIP.
