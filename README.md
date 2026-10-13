# Polymodel final combined version

This folder combines the testing handoffs, sample-efficiency study and an expanded-bound M6 review (best converged estimate included; numerical sensitivity documented). The original incoming files are untouched.

## Start here

Keep this entire folder together when sharing it. In Spyder, open `submission_workflow.py` and press Run. It generates predictions, the selected campaign and presentation figures from saved fitted/bootstrap parameters. Expensive bootstrap training is not repeated.

From a terminal with the required Python environment:

```
just install
just predict
just suggest
just replicate
```

Without Just, install `requirements.txt` and the local package (`python -m pip install -e .`), then run `python submission_workflow.py all`.

## Final outputs

- `submission/predictions.csv`: 384 concentration predictions with 90% predictive intervals (parameter uncertainty, model mixture and assay noise).
- `submission/hypotheses.yaml`: six-model ranking, conditional AIC weights and reasons.
- `submission/campaign.yaml`: the authoritative campaign: three experiments, 18 samples, 1.500 g A.
- `submission/figures/`: regenerated presentation figures, including held-out interval coverage.

The campaign source is `analysis/artifacts/selected_campaign.json`. Exploratory campaigns under `analysis/output/` are alternatives, not competing submission files. The 12-sample option performed similarly in assumed simulations; the selected 18-sample design retains additional early measurements for checking unexpected kinetics.

## What has been validated

M2 and M4 remain effectively tied. Five leave-one-experiment-out tests fit four runs and predict the fifth. Each fold/model has 100 converged saved bootstrap fits, without parameter-bound flags. Quantified-observation interval coverage is A 88.5%, B 96.2%, P 90.4%, Q 83.3%; overall 90.8%. Q censoring is evaluated separately, never treated as zero. Model selection is conditional on the M2/M4 pair chosen earlier. These five correlated experiments do not establish calibration outside the historical charge/temperature range.

`analysis/check_integrity.py` checks the supplied candidate tests, all-six-model solver agreement, censored likelihood and exclusion handling. `analysis/validate_bootstrap_full.py` regenerates validation intervals using saved training-only bootstrap fits. Missing cached samples trigger expensive additional fitting; `just replicate` verifies their completeness first.

## Further development

`analysis/compare_system_B.py` performs new independent fits, using expanded M6 rate bounds and accepting only converged finite solutions. Results go to `analysis/output/refit_system_B/`, so a new fit cannot silently invalidate the saved prediction bootstraps. Rebuild bootstrap samples and validation before promoting materially changed fits. `analysis/review_m6.py` is the focused M6 sensitivity check. Rate bounds are modelling assumptions, not known chemical limits.

`DEVELOPMENT_STATUS.md` records remaining scientific limitations. `WORKFLOW_VERIFICATION.md` records checks performed on this combined version. `SHA256_MANIFEST.json` identifies the packaged files.
