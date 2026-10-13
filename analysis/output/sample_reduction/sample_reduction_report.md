# Person 4: Sample-reduction assessment (independent simulations)

## Scope and method

The reference three reaction conditions are frozen in `output/sample_reduction/reference_matched_A_18.json`, copied from the original 18-sample `analysis/artifacts/selected_campaign.json`. This comparison remains reproducible even if the official campaign is revised later.
We optimize only a subset of the existing six sample times per run. E2 and E3 always share the same schedule;
each run retains >=1 early sample (<=10 min) and >=1 later sample (>=40 min).
The 12-, 9-, and 6-sample designs are found by exhaustive search among those permitted subsets;
the objective is fractional model uncertainty eliminated under noise x3/parameter spread x2,
plus 0.15 times the worst leave-one-run-out fractional uncertainty eliminated.
Selection uses 150 generated datasets per model; **all reported statistics** are computed with a different
random seed and 1200 independent simulated datasets per model/scenario.
All species A/B/P/Q are assayed at every selected time; Q uses left-censoring at LOQ = 3e-5 M.
A shared bootstrap-parameter index is used across all experiment times and runs in each simulated world;
bootstrap mixture likelihoods also preserve that parameter dependence.
AIC weights are normalized over M2/M4 and are **not Bayesian posterior probabilities**.
Noise x1/x2/x3 and bootstrap-spread x1/x2 are sensitivity assumptions, not real observed performance.

## Results on independent evaluation worlds

| Samples | Worst information removed | Worst classification | Worst lost-run classification | Nominal information removed |
|---:|---:|---:|---:|---:|
| 18 | 88.5% | 97.0% | 86.9% | 100.0% |
| 12 | 88.4% | 97.0% | 87.0% | 100.0% |
| 9 | 86.2% | 96.5% | 85.8% | 100.0% |
| 6 | 76.2% | 93.4% | 81.1% | 100.0% |

Each row takes its minimum across six stated scenarios independently.
Tiny reversals (e.g., 12-sample score slightly above 18) arise from finite Monte Carlo error; they do not demonstrate that discarding data improves theoretical expected information.
A dropped-run score is only a robustness diagnostic: two-run campaigns would **not** meet the 3-run submission minimum.

## Sample schedules

**18 total:** E1 `[2.0, 5.0, 10.0, 20.0, 40.0, 60.0]`, E2/E3 `[2.0, 5.0, 10.0, 40.0, 60.0, 180.0]`.
**12 total:** E1 `[10.0, 20.0, 40.0, 60.0]`, E2/E3 `[10.0, 40.0, 60.0, 180.0]`.
**9 total:** E1 `[10.0, 40.0, 60.0]`, E2/E3 `[10.0, 60.0, 180.0]`.
**6 total:** E1 `[10.0, 60.0]`, E2/E3 `[5.0, 180.0]`.

## Interpretation / limitations

- Near-one-bit information at nominal noise does not prove experimental success; discrimination is **conditional on M2/M4 and the bootstrap ensemble**.
- Material usage remains **1.50 g A** in every candidate; all schedules satisfy temperature, concentration, duration, assay and total-sample constraints.
- Comparing nominal EIG alone will overvalue aggressive sample removal because model entropy quickly saturates near one bit.
- High-noise and widened-parameter scenarios are meaningful stress tests; all are recognised-noise cases and do **not** cover systematic bias or model misspecification.
- E2/E3 pairing supports a Q/P charge-dependence check. The P and Q measurements are simultaneous assays within each sample.
- More samples also improve time-kinetics checking and ability to detect wrong models; these benefits are not fully captured by the two-model information metric.
- For safety, neither the `submission/campaign.yaml` nor `analysis/artifacts/selected_campaign.json` has been changed.
- The alternative YAMLs are ready-to-inspect candidates, **not approved replacements**. To select one, update `analysis/artifacts/selected_campaign.json` then run `just suggest`.
- Samples are Monte Carlo simulations; results are reproducible using `python analysis/review_sample_efficiency.py` with output seed set as documented. Parameter sets clipped in spread x2: [0, 0].

## Decision principle

Prefer the smallest sample count that preserves acceptable stressed classification and worst lost-run performance, **not** just nominal EIG. Keep the official 18-sample campaign if that trade-off is not explicitly accepted by the team.
