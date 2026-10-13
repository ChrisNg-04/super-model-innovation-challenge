# Proposed confirmatory campaign

This is a proposed single-block campaign, not a completed experiment.

## Strategy

Finite-grid, uncertainty-aware discrimination of M2 vs M4. Greedy joint expected-information gain per gram preserves shared parameter uncertainty across experiments. Selected schedule strategy: uncertainty_short_times. Shorter, logarithmically spread sampling was compared with long plateau sampling. All experiments are committed together; no observations are fed back during this event.

M2 impurity rate is k3*I; M4 impurity rate is k3*I*A. Changing initial A and B tests this distinction.
Screened 168 conditions with 20 bootstrap fits/model; shortlisted designs use all 60 fits/model.
Independent evaluation uses 600 simulated campaign outcomes, weighted by the current model weights.
Each parameter sample is shared across all observations in a simulated campaign.

## Proposed experiments

| Run | Temperature C | A0 M | B0 M | Sampling times min | A cost g |
|---|---:|---:|---:|---|---:|
| E1 | 95 | 0.1 | 1.5 | 2, 5, 10, 20, 40, 60 | 0.125 |
| E2 | 75 | 0.1 | 1.5 | 2, 5, 10, 40, 60, 180 | 0.125 |
| E3 | 85 | 0.1 | 0.75 | 2, 5, 10, 40, 60, 180 | 0.125 |

Budget: 3 experiments; 0.375 g A; 18 samples. All four observable species assayed.

## Explored alternatives

Temperature-only: repeats the historical charge at 65/75/85 C.
Point-estimate design: maximises best-fit separation per gram while ignoring parameter uncertainty.
Uncertainty-aware design: greedy joint information per gram, at least three experiments; stops when information saturates.
Short-time variant: keeps the selected conditions but spreads samples logarithmically up to roughly 95% product/impurity formation.
Final schedule: uncertainty_short_times; selected after checking retained information in fresh simulated campaigns.

| Strategy | Experiments | Samples | A g | Information nats | Fraction uncertainty removed | Simulated classification |
|---|---:|---:|---:|---:|---:|---:|
| temperature_only | 3 | 18 | 3.750 | 0.0230 | 0.033 | 0.566 |
| point_estimates | 3 | 18 | 0.938 | 0.6929 | 1.000 | 1.000 |
| uncertainty_aware | 3 | 18 | 0.375 | 0.6929 | 1.000 | 1.000 |
| uncertainty_short_times | 3 | 18 | 0.375 | 0.6925 | 1.000 | 1.000 |

## Limitations and possible failures

- Conditional on M2/M4 and fixed AIC weights; other mechanisms and model-weight uncertainty are omitted.
- Saved bootstrap samples and their extrapolation are not yet empirically calibrated.
- Finite grid and greedy selection do not guarantee the globally optimal campaign.
- Time selection uses a screening heuristic with minimum time separation, not exhaustive joint time optimisation.
- Bootstrap parameter uncertainty is shared across campaign samples; assay errors are assumed independent.
- Near-duplicate experiments may improve discrimination under independent assay noise but do not test batch/systematic errors.
- Simulated information and classification are planning metrics, not observed experimental performance.
- Budget differences between strategies must be reported; they are not necessarily equally expensive campaigns.
- The follow-up assay is assumed to have the supplied Q noise and quantitation limit at every selected sample.

Do not equate these simulated discrimination probabilities with guarantees about real experimental results.
A low predicted Q concentration may be reported below LOQ; the information calculation includes that event.
No campaign observations are used to alter the held-out predictions or other current deliverables.
