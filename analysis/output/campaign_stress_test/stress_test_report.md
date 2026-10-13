# Campaign stress-test and matched-control comparison

These are planning simulations, not experimental observations or guaranteed performance.

## Designs

Original: three low-A experiments, 0.375 g A, 18 samples.
Matched control: replaces E3 with 75 C, A0=1.0 M, B0=1.5 M, at the same sampling times as low-A E2.
This isolates the effect of A in the initial conditions. It does not hold reaction progress or I constant.
Matched lean: same conditions, but three samples per experiment rather than six.

M2 predicts a temperature-specific Q/P ratio independent of initial charge (with zero initial P/Q).
M4's impurity branch depends on the changing concentration of A, so its integrated Q/P ratio can change with charge.
Comparing Q/P helps distinguish branching from simply making more product at higher initial A.

## Stress assumptions

Six scenarios combine assay standard deviations x1/x2/x3 and parameter-spread factors x1/x2.
Broader parameter variation expands deviations in log k and Ea/100 while retaining sample correlations.
Parameter sets are clipped to the original bounds; clipping counts are reported in the CSV.
Noise increases are applied both to simulated outcomes and the analysis likelihood: these are recognised-noise scenarios.
For each design/scenario, 500 campaigns are simulated; parameter samples are shared across all experiments within a campaign.
Each experiment is also removed in turn to represent a lost/uninformative run. Those are diagnostics, not legal two-run proposals.

## Worst-case results across these scenarios

| Design | A g | Samples | Min uncertainty removed | Min simulated classification | Worst missing-run classification |
|---|---:|---:|---:|---:|---:|
| matched_A_control | 1.500 | 18 | 0.909 | 0.984 | 0.895 |
| matched_control_lean | 1.500 | 9 | 0.802 | 0.941 | 0.839 |
| original_low_A | 0.375 | 18 | 0.746 | 0.946 | 0.888 |

## Trade-offs and limitations

- A matched control gives a more direct, explainable test of concentration dependence, at higher material cost.
- The lean version uses fewer assays but may lose robustness when noise increases or one run is uninformative.
- These distributions are conditional on M2/M4 and their fixed AIC weights; other mechanisms are not represented.
- Broader parameter variation is a sensitivity test, not independent calibration of uncertainty.
- All historical experiments share one charge; predictions at these new stoichiometries are extrapolations.
- Systematic assay bias, batch effects, time-correlated noise and rig limitations are not included.
- A finite simulation with very high classification does not prove a real campaign will succeed.
- The Q/P plot shows parameter uncertainty only; assay uncertainty and censoring are included in the information calculation.

The alternative YAML files are for comparison. The original proposal has not been overwritten.
