"""Compare a low-A campaign with a matched-A control and stress assumptions.

All outcomes are simulated planning checks, not real observations. Increased
noise is known to the likelihood in those scenarios; unrecognised systematic
bias is not tested. Expanded parameter variation is a sensitivity assumption,
not a recalibrated posterior or bootstrap distribution.
"""

import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import log_ndtr, logsumexp, xlogy

import reaction_fitting as fitting

folder = Path(__file__).resolve().parent
saved = folder / "output" / "system_B"
proposal_file = folder / "output" / "campaign_design" / "proposed_campaign.yaml"
output = folder / "output" / "campaign_stress_test"
output.mkdir(parents=True, exist_ok=True)
proposal = json.loads(proposal_file.read_text())
original = proposal["experiments"]
if len(original) != 3:
    raise ValueError("This comparison expects the current three-experiment proposal.")
matched = copy.deepcopy(original)
# Replace E3 with a control matched to E2 in temperature, B and sampling times.
matched[2] = copy.deepcopy(original[1])
matched[2]["run_id"] = "E3_high_A_control"
matched[2]["A0_M"] = 1.0
lean = copy.deepcopy(matched)
for experiment in lean:
    times = experiment["sample_times_min"]
    experiment["sample_times_min"] = [times[0], times[len(times) // 2], times[-1]]
designs = {"original_low_A": original, "matched_A_control": matched, "matched_control_lean": lean}


def budget(experiments):
    material = sum(1.25 * experiment["A0_M"] for experiment in experiments)
    count = sum(len(experiment["sample_times_min"]) for experiment in experiments)
    if not (3 <= len(experiments) <= 10 and material <= 5 + 1e-9 and count <= 60):
        raise ValueError("Campaign exceeds budget.")
    for experiment in experiments:
        times = experiment["sample_times_min"]
        if not (50 <= experiment["temperature_C"] <= 120 and .1 <= experiment["A0_M"] <= 1.5
                and .1 <= experiment["B0_M"] <= 1.5 and all(0 <= t <= 960 for t in times)
                and times == sorted(set(times)) and set(experiment["assay"]) == set(fitting.SPECIES)):
            raise ValueError("Invalid experiment settings.")
    return material, count


for name, experiments in designs.items():
    material, count = budget(experiments)
    print(f"{name}: {material:.3f} g A; {count} samples", flush=True)
models = ["M2", "M4"]
bootstrap = json.loads((saved / "bootstrap_parameters.json").read_text())
ranking = pd.read_csv(saved / "model_comparison.csv").set_index("model")
weights = ranking.loc[models, "relative_AIC_weight"].to_numpy(float)
weights = weights / weights.sum()
parameter_count = len(bootstrap[models[0]])
if parameter_count < 54 or len(bootstrap[models[1]]) != parameter_count:
    raise ValueError("Incomplete bootstrap distributions.")
relative = np.array([.03, .03, .03, .15])
absolute = np.array([.0035, .0035, .0035, .000022])
entropy_before = -np.sum(xlogy(weights, weights))
fingerprint = hashlib.sha256(b"".join(path.read_bytes() for path in [
    Path(__file__), folder / "reaction_fitting.py", fitting.PROJECT / "src" / "candidates" / "models.py",
    saved / "bootstrap_parameters.json", saved / "model_comparison.csv", proposal_file
])).hexdigest()


def varied_parameters(model, factor):
    names = fitting.CANDIDATES[model].param_names
    matrix = np.array([[np.log(p[n]) if n.startswith("k") else p[n] / 100 for n in names]
                       for p in bootstrap[model]])
    varied = matrix.mean(axis=0) + factor * (matrix - matrix.mean(axis=0))
    lower = np.array([np.log(1e-10) if n.startswith("k") else 0 for n in names])
    upper = np.array([np.log(10) if n.startswith("k") else 2 for n in names])
    bounded = np.clip(varied, lower, upper)
    clipped = int(np.any(varied != bounded, axis=1).sum())
    return [fitting.unpack(model, row) for row in bounded], clipped


prediction_cache = {}


def predictions(experiment, factor):
    key = json.dumps({"temperature": experiment["temperature_C"], "A": experiment["A0_M"],
                      "B": experiment["B0_M"], "times": experiment["sample_times_min"], "factor": factor}, sort_keys=True)
    if key in prediction_cache:
        return prediction_cache[key]
    cache_file = output / ("curves_" + hashlib.sha256((fingerprint + key).encode()).hexdigest()[:20] + ".npz")
    if cache_file.exists():
        with np.load(cache_file) as stored:
            result = stored["means"]
            clipping = stored["clipping"].tolist()
    else:
        sample_times = np.array(experiment["sample_times_min"], float)
        times = np.unique(np.r_[0., sample_times])
        run = {"temperature": experiment["temperature_C"], "A0": experiment["A0_M"], "B0": experiment["B0_M"],
               "times": times, "ti": np.repeat(np.searchsorted(times, sample_times), 4),
               "si": np.tile(np.arange(4), len(sample_times))}
        arrays, clipping = [], []
        for model in models:
            parameters, clipped = varied_parameters(model, factor)
            arrays.append([fitting.predict_run(model, p, run) for p in parameters])
            clipping.append(clipped)
        result = np.asarray(arrays)
        np.savez_compressed(cache_file, means=result, clipping=np.asarray(clipping))
    prediction_cache[key] = (result, clipping)
    return result, clipping


def metrics(joint, truth, world_weights):
    log_scores = np.column_stack([
        logsumexp(joint[:, i * parameter_count:(i + 1) * parameter_count], axis=1)
        - np.log(parameter_count) + np.log(weights[i]) for i in range(2)])
    probabilities = np.exp(log_scores - logsumexp(log_scores, axis=1)[:, None])
    entropy = -np.sum(xlogy(probabilities, probabilities), axis=1)
    information = entropy_before - np.sum(world_weights * entropy)
    accuracy = np.sum(world_weights * (probabilities.argmax(axis=1) == truth))
    # Monte Carlo error only; does not describe uncertainty in model assumptions.
    half = len(truth) // 2
    standard_error = np.sqrt(sum(weights[i] ** 2 * entropy[i * half:(i + 1) * half].var(ddof=1) / half
                                 for i in range(2)))
    return float(information / entropy_before), float(accuracy), float(standard_error / entropy_before)


started = time.perf_counter()
records = []
worlds_per_model = 250
for variation in [1., 2.]:
    for noise_multiplier in [1., 2., 3.]:
        print(f"Scenario: parameter spread x{variation:g}, assay noise x{noise_multiplier:g}", flush=True)
        # Shared underlying parameter draws across the designs and experiments.
        rng_parameters = np.random.default_rng(419901)
        truth = np.repeat(np.arange(2), worlds_per_model)
        world_parameters = rng_parameters.integers(0, parameter_count, size=len(truth))
        world_weights = np.repeat(weights / worlds_per_model, worlds_per_model)
        for name, experiments in designs.items():
            contributions = []
            clipped_counts = [0, 0]
            for experiment in experiments:
                mean, clipping = predictions(experiment, variation)
                clipped_counts = [max(a, b) for a, b in zip(clipped_counts, clipping)]
                n_times = len(experiment["sample_times_min"])
                sigma = noise_multiplier * np.sqrt((mean * np.tile(relative, n_times)) ** 2
                                                  + np.tile(absolute, n_times) ** 2)
                true_mean = mean[truth, world_parameters]
                true_sigma = sigma[truth, world_parameters]
                # Reuse standard-normal noise for identical physical experiments in each design.
                physical_key = json.dumps({k: experiment[k] for k in ["temperature_C", "A0_M", "B0_M", "sample_times_min"]}, sort_keys=True)
                noise_seed = int(hashlib.sha256(physical_key.encode()).hexdigest()[:8], 16)
                rng = np.random.default_rng(noise_seed)
                observed = true_mean + true_sigma * rng.normal(size=true_mean.shape)
                q = np.tile(np.arange(4), n_times) == 3
                censored = (observed < 3e-5) & q
                component_mean = mean.reshape(2 * parameter_count, -1)
                component_sigma = sigma.reshape(2 * parameter_count, -1)
                z = (observed[:, None, :] - component_mean[None, :, :]) / component_sigma[None, :, :]
                loglike = -.5 * z ** 2 - np.log(component_sigma[None, :, :]) - .5 * np.log(2 * np.pi)
                cdf = log_ndtr((3e-5 - component_mean) / component_sigma)
                contributions.append(np.where(censored[:, None, :], cdf[None, :, :], loglike).sum(axis=2))
            joint = sum(contributions)
            information, accuracy, error = metrics(joint, truth, world_weights)
            dropout_information = []
            dropout_accuracy = []
            for missing in range(len(experiments)):
                remaining = sum(contribution for i, contribution in enumerate(contributions) if i != missing)
                removed_information, removed_accuracy, _ = metrics(remaining, truth, world_weights)
                dropout_information.append(removed_information)
                dropout_accuracy.append(removed_accuracy)
            material, count = budget(experiments)
            records.append({"design": name, "noise_multiplier": noise_multiplier, "parameter_spread_multiplier": variation,
                            "material_g": material, "samples": count, "fraction_uncertainty_removed": information,
                            "information_MC_standard_error": error, "simulated_classification": accuracy,
                            "worst_dropout_information": min(dropout_information),
                            "worst_dropout_classification": min(dropout_accuracy),
                            "clipped_parameter_sets_M2": clipped_counts[0], "clipped_parameter_sets_M4": clipped_counts[1]})
            pd.DataFrame(records).to_csv(output / "stress_comparison.csv", index=False)

results = pd.DataFrame(records)
summary = results.groupby("design").agg(material_g=("material_g", "first"), samples=("samples", "first"),
                                       minimum_information=("fraction_uncertainty_removed", "min"),
                                       minimum_classification=("simulated_classification", "min"),
                                       worst_dropout_classification=("worst_dropout_classification", "min"))
summary.to_csv(output / "worst_case_summary.csv")
print("\nWORST CASES ACROSS THE SIX ASSUMED SCENARIOS:")
print(summary.to_string(float_format=lambda value: f"{value:.4f}"))
print("These are conditional simulations, not real success rates or final interval-calibration results.")
print("Increased assay noise is assumed known; systematic biases and other mechanisms are not tested.")
print("The original proposal is preserved. The matched design is an alternative, not automatically the final campaign.")

for name in ["matched_A_control", "matched_control_lean"]:
    candidate = {"rationale": "Matched low/high-A comparison at fixed temperature, B and sampling times. "
                 "Alternative for team review; see stress-test report for assumptions and limitations.",
                 "experiments": designs[name]}
    (output / f"{name}.yaml").write_text(json.dumps(candidate, indent=2))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = {"original_low_A": "tab:blue", "matched_A_control": "tab:orange", "matched_control_lean": "tab:green"}
for ax, variation in zip(axes, [1., 2.]):
    for name in designs:
        rows = results[(results["design"] == name) & (results["parameter_spread_multiplier"] == variation)]
        ax.errorbar(rows["noise_multiplier"], rows["fraction_uncertainty_removed"],
                    yerr=2 * rows["information_MC_standard_error"], marker="o", label=name, color=colors[name])
    ax.set_title(f"Parameter spread x{variation:g}")
    ax.set_xlabel("Assay-noise multiplier")
    ax.set_ylabel("Expected fraction of M2/M4 uncertainty removed")
    ax.set_xticks([1, 2, 3])
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
fig.suptitle("Campaign sensitivity - error bars show simulation error only")
fig.tight_layout()
fig.savefig(output / "campaign_stress_comparison.png", dpi=180)

# Show the mechanistic contrast at the common 75 C, B0=1.5 M condition.
fig2, ax = plt.subplots(figsize=(8, 5))
for a, experiment in zip([.1, 1.], [matched[1], matched[2]]):
    curves, _ = predictions(experiment, 1.)
    for i, model in enumerate(models):
        p = curves[i, :, -2]
        q = curves[i, :, -1]
        ratio = q / np.maximum(p, 1e-12)
        low, middle, high = np.quantile(ratio, [.05, .5, .95])
        ax.errorbar(a + (i - .5) * .03, middle, yerr=[[middle - low], [high - middle]],
                    fmt="o", color=["tab:blue", "tab:orange"][i], label=model if a == .1 else None, capsize=4)
ax.set_xlabel("Starting A concentration (mol/L)")
ax.set_ylabel("Predicted molar Q/P ratio at 180 min")
ax.set_title("Matched control: 75 C, B0=1.5 M")
ax.legend()
ax.grid(alpha=.2)
fig2.tight_layout()
fig2.savefig(output / "matched_A_Q_to_P_ratio.png", dpi=180)

lines = ["# Campaign stress-test and matched-control comparison", "",
         "These are planning simulations, not experimental observations or guaranteed performance.", "",
         "## Designs", "", "Original: three low-A experiments, 0.375 g A, 18 samples.",
         "Matched control: replaces E3 with 75 C, A0=1.0 M, B0=1.5 M, at the same sampling times as low-A E2.",
         "This isolates the effect of A in the initial conditions. It does not hold reaction progress or I constant.",
         "Matched lean: same conditions, but three samples per experiment rather than six.", "",
         "M2 predicts a temperature-specific Q/P ratio independent of initial charge (with zero initial P/Q).",
         "M4's impurity branch depends on the changing concentration of A, so its integrated Q/P ratio can change with charge.",
         "Comparing Q/P helps distinguish branching from simply making more product at higher initial A.", "",
         "## Stress assumptions", "",
         "Six scenarios combine assay standard deviations x1/x2/x3 and parameter-spread factors x1/x2.",
         "Broader parameter variation expands deviations in log k and Ea/100 while retaining sample correlations.",
         "Parameter sets are clipped to the original bounds; clipping counts are reported in the CSV.",
         "Noise increases are applied both to simulated outcomes and the analysis likelihood: these are recognised-noise scenarios.",
         "For each design/scenario, 500 campaigns are simulated; parameter samples are shared across all experiments within a campaign.",
         "Each experiment is also removed in turn to represent a lost/uninformative run. Those are diagnostics, not legal two-run proposals.", "",
         "## Worst-case results across these scenarios", "",
         "| Design | A g | Samples | Min uncertainty removed | Min simulated classification | Worst missing-run classification |",
         "|---|---:|---:|---:|---:|---:|"]
for name, row in summary.iterrows():
    lines.append(f"| {name} | {row['material_g']:.3f} | {row['samples']:.0f} | {row['minimum_information']:.3f} | "
                 f"{row['minimum_classification']:.3f} | {row['worst_dropout_classification']:.3f} |")
lines.extend(["", "## Trade-offs and limitations", "",
              "- A matched control gives a more direct, explainable test of concentration dependence, at higher material cost.",
              "- The lean version uses fewer assays but may lose robustness when noise increases or one run is uninformative.",
              "- These distributions are conditional on M2/M4 and their fixed AIC weights; other mechanisms are not represented.",
              "- Broader parameter variation is a sensitivity test, not independent calibration of uncertainty.",
              "- All historical experiments share one charge; predictions at these new stoichiometries are extrapolations.",
              "- Systematic assay bias, batch effects, time-correlated noise and rig limitations are not included.",
              "- A finite simulation with very high classification does not prove a real campaign will succeed.",
              "- The Q/P plot shows parameter uncertainty only; assay uncertainty and censoring are included in the information calculation.",
              "", "The alternative YAML files are for comparison. The original proposal has not been overwritten."])
(output / "stress_test_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nElapsed time: {time.perf_counter() - started:.1f} seconds")
print(f"Results: {output}")
if "agg" not in plt.get_backend().lower():
    plt.show()
