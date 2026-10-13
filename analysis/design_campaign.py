"""Compare campaign strategies using M2/M4 predictions, not new experiments.

Screen a finite grid, then evaluate shortlisted campaigns with a joint
bootstrap predictive likelihood. The same parameter sample is shared across
all times and experiments in a simulated campaign, preserving correlation.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import log_ndtr, logsumexp, xlogy

import reaction_fitting as fitting

folder = Path(__file__).resolve().parent
output = folder / "output" / "campaign_design"
output.mkdir(parents=True, exist_ok=True)
saved = folder / "output" / "system_B"
parameters = json.loads((saved / "fitted_parameters.json").read_text())
bootstrap = json.loads((saved / "bootstrap_parameters.json").read_text())
models = ["M2", "M4"]
ranking = pd.read_csv(saved / "model_comparison.csv").set_index("model")
weights = ranking.loc[models, "relative_AIC_weight"].to_numpy(float)
weights = weights / weights.sum()
if any(len(bootstrap[m]) < 54 for m in models):
    raise ValueError("Incomplete bootstrap fits.")
if any(len(bootstrap[m]) != 60 for m in models):
    raise ValueError("This design comparison expects the current 60 fits per model.")
fingerprint = hashlib.sha256(b"".join(p.read_bytes() for p in [
    Path(__file__), folder / "reaction_fitting.py", fitting.PROJECT / "src" / "candidates" / "models.py",
    saved / "fitted_parameters.json", saved / "bootstrap_parameters.json", saved / "model_comparison.csv"
])).hexdigest()
times = np.array([0, 2, 5, 10, 20, 40, 60, 90, 120, 180, 240, 360, 480, 600, 720, 960], float)
relative = np.array([.03, .03, .03, .15])
absolute = np.array([.0035, .0035, .0035, .000022])
started = time.perf_counter()


def make_run(temperature, a, b):
    return {"temperature": float(temperature), "A0": float(a), "B0": float(b), "times": times,
            "ti": np.repeat(np.arange(len(times)), 4), "si": np.tile(np.arange(4), len(times))}


def simulate_array(model, values, run):
    return fitting.predict_run(model, values, run).reshape(len(times), 4)


def sample_times(score):
    # Retain a spread of times instead of six almost-identical plateau samples.
    selected = []
    for index in np.argsort(score[1:])[::-1] + 1:
        if all(abs(np.log(times[index] / times[other])) >= np.log(1.6) for other in selected):
            selected.append(int(index))
        if len(selected) == 6:
            break
    if len(selected) < 6:
        for index in np.argsort(score[1:])[::-1] + 1:
            if int(index) not in selected:
                selected.append(int(index))
            if len(selected) == 6:
                break
    return sorted(selected)


candidates = []
print("Screening 168 allowed condition sets; no real experiments are being run.", flush=True)
for temperature in [50, 65, 75, 85, 95, 110, 120]:
    for a in [.1, .25, .5, .75, 1., 1.5]:
        for b in [.1, .25, .75, 1.5]:
            run = make_run(temperature, a, b)
            index = len(candidates)
            cache = output / f"screen_{index:03d}.npz"
            if cache.exists():
                cached = np.load(cache)
                compatible = str(cached["fingerprint"]) == fingerprint
            else:
                compatible = False
            if compatible:
                means, variance, point = cached["means"], cached["variance"], cached["point"]
            else:
                point = np.array([simulate_array(m, parameters[m], run) for m in models])
                screening = np.array([[simulate_array(m, bootstrap[m][i], run)
                                       for i in np.linspace(0, 59, 20, dtype=int)] for m in models])
                means = screening.mean(axis=1)
                variance = screening.var(axis=1, ddof=1)
                np.savez_compressed(cache, fingerprint=fingerprint, means=means, variance=variance, point=point)
            assay_variance = (relative * means) ** 2 + absolute ** 2
            robust = np.sum((means[0] - means[1]) ** 2 / (variance.sum(axis=0) + assay_variance.sum(axis=0)), axis=1)
            point_noise = (relative * point) ** 2 + absolute ** 2
            point_score = np.sum((point[0] - point[1]) ** 2 / point_noise.sum(axis=0), axis=1)
            robust_indices = sample_times(robust)
            point_indices = sample_times(point_score)
            candidates.append({"id": index, "temperature": temperature, "A0": a, "B0": b,
                               "cost_g": 1.25 * a, "run": run, "robust_indices": robust_indices,
                               "point_indices": point_indices, "robust_score": float(robust[robust_indices].sum()),
                               "point_score": float(point_score[point_indices].sum())})
            if (index + 1) % 24 == 0:
                print(f"Screened {index + 1}/168", flush=True)

grid = pd.DataFrame([{k: c[k] for k in ["id", "temperature", "A0", "B0", "cost_g", "robust_score", "point_score"]}
                     for c in candidates])
grid.to_csv(output / "condition_screen.csv", index=False)
by_id = {c["id"]: c for c in candidates}
baseline = [next(c["id"] for c in candidates if c["temperature"] == t and c["A0"] == 1. and c["B0"] == .25)
            for t in [65, 75, 85]]

# Point-estimate strategy: select high separation per gram, without parameter uncertainty.
point_ids = []
material = 0.
for candidate in sorted(candidates, key=lambda c: c["point_score"] / c["cost_g"], reverse=True):
    if material + candidate["cost_g"] <= 5 + 1e-9:
        point_ids.append(candidate["id"])
        material += candidate["cost_g"]
    if len(point_ids) == 3:
        break

# A finite shortlist avoids exhaustive joint optimisation of the full design space.
shortlist = set(grid.nlargest(16, "robust_score")["id"].astype(int))
value_grid = grid.assign(value=grid["robust_score"] / grid["cost_g"])
shortlist.update(value_grid.nlargest(16, "value")["id"].astype(int))
shortlist.update(point_ids + baseline)
for _, rows in grid.groupby("temperature"):
    shortlist.add(int(rows.nlargest(1, "robust_score")["id"].iloc[0]))


def full_predictions(candidate_id):
    cache = output / f"full_{candidate_id:03d}.npz"
    if cache.exists():
        cached = np.load(cache)
        if str(cached["fingerprint"]) == fingerprint:
            return cached["predictions"]
    run = by_id[candidate_id]["run"]
    result = np.array([[simulate_array(m, p, run) for p in bootstrap[m]] for m in models])
    np.savez_compressed(cache, fingerprint=fingerprint, predictions=result)
    return result


print(f"Evaluating joint information for {len(shortlist)} shortlisted conditions using all 60 fits per model.", flush=True)
full = {i: full_predictions(i) for i in sorted(shortlist)}
prior_entropy = -np.sum(xlogy(weights, weights))


class CampaignEvaluator:
    def __init__(self, worlds_per_model, seed):
        self.rng = np.random.default_rng(seed)
        self.worlds_per_model = worlds_per_model
        self.truth = np.repeat(np.arange(2), worlds_per_model)
        self.world_parameters = self.rng.integers(0, 60, size=2 * worlds_per_model)
        self.world_weights = np.repeat(weights / worlds_per_model, worlds_per_model)
        self.cache = {}

    def likelihood(self, candidate_id, indices):
        key = (candidate_id, tuple(indices))
        if key in self.cache:
            return self.cache[key]
        mean = full[candidate_id][:, :, indices, :].reshape(2, 60, len(indices) * 4)
        sigmas = np.sqrt((mean * np.tile(relative, len(indices))) ** 2 + np.tile(absolute, len(indices)) ** 2)
        true_mean = mean[self.truth, self.world_parameters]
        true_sigma = sigmas[self.truth, self.world_parameters]
        observed = self.rng.normal(true_mean, true_sigma)
        censored = (observed < 3e-5) & (np.tile(np.arange(4), len(indices)) == 3)
        component_mean = mean.reshape(120, -1)
        component_sigma = sigmas.reshape(120, -1)
        z = (observed[:, None, :] - component_mean[None, :, :]) / component_sigma[None, :, :]
        log_likelihood = -.5 * z ** 2 - np.log(component_sigma[None, :, :]) - .5 * np.log(2 * np.pi)
        below_limit = log_ndtr((3e-5 - component_mean) / component_sigma)
        log_likelihood = np.where(censored[:, None, :], below_limit[None, :, :], log_likelihood).sum(axis=2)
        self.cache[key] = log_likelihood
        return log_likelihood

    def evaluate(self, log_likelihood):
        model_log_scores = np.column_stack([
            logsumexp(log_likelihood[:, i * 60:(i + 1) * 60], axis=1) - np.log(60) + np.log(weights[i])
            for i in range(2)])
        posterior = np.exp(model_log_scores - logsumexp(model_log_scores, axis=1)[:, None])
        entropy = -np.sum(xlogy(posterior, posterior), axis=1)
        information = prior_entropy - np.sum(self.world_weights * entropy)
        accuracy = np.sum(self.world_weights * (posterior.argmax(axis=1) == self.truth))
        return float(information), float(accuracy)


evaluator = CampaignEvaluator(80, 41977)
selected = []
joint = np.zeros((160, 120))
material = 0.
selection_steps = []
current_information = 0.
while len(selected) < 10:
    options = []
    for candidate_id in sorted(shortlist - set(selected)):
        candidate = by_id[candidate_id]
        if material + candidate["cost_g"] > 5 + 1e-9:
            continue
        contribution = evaluator.likelihood(candidate_id, candidate["robust_indices"])
        information, accuracy = evaluator.evaluate(joint + contribution)
        gain = information - current_information
        options.append((gain / candidate["cost_g"], gain, information, accuracy, candidate_id, contribution))
    if not options:
        break
    best = max(options, key=lambda item: item[0])
    if len(selected) >= 3 and (current_information >= .95 * prior_entropy or best[1] < .005):
        break
    _, gain, current_information, accuracy, candidate_id, contribution = best
    selected.append(candidate_id)
    material += by_id[candidate_id]["cost_g"]
    joint += contribution
    selection_steps.append({"step": len(selected), "candidate_id": candidate_id, "marginal_information_nats": gain,
                            "total_information_nats": current_information, "material_g": material})
    print(f"Selected {len(selected)} experiments: information={current_information:.3f} nats, A={material:.3f} g", flush=True)
if len(selected) < 3:
    raise RuntimeError("No legal minimum-size campaign found.")

baseline_indices = [int(np.flatnonzero(times == t)[0]) for t in [0, 120, 240, 480, 720, 960]]
strategies = {"temperature_only": [(i, baseline_indices) for i in baseline],
              "point_estimates": [(i, by_id[i]["point_indices"]) for i in point_ids],
              "uncertainty_aware": [(i, by_id[i]["robust_indices"]) for i in selected]}

# Test a shorter schedule with early kinetic observations and fewer late plateaus.
short_schedule = []
for candidate_id in selected:
    mean_curve = full[candidate_id].mean(axis=1)
    needed = []
    for model_index in range(2):
        for chemical in [2, 3]:
            curve = mean_curve[model_index, :, chemical]
            reached = np.flatnonzero(curve >= .95 * curve[-1])
            needed.append(int(reached[0]) if len(reached) else len(times) - 1)
    last = max(needed)
    available = np.arange(1, last + 1)
    desired = np.geomspace(times[1], times[last], 6)
    indices = sorted(set(int(available[np.argmin(abs(np.log(times[available] / t)))]) for t in desired))
    for index in available:
        if len(indices) >= 6:
            break
        if int(index) not in indices:
            indices.append(int(index))
    short_schedule.append((candidate_id, sorted(indices)))
strategies["uncertainty_short_times"] = short_schedule

# Evaluate with fresh simulated campaigns, independently of greedy selection.
assessor = CampaignEvaluator(300, 41988)
comparison = []
for name, schedule in strategies.items():
    score = np.zeros((600, 120))
    for candidate_id, indices in schedule:
        score += assessor.likelihood(candidate_id, indices)
    information, accuracy = assessor.evaluate(score)
    comparison.append({"strategy": name, "experiments": len(schedule), "samples": sum(len(t) for _, t in schedule),
                       "material_g": sum(by_id[i]["cost_g"] for i, _ in schedule),
                       "expected_information_nats": information, "fraction_prior_entropy_removed": information / prior_entropy,
                       "simulated_model_classification": accuracy})
comparison = pd.DataFrame(comparison)
comparison.to_csv(output / "strategy_comparison.csv", index=False)
pd.DataFrame(selection_steps).to_csv(output / "selection_steps.csv", index=False)

long_information = comparison.set_index("strategy").loc["uncertainty_aware", "expected_information_nats"]
short_information = comparison.set_index("strategy").loc["uncertainty_short_times", "expected_information_nats"]
if short_information >= .95 * prior_entropy and short_information >= long_information - .01:
    final_strategy = "uncertainty_short_times"
else:
    final_strategy = "uncertainty_aware"
final_schedule = strategies[final_strategy]

experiments = []
for number, (candidate_id, indices) in enumerate(final_schedule, 1):
    candidate = by_id[candidate_id]
    experiments.append({"run_id": f"E{number}", "temperature_C": float(candidate["temperature"]),
                        "A0_M": float(candidate["A0"]), "B0_M": float(candidate["B0"]),
                        "sample_times_min": [float(times[i]) for i in indices], "assay": fitting.SPECIES})
cost = sum(1.25 * experiment["A0_M"] for experiment in experiments)
samples = sum(len(experiment["sample_times_min"]) for experiment in experiments)
assert 3 <= len(experiments) <= 10 and cost <= 5 + 1e-9 and samples <= 60
for experiment in experiments:
    assert 50 <= experiment["temperature_C"] <= 120
    assert .1 <= experiment["A0_M"] <= 1.5 and .1 <= experiment["B0_M"] <= 1.5
    assert all(0 <= t <= 960 for t in experiment["sample_times_min"])
    assert experiment["sample_times_min"] == sorted(set(experiment["sample_times_min"]))
campaign = {"rationale": "Finite-grid, uncertainty-aware discrimination of M2 vs M4. "
            "Greedy joint expected-information gain per gram preserves shared parameter uncertainty across experiments. "
            f"Selected schedule strategy: {final_strategy}. "
            "Shorter, logarithmically spread sampling was compared with long plateau sampling. "
            "All experiments are committed together; no observations are fed back during this event.",
            "experiments": experiments}
# JSON syntax is valid YAML; the organisers' yaml.safe_load can read this file.
(output / "proposed_campaign.yaml").write_text(json.dumps(campaign, indent=2))
(output / "budget_check.json").write_text(json.dumps({"experiments": len(experiments), "material_g": cost,
                                                     "samples": samples, "legal": True}, indent=2))

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(comparison["strategy"], comparison["fraction_prior_entropy_removed"], color=["gray", "tab:orange", "tab:blue", "tab:green"])
ax.tick_params(axis="x", labelrotation=15)
ax.set_ylabel("Expected fraction of M2/M4 uncertainty removed")
ax.set_title("Simulated campaign comparison - conditional on fitted M2/M4")
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=.2)
fig.tight_layout()
fig.savefig(output / "strategy_comparison.png", dpi=180)

fig2, axes = plt.subplots(len(experiments), 1, figsize=(10, 3.3 * len(experiments)), squeeze=False)
for ax, (candidate_id, indices), experiment in zip(axes.flat, final_schedule, experiments):
    prediction = full[candidate_id]
    for i, model in enumerate(models):
        curves = prediction[i, :, :, 3]
        low, high = np.quantile(curves, [.05, .95], axis=0)
        shown = times <= experiment["sample_times_min"][-1]
        ax.plot(times[shown], curves.mean(axis=0)[shown], label=model)
        ax.fill_between(times[shown], low[shown], high[shown], alpha=.2)
    ax.axhline(3e-5, color="gray", linestyle=":", label="Q LOQ")
    for t in experiment["sample_times_min"]:
        ax.axvline(t, color="gray", alpha=.2)
    ax.set_title(f"{experiment['run_id']}: {experiment['temperature_C']:g} deg C, A0={experiment['A0_M']:g} M, B0={experiment['B0_M']:g} M")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Q concentration (mol/L)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
fig2.suptitle("Proposed experiments: parameter-uncertainty bands, excluding assay noise")
fig2.tight_layout(rect=(0, 0, 1, .98))
fig2.savefig(output / "proposed_Q_profiles.png", dpi=180)

limitations = [
    "Conditional on M2/M4 and fixed AIC weights; other mechanisms and model-weight uncertainty are omitted.",
    "Saved bootstrap samples and their extrapolation are not yet empirically calibrated.",
    "Finite grid and greedy selection do not guarantee the globally optimal campaign.",
    "Time selection uses a screening heuristic with minimum time separation, not exhaustive joint time optimisation.",
    "Bootstrap parameter uncertainty is shared across campaign samples; assay errors are assumed independent.",
    "Near-duplicate experiments may improve discrimination under independent assay noise but do not test batch/systematic errors.",
    "Simulated information and classification are planning metrics, not observed experimental performance.",
    "Budget differences between strategies must be reported; they are not necessarily equally expensive campaigns.",
    "The follow-up assay is assumed to have the supplied Q noise and quantitation limit at every selected sample.",
]
lines = ["# Proposed confirmatory campaign", "", "This is a proposed single-block campaign, not a completed experiment.",
         "", "## Strategy", "", campaign["rationale"], "",
         "M2 impurity rate is k3*I; M4 impurity rate is k3*I*A. Changing initial A and B tests this distinction.",
         "Screened 168 conditions with 20 bootstrap fits/model; shortlisted designs use all 60 fits/model.",
         "Independent evaluation uses 600 simulated campaign outcomes, weighted by the current model weights.",
         "Each parameter sample is shared across all observations in a simulated campaign.", "",
         "## Proposed experiments", "", "| Run | Temperature C | A0 M | B0 M | Sampling times min | A cost g |",
         "|---|---:|---:|---:|---|---:|"]
for experiment in experiments:
    lines.append(f"| {experiment['run_id']} | {experiment['temperature_C']:g} | {experiment['A0_M']:g} | {experiment['B0_M']:g} | "
                 + ", ".join(f"{t:g}" for t in experiment["sample_times_min"]) + f" | {1.25 * experiment['A0_M']:.3f} |")
lines.extend(["", f"Budget: {len(experiments)} experiments; {cost:.3f} g A; {samples} samples. All four observable species assayed.",
              "", "## Explored alternatives", "", "Temperature-only: repeats the historical charge at 65/75/85 C.",
              "Point-estimate design: maximises best-fit separation per gram while ignoring parameter uncertainty.",
              "Uncertainty-aware design: greedy joint information per gram, at least three experiments; stops when information saturates.",
              "Short-time variant: keeps the selected conditions but spreads samples logarithmically up to roughly 95% product/impurity formation.",
              f"Final schedule: {final_strategy}; selected after checking retained information in fresh simulated campaigns.",
              "", "| Strategy | Experiments | Samples | A g | Information nats | Fraction uncertainty removed | Simulated classification |",
              "|---|---:|---:|---:|---:|---:|---:|"])
for row in comparison.to_dict("records"):
    lines.append(f"| {row['strategy']} | {row['experiments']} | {row['samples']} | {row['material_g']:.3f} | "
                 f"{row['expected_information_nats']:.4f} | {row['fraction_prior_entropy_removed']:.3f} | {row['simulated_model_classification']:.3f} |")
lines.extend(["", "## Limitations and possible failures", ""] + ["- " + item for item in limitations])
lines.extend(["", "Do not equate these simulated discrimination probabilities with guarantees about real experimental results.",
              "A low predicted Q concentration may be reported below LOQ; the information calculation includes that event.",
              "No campaign observations are used to alter the held-out predictions or other current deliverables."])
(output / "campaign_rationale.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\nSTRATEGY COMPARISON:")
print(comparison.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nPROPOSED CAMPAIGN:")
print(json.dumps(campaign, indent=2))
print(f"\nBudget passed: {cost:.3f} g A, {samples} samples, {len(experiments)} experiments.")
print(f"Elapsed time: {time.perf_counter() - started:.1f} seconds")
print(f"Results: {output}")
print("Planning scores are conditional simulations, not measured performance or guarantees.")
if "agg" not in plt.get_backend().lower():
    plt.show()
