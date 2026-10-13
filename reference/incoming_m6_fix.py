"""Preliminary likelihood comparison of the six supplied System B models.

Uses the specified noise standard deviation at the model concentration,
and a Gaussian left-censoring likelihood for <LOQ results. AIC weights
are relative model-comparison weights, not Bayesian posterior probabilities.

M6 CONVERGENCE FIX (changes marked "M6 CHANGED"): new starting guess for M6, a higher upper
limit on its rate constants, and more restarts for M6. Results are saved with a "_m6fix"
suffix so the original output files are not overwritten.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import log_ndtr

project = Path(__file__).resolve().parent / "ucl-hackathon-student-main" / "ucl-hackathon-student-main"
sys.path.insert(0, str(project / "src"))
from candidates import CANDIDATES, Conditions, simulate

output = Path(__file__).resolve().parent / "output" / "system_B"
output.mkdir(parents=True, exist_ok=True)
data = pd.read_csv(project / "data" / "system_B" / "measurements.csv")
data["censored"] = data["censored"].astype(str).str.lower().map({"true": True, "false": False})
if data["censored"].isna().any():
    raise ValueError("Unrecognised censoring flag.")
data["numeric_value"] = pd.to_numeric(data["value"], errors="coerce")
if data.loc[~data["censored"], "numeric_value"].isna().any():
    raise ValueError("An uncensored measurement is not numeric.")
if (data.loc[data["censored"], "species"] != "Q").any():
    raise ValueError("This script expects censoring only for Q.")

species_order = ["A", "B", "P", "Q"]
experiments = []
for run_id, rows in data.groupby("run_id"):
    first = rows.iloc[0]
    times = np.unique(np.r_[0.0, rows["time_min"].to_numpy()])
    conditions = Conditions(float(first["temperature_C"]), float(first["A0_M"]),
                            float(first["B0_M"]), times)
    experiments.append({
        "run_id": run_id, "rows": rows, "conditions": conditions,
        "time_index": np.searchsorted(times, rows["time_min"].to_numpy()),
        "species_index": np.array([species_order.index(s) for s in rows["species"]]),
        "observed": rows["numeric_value"].to_numpy(),
        "censored": rows["censored"].to_numpy(dtype=bool),
        "rel": rows["species"].map({"A": .03, "B": .03, "P": .03, "Q": .15}).to_numpy(),
        "absolute": rows["species"].map({"A": .0035, "B": .0035, "P": .0035, "Q": .000022}).to_numpy(),
    })


def unpack(model, values):
    # Scale activation energies to make optimisation better conditioned.
    return {name: float(np.exp(value) if name.startswith("k") else 100 * value)
            for name, value in zip(CANDIDATES[model].param_names, values)}


def negative_log_likelihood(values, model):
    parameters = unpack(model, values)
    total = 0.0
    try:
        for experiment in experiments:
            curve = simulate(model, experiment["conditions"], parameters)
            means = curve[species_order].to_numpy()[experiment["time_index"], experiment["species_index"]]
            sigma = np.sqrt((experiment["rel"] * means) ** 2 + experiment["absolute"] ** 2)
            exact = ~experiment["censored"]
            z = (experiment["observed"][exact] - means[exact]) / sigma[exact]
            total += np.sum(.5 * z ** 2 + np.log(sigma[exact]) + .5 * np.log(2 * np.pi))
            # Probability that an assay result lies below the quantitation limit.
            censored = experiment["censored"]
            total -= np.sum(log_ndtr((3e-5 - means[censored]) / sigma[censored]))
    except (RuntimeError, ValueError, FloatingPointError):
        return 1e30
    return float(total) if np.isfinite(total) else 1e30


initial_rates = {
    "M1": [.02, .00002], "M2": [.02, .03, .00003],
    "M3": [.02, .000002], "M4": [.02, .03, .00004],
    # M6 CHANGED: was [.01, .1, .00002], which stalled at NLL ~66,000. All M6 fits put
    # k1 near 0.001 and k2 large (I reacts with B almost instantly), so start there.
    "M5": [.02, .000003], "M6": [.001, 10.0, .00002],
}

# M6 CHANGED: upper limit for k@75C. Default stays 10; M6's k2 pinned at that limit.
K_UPPER_DEFAULT = 10.0
K_UPPER = {"M6": 1000.0}

# M6 CHANGED: number of random restarts (on top of the starting guess).
N_RANDOM_STARTS = 3
N_RANDOM_STARTS_M6 = 3   # M6 fits take ~1-2 minutes each, so this adds several minutes
rng = np.random.default_rng(419)
results = []
fitted_parameters = {}
print("Fitting six models. This may take several minutes.", flush=True)
print("Assumed bounds: k@75C = 1e-10 to 10 (1e-10 to 1000 for M6); Ea = 0 to 200 kJ/mol.", flush=True)

for model, candidate in CANDIDATES.items():
    names = candidate.param_names
    initial = np.array([np.log(initial_rates[model][i // 2]) if name.startswith("k") else .6
                        for i, name in enumerate(names)])
    k_max = K_UPPER.get(model, K_UPPER_DEFAULT)   # M6 CHANGED: per-model upper limit
    bounds = [(np.log(1e-10), np.log(k_max)) if name.startswith("k") else (0, 2) for name in names]
    starts = [initial]
    for _ in range(N_RANDOM_STARTS_M6 if model == "M6" else N_RANDOM_STARTS):   # M6 CHANGED
        trial = initial.copy()
        trial[0::2] += rng.normal(0, 1.5, len(trial[0::2]))
        trial[1::2] = rng.uniform(.2, 1.2, len(trial[1::2]))
        starts.append(np.array([np.clip(v, low + 1e-5, high - 1e-5) for v, (low, high) in zip(trial, bounds)]))
    attempts = []
    for number, start in enumerate(starts, 1):
        print(f"{model}: starting guess {number}/{len(starts)}", flush=True)
        fit = minimize(negative_log_likelihood, start, args=(model,), method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500, "ftol": 1e-10, "maxls": 30})
        attempts.append(fit)
    valid = [fit for fit in attempts if np.isfinite(fit.fun) and fit.fun < 1e29]
    if not valid:
        raise RuntimeError(f"All fits failed for {model}. Send the console output.")
    best = min(valid, key=lambda fit: fit.fun)
    boundary = any(abs(value - low) < .001 or abs(value - high) < .001
                   for value, (low, high) in zip(best.x, bounds))
    fitted_parameters[model] = unpack(model, best.x)
    results.append({"model": model, "parameters": len(names), "negative_log_likelihood": best.fun,
                    "AIC": 2 * best.fun + 2 * len(names), "converged": bool(best.success),
                    "at_parameter_bound": boundary,
                    "start_NLL_range": max(f.fun for f in valid) - min(f.fun for f in valid),
                    "optimizer_message": str(best.message)})
    # Save progress after every model, so completed work is retained.
    pd.DataFrame(results).to_csv(output / "comparison_progress_m6fix.csv", index=False)
    print(f"{model} completed; converged={best.success}, at bound={boundary}", flush=True)

ranking = pd.DataFrame(results).sort_values("AIC").reset_index(drop=True)
ranking["delta_AIC"] = ranking["AIC"] - ranking["AIC"].min()
weights = np.exp(-.5 * ranking["delta_AIC"].to_numpy())
ranking["relative_AIC_weight"] = weights / weights.sum()
ranking.to_csv(output / "model_comparison_m6fix.csv", index=False)
(output / "fitted_parameters_m6fix.json").write_text(json.dumps(fitted_parameters, indent=2))
print("\nPRELIMINARY comparison: lower AIC is better.")
print(ranking[["model", "parameters", "AIC", "delta_AIC", "relative_AIC_weight", "converged", "at_parameter_bound"]]
      .to_string(index=False, float_format=lambda value: f"{value:.4g}"))


# Overlay the best two preliminary models for a visual comparison.
top_models = ranking["model"].iloc[:2].tolist()
fig, axes = plt.subplots(len(experiments), 2, figsize=(12, 3 * len(experiments)), squeeze=False)
colors = {"A": "tab:blue", "B": "tab:orange", "P": "tab:green", "Q": "tab:red"}
for row_index, experiment in enumerate(experiments):
    rows, conditions = experiment["rows"], experiment["conditions"]
    smooth = Conditions(conditions.temperature_c, conditions.A0, conditions.B0,
                        np.linspace(0, rows["time_min"].max(), 160))
    for column, species_list in enumerate((["A", "B", "P"], ["Q"])):
        ax = axes[row_index, column]
        for species in species_list:
            measured = rows[(rows["species"] == species) & ~rows["censored"]]
            ax.scatter(measured["time_min"], measured["numeric_value"], color=colors[species], s=20,
                       label=f"{species} measured")
        for model, style in zip(top_models, ["-", "--"]):
            curve = simulate(model, smooth, fitted_parameters[model])
            for species in species_list:
                ax.plot(curve["time_min"], curve[species], style, color=colors[species], label=f"{species}: {model}")
        if column == 1:
            censored_q = rows[(rows["species"] == "Q") & rows["censored"]]
            ax.axhline(3e-5, color="gray", linestyle=":", label="LOQ")
            ax.scatter(censored_q["time_min"], np.full(len(censored_q), 3e-5), marker="v",
                       facecolors="none", edgecolors="tab:red", label="Below LOQ (upper bound)")
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.set_title(f"{experiment['run_id']}: {conditions.temperature_c:g} deg C")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Concentration (mol/L)")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
fig.suptitle("System B: best two preliminary fits")
fig.tight_layout(rect=(0, 0, 1, .98))
fig.savefig(output / "top_two_model_fits_m6fix.png", dpi=180)
print(f"\nResults saved in: {output}")
plt.show()
