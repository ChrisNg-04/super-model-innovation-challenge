"""Exploratory model-averaged held-out predictions with bootstrap intervals.

These intervals are not yet empirically validated for 90% coverage.
Models and AIC weights are fixed at the saved comparison results.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import log_ndtr

BOOTSTRAPS_PER_MODEL = 60  # Initial exercise; increase after checking runtime and stability.
PREDICTIVE_DRAWS = 4000
rng = np.random.default_rng(4192026)
folder = Path(__file__).resolve().parent
project = folder.parent
output = folder / "output" / "system_B"
sys.path.insert(0, str(project / "src"))
from candidates import CANDIDATES, Conditions, simulate

ranking = pd.read_csv(output / "model_comparison.csv")
saved_parameters = json.loads((output / "fitted_parameters.json").read_text())
models = ["M2", "M4"]
weights = ranking.set_index("model").loc[models, "relative_AIC_weight"].to_numpy(float)
weights = weights / weights.sum()
data = pd.read_csv(project / "data" / "system_B" / "measurements.csv")
data["censored"] = data["censored"].astype(str).str.lower().map({"true": True, "false": False})
if data["censored"].isna().any():
    raise ValueError("Unrecognised censoring flag.")
data["observed"] = pd.to_numeric(data["value"], errors="coerce")
if data.loc[~data["censored"], "observed"].isna().any():
    raise ValueError("An uncensored observation has no numeric concentration.")
species_order = ["A", "B", "P", "Q"]

experiments = []
for run_id, rows in data.groupby("run_id"):
    first = rows.iloc[0]
    times = np.unique(np.r_[0.0, rows["time_min"].to_numpy()])
    experiments.append({
        "conditions": Conditions(float(first["temperature_C"]), float(first["A0_M"]), float(first["B0_M"]), times),
        "ti": np.searchsorted(times, rows["time_min"].to_numpy()),
        "si": np.array([species_order.index(s) for s in rows["species"]]),
        "q": (rows["species"] == "Q").to_numpy(),
        "relative": rows["species"].map({"A": .03, "B": .03, "P": .03, "Q": .15}).to_numpy(),
        "absolute": rows["species"].map({"A": .0035, "B": .0035, "P": .0035, "Q": .000022}).to_numpy(),
    })


def unpack(model, values):
    return {name: float(np.exp(value) if name.startswith("k") else 100 * value)
            for name, value in zip(CANDIDATES[model].param_names, values)}


def measured_means(model, parameters, experiment):
    curve = simulate(model, experiment["conditions"], parameters)
    return curve[species_order].to_numpy()[experiment["ti"], experiment["si"]]


def nll(values, model, bootstrap_observations):
    total = 0.0
    parameters = unpack(model, values)
    try:
        for experiment, (observed, censored) in zip(experiments, bootstrap_observations):
            means = measured_means(model, parameters, experiment)
            sigma = np.sqrt((experiment["relative"] * means) ** 2 + experiment["absolute"] ** 2)
            exact = ~censored
            z = (observed[exact] - means[exact]) / sigma[exact]
            total += np.sum(.5 * z ** 2 + np.log(sigma[exact]) + .5 * np.log(2 * np.pi))
            total -= np.sum(log_ndtr((3e-5 - means[censored]) / sigma[censored]))
    except (RuntimeError, ValueError, FloatingPointError):
        return 1e30
    return float(total) if np.isfinite(total) else 1e30


heldout = pd.read_csv(project / "data" / "heldout_conditions.csv")
heldout_groups = []
keys = []
for run_id, rows in heldout.groupby("run_id", sort=False):
    if len(rows[["temperature_C", "A0_M", "B0_M"]].drop_duplicates()) != 1:
        raise ValueError(f"Inconsistent conditions for {run_id}.")
    first = rows.iloc[0]
    times = np.unique(np.r_[0.0, rows["time_min"].to_numpy()])
    heldout_groups.append((rows, Conditions(float(first["temperature_C"]), float(first["A0_M"]),
                                           float(first["B0_M"]), times)))
    for t in rows["time_min"]:
        for species in species_order:
            keys.append({"run_id": run_id, "time_min": float(t), "species": species})


def heldout_means(model, parameters):
    values = []
    for rows, conditions in heldout_groups:
        curve = simulate(model, conditions, parameters).set_index("time_min")
        for t in rows["time_min"]:
            values.extend(curve.loc[t, species_order].to_numpy(float))
    return np.asarray(values)


started = time.perf_counter()
bootstrap_parameters = {}
ensemble_means = {}
diagnostic_rows = []
print("Building exploratory 90% prediction intervals; this may take several minutes.", flush=True)
print("Each bootstrap creates a noisy copy of the historical experiments and refits it.", flush=True)
for model in models:
    names = CANDIDATES[model].param_names
    start = np.array([np.log(saved_parameters[model][name]) if name.startswith("k")
                      else saved_parameters[model][name] / 100 for name in names])
    bounds = [(np.log(1e-10), np.log(10)) if name.startswith("k") else (0, 2) for name in names]
    fitted_mean_sigma = []
    for experiment in experiments:
        mean = measured_means(model, saved_parameters[model], experiment)
        sigma = np.sqrt((experiment["relative"] * mean) ** 2 + experiment["absolute"] ** 2)
        fitted_mean_sigma.append((mean, sigma))
    samples = []
    failures = 0
    boundary_count = 0
    for number in range(BOOTSTRAPS_PER_MODEL):
        observations = []
        for experiment, (mean, sigma) in zip(experiments, fitted_mean_sigma):
            observed = rng.normal(mean, sigma)
            censored = experiment["q"] & (observed < 3e-5)
            observations.append((observed, censored))
        fit = minimize(nll, start, args=(model, observations), method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 400, "ftol": 1e-9, "maxls": 30})
        if not fit.success or not np.isfinite(fit.fun) or fit.fun >= 1e29:
            # Retry the same noisy dataset, rather than silently accepting a failed fit.
            fit = minimize(nll, start, args=(model, observations), method="Powell", bounds=bounds,
                           options={"maxiter": 150, "ftol": 1e-7, "xtol": 1e-5})
        if fit.success and np.isfinite(fit.fun) and fit.fun < 1e29:
            samples.append(unpack(model, fit.x))
            boundary_count += int(any(abs(v - low) < .001 or abs(v - high) < .001
                                      for v, (low, high) in zip(fit.x, bounds)))
        else:
            failures += 1
        if (number + 1) % 5 == 0:
            print(f"{model}: {number + 1}/{BOOTSTRAPS_PER_MODEL}, accepted={len(samples)}, failures={failures}", flush=True)
    bootstrap_parameters[model] = samples
    diagnostic_rows.append({"model": model, "attempted": BOOTSTRAPS_PER_MODEL,
                            "accepted": len(samples), "failed": failures, "fits_at_bound": boundary_count})
    (output / "bootstrap_parameters.json").write_text(json.dumps(bootstrap_parameters, indent=2))
    pd.DataFrame(diagnostic_rows).to_csv(output / "bootstrap_diagnostics.csv", index=False)
    if len(samples) < .9 * BOOTSTRAPS_PER_MODEL:
        raise RuntimeError(f"Too many failed fits for {model}; inspect bootstrap_diagnostics.csv before using intervals.")
    ensemble_means[model] = np.array([heldout_means(model, p) for p in samples])

prediction = sum(weight * heldout_means(model, saved_parameters[model])
                 for model, weight in zip(models, weights))
chosen_model = rng.choice(len(models), size=PREDICTIVE_DRAWS, p=weights)
draw_means = np.empty((PREDICTIVE_DRAWS, len(keys)))
for model_index, model in enumerate(models):
    indices = np.flatnonzero(chosen_model == model_index)
    selected = rng.integers(0, len(ensemble_means[model]), size=len(indices))
    draw_means[indices] = ensemble_means[model][selected]
key_table = pd.DataFrame(keys)
relative = key_table["species"].map({"A": .03, "B": .03, "P": .03, "Q": .15}).to_numpy()
absolute = key_table["species"].map({"A": .0035, "B": .0035, "P": .0035, "Q": .000022}).to_numpy()
sigma = np.sqrt((relative * draw_means) ** 2 + absolute ** 2)
predictive_draws = np.maximum(0, rng.normal(draw_means, sigma))
lower, upper = np.quantile(predictive_draws, [.05, .95], axis=0)
result = key_table.assign(prediction=prediction, lower90=lower, upper90=upper)
if not np.isfinite(result[["prediction", "lower90", "upper90"]].to_numpy()).all():
    raise RuntimeError("Nonfinite predictions found.")
if (result["lower90"] > result["upper90"]).any():
    raise RuntimeError("Invalid interval ordering.")
if result.duplicated(["run_id", "time_min", "species"]).any():
    raise RuntimeError("Duplicate prediction keys found.")
result.to_csv(output / "preliminary_predictions.csv", index=False)
print("\nFirst predictions:")
print(result.head(12).to_string(index=False))
print(f"\nWrote {len(result)} predictions to: {output / 'preliminary_predictions.csv'}")
print(f"Elapsed time: {time.perf_counter() - started:.1f} seconds")
print("\nIMPORTANT: exploratory intervals, not yet checked for empirical 90% coverage.")
print("They assume M2/M4, fixed AIC weights, the specified assay noise and the current parameter bounds.")
print("The lower endpoint is clipped to zero for physically nonnegative concentrations.")
print("Near-zero coverage can therefore be conservative, rather than exactly 90%.")
print("Check bootstrap failures/bounds, interval stability and withheld-run coverage before submission.")

first_run = heldout["run_id"].iloc[0]
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, species in zip(axes.flat, species_order):
    rows = result[(result["run_id"] == first_run) & (result["species"] == species)].sort_values("time_min")
    ax.plot(rows["time_min"], rows["prediction"], "o-", label="Model-averaged prediction")
    ax.fill_between(rows["time_min"].to_numpy(), rows["lower90"].to_numpy(), rows["upper90"].to_numpy(),
                    alpha=.25, label="Exploratory 90% prediction interval")
    ax.set_title(f"{first_run}: species {species}")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Concentration (mol/L)")
    ax.grid(alpha=.2)
    if species == "Q":
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
axes.flat[0].legend(fontsize=8)
fig.suptitle("Held-out predictions: M2/M4 mixture with bootstrap uncertainty")
fig.tight_layout()
fig.savefig(output / "heldout_prediction_example.png", dpi=180)
if "agg" not in plt.get_backend().lower():
    plt.show()
