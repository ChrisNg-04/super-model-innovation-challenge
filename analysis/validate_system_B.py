"""Fresh fits with one historical experiment excluded at a time.

This checks point prediction and held-out likelihood for the preselected M2/M4
pair. It does not validate the final bootstrap intervals. Assay-only interval
coverage is a diagnostic baseline that omits parameter uncertainty.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logsumexp, ndtr

import reaction_fitting as fitting

MODELS = ["M2", "M4"]  # Preselected from full-data exploration; interpretation is conditional on this pair.
STARTS = 3
folder = Path(__file__).resolve().parent
output = folder / "output" / "system_B" / "validation"
output.mkdir(parents=True, exist_ok=True)
data = fitting.load_data()
data_path = fitting.PROJECT / "data" / "system_B" / "measurements.csv"
source_hash = hashlib.sha256(data_path.read_bytes() + (folder / "reaction_fitting.py").read_bytes()
                             + Path(__file__).read_bytes()).hexdigest()
started = time.perf_counter()
fold_records = []
prediction_tables = []
print("Fresh fits: each experiment is hidden in turn. No full-data fits or bootstraps are reused.", flush=True)
print("This checks the preselected M2/M4 pair, not independent six-model selection.", flush=True)

for fold_number, excluded in enumerate(sorted(data["run_id"].unique())):
    heldout_data = data[data["run_id"] == excluded].copy()
    run = fitting.prepare_runs(heldout_data)[0]
    fitted = []
    for model_number, model in enumerate(MODELS):
        seed = 419 + fold_number * 10 + model_number
        cache = output / f"fit_without_{excluded}_{model}.json"
        key = {"source_hash": source_hash, "starts": STARTS, "seed": seed,
               "model": model, "excluded": excluded}
        cached = json.loads(cache.read_text()) if cache.exists() else None
        if cached and cached.get("key") == key:
            result = cached["fit"]
            print(f"Reuse validated cache: exclude {excluded}, model {model}", flush=True)
        else:
            print(f"Fitting {model} without {excluded}: {STARTS} independent guesses...", flush=True)
            result = fitting.fit_model(data, model, exclude_runs=[excluded], starts=STARTS, seed=seed)
            cache.write_text(json.dumps({"key": key, "fit": result}, indent=2))
        if excluded in result["training_runs"]:
            raise RuntimeError("Validation leakage detected.")
        fitted.append(result)
    aic = np.array([result["AIC"] for result in fitted])
    weights = np.exp(-.5 * (aic - aic.min()))
    weights = weights / weights.sum()
    means = np.array([fitting.predict_run(result["model"], result["parameters"], run) for result in fitted])
    log_scores = np.array([fitting.log_likelihood_rows(mean, run) for mean in means])
    for result, weight, scores in zip(fitted, weights, log_scores):
        fold_records.append({"excluded": excluded, "model": result["model"],
                             "training_runs": ",".join(result["training_runs"]),
                             "training_AIC_weight": float(weight), "test_NLL": float(-scores.sum()),
                             "at_parameter_bound": result["at_parameter_bound"],
                             "converged_starts": result["converged_starts"],
                             "starts_near_best": result["starts_near_best"],
                             "start_NLL_range": result["start_NLL_range"]})
        if result["starts_near_best"] < 2:
            print(f"WARNING: {result['model']} excluding {excluded} has only one start near its best fit.", flush=True)
        elif result["start_NLL_range"] > 1:
            print(f"NOTE: {result['model']} excluding {excluded}: some guesses found poorer optima; "
                  f"{result['starts_near_best']} reached the best fit.", flush=True)
    central = weights @ means
    # Exact mixture likelihood for quantified and below-limit observations.
    mixture_log_score = logsumexp(np.log(weights)[:, None] + log_scores, axis=0)
    checked = heldout_data.copy()
    checked["prediction"] = central
    checked["sigma"] = fitting.noise_sigma(central, run)
    checked["scaled_residual"] = (central - run["observed"]) / checked["sigma"].to_numpy()
    checked["heldout_log_score"] = mixture_log_score
    sigmas = np.array([fitting.noise_sigma(mean, run) for mean in means])
    checked["probability_below_LOQ"] = np.sum(weights[:, None] * ndtr((3e-5 - means) / sigmas), axis=0)
    # Screening intervals only: draw assay noise around fitted curves.
    # These deliberately exclude bootstrap parameter uncertainty.
    rng = np.random.default_rng(9900 + fold_number)
    indices = rng.choice(len(MODELS), size=4000, p=weights)
    draws = np.maximum(0, rng.normal(means[indices], sigmas[indices]))
    lower, upper = np.quantile(draws, [.05, .95], axis=0)
    checked["noise_only_lower90"] = lower
    checked["noise_only_upper90"] = upper
    checked["inside_noise_only_interval"] = np.where(
        ~run["censored"], (run["observed"] >= lower) & (run["observed"] <= upper), np.nan)
    prediction_tables.append(checked)
    pd.DataFrame(fold_records).to_csv(output / "fit_checks.csv", index=False)
    pd.concat(prediction_tables).to_csv(output / "excluded_run_predictions.csv", index=False)
    print(f"Completed {excluded}; held-out mixture NLL={-mixture_log_score.sum():.3f}", flush=True)

predictions = pd.concat(prediction_tables, ignore_index=True)
exact = predictions[~predictions["censored"]]
summary_rows = []
for species, rows in exact.groupby("species"):
    summary_rows.append({"species": species, "measurements": len(rows),
                         "mean_scaled_error": rows["scaled_residual"].mean(),
                         "RMS_scaled_error": np.sqrt(np.mean(rows["scaled_residual"] ** 2)),
                         "noise_only_coverage": rows["inside_noise_only_interval"].mean()})
summary = pd.DataFrame(summary_rows)
summary.to_csv(output / "point_prediction_summary.csv", index=False)
print("\nFRESH-FIT CHECKS:")
print(pd.DataFrame(fold_records).to_string(index=False, float_format=lambda value: f"{value:.4g}"))
print("\nEXCLUDED-EXPERIMENT PREDICTION SUMMARY:")
print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
print("\nBelow-limit Q checks:")
print(predictions[predictions["censored"]][["run_id", "time_min", "prediction", "probability_below_LOQ"]]
      .to_string(index=False, float_format=lambda value: f"{value:.5g}"))
print("\nNoise-only coverage is NOT coverage of the submitted/bootstrap prediction intervals.")
print("It omits parameter uncertainty; Q coverage is conditional on observations being quantified.")
print("M2/M4 were selected during full-data exploration, so this check is conditional on that pair.")
print("Edge temperatures test limited extrapolation; new starting concentrations remain unvalidated.")
print(f"Elapsed time: {time.perf_counter() - started:.1f} seconds")
print(f"Results: {output}")

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
for ax, species in zip(axes.flat, fitting.SPECIES):
    ax.axhspan(-2, 2, color="gray", alpha=.1)
    ax.axhline(0, color="black", linewidth=1)
    for run_id, rows in exact[exact["species"] == species].groupby("run_id"):
        ax.scatter(rows["time_min"], rows["scaled_residual"], label=f"Excluded {run_id}", s=22)
    ax.set_title(f"Species {species}")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("(Prediction - measurement) / assay noise SD")
    ax.grid(alpha=.2)
axes.flat[0].legend(fontsize=8)
fig.suptitle("Predictions for experiments excluded during fitting")
fig.tight_layout()
fig.savefig(output / "excluded_run_residuals.png", dpi=180)
if "agg" not in plt.get_backend().lower():
    plt.show()
