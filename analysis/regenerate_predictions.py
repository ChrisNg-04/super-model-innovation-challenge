"""Regenerate preliminary predictions using the saved bootstrap fits.

Run predict_system_B.py again only if training data, fitted parameters,
model selection, noise assumptions or fitting bounds change. This script
does not fit anything and does not validate interval calibration.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

started = time.perf_counter()
folder = Path(__file__).resolve().parent
project = folder.parent
output = folder / "output" / "system_B"
sys.path.insert(0, str(project / "src"))
from candidates import Conditions, simulate

parameters = json.loads((output / "fitted_parameters.json").read_text())
bootstraps = json.loads((output / "bootstrap_parameters.json").read_text())
diagnostics = pd.read_csv(output / "bootstrap_diagnostics.csv").set_index("model")
ranking = pd.read_csv(output / "model_comparison.csv").set_index("model")
models = ["M2", "M4"]
weights = ranking.loc[models, "relative_AIC_weight"].to_numpy(float)
weights = weights / weights.sum()
species = ["A", "B", "P", "Q"]
rng = np.random.default_rng(4192027)
heldout = pd.read_csv(project / "data" / "heldout_conditions.csv")
groups = []
keys = []
for run_id, rows in heldout.groupby("run_id", sort=False):
    if len(rows[["temperature_C", "A0_M", "B0_M"]].drop_duplicates()) != 1:
        raise ValueError(f"Inconsistent conditions in {run_id}.")
    first = rows.iloc[0]
    conditions = Conditions(float(first["temperature_C"]), float(first["A0_M"]), float(first["B0_M"]),
                            np.unique(np.r_[0, rows["time_min"].to_numpy()]))
    groups.append((rows, conditions))
    for t in rows["time_min"]:
        for chemical in species:
            keys.append({"run_id": run_id, "time_min": float(t), "species": chemical})


def predict(model, values):
    result = []
    for rows, conditions in groups:
        curve = simulate(model, conditions, values).set_index("time_min")
        for t in rows["time_min"]:
            result.extend(curve.loc[t, species].to_numpy(float))
    return np.asarray(result)


print("Using saved bootstrap fits; no fitting will be repeated.", flush=True)
ensembles = {}
central = np.zeros(len(keys))
for model, weight in zip(models, weights):
    samples = bootstraps[model]
    check = diagnostics.loc[model]
    if len(samples) != int(check["accepted"]) or len(samples) < .9 * check["attempted"]:
        raise ValueError(f"Incomplete bootstrap results for {model}.")
    print(f"Predicting held-out conditions with {len(samples)} saved {model} fits...", flush=True)
    ensembles[model] = np.array([predict(model, values) for values in samples])
    central += weight * predict(model, parameters[model])

number_of_draws = 4000
chosen = rng.choice(len(models), size=number_of_draws, p=weights)
means = np.empty((number_of_draws, len(keys)))
for index, model in enumerate(models):
    indices = np.flatnonzero(chosen == index)
    selected = rng.integers(0, len(ensembles[model]), size=len(indices))
    means[indices] = ensembles[model][selected]
result = pd.DataFrame(keys)
relative = result["species"].map({"A": .03, "B": .03, "P": .03, "Q": .15}).to_numpy()
absolute = result["species"].map({"A": .0035, "B": .0035, "P": .0035, "Q": .000022}).to_numpy()
sigma = np.sqrt((relative * means) ** 2 + absolute ** 2)
draws = np.maximum(0, rng.normal(means, sigma))
lower, upper = np.quantile(draws, [.05, .95], axis=0)
result = result.assign(prediction=central, lower90=lower, upper90=upper)
if not np.isfinite(result[["prediction", "lower90", "upper90"]].to_numpy()).all():
    raise ValueError("Nonfinite predictions.")
if result.duplicated(["run_id", "time_min", "species"]).any() or (result["lower90"] > result["upper90"]).any():
    raise ValueError("Invalid prediction keys or intervals.")
# Preserve the original bootstrap-run CSV for comparison.
target = output / "regenerated_predictions.csv"
result.to_csv(target, index=False)
print(f"\nWrote {len(result)} predictions to: {target}")
print(f"Elapsed time: {time.perf_counter() - started:.1f} seconds")
print("Five-run conditional validation is included; extrapolation coverage is unverified.")
print("The new fixed random seed can cause small differences in interval endpoints.")

run_id = heldout["run_id"].iloc[0]
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, chemical in zip(axes.flat, species):
    rows = result[(result["run_id"] == run_id) & (result["species"] == chemical)].sort_values("time_min")
    ax.plot(rows["time_min"], rows["prediction"], "o-", label="Model-averaged prediction")
    ax.fill_between(rows["time_min"].to_numpy(), rows["lower90"].to_numpy(), rows["upper90"].to_numpy(),
                    alpha=.25, label="Exploratory 90% prediction interval")
    ax.set_title(f"{run_id}: {chemical}")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Concentration (mol/L)")
    ax.grid(alpha=.2)
    if chemical == "Q":
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
axes.flat[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(output / "regenerated_prediction_example.png", dpi=180)
if "agg" not in plt.get_backend().lower():
    plt.show()
