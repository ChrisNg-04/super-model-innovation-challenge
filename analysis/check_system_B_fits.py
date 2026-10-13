"""Inspect saved System B fits without repeating the optimisation.

Residual plots use only quantified observations. Censored observations
are checked separately through their probability of falling below LOQ.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import ndtr

folder = Path(__file__).resolve().parent
project = folder.parent
output = folder / "output" / "system_B"
sys.path.insert(0, str(project / "src"))
from candidates import Conditions, simulate

ranking = pd.read_csv(output / "model_comparison.csv")
parameters = json.loads((output / "fitted_parameters.json").read_text())
data = pd.read_csv(project / "data" / "system_B" / "measurements.csv")
data["censored"] = data["censored"].astype(str).str.lower().map({"true": True, "false": False})
if data["censored"].isna().any():
    raise ValueError("Unrecognised censoring flag.")
data["observed"] = pd.to_numeric(data["value"], errors="coerce")
if data.loc[~data["censored"], "observed"].isna().any():
    raise ValueError("An uncensored observation has no numeric value.")
models = ranking["model"].iloc[:2].tolist()

print("OPTIMISATION CHECK - all six models")
print(ranking[["model", "converged", "at_parameter_bound", "start_NLL_range"]]
      .to_string(index=False, float_format=lambda value: f"{value:.6g}"))
print("start_NLL_range = difference between the best and worst finite fits from the saved starting guesses (six for the reviewed M6).")
print("A small range is encouraging. A large range means some guesses found poorer solutions.")
print("A large range does not by itself mean that the best solution is wrong.")

tables = []
for model in models:
    for run_id, rows in data.groupby("run_id"):
        first = rows.iloc[0]
        times = np.unique(np.r_[0.0, rows["time_min"].to_numpy()])
        conditions = Conditions(float(first["temperature_C"]), float(first["A0_M"]),
                                float(first["B0_M"]), times)
        curve = simulate(model, conditions, parameters[model]).set_index("time_min")
        checked = rows.copy()
        checked["model"] = model
        checked["predicted"] = [curve.loc[time, species]
                                for time, species in zip(rows["time_min"], rows["species"])]
        relative = checked["species"].map({"A": .03, "B": .03, "P": .03, "Q": .15})
        absolute = checked["species"].map({"A": .0035, "B": .0035, "P": .0035, "Q": .000022})
        checked["sigma"] = np.sqrt((relative * checked["predicted"]) ** 2 + absolute ** 2)
        checked["scaled_residual"] = (checked["predicted"] - checked["observed"]) / checked["sigma"]
        checked["probability_below_LOQ"] = np.nan
        censored = checked["censored"]
        checked.loc[censored, "probability_below_LOQ"] = ndtr(
            (3e-5 - checked.loc[censored, "predicted"]) / checked.loc[censored, "sigma"]
        )
        tables.append(checked)
diagnostics = pd.concat(tables, ignore_index=True)
exact = diagnostics[~diagnostics["censored"]].copy()

summary_rows = []
for (model, species), rows in exact.groupby(["model", "species"]):
    errors = rows["scaled_residual"].to_numpy()
    summary_rows.append({"model": model, "species": species, "measurements": len(errors),
                         "mean_scaled_error": errors.mean(),
                         "RMS_scaled_error": np.sqrt(np.mean(errors ** 2)),
                         "largest_absolute_error": np.max(np.abs(errors))})
summary = pd.DataFrame(summary_rows)
print("\nQUANTIFIED MEASUREMENTS - includes time-zero observations")
print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
print("Positive residual means the model predicts too high; negative means too low.")
print("Q summaries omit below-limit observations and are affected by that selection.")
print("RMS near one is a rough check, not proof that a model is correct.")

censored_table = diagnostics[diagnostics["censored"]][
    ["model", "run_id", "time_min", "predicted", "sigma", "probability_below_LOQ"]
]
print("\nBELOW-LIMIT Q MEASUREMENTS")
print(censored_table.to_string(index=False, float_format=lambda value: f"{value:.5g}"))
print("A small probability means a recorded below-limit result is surprising under that model.")

print("\nLARGEST QUANTIFIED ERRORS (six per model)")
for model in models:
    rows = exact[exact["model"] == model].copy()
    rows["absolute_error"] = rows["scaled_residual"].abs()
    worst = rows.nlargest(6, "absolute_error")
    print(f"\n{model}")
    print(worst[["run_id", "time_min", "species", "observed", "predicted", "scaled_residual"]]
          .to_string(index=False, float_format=lambda value: f"{value:.5g}"))

fig, axes = plt.subplots(4, len(models), figsize=(12, 12), sharey=True, squeeze=False)
for column, model in enumerate(models):
    for row_index, species in enumerate(["A", "B", "P", "Q"]):
        ax = axes[row_index, column]
        ax.axhspan(-2, 2, color="gray", alpha=.1)
        ax.axhline(0, color="black", linewidth=1)
        ax.axhline(-2, color="gray", linestyle="--", linewidth=.8)
        ax.axhline(2, color="gray", linestyle="--", linewidth=.8)
        rows = exact[(exact["model"] == model) & (exact["species"] == species)]
        for run_id, run in rows.groupby("run_id"):
            ax.scatter(run["time_min"], run["scaled_residual"], s=22, label=run_id)
        ax.set_title(f"{model}: species {species}")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("(Prediction - measurement) / noise SD")
        ax.grid(alpha=.2)
axes[0, 0].legend(title="Experiment", fontsize=8)
fig.suptitle("System B: quantified-measurement residuals for the leading models")
fig.tight_layout(rect=(0, 0, 1, .98))
fig.savefig(output / "leading_model_residuals.png", dpi=180)
diagnostics.to_csv(output / "leading_model_diagnostics.csv", index=False)
summary.to_csv(output / "leading_model_residual_summary.csv", index=False)
censored_table.to_csv(output / "leading_model_censoring_check.csv", index=False)
print(f"\nDiagnostics saved in: {output}")
if "agg" not in plt.get_backend().lower():
    plt.show()
