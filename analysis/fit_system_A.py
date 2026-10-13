"""First fitting exercise: fit the disclosed M1 model to System A.

This uses approximate weights based on measured concentrations. It does
not calculate parameter uncertainty or establish a mechanism ranking.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

project = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project / "src"))
from candidates import Conditions, simulate

data = pd.read_csv(project / "data" / "system_A" / "measurements.csv")
if data["censored"].any():
    raise ValueError("This introductory script only handles uncensored System A data.")

# At time zero, starting concentrations are fixed by the experiment.
# Those rows cannot help estimate the reaction rate constants.
fitting_data = data[(data["time_min"] > 0) & (data["value"] > 0)].copy()

# System A has 2% relative noise on A/B/P and 5% on Q.
# Observed concentrations give approximate weights for this first exercise.
relative_noise = fitting_data["species"].map({"A": 0.02, "B": 0.02, "P": 0.02, "Q": 0.05})
fitting_data["sigma"] = relative_noise * fitting_data["value"]

experiments = []
for run_id, rows in fitting_data.groupby("run_id"):
    rows = rows.copy()
    first = rows.iloc[0]
    conditions = Conditions(
        temperature_c=float(first["temperature_C"]),
        A0=float(first["A0_M"]),
        B0=float(first["B0_M"]),
        times_min=np.unique(np.r_[0.0, rows["time_min"].to_numpy()]),
    )
    experiments.append((run_id, rows, conditions))


def unpack(values):
    # Optimise log(rate constants) to keep both constants positive.
    return {"k1": np.exp(values[0]), "Ea1": values[1],
            "k2": np.exp(values[2]), "Ea2": values[3]}


def residuals(values):
    parameters = unpack(values)
    errors = []
    for run_id, rows, conditions in experiments:
        prediction = simulate("M1", conditions, parameters).set_index("time_min")
        predicted_values = np.array([
            prediction.loc[time, species]
            for time, species in zip(rows["time_min"], rows["species"])
        ])
        errors.extend((predicted_values - rows["value"].to_numpy()) / rows["sigma"].to_numpy())
    return np.asarray(errors)


print("Fitting M1 to all five System A experiments...")
# Try several guesses to reduce dependence on the starting point.
guesses = [(0.01, 60, 0.001, 60), (0.03, 40, 0.003, 80), (0.005, 80, 0.0005, 40)]
fits = []
for k1, ea1, k2, ea2 in guesses:
    fit = least_squares(
        residuals, [np.log(k1), ea1, np.log(k2), ea2],
        bounds=([np.log(1e-8), 0, np.log(1e-8), 0],
                [np.log(10), 200, np.log(10), 200]),
        x_scale="jac", max_nfev=300,
    )
    if fit.success:
        fits.append(fit)
if not fits:
    raise RuntimeError("The fitting did not converge. Send the console output for review.")
best_fit = min(fits, key=lambda fit: fit.cost)
parameters = unpack(best_fit.x)

print("\nEstimated rate constants at 75 deg C (L/mol/min):")
print(f"k1 = {parameters['k1']:.6g}")
print(f"k2 = {parameters['k2']:.6g}")
print("\nEstimated activation energies (kJ/mol):")
print(f"Ea1 = {parameters['Ea1']:.3f}")
print(f"Ea2 = {parameters['Ea2']:.3f}")
print("\nThese are preliminary estimates; uncertainty has not been calculated.")

# Compare measurements (dots) and fitted curves (lines) for every experiment.
fig, axes = plt.subplots(3, 2, figsize=(11, 10))
colors = {"A": "tab:blue", "B": "tab:orange", "P": "tab:green", "Q": "tab:red"}
for ax, (run_id, _, conditions) in zip(axes.flat, experiments):
    rows = data[data["run_id"] == run_id]
    smooth_conditions = Conditions(
        temperature_c=conditions.temperature_c, A0=conditions.A0, B0=conditions.B0,
        times_min=np.linspace(0, rows["time_min"].max(), 150),
    )
    curve = simulate("M1", smooth_conditions, parameters)
    for species, color in colors.items():
        observed = rows[rows["species"] == species]
        ax.scatter(observed["time_min"], observed["value"], color=color, s=15)
        ax.plot(curve["time_min"], curve[species], color=color, label=species)
    ax.set_title(f"{run_id}: {conditions.temperature_c:g} deg C, A0={conditions.A0:g}, B0={conditions.B0:g} M")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Concentration (mol/L)")
    ax.grid(alpha=0.2)
axes.flat[0].legend(title="Species")
axes.flat[-1].axis("off")
fig.suptitle("System A: measurements (dots) and fitted M1 curves (lines)")
fig.tight_layout()

# Residuals reveal errors that concentration plots can hide.
# Positive residual = model prediction is above the measurement.
diagnostic_tables = []
for run_id, rows, conditions in experiments:
    prediction = simulate("M1", conditions, parameters).set_index("time_min")
    checked = rows.copy()
    checked["predicted"] = [
        prediction.loc[time, species]
        for time, species in zip(checked["time_min"], checked["species"])
    ]
    checked["residual"] = checked["predicted"] - checked["value"]
    checked["scaled_residual"] = checked["residual"] / checked["sigma"]
    diagnostic_tables.append(checked)
diagnostics = pd.concat(diagnostic_tables, ignore_index=True)

summary_rows = []
for species, rows in diagnostics.groupby("species"):
    errors = rows["scaled_residual"].to_numpy()
    summary_rows.append({
        "species": species,
        "measurements": len(errors),
        "mean_scaled_error": errors.mean(),
        "RMS_scaled_error": np.sqrt(np.mean(errors ** 2)),
        "largest_absolute_scaled_error": np.max(np.abs(errors)),
    })
summary = pd.DataFrame(summary_rows)
print("\nResidual check (uses the same approximate weights as fitting):")
print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
print("Positive error means the model predicts too high; negative means too low.")
print("Time-zero and nonpositive measurements are excluded, as in the fitting.")

residual_fig, residual_axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
for ax, species in zip(residual_axes.flat, ["A", "B", "P", "Q"]):
    ax.axhspan(-2, 2, color="gray", alpha=0.10)
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(2, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(-2, color="gray", linestyle="--", linewidth=0.8)
    species_rows = diagnostics[diagnostics["species"] == species]
    for run_id, rows in species_rows.groupby("run_id"):
        ax.scatter(rows["time_min"], rows["scaled_residual"], label=run_id, s=22)
    ax.set_title(f"Species {species}")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("(Prediction - measurement) / estimated noise")
    ax.grid(alpha=0.2)
residual_axes.flat[0].legend(title="Experiment", fontsize=8)
residual_fig.suptitle("System A: residuals after fitting M1")
residual_fig.tight_layout()

output = Path(__file__).resolve().parent / "output" / "system_A"
output.mkdir(parents=True, exist_ok=True)
fig.savefig(output / "fitted_curves.png", dpi=180)
residual_fig.savefig(output / "residuals.png", dpi=180)
diagnostics.to_csv(output / "residuals.csv", index=False)
summary.to_csv(output / "residual_summary.csv", index=False)
print(f"\nGraphs and residual tables saved in: {output}")
if "agg" not in plt.get_backend().lower():
    plt.show()
