"""Inspect the blind data before fitting any candidate mechanisms."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

project = Path(__file__).resolve().parent.parent
data = pd.read_csv(project / "data" / "system_B" / "measurements.csv")

# Keep below-limit measurements distinct from ordinary numeric results.
data["censored"] = data["censored"].astype(str).str.lower().map({"true": True, "false": False})
if data["censored"].isna().any():
    raise ValueError("Unrecognised value in the censored column.")
data["concentration"] = pd.to_numeric(data["value"], errors="coerce")
if data.loc[~data["censored"], "concentration"].isna().any():
    raise ValueError("An uncensored measurement has no numeric concentration.")

conditions = data.groupby("run_id")[["temperature_C", "A0_M", "B0_M"]].first()
print("System B experimental conditions:")
print(conditions.to_string())
print("\nMeasurement counts:")
print(data.groupby(["run_id", "species"]).size().unstack(fill_value=0).to_string())
print("\nBelow-limit measurements:")
print(data.groupby("run_id")["censored"].sum().to_string())
print("\nMissing rows mean no measurement. Below-limit rows are measurements, not zeros.")

run_ids = list(conditions.index)
fig, axes = plt.subplots(len(run_ids), 2, figsize=(12, 3 * len(run_ids)), squeeze=False)
colors = {"A": "tab:blue", "B": "tab:orange", "P": "tab:green"}
loq = 3.0e-5

for row_index, run_id in enumerate(run_ids):
    run = data[data["run_id"] == run_id]
    main_ax, trace_ax = axes[row_index]
    temperature = conditions.loc[run_id, "temperature_C"]

    for species, color in colors.items():
        rows = run[(run["species"] == species) & ~run["censored"]].sort_values("time_min")
        main_ax.plot(rows["time_min"], rows["concentration"], "o-", color=color, label=species, markersize=4)
    main_ax.set_title(f"{run_id}: {temperature:g} deg C - A, B and P")
    main_ax.set_ylabel("Concentration (mol/L)")
    main_ax.legend()

    q = run[run["species"] == "Q"].sort_values("time_min")
    quantified = q[~q["censored"]]
    below_limit = q[q["censored"]]
    # Plot Q as points because it was only assayed at selected times.
    trace_ax.scatter(quantified["time_min"], quantified["concentration"], color="tab:red", label="Quantified Q")
    trace_ax.axhline(loq, color="gray", linestyle="--", label="Quantitation limit")
    if not below_limit.empty:
        trace_ax.scatter(below_limit["time_min"], np.full(len(below_limit), loq),
                         marker="v", facecolors="none", edgecolors="tab:red",
                         label="Q below limit (marker is not its value)")
    trace_ax.set_title(f"{run_id}: impurity Q")
    trace_ax.set_ylabel("Q concentration (mol/L)")
    trace_ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    trace_ax.set_ylim(bottom=0)
    trace_ax.legend(fontsize=8)

    for ax in (main_ax, trace_ax):
        ax.set_xlabel("Time (min)")
        ax.grid(alpha=0.2)

fig.suptitle("System B: observed data only - no mechanism fitted yet")
fig.tight_layout(rect=(0, 0, 1, 0.98))
output = Path(__file__).resolve().parent / "output" / "system_B"
output.mkdir(parents=True, exist_ok=True)
fig.savefig(output / "measurements_overview.png", dpi=180)
conditions.to_csv(output / "historical_conditions.csv")
print(f"\nOverview saved in: {output}")
print("Below-limit Q markers are drawn at the limit only to indicate an upper bound.")
if "agg" not in plt.get_backend().lower():
    plt.show()
