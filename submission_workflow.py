"""Portable submission output generation from committed analysis artifacts.

Saved bootstrap samples avoid repeating expensive fitting. Five-run conditional
coverage checks are included; extrapolation calibration remains unverified.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

# Conda-based Windows Python distributions need their native DLL directory
# when launched from a plain terminal instead of an activated environment.
# Detect it from this interpreter; ordinary Python installations need no change.
_DLL_HANDLES = []
if os.name == "nt":
    runtime_paths = []
    for prefix in dict.fromkeys([sys.prefix, sys.base_prefix]):
        native = Path(prefix) / "Library" / "bin"
        if native.is_dir():
            runtime_paths.extend([str(Path(prefix)), str(native), str(Path(prefix) / "Scripts")])
            if hasattr(os, "add_dll_directory"):
                _DLL_HANDLES.append(os.add_dll_directory(str(native)))
    if runtime_paths:
        os.environ["PATH"] = os.pathsep.join(runtime_paths + [os.environ.get("PATH", "")])

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis"
SUBMISSION = ROOT / "submission"
FIGURES = SUBMISSION / "figures"


def check_inputs():
    manifest = json.loads((ANALYSIS / "artifacts/saved_fit_manifest.json").read_text())
    for relative, expected in manifest.items():
        path = ROOT / relative
        normalized = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        if hashlib.sha256(normalized).hexdigest() != expected:
            raise RuntimeError(f"Saved-fit inputs changed: {relative}. Refit/update artifacts before generating outputs.")
    SUBMISSION.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)


def run_analysis(name):
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    subprocess.run([sys.executable, str(ANALYSIS / name)], cwd=ROOT, env=environment, check=True)


def predict():
    import pandas as pd
    run_analysis("regenerate_predictions.py")
    source = ANALYSIS / "output/system_B/regenerated_predictions.csv"
    predictions = pd.read_csv(source)
    conditions = pd.read_csv(ROOT / "data/heldout_conditions.csv")
    expected = {(row.run_id, float(row.time_min), species)
                for row in conditions.itertuples() for species in ["A", "B", "P", "Q"]}
    actual = {(row.run_id, float(row.time_min), row.species) for row in predictions.itertuples()}
    if actual != expected or len(predictions) != len(expected):
        raise RuntimeError("Prediction rows do not match requested held-out conditions.")
    shutil.copy2(source, SUBMISSION / "predictions.csv")
    write_yaml(SUBMISSION / "hypotheses.yaml", json.loads((ANALYSIS / "artifacts/hypotheses.json").read_text()))
    print(f"Prediction output checked: {len(predictions)} rows.")


def write_yaml(path, content):
    import yaml
    path.write_text(yaml.safe_dump(content, sort_keys=False, allow_unicode=True), encoding="utf-8")


def suggest():
    campaign = json.loads((ANALYSIS / "artifacts/selected_campaign.json").read_text())
    experiments = campaign["experiments"]
    material = sum(1.25 * experiment["A0_M"] for experiment in experiments)
    samples = sum(len(experiment["sample_times_min"]) for experiment in experiments)
    if not (3 <= len(experiments) <= 10 and material <= 5 and samples <= 60):
        raise RuntimeError("Campaign exceeds budget.")
    if len({experiment["run_id"] for experiment in experiments}) != len(experiments):
        raise RuntimeError("Duplicate experiment IDs.")
    for experiment in experiments:
        times = experiment["sample_times_min"]
        if not (50 <= experiment["temperature_C"] <= 120 and .1 <= experiment["A0_M"] <= 1.5
                and .1 <= experiment["B0_M"] <= 1.5 and times == sorted(set(times)) and times
                and all(0 <= t <= 960 for t in times) and set(experiment["assay"]) == {"A", "B", "P", "Q"}):
            raise RuntimeError(f"Invalid conditions in {experiment['run_id']}.")
    write_yaml(SUBMISSION / "campaign.yaml", campaign)
    print(f"Selected matched-control campaign: {material:.3f} g A, {samples} samples, {len(experiments)} experiments.")


def draw_saved_fit_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(ROOT / "src"))
    from candidates import Conditions, simulate
    parameters = json.loads((ANALYSIS / "output/system_B/fitted_parameters.json").read_text())
    bootstrap = json.loads((ANALYSIS / "output/system_B/bootstrap_parameters.json").read_text())
    data = pd.read_csv(ROOT / "data/system_B/measurements.csv")
    data["numeric"] = pd.to_numeric(data["value"], errors="coerce")
    runs = list(data.groupby("run_id"))
    fig, axes = plt.subplots(len(runs), 2, figsize=(12, 3 * len(runs)), squeeze=False)
    colors = {"A": "tab:blue", "B": "tab:orange", "P": "tab:green", "Q": "tab:red"}
    for row_index, (run_id, rows) in enumerate(runs):
        first = rows.iloc[0]
        conditions = Conditions(float(first["temperature_C"]), float(first["A0_M"]), float(first["B0_M"]),
                                np.linspace(0, rows["time_min"].max(), 160))
        for column, species_list in enumerate((["A", "B", "P"], ["Q"])):
            ax = axes[row_index, column]
            for species in species_list:
                measured = rows[(rows["species"] == species) & rows["numeric"].notna()]
                ax.scatter(measured["time_min"], measured["numeric"], color=colors[species], s=15)
            for model, style in zip(["M2", "M4"], ["--", "-"]):
                curve = simulate(model, conditions, parameters[model])
                for species in species_list:
                    ax.plot(curve["time_min"], curve[species], style, color=colors[species], label=f"{species}: {model}")
            if column == 1:
                censored = rows[(rows["species"] == "Q") & rows["numeric"].isna()]
                ax.axhline(3e-5, color="gray", linestyle=":", label="Q LOQ")
                ax.scatter(censored["time_min"], np.full(len(censored), 3e-5), marker="v", facecolors="none", edgecolors="tab:red")
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.set_title(f"{run_id}: {conditions.temperature_c:g} C")
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("Concentration (mol/L)")
            ax.legend(fontsize=7)
            ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "system_B_leading_fits.png", dpi=180)
    plt.close(fig)

    campaign = json.loads((ANALYSIS / "artifacts/selected_campaign.json").read_text())
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    for ax, experiment in zip(axes, campaign["experiments"]):
        end = max(experiment["sample_times_min"])
        conditions = Conditions(experiment["temperature_C"], experiment["A0_M"], experiment["B0_M"],
                                np.linspace(0, end, 160))
        for model in ["M2", "M4"]:
            curves = np.array([simulate(model, conditions, p)["Q"].to_numpy() for p in bootstrap[model]])
            low, middle, high = np.quantile(curves, [.05, .5, .95], axis=0)
            ax.plot(conditions.times_min, middle, label=model)
            ax.fill_between(conditions.times_min, low, high, alpha=.2)
        for t in experiment["sample_times_min"]:
            ax.axvline(t, color="gray", alpha=.2)
        ax.axhline(3e-5, color="gray", linestyle=":", label="Q LOQ")
        ax.set_title(f"{experiment['run_id']}: {experiment['temperature_C']:g} C, A0={experiment['A0_M']:g}, B0={experiment['B0_M']:g} M")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Q concentration (mol/L)")
        ax.legend()
        ax.grid(alpha=.2)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.suptitle("Selected matched-control campaign - parameter uncertainty, no assay noise")
    fig.tight_layout()
    fig.savefig(FIGURES / "selected_campaign_Q_profiles.png", dpi=180)
    plt.close(fig)


def replicate():
    for name in ["fit_system_A.py", "explore_system_B.py", "check_system_B_fits.py",
                 "regenerate_predictions.py", "design_campaign.py", "stress_test_campaign.py",
                 "review_sample_efficiency.py", "report_validation.py", "validate_bootstrap_full.py", "report_validation.py", "report_m6.py"]:
        run_analysis(name)
    mappings = {
        "system_A/fitted_curves.png": "system_A_fitted_curves.png",
        "system_A/residuals.png": "system_A_residuals.png",
        "system_B/measurements_overview.png": "system_B_measurements.png",
        "system_B/leading_model_residuals.png": "system_B_residuals.png",
        "system_B/regenerated_prediction_example.png": "heldout_prediction_example.png",
        "campaign_design/strategy_comparison.png": "campaign_strategy_comparison.png",
        "campaign_stress_test/campaign_stress_comparison.png": "campaign_stress_comparison.png",
        "campaign_stress_test/matched_A_Q_to_P_ratio.png": "matched_A_Q_to_P_ratio.png",
        "sample_reduction/sample_reduction_stress.png": "sample_reduction_stress.png",
        "sample_reduction/sampling_schedules.png": "sampling_schedules.png",
        "system_B/validation/validation_coverage.png": "validation_coverage.png",
        "system_B/m6_review.png": "m6_review.png",
    }
    for source, target in mappings.items():
        shutil.copy2(ANALYSIS / "output" / source, FIGURES / target)
    draw_saved_fit_figures()
    print("Regenerated 14 supported figures in submission/figures.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["predict", "suggest", "replicate", "all"], nargs="?", default="all")
    command = parser.parse_args().command
    check_inputs()
    started = time.perf_counter()
    actions = {"predict": predict, "suggest": suggest, "replicate": replicate}
    for name in (["predict", "suggest", "replicate"] if command == "all" else [command]):
        actions[name]()
    print(f"Workflow {command} completed in {time.perf_counter() - started:.1f} seconds.")
    print("Five-run interval validation is included; extrapolation coverage remains unverified.")


if __name__ == "__main__":
    main()
