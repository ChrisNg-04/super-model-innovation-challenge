"""Audit saved training-only validation artifacts and regenerate their coverage report."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import reaction_fitting as fitting
out = Path(__file__).resolve().parent / "output/system_B/validation"
data = fitting.load_data()
for excluded in sorted(data.run_id.unique()):
    for model in ("M2", "M4"):
        fit = json.loads((out / f"fit_without_{excluded}_{model}.json").read_text())["fit"]
        if set(fit["training_runs"]) != set(data.run_id.unique()) - {excluded} or excluded not in fit["excluded_runs"]:
            raise ValueError("Invalid training/test separation")
        training = data[data.run_id != excluded]
        runs = fitting.prepare_runs(training)
        score = -sum(fitting.log_likelihood_rows(fitting.predict_run(model, fit["parameters"], run),run).sum() for run in runs)
        if abs(score - fit["negative_log_likelihood"]) > .01:
            raise ValueError("Saved fold likelihood does not reproduce")
        samples = pd.read_csv(out / f"bootstrap_full_{excluded}_{model}.csv")
        good = samples.converged.astype(str).str.lower().eq("true")
        if len(samples) != 100 or not good.all() or samples[list(fit["parameters"])].isna().any().any():
            raise ValueError("Incomplete incoming validation bootstrap")
rows = pd.read_csv(out / "bootstrap_heldout_intervals.csv")
actual = {(str(x.heldout), float(x.time_min),str(x.species)) for x in rows.itertuples()}
expected = {(str(x.run_id),float(x.time_min),str(x.species)) for x in data.itertuples()}
if actual != expected or len(rows) != len(data):
    raise ValueError("Validation rows do not match measurements")
joined = rows.merge(data, left_on=["heldout","time_min","species"],right_on=["run_id","time_min","species"],suffixes=("_report","_data"),validate="one_to_one")
if not (joined.censored_report == joined.censored_data).all():
    raise ValueError("Censoring flags changed")
exact = joined[~joined.censored_data].copy()
if not np.allclose(exact.observed_M,exact.observed):
    raise ValueError("Validation observations changed")
exact["inside90"] = (exact.lower90_M <= exact.observed) & (exact.observed <= exact.upper90_M)
summary = exact.groupby("species").agg(n=("inside90","size"),coverage=("inside90","mean")).reset_index()
summary.to_csv(out / "bootstrap_coverage_by_species.csv",index=False)
fig, ax = plt.subplots(figsize=(7,4))
ax.bar(summary.species,summary.coverage)
ax.axhline(.9,color="black",linestyle="--",label="Nominal 90%")
ax.set_ylim(0,1);ax.set_ylabel("Fraction of quantified held-out measurements covered")
ax.set_title("Five-run validation, conditional on M2/M4")
ax.legend();fig.tight_layout();fig.savefig(out / "validation_coverage.png",dpi=180);plt.close(fig)
print(summary.to_string(index=False))
print("Q excludes censored values; five correlated experiments do not establish extrapolation calibration.")
