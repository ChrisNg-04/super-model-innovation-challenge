"""Reproduce M6 residual evidence from the reviewed saved fit, without refitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import reaction_fitting as f
from candidates import arrhenius
out=Path(__file__).resolve().parent / "output/system_B"
review=json.loads((out/"m6_expanded_bound_review.json").read_text())
params=review["parameters"]
records=[]
fig, axes=plt.subplots(1,2,figsize=(10,4))
for run in f.prepare_runs(f.load_data()):
    mean=f.predict_run("M6",params,run)
    exact=~run["censored"]
    rows=run["rows"].copy()
    rows["scaled_residual"]=(mean-run["observed"])/f.noise_sigma(mean,run)
    records.extend(rows[exact].to_dict("records"))
    a=rows.species.eq("A").to_numpy()
    expected=run["A0"]*np.exp(-arrhenius(params["k1"],params["Ea1"],run["temperature"])*rows.loc[a,"time_min"].to_numpy())
    assert np.allclose(mean[a],expected,atol=1e-6,rtol=1e-6)
    axes[0].plot(rows.loc[a,"time_min"],mean[a],label=run["run_id"])
    axes[0].scatter(rows.loc[a,"time_min"],rows.loc[a,"observed"],s=12)
summary=pd.DataFrame(records).groupby("species").scaled_residual.agg(mean="mean",RMS=lambda x:np.sqrt(np.mean(x**2)))
summary.to_csv(out/"m6_review_residual_summary.csv")
axes[0].set(xlabel="Time (min)",ylabel="A concentration (mol/L)",title="Reviewed M6 curves and measured A");axes[0].legend()
axes[1].bar(summary.index,summary.RMS);axes[1].axhline(1,color="black",linestyle="--")
axes[1].set(ylabel="RMS residual in assay-noise units",title="Quantified measurements only")
fig.suptitle("M6: poor fit plus remaining numerical sensitivity")
fig.tight_layout();fig.savefig(out/"m6_review.png",dpi=180)
print(summary.to_string())
