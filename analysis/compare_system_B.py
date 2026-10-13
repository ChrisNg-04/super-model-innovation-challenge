"""Refit all six candidates without overwriting the validated prediction artifacts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import reaction_fitting as fitting
out = Path(__file__).resolve().parent / "output/refit_system_B"
out.mkdir(parents=True, exist_ok=True)
data = fitting.load_data()
results = []
for model in fitting.CANDIDATES:
    print(f"Fitting {model}...", flush=True)
    result = fitting.fit_model(data, model, starts=6 if model == "M6" else 4,
                              rate_bounds=(1e-10,1000 if model == "M6" else 10), maxiter=700)
    results.append(result)
    (out / f"{model}.json").write_text(json.dumps(result, indent=2))
ranking = pd.DataFrame([{k:v for k,v in result.items() if k not in
                         ("parameters","start_diagnostics","training_runs","excluded_runs")} for result in results])
ranking = ranking.sort_values("AIC")
ranking["delta_AIC"] = ranking.AIC - ranking.AIC.min()
w = np.exp(-.5 * ranking.delta_AIC.to_numpy())
ranking["relative_AIC_weight"] = w / w.sum()
ranking.to_csv(out / "model_comparison.csv", index=False)
print(ranking.to_string(index=False))
print("New fits saved separately. Rebuild bootstrap samples and validation before promoting changed fits.")
