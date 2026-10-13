"""Numerical and scientific regression checks; no refitting required."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from scipy.special import log_ndtr
import reaction_fitting as f
from candidates import Conditions, simulate
root = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("candidate_tests",root / "tests/test_candidates.py")
t = importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
count=0
for name in dir(t):
    if name.startswith("test_"):
        getattr(t,name)();count+=1
params=json.loads((root/"analysis/output/system_B/fitted_parameters.json").read_text())
runs=f.prepare_runs(f.load_data())
for model in f.CANDIDATES:
    for run in runs:
        official=simulate(model,Conditions(run["temperature"],run["A0"],run["B0"],run["times"]),params[model])
        wanted=official[f.SPECIES].to_numpy()[run["ti"],run["si"]]
        predicted=f.predict_run(model,params[model],run)
        if np.max(np.abs(wanted-predicted)/f.noise_sigma(predicted,run)) > .001:
            raise AssertionError(f"ODE implementation differs: {model}")
run=runs[0];mean=f.predict_run("M2",params["M2"],run)
ll=f.log_likelihood_rows(mean,run);mask=run["censored"]
assert mask.any()
assert np.allclose(ll[mask],log_ndtr((3e-5-mean[mask])/f.noise_sigma(mean,run)[mask]))
try:
    f.fit_model(f.load_data(),"M2",exclude_runs=("UNKNOWN",),starts=1)
except ValueError:
    pass
else:
    raise AssertionError("Unknown excluded run accepted")
print(f"Passed {count} supplied candidate tests, all-six-model ODE equivalence, censor likelihood and exclusion checks.")
