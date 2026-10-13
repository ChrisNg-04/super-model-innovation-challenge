"""Expanded-bound M6 sensitivity check; preserves the original six-model results."""
import json
from pathlib import Path
import reaction_fitting as fitting
ROOT = Path(__file__).resolve().parent
result = fitting.fit_model(fitting.load_data(), "M6", starts=6, seed=419, rate_bounds=(1e-10,1000), maxiter=700)
result["rate_bounds"] = [1e-10,1000]
result["energy_bounds_kJ_mol"] = [0,200]
path = ROOT / "output/system_B/m6_expanded_bound_review.json"
path.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
