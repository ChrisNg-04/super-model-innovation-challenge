"""Reusable, censor-aware fitting for the supplied reaction mechanisms.

Importing this module does not run a fit or write results. Parameter bounds
are explicit assumptions. Direct array-based ODE evaluation avoids repeatedly
building pandas tables inside the optimisation loop.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.special import log_ndtr

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
from candidates import CANDIDATES

SPECIES = ["A", "B", "P", "Q"]
INITIAL_RATES = {"M1": [.02, .00002], "M2": [.02, .03, .00003], "M3": [.02, .000002],
                 "M4": [.02, .03, .00004], "M5": [.02, .000003], "M6": [.001, 10, .00002]}
RTOL = 1e-7
ATOL = 1e-10


def load_data(path=None):
    data = pd.read_csv(path or PROJECT / "data" / "system_B" / "measurements.csv")
    data["censored"] = data["censored"].astype(str).str.lower().map({"true": True, "false": False})
    if data["censored"].isna().any():
        raise ValueError("Unrecognised censoring flag.")
    data["observed"] = pd.to_numeric(data["value"], errors="coerce")
    if data.loc[~data["censored"], "observed"].isna().any():
        raise ValueError("An uncensored observation is not numeric.")
    if (data.loc[data["censored"], "species"] != "Q").any():
        raise ValueError("Expected censoring only for Q.")
    return data


def prepare_runs(data):
    runs = []
    for run_id, rows in data.groupby("run_id"):
        if len(rows[["temperature_C", "A0_M", "B0_M"]].drop_duplicates()) != 1:
            raise ValueError(f"Inconsistent conditions in {run_id}.")
        first = rows.iloc[0]
        times = np.unique(np.r_[0.0, rows["time_min"].to_numpy()])
        runs.append({"run_id": run_id, "rows": rows.copy(), "times": times,
                     "temperature": float(first["temperature_C"]),
                     "A0": float(first["A0_M"]), "B0": float(first["B0_M"]),
                     "ti": np.searchsorted(times, rows["time_min"].to_numpy()),
                     "si": np.array([SPECIES.index(s) for s in rows["species"]]),
                     "observed": rows["observed"].to_numpy(float),
                     "censored": rows["censored"].to_numpy(bool),
                     "relative": rows["species"].map({"A": .03, "B": .03, "P": .03, "Q": .15}).to_numpy(),
                     "absolute": rows["species"].map({"A": .0035, "B": .0035, "P": .0035, "Q": .000022}).to_numpy()})
    if not runs:
        raise ValueError("No training experiments.")
    return runs


def predict_run(model, parameters, run):
    candidate = CANDIDATES[model]
    y0 = np.array([run["A0"] if s == "A" else run["B0"] if s == "B" else 0.
                   for s in candidate.species])
    if run["times"][-1] == 0:
        values = y0[:, None]
    else:
        solution = solve_ivp(lambda t, y: candidate.rhs(y, parameters, run["temperature"]),
                             (0., float(run["times"][-1])), y0, t_eval=run["times"],
                             method="LSODA", rtol=RTOL, atol=ATOL)
        if not solution.success:
            raise RuntimeError(solution.message)
        values = solution.y
    observed_columns = np.array([candidate.species.index(s) for s in SPECIES])
    curve = np.maximum(values[observed_columns].T, 0.)
    return curve[run["ti"], run["si"]]


def noise_sigma(mean, run):
    return np.sqrt((run["relative"] * mean) ** 2 + run["absolute"] ** 2)


def log_likelihood_rows(mean, run):
    sigma = noise_sigma(mean, run)
    exact = ~run["censored"]
    result = np.empty(len(mean))
    z = (run["observed"][exact] - mean[exact]) / sigma[exact]
    result[exact] = -.5 * z ** 2 - np.log(sigma[exact]) - .5 * np.log(2 * np.pi)
    result[~exact] = log_ndtr((3e-5 - mean[~exact]) / sigma[~exact])
    return result


def unpack(model, values):
    return {name: float(np.exp(value) if name.startswith("k") else value * 100)
            for name, value in zip(CANDIDATES[model].param_names, values)}


def objective(values, model, runs):
    parameters = unpack(model, values)
    try:
        score = -sum(np.sum(log_likelihood_rows(predict_run(model, parameters, run), run)) for run in runs)
    except (ValueError, RuntimeError, FloatingPointError):
        return 1e30
    return float(score) if np.isfinite(score) else 1e30


def fit_model(data, model, exclude_runs=(), starts=3, seed=419, rate_bounds=None,
              energy_bounds=(0, 200), maxiter=400):
    """Fit only the permitted rows, with independent initial guesses.

    No all-data fit or saved bootstrap parameters are used as starting guesses.
    """
    if model not in CANDIDATES or starts < 1:
        raise ValueError("Invalid model or starting-guess count.")
    rate_bounds = rate_bounds or (1e-10, 1000 if model == "M6" else 10)
    if not (0 < rate_bounds[0] < rate_bounds[1] and energy_bounds[0] < energy_bounds[1]):
        raise ValueError("Invalid parameter bounds.")
    excluded = set(exclude_runs)
    unknown = excluded - set(data["run_id"])
    if unknown:
        raise ValueError(f"Unknown excluded experiments: {sorted(unknown)}")
    training = data[~data["run_id"].isin(excluded)].copy()
    runs = prepare_runs(training)
    if set(training["run_id"]) & excluded:
        raise RuntimeError("Excluded observations leaked into training.")
    names = CANDIDATES[model].param_names
    bounds = [(np.log(rate_bounds[0]), np.log(rate_bounds[1])) if name.startswith("k")
              else (energy_bounds[0] / 100, energy_bounds[1] / 100) for name in names]
    initial = np.array([np.log(INITIAL_RATES[model][i // 2]) if name.startswith("k") else .6
                        for i, name in enumerate(names)])
    rng = np.random.default_rng(seed)
    guesses = [initial]
    for _ in range(starts - 1):
        trial = initial.copy()
        trial[0::2] += rng.normal(0, 1.5, len(trial[0::2]))
        trial[1::2] = rng.uniform(.2, 1.2, len(trial[1::2]))
        guesses.append(trial)
    attempts = []
    for guess in guesses:
        guess = np.array([np.clip(value, low + 1e-6, high - 1e-6)
                          for value, (low, high) in zip(guess, bounds)])
        fit = minimize(objective, guess, args=(model, runs), method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "ftol": 1e-10, "maxls": 30})
        attempts.append(fit)
    valid = [fit for fit in attempts if fit.success and np.isfinite(fit.fun) and fit.fun < 1e29]
    if not valid:
        raise RuntimeError(f"No converged {model} fit. Messages: {[str(f.message) for f in attempts]}")
    best = min(valid, key=lambda fit: fit.fun)
    finite = [fit.fun for fit in attempts if np.isfinite(fit.fun) and fit.fun < 1e29]
    return {"model": model, "parameters": unpack(model, best.x), "training_runs": sorted(set(training["run_id"])),
            "excluded_runs": sorted(excluded), "rate_bounds": list(rate_bounds),
            "energy_bounds_kJ_mol": list(energy_bounds), "negative_log_likelihood": float(best.fun),
            "AIC": float(2 * best.fun + 2 * len(names)), "converged": True,
            "converged_starts": len(valid), "attempted_starts": starts,
            "starts_near_best": sum(abs(fit.fun - best.fun) < .05 for fit in valid),
            "start_diagnostics": [{"NLL": float(fit.fun), "converged": bool(fit.success),
                                   "message": str(fit.message)} for fit in attempts],
            "start_NLL_range": float(max(finite) - min(finite)),
            "at_parameter_bound": any(abs(v - low) < .001 or abs(v - high) < .001
                                      for v, (low, high) in zip(best.x, bounds))}
