"""Leave-one-experiment-out 90% predictive interval diagnostic for System B.

Uses the existing reaction_fitting.py and 10 previously saved held-out fits.
No held-out measurements are used to fit bootstrap parameters or select model weights.
This is conditional on the preselected M2/M4 pair; it is not independent model selection.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import reaction_fitting as fitting

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output' / 'system_B' / 'validation'
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ('M2', 'M4')
EXPERIMENTS = ('B1', 'B2', 'B3', 'B4', 'B5')
REPETITIONS = 100  # Initial interval estimate; increase to 200+ for a sensitivity check.
STARTS = 3
DRAWS = 4000
LOQ = 3e-5
RNG = np.random.default_rng(20261011)

import sys
sys.path.insert(0, str(ROOT.parent))
from submission_workflow import check_inputs
check_inputs()
data = fitting.load_data()
assert set(EXPERIMENTS) == set(data['run_id'].unique())


def load_fit(excluded, model):
    path = OUT / f'fit_without_{excluded}_{model}.json'
    result = json.loads(path.read_text())['fit']
    assert result['model'] == model
    assert excluded in result['excluded_runs']
    assert excluded not in result['training_runs']
    return result


def bootstrap_one(excluded, model, fit):
    path = OUT / f'bootstrap_full_{excluded}_{model}.csv'
    names = list(fit['parameters'])
    if path.exists():
        saved = pd.read_csv(path)
        if len(saved) >= REPETITIONS and all(name in saved.columns for name in names):
            print(f'Reusing {excluded}/{model}: {len(saved)} rows', flush=True)
            return saved.iloc[:REPETITIONS].copy()
        if len(saved) and not all(name in saved.columns for name in names):
            raise ValueError(f'Unexpected columns in {path}')
        records = saved.to_dict('records')
    else:
        records = []
    training = data[data['run_id'] != excluded].copy()
    assert excluded not in training['run_id'].unique()
    runs = fitting.prepare_runs(training)
    for i in range(len(records), REPETITIONS):
        bootstrap_rng = np.random.default_rng(20261011 + 10000 * EXPERIMENTS.index(excluded) + 1000 * MODELS.index(model) + i)
        artificial = training.copy()
        for run in runs:
            mean = fitting.predict_run(model, fit['parameters'], run)
            sigma = fitting.noise_sigma(mean, run)
            simulated = bootstrap_rng.normal(mean, sigma)
            censored = (run['rows']['species'].to_numpy() == 'Q') & (simulated < LOQ)
            idx = run['rows'].index
            artificial.loc[idx, 'observed'] = simulated
            artificial.loc[idx, 'censored'] = censored
        assert excluded not in artificial['run_id'].unique()
        try:
            result = fitting.fit_model(artificial, model, starts=STARTS,
                                       seed=20261011 + 10000 * EXPERIMENTS.index(excluded)
                                       + 1000 * MODELS.index(model) + i)
            row = dict(repetition=i+1, converged=True,
                       converged_starts=result['converged_starts'],
                       at_parameter_bound=result['at_parameter_bound'],
                       **result['parameters'])
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            row = dict(repetition=i+1, converged=False,
                       converged_starts=0, at_parameter_bound=False,
                       error=str(exc))
        records.append(row)
        pd.DataFrame(records).to_csv(path, index=False)  # Resume after interruption.
        print(f'{excluded}/{model} bootstrap {i+1}/{REPETITIONS}: '
              f'{"OK" if row["converged"] else "FAILED"}', flush=True)
    return pd.DataFrame(records)


results = []
summary = []
for excluded in EXPERIMENTS:
    fits = {model: load_fit(excluded, model) for model in MODELS}
    aic = np.array([fits[model]['AIC'] for model in MODELS])
    weights = np.exp(-.5 * (aic - aic.min()))
    weights /= weights.sum()
    bootstrap = {}
    for model in MODELS:
        samples = bootstrap_one(excluded, model, fits[model])
        good = samples[samples['converged'].astype(str).str.lower() == 'true'].copy()
        names = list(fits[model]['parameters'])
        good = good.dropna(subset=names)
        if len(good) < max(30, REPETITIONS // 2):
            raise RuntimeError(f'Insufficient converged bootstraps for {excluded}/{model}: {len(good)}')
        bootstrap[model] = good
        summary.append(dict(heldout=excluded, model=model, attempted=len(samples),
                            converged=len(good), at_bound=int((good['at_parameter_bound'].astype(str).str.lower() == 'true').sum()),
                            training_AIC_weight=weights[MODELS.index(model)]))
    heldout = fitting.prepare_runs(data[data['run_id'] == excluded])[0]
    model_choices = RNG.choice(len(MODELS), size=DRAWS, p=weights)
    predictions = np.empty((DRAWS, len(heldout['observed'])))
    for j, model in enumerate(MODELS):
        positions = np.flatnonzero(model_choices == j)
        good = bootstrap[model]
        names = list(fits[model]['parameters'])
        choices = RNG.integers(0, len(good), size=len(positions))
        # Cache predictions for identical sampled parameter vectors.
        for choice in np.unique(choices):
            params = {name: float(good.iloc[choice][name]) for name in names}
            mean = fitting.predict_run(model, params, heldout)
            sigma = fitting.noise_sigma(mean, heldout)
            take = positions[choices == choice]
            predictions[take] = np.maximum(0, RNG.normal(mean, sigma, size=(len(take), len(mean))))
    lower, upper = np.quantile(predictions, [.05, .95], axis=0)
    rows = heldout['rows']
    for i, (_, row) in enumerate(rows.iterrows()):
        censored = bool(row['censored'])
        observed = float(row['observed']) if not censored else np.nan
        inside = bool(lower[i] <= observed <= upper[i]) if not censored else np.nan
        # Censored Q means only 'below LOQ' is observed, not an exact concentration.
        results.append(dict(heldout=excluded, species=row['species'], time_min=row['time_min'],
                            censored=censored, observed_M=observed,
                            lower90_M=lower[i], upper90_M=upper[i],
                            inside90=inside,
                            predicted_below_LOQ=float(np.mean(predictions[:, i] < LOQ)) if row['species'] == 'Q' else np.nan))
    pd.DataFrame(results).to_csv(OUT / 'bootstrap_heldout_intervals.csv', index=False)
    pd.DataFrame(summary).to_csv(OUT / 'bootstrap_full_diagnostics.csv', index=False)
    print(f'Completed held-out {excluded}; weights M2={weights[0]:.3f}, M4={weights[1]:.3f}', flush=True)

pred = pd.DataFrame(results)
exact = pred[~pred['censored']].copy()
coverage = exact.groupby('species').agg(n=('inside90', 'size'), coverage=('inside90', 'mean')).reset_index()
coverage.to_csv(OUT / 'bootstrap_coverage_by_species.csv', index=False)
print('\n90% interval empirical coverage on QUANTIFIED held-out observations:')
print(coverage.to_string(index=False))
print(f'Overall: {exact["inside90"].mean():.3f} over {len(exact)} observations')
print('Censored Q observations excluded from exact-value coverage; see predicted_below_LOQ.')
print('Note: timepoints within an experiment are correlated; this is a small five-run diagnostic.')
print('Output folder:', OUT)
