"""Person 4: reproducible, independent sample-time reduction assessment.

Run from the handoff root: python review_sample_efficiency.py
Uses the committed three-run matched-A campaign, M2/M4 bootstrap ensemble,
Gaussian assay noise and Gaussian Q left-censoring at 3e-5 M.
Optimises the sampling subset on a training Monte Carlo seed and reports results
on independent evaluation seeds. Does NOT rewrite submission/campaign.yaml.
"""
from __future__ import annotations
import hashlib
import itertools
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import log_ndtr, logsumexp, xlogy

import reaction_fitting as fitting

BASE = Path(__file__).resolve().parent
OUT = BASE / 'output' / 'sample_reduction'
OUT.mkdir(parents=True, exist_ok=True)
SAVED = BASE / 'output' / 'system_B'
ROOT = fitting.PROJECT
# Frozen 18-sample reference: remains valid if the official selected campaign is later changed.
CAMPAIGN_FILE = OUT / 'reference_matched_A_18.json'
MODELS = ('M2', 'M4')
REL = np.array([.03, .03, .03, .15])
ABS = np.array([.0035, .0035, .0035, 2.2e-5])
Q_LOQ = 3e-5
ENSEMBLE = json.loads((SAVED / 'bootstrap_parameters.json').read_text())
CAMPAIGN = json.loads(CAMPAIGN_FILE.read_text())
EXPERIMENTS = CAMPAIGN['experiments']
assert len(EXPERIMENTS) == 3, 'Current review assumes 3-run matched-A campaign.'
assert len(EXPERIMENTS[0]['sample_times_min']) == len(EXPERIMENTS[1]['sample_times_min']) == len(EXPERIMENTS[2]['sample_times_min']) == 6
assert EXPERIMENTS[1]['sample_times_min'] == EXPERIMENTS[2]['sample_times_min'], 'E2 and E3 must be matched.'
N = len(ENSEMBLE['M2'])
assert N == len(ENSEMBLE['M4']) == 60, 'Expect the saved 60-draw bootstrap ensemble.'
RANK = pd.read_csv(SAVED / 'model_comparison.csv').set_index('model')
PRIOR = RANK.loc[list(MODELS), 'relative_AIC_weight'].to_numpy(float)
PRIOR = PRIOR / PRIOR.sum()
H0 = -np.sum(xlogy(PRIOR, PRIOR))


def param_variation(model, factor):
    names = fitting.CANDIDATES[model].param_names
    m = np.array([[np.log(p[n]) if n.startswith('k') else p[n] / 100 for n in names]
                  for p in ENSEMBLE[model]], float)
    varied = m.mean(axis=0) + factor * (m - m.mean(axis=0))
    lo = np.array([np.log(1e-10) if n.startswith('k') else 0 for n in names])
    hi = np.array([np.log(10) if n.startswith('k') else 2 for n in names])
    clipped = np.clip(varied, lo, hi)
    return [fitting.unpack(model, row) for row in clipped], int(np.any(varied != clipped, axis=1).sum())


def prediction_ensemble(spread):
    """(2 models, 60 bootstrap draws, 3 experiments, 6 times, 4 assays)."""
    cache_key = hashlib.sha256(b''.join(p.read_bytes() for p in [CAMPAIGN_FILE, SAVED / 'bootstrap_parameters.json',
                     BASE / 'reaction_fitting.py', fitting.PROJECT / 'src/candidates/models.py']) + str(spread).encode()).hexdigest()[:18]
    cache = OUT / ('cache_means_' + cache_key + '.npz')
    if cache.exists():
        with np.load(cache) as loaded:
            return loaded['means'], loaded['clipped'].tolist()
    curves, clipping = [], []
    for model in MODELS:
        params, n_clip = param_variation(model, spread)
        clipping.append(n_clip)
        group = []
        for p in params:
            values = []
            for exp in EXPERIMENTS:
                ts = np.array(exp['sample_times_min'], float)
                time_grid = np.unique(np.r_[0., ts])
                run = {'temperature': float(exp['temperature_C']), 'A0': float(exp['A0_M']),
                       'B0': float(exp['B0_M']), 'times': time_grid,
                       'ti': np.repeat(np.searchsorted(time_grid, ts), 4),
                       'si': np.tile(np.arange(4), len(ts))}
                values.append(fitting.predict_run(model, p, run).reshape(len(ts), 4))
            group.append(values)
        curves.append(group)
    result = np.array(curves, float)
    np.savez_compressed(cache, means=result, clipped=clipping)
    return result, clipping


def simulated_likelihood(means, noise_mult, worlds_per_model, seed, species_idx=(0, 1, 2, 3)):
    """Per-sampling-time joint predictive log likelihood; shared parameter per whole campaign.

    Shape: (3 runs, 6 times, 2*worlds_per_model worlds, 120 hypothesis parameter draws).
    The same simulated world/assay-noise realisation is used by EVERY competing subset.
    """
    rng = np.random.default_rng(seed)
    truth = np.repeat([0, 1], worlds_per_model)
    world_index = rng.integers(0, N, size=2 * worlds_per_model)
    sigma = noise_mult * np.sqrt((means * REL)**2 + ABS**2)
    actual_mean = means[truth, world_index]
    actual_sigma = sigma[truth, world_index]
    observed = actual_mean + actual_sigma * rng.standard_normal(size=actual_mean.shape)
    # (worlds, 3 experiments, 6 times, 4 assays)
    mu = means.reshape(2 * N, 3, 6, 4)
    scale = sigma.reshape(2 * N, 3, 6, 4)
    components = []
    for run in range(3):
        steps = []
        for sample in range(6):
            ob = observed[:, run, sample, species_idx]
            center = mu[:, run, sample, :][:, species_idx]
            sd = scale[:, run, sample, :][:, species_idx]
            z = (ob[:, None, :] - center[None, :, :]) / sd[None, :, :]
            ll = -.5*z**2 - np.log(sd[None, :, :]) - .5*np.log(2*np.pi)
            if 3 in species_idx:
                qcol = tuple(species_idx).index(3)
                censored = ob[:, qcol] < Q_LOQ
                ll[:, :, qcol] = np.where(censored[:, None],
                                          log_ndtr((Q_LOQ - center[None, :, qcol]) / sd[None, :, qcol]),
                                          ll[:, :, qcol])
            steps.append(ll.sum(axis=2))
        components.append(np.stack(steps, axis=0))
    return np.stack(components, axis=0), truth


def score_joint(joint, truth, worlds_per_model):
    prob_log = np.column_stack([
        logsumexp(joint[:, i*N:(i+1)*N], axis=1) - np.log(N) + np.log(PRIOR[i])
        for i in range(2)])
    posterior = np.exp(prob_log - logsumexp(prob_log, axis=1)[:, None])
    h = -np.sum(xlogy(posterior, posterior), axis=1)
    world_weights = np.repeat(PRIOR / worlds_per_model, worlds_per_model)
    frac = (H0 - (world_weights*h).sum()) / H0
    accuracy = (world_weights * (posterior.argmax(axis=1) == truth)).sum()
    # MC standard error: weighted variance of the conditional entropy per truth model.
    se_h = np.sqrt(sum(PRIOR[i]**2 * h[i*worlds_per_model:(i+1)*worlds_per_model].var(ddof=1) / worlds_per_model
                       for i in range(2))) / H0
    return float(frac), float(accuracy), float(se_h)


def extract(components, indices):
    e1, matched = indices
    return (components[0, list(e1)].sum(axis=0) +
            components[1, list(matched)].sum(axis=0) +
            components[2, list(matched)].sum(axis=0))


def dropout_scores(components, indices, truth, n_worlds):
    e1, matched = indices
    contributions = [components[0, list(e1)].sum(axis=0),
                     components[1, list(matched)].sum(axis=0),
                     components[2, list(matched)].sum(axis=0)]
    leave = [score_joint(sum(v for j, v in enumerate(contributions) if j != i), truth, n_worlds)
             for i in range(3)]
    return min(x[0] for x in leave), min(x[1] for x in leave)


def eligible_combinations(exp, k):
    times = np.array(exp['sample_times_min'])
    valid = []
    for indices in itertools.combinations(range(6), k):
        selected = times[list(indices)]
        if not np.any(selected <= 10) or not np.any(selected >= 40):
            continue
        valid.append(indices)
    return valid


def select_schedule(k, training_components, truth, n_worlds):
    """Exhaustive discrete subset search, held-to 4/3/2 samples per run.

    Maximises a pre-declared conservative score: fraction uncertainty removed under
    noise x3/parameter spread x2, plus 0.15 * worst leave-one-experiment-out
    fraction uncertainty removed.  The independent evaluation seed is not used
    to pick the schedule.
    """
    options_e1 = eligible_combinations(EXPERIMENTS[0], k)
    options_pair = eligible_combinations(EXPERIMENTS[1], k)
    best = (-np.inf, None, None)
    c = training_components
    for e1 in options_e1:
        part_e1 = c[0, list(e1)].sum(axis=0)
        for pair in options_pair:
            part_e2 = c[1, list(pair)].sum(axis=0)
            part_e3 = c[2, list(pair)].sum(axis=0)
            j = part_e1 + part_e2 + part_e3
            frac, accuracy, _ = score_joint(j, truth, n_worlds)
            dropout_frac = min(score_joint(j - p, truth, n_worlds)[0] for p in (part_e1, part_e2, part_e3))
            objective = frac + 0.15 * dropout_frac
            if objective > best[0] + 1e-12:
                best = (objective, (e1, pair), (frac, accuracy, dropout_frac))
    return best


def schedule_to_experiments(indices):
    out = []
    for i, exp in enumerate(EXPERIMENTS):
        copy = {k: v for k, v in exp.items()}
        copy['sample_times_min'] = [exp['sample_times_min'][j] for j in indices[0 if i == 0 else 1]]
        out.append(copy)
    return out


def validate_campaign(experiments):
    assert 3 <= len(experiments) <= 10
    cost = sum(1.25 * float(r['A0_M']) for r in experiments)
    samples = sum(len(r['sample_times_min']) for r in experiments)
    assert cost <= 5 and samples <= 60
    assert len({r['run_id'] for r in experiments}) == len(experiments)
    for e in experiments:
        assert 50 <= e['temperature_C'] <= 120 and .1 <= e['A0_M'] <= 1.5 and .1 <= e['B0_M'] <= 1.5
        assert e['sample_times_min'] == sorted(set(e['sample_times_min']))
        assert all(0 <= t <= 960 for t in e['sample_times_min'])
        assert set(e['assay']) == set(fitting.SPECIES)
    return float(cost), int(samples)


def main():
    tic = time.perf_counter()
    print('Sampling efficiency: preparing M2/M4 bootstrap predictions...', flush=True)
    ensemble_2, clipped = prediction_ensemble(2.0)
    train_n = 150
    train_c, train_truth = simulated_likelihood(ensemble_2, 3., train_n, 199304)
    schedules = {}
    for k in (6, 4, 3, 2):
        if k == 6:
            sched = (tuple(range(6)), tuple(range(6)))
            info = (None, None, None)
        else:
            _, sched, info = select_schedule(k, train_c, train_truth, train_n)
        schedules[k] = sched
        experiments = schedule_to_experiments(sched)
        validate_campaign(experiments)
        print(f'{k} samples/run: E1={experiments[0]["sample_times_min"]}, E2/E3={experiments[1]["sample_times_min"]} (train {info})', flush=True)

    # Independent evaluation: distributions/noise scenarios, distinct RNG seed.
    scenarios = [(spread, noise) for spread in (1., 2.) for noise in (1., 2., 3.)]
    records, clips = [], {}
    for spread, noise in scenarios:
        ensemble, nclip = prediction_ensemble(spread)
        clips[spread] = nclip
        n_worlds = 1200
        terms, truth = simulated_likelihood(ensemble, noise, n_worlds, 620327 + int(spread)*100 + int(noise))
        terms_q, truth_q = simulated_likelihood(ensemble, noise, n_worlds,
                                               620327 + int(spread)*100 + int(noise), species_idx=(3,))
        for k, sched in schedules.items():
            frac, acc, se = score_joint(extract(terms, sched), truth, n_worlds)
            dropout_frac, dropout_acc = dropout_scores(terms, sched, truth, n_worlds)
            q_frac, q_acc, _ = score_joint(extract(terms_q, sched), truth_q, n_worlds)
            records.append({'samples_per_run': k, 'samples_total': k*3,
                            'parameter_spread_multiplier': spread, 'noise_multiplier': noise,
                            'fraction_uncertainty_removed': frac, 'information_MC_se': se,
                            'simulated_classification': acc, 'worst_dropout_information': dropout_frac,
                            'worst_dropout_classification': dropout_acc,
                            'Q_only_fraction_uncertainty_removed': q_frac, 'Q_only_classification': q_acc,
                            'A_material_g': sum(1.25 * r['A0_M'] for r in EXPERIMENTS)})
        print(f'Independent evaluation completed: parameter x{spread:g}, assay noise x{noise:g}', flush=True)

    df = pd.DataFrame(records).sort_values(['samples_total', 'parameter_spread_multiplier', 'noise_multiplier'], ascending=[False, True, True])
    df.to_csv(OUT / 'independent_scenario_comparison.csv', index=False)
    summaries = []
    for k, records_k in df.groupby('samples_per_run', sort=False):
        summaries.append({'samples_per_run': int(k), 'samples_total': int(k*3),
                          'min_information_removed': records_k.fraction_uncertainty_removed.min(),
                          'min_classification': records_k.simulated_classification.min(),
                          'min_dropout_classification': records_k.worst_dropout_classification.min(),
                          'nominal_information_removed': records_k[(records_k.noise_multiplier==1)&(records_k.parameter_spread_multiplier==1)].fraction_uncertainty_removed.iloc[0],
                          'nominal_classification': records_k[(records_k.noise_multiplier==1)&(records_k.parameter_spread_multiplier==1)].simulated_classification.iloc[0],
                          'A_material_g': 1.5})
    summary = pd.DataFrame(summaries).sort_values('samples_total', ascending=False)
    summary.to_csv(OUT / 'sample_reduction_summary.csv', index=False)

    # Save explicit independent candidate schedules; default final submission left untouched.
    import yaml
    for k, indices in schedules.items():
        candidate = dict(CAMPAIGN)
        candidate['rationale'] = (f'PERSON4 ALTERNATIVE (not automatically selected): {3*k} analytical samples, '
           'same three reactions and material as official 18-sample matched-A campaign. '
           'Time-point subset chosen by independent bootstrap simulation, with E2/E3 paired '
           'and each run retaining early and late samples. Refer to the independent stress-test report.')
        candidate['experiments'] = schedule_to_experiments(indices)
        yaml_path = OUT / f'matched_control_{3*k:02d}_samples.yaml'
        yaml_path.write_text(yaml.safe_dump(candidate, sort_keys=False), encoding='utf-8')

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for k, rows in df.groupby('samples_per_run'):
        rows = rows[rows.parameter_spread_multiplier==2].sort_values('noise_multiplier')
        ax.plot(rows.noise_multiplier, 100*rows.fraction_uncertainty_removed,
                marker='o', label=f'{k*3} samples ({k}/run)')
    ax.set(xlabel='Assay noise multiplier', ylabel='Fraction of model uncertainty removed (%)',
           title='Independent M2/M4 model-discrimination stress test')
    ax.set_xticks([1, 2, 3]); ax.set_ylim(0, 105)
    ax.grid(alpha=.25); ax.legend(title='Parameter spread x2')
    fig.tight_layout(); fig.savefig(OUT / 'sample_reduction_stress.png', dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for row_index, e in enumerate(EXPERIMENTS):
        y = len(EXPERIMENTS)-row_index
        for k in (6, 4, 3, 2):
            row = schedules[k][0 if row_index==0 else 1]
            t = np.array(e['sample_times_min'])[list(row)]
            ax.scatter(t, np.full(len(t), y + (k-4)*.055), label=f'{3*k} samples' if row_index==0 else None, s=45)
    ax.set_xscale('symlog', linthresh=3)
    ax.set_yticks([1, 2, 3]); ax.set_yticklabels(['E3 (matched A)', 'E2 (low A)', 'E1 (95 C)'])
    ax.set_xlabel('Sampling time (min)'); ax.set_title('Proposed retained sampling times')
    ax.grid(alpha=.2); ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(.5, -.15))
    fig.tight_layout(); fig.savefig(OUT / 'sampling_schedules.png', dpi=170, bbox_inches='tight'); plt.close(fig)

    report = ['# Person 4: Sample-reduction assessment (independent simulations)', '',
     '## Scope and method', '',
     'The reference three reaction conditions are frozen in `output/sample_reduction/reference_matched_A_18.json`, copied from the original 18-sample `analysis/artifacts/selected_campaign.json`. This comparison remains reproducible even if the official campaign is revised later.',
     'We optimize only a subset of the existing six sample times per run. E2 and E3 always share the same schedule;',
     'each run retains >=1 early sample (<=10 min) and >=1 later sample (>=40 min).',
     'The 12-, 9-, and 6-sample designs are found by exhaustive search among those permitted subsets;',
     'the objective is fractional model uncertainty eliminated under noise x3/parameter spread x2,',
     'plus 0.15 times the worst leave-one-run-out fractional uncertainty eliminated.',
     'Selection uses 150 generated datasets per model; **all reported statistics** are computed with a different',
     'random seed and 1200 independent simulated datasets per model/scenario.',
     'All species A/B/P/Q are assayed at every selected time; Q uses left-censoring at LOQ = 3e-5 M.',
     'A shared bootstrap-parameter index is used across all experiment times and runs in each simulated world;',
     'bootstrap mixture likelihoods also preserve that parameter dependence.',
     'AIC weights are normalized over M2/M4 and are **not Bayesian posterior probabilities**.',
     'Noise x1/x2/x3 and bootstrap-spread x1/x2 are sensitivity assumptions, not real observed performance.', '',
     '## Results on independent evaluation worlds', '',
     '| Samples | Worst information removed | Worst classification | Worst lost-run classification | Nominal information removed |',
     '|---:|---:|---:|---:|---:|']
    for r in summary.itertuples():
        report.append(f'| {r.samples_total} | {r.min_information_removed:.1%} | {r.min_classification:.1%} | {r.min_dropout_classification:.1%} | {r.nominal_information_removed:.1%} |')
    report += ['', 'Each row takes its minimum across six stated scenarios independently.',
               'Tiny reversals (e.g., 12-sample score slightly above 18) arise from finite Monte Carlo error; they do not demonstrate that discarding data improves theoretical expected information.',
               'A dropped-run score is only a robustness diagnostic: two-run campaigns would **not** meet the 3-run submission minimum.', '',
               '## Sample schedules', '']
    for k in (6,4,3,2):
        exps = schedule_to_experiments(schedules[k])
        report.append(f'**{3*k} total:** E1 `{exps[0]["sample_times_min"]}`, E2/E3 `{exps[1]["sample_times_min"]}`.')
    report += ['', '## Interpretation / limitations', '',
       '- Near-one-bit information at nominal noise does not prove experimental success; discrimination is **conditional on M2/M4 and the bootstrap ensemble**.',
       '- Material usage remains **1.50 g A** in every candidate; all schedules satisfy temperature, concentration, duration, assay and total-sample constraints.',
       '- Comparing nominal EIG alone will overvalue aggressive sample removal because model entropy quickly saturates near one bit.',
       '- High-noise and widened-parameter scenarios are meaningful stress tests; all are recognised-noise cases and do **not** cover systematic bias or model misspecification.',
       '- E2/E3 pairing supports a Q/P charge-dependence check. The P and Q measurements are simultaneous assays within each sample.',
       '- More samples also improve time-kinetics checking and ability to detect wrong models; these benefits are not fully captured by the two-model information metric.',
       '- For safety, neither the `submission/campaign.yaml` nor `analysis/artifacts/selected_campaign.json` has been changed.',
       '- The alternative YAMLs are ready-to-inspect candidates, **not approved replacements**. To select one, update `analysis/artifacts/selected_campaign.json` then run `just suggest`.',
       f'- Samples are Monte Carlo simulations; results are reproducible using `python analysis/review_sample_efficiency.py` with output seed set as documented. Parameter sets clipped in spread x2: {clipped}.',
       '', '## Decision principle', '',
       'Prefer the smallest sample count that preserves acceptable stressed classification and worst lost-run performance, **not** just nominal EIG. Keep the official 18-sample campaign if that trade-off is not explicitly accepted by the team.', '']
    (OUT/'sample_reduction_report.md').write_text('\n'.join(report), encoding='utf-8')
    print(summary.to_string(index=False, float_format=lambda x:f'{x:.4f}'), flush=True)
    print(f'Written {OUT} in {time.perf_counter()-tic:.1f}s', flush=True)

if __name__ == '__main__':
    main()
