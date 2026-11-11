#!/usr/bin/env python3
"""Generate Jay's cases and analyze each pair with the empirical PAGDVE function.

Sequential laptop runner. Saved generating parameters replay every activity
matrix; per-comparison NPZ files retain PA, GD, all eight VE statistics, fitted
bases, evaluation covariances, and diagnostics. No extra random/split-half
controls are run inside a simulated comparison.

--common-unit-cache refits the generating FA models on the actual ordered
common units returned by get_paired_df, instead of matching neuron counts only.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import subspace_alignment as empirical
from scripts.subspace_simulation.generate_activity import generate_activity
from scripts.subspace_simulation.simulate_fixed_rank import common_neuron_counts, random_bases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params', default=str(ROOT / 'models/subspace_simulation/simulation_generating_params.pkl'))
    parser.add_argument('--models', default=str(ROOT / 'models/fa_thres_cv_regions.pkl'))
    parser.add_argument('--pair-manifest', default=str(ROOT / 'data/subspace_alignment/seed0_n500/manifest.json'))
    parser.add_argument('--common-unit-cache', type=Path, default=None,
                        help='Read cached df_all and refit generating FA on each pair\'s actual common units.')
    parser.add_argument('--output', '--data-dir', default=str(ROOT / 'data/subspace_simulation'),
                        help='Data destination; must be empty to preserve existing results.')
    parser.add_argument('--model-dir', default=str(ROOT / 'models/subspace_simulation'),
                        help='Destination for common-unit FA parameters, generating parameters, and fitted bases.')
    parser.add_argument('--task-pair', default='Pursuit/Rotations',
                        help='Two task names separated by /, or all for every available task pair.')
    parser.add_argument('--replicates', type=int, default=40)
    parser.add_argument('--samples', type=int, default=14300)
    parser.add_argument('--seed', type=int, default=20261026)
    parser.add_argument('--geometry-seed', type=int, default=20261023)
    args = parser.parse_args()
    if args.replicates < 1 or args.samples < 2 or args.seed < 0 or args.geometry_seed < 0:
        parser.error('Positive repeats, at least two samples, and nonnegative seeds required.')
    tasks = None if args.task_pair.lower() == 'all' else set(args.task_pair.split('/'))
    if tasks is not None and len(tasks) != 2:
        parser.error('--task-pair must be all or contain two task names separated by /.')
    out = Path(args.output).resolve()
    model_out = Path(args.model_dir).resolve()
    if out == model_out or out in model_out.parents or model_out in out.parents:
        raise ValueError('Data and models must use separate, non-nested directories.')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Choose a new output directory; preserving {out}.')
    if model_out.exists() and any(p.name != 'simulation_generating_params.pkl' for p in model_out.iterdir()):
        raise FileExistsError(f'Choose a new model directory; preserving {model_out}.')
    out.mkdir(parents=True, exist_ok=True)
    model_out.mkdir(parents=True, exist_ok=True)
    tables, provenance = out / 'tables', out / 'provenance'
    tables.mkdir()
    provenance.mkdir()
    started = time.monotonic()

    def log(message):
        print(message, flush=True)
        with (provenance / 'run.log').open('a') as stream:
            stream.write(message + '\n')

    # Record inputs/settings/source before generating or fitting anything.
    analysis = dict(n_rands=0, n_shufs=0, shared_var_thresh=.95,
                    max_iter=10000, tol=1e-4, svd_method='lapack')
    input_names = ('params', 'common_unit_cache') if args.common_unit_cache else ('params', 'models', 'pair_manifest')
    parameter_source = 'common_unit_fa' if args.common_unit_cache else 'full_session_fa'
    config = dict({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                  analysis=analysis, versions=empirical.versions(),
                  input_sha256={k: empirical.sha256(getattr(args, k))
                                for k in input_names},
                  seed_rule='SeedSequence([seed, original_pair_index, replicate, task_index]); shared across cases',
                  rank_rule='Empirical 95% shared-variance retention, independently for each task and simulated comparison',
                  N_source='empirical_common_count', parameter_source=parameter_source,
                  generating_rank_rule='2 * floor(min(common-unit retained ranks)/2)' if args.common_unit_cache
                                       else 'D from supplied generating parameters',
                  layout_version=2, directories=dict(data='.', models=os.path.relpath(model_out, out),
                    plots=os.path.relpath(ROOT / 'plots/subspace_simulation', out)))
    sources = [Path(__file__), ROOT / 'subspace_comparison.py', ROOT / 'utils.py', ROOT / 'decomposition.py',
               Path(empirical.__file__), Path(__file__).with_name('generate_activity.py'),
               Path(__file__).with_name('simulate_fixed_rank.py')]
    if args.common_unit_cache:
        sources.append(ROOT / 'data_loader.py')
    source_dir = provenance / 'source'
    source_dir.mkdir()
    config['source_sha256'] = {}
    for path in sources:
        shutil.copy2(path, source_dir / path.name)
        config['source_sha256'][path.name] = empirical.sha256(path)
    if not args.common_unit_cache:
        shutil.copy2(args.pair_manifest, provenance / 'empirical_manifest.json')
    (provenance / 'config.json').write_text(json.dumps(config, indent=2))

    with Path(args.params).open('rb') as stream:
        supplied = pickle.load(stream)
    if args.common_unit_cache:
        from utils import get_paired_df
        from decomposition import FAResults
        # Same cache and pairing function as main_analysis.py. This
        # mutates Session objects only in memory; the cache is never rewritten.
        df_all = pd.read_hdf(args.common_unit_cache, key='all')
        df_pairs = get_paired_df(df_all)
        paired = {tuple(sorted((row.task_1.session_key, row.task_2.session_key))): row
                  for row in df_pairs.itertuples()
                  if tasks is None or {row.task_1.task.capitalize(), row.task_2.task.capitalize()} == tasks}
        common = {key: len(row.common) for key, row in paired.items()}
    else:
        common = common_neuron_counts(args.pair_manifest)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            models = joblib.load(args.models)
    selected = [(a, b) for key, records in supplied.items() if tasks is None or set(key[:2]) == tasks
                for a, b in records if a['replicate'] < args.replicates
                and tuple(sorted((a['session'], b['session']))) in common]
    selected.sort(key=lambda ab: (ab[0]['pair_index'], ab[0]['replicate'], ab[0]['case']))
    pair_ids = sorted({a['pair_id'] for a, b in selected})
    if not pair_ids:
        raise ValueError('No supplied parameter pairs match the task pair and empirical cohort.')
    expected = {(pair, rep, case) for pair in pair_ids for rep in range(args.replicates) for case in (1, 2, 3)}
    actual = [(a['pair_id'], a['replicate'], a['case']) for a, b in selected]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Supplied parameters do not contain exactly the requested repeats and cases.')

    common_fits, common_metadata = {}, {}
    if args.common_unit_cache:
        selected_keys = {tuple(sorted((a['session'], b['session']))) for a, b in selected}
        if selected_keys != set(paired):
            raise ValueError('Supplied parameter identities do not cover every cached pair for these tasks.')
        log(f'Fitting actual common-unit data for {len(pair_ids)} session pairs from {args.common_unit_cache}.')
        with threadpool_limits(limits=1):
            for a, b in selected:
                pair_id = a['pair_id']
                if pair_id in common_fits:
                    continue
                row = paired[tuple(sorted((a['session'], b['session'])))]
                sessions = {t.session_key: t for t in (row.task_1, row.task_2)}
                channels = sorted(row.common)  # All common units, not common[i:i+1].
                if len(channels) < 2 or len(set(channels)) != len(channels):
                    raise ValueError(f'{pair_id}: invalid common-unit identifiers.')
                fits, records = [], []
                for side, p in enumerate((a, b)):
                    task = sessions[p['session']]
                    field = 'filename' if task.matfile_type == 'mat' else 'seq_num'
                    labels = task.df_chan[field].tolist()
                    if len(labels) != len(set(labels)):
                        raise ValueError(f'{task.session_key}: duplicate channel identifiers.')
                    positions = {label: i for i, label in enumerate(labels)}
                    ix = np.asarray([positions[channel] for channel in channels])
                    # find_y uses positional columns for these cached .mat sessions.
                    # Its hdf5 branch ignores idx, so explicitly subset in that case.
                    task.find_y(ix)
                    if task.matfile_type != 'mat':
                        task.Y = task.spikes_per_bin[:, ix]
                    task.get_spikes_mu_sd(verbose=False)
                    Y, Z = task.Y.copy(), task.Y_norm.copy()
                    np.testing.assert_array_equal(task.df_chan[field].iloc[ix].to_numpy(), channels)
                    if Y.shape[1] != len(channels) or not np.isfinite(Z).all():
                        raise ValueError(f'{pair_id}: invalid common-unit data matrix.')
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        fit = FAResults().fa_fit(Z, max_iter=10000, tol=1e-4,
                                                svd_method='lapack', shared_var_thresh=.95)
                    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                        raise RuntimeError(f'{pair_id}/{task.session_key}: common-unit FA did not converge.')
                    record = dict(session=task.session_key, task=p['task'], channel_field=field,
                        channels=[str(c) for c in channels], source_indices=ix.tolist(),
                        regions=task.df_chan.region.iloc[ix].astype(str).tolist(),
                        shape=list(Y.shape), retained_rank=int(fit.d_shared),
                        fitted_rank=int(fit.n_components), n_iter=int(fit.fa.n_iter_), converged=True,
                        warnings=[str(w.message) for w in caught],
                        input_sha256=hashlib.sha256(Y.tobytes()).hexdigest(),
                        bin_size=getattr(task, 'bin_size', None), event_mode=getattr(task, 'mode', None),
                        max_gap_dur_secs=getattr(task, 'max_gap_dur_secs', None))
                    spectral = np.linalg.svd(fit.fa.components_, compute_uv=False)**2
                    dest = model_out / 'common_unit_fits' / pair_id
                    dest.mkdir(parents=True, exist_ok=True)
                    model_path = dest / f'task_{side+1}.npz'
                    input_path = out / 'inputs' / pair_id / f'task_{side+1}.npz'
                    input_path.parent.mkdir(parents=True, exist_ok=True)
                    record.update(input_path=str(input_path.relative_to(out)),
                                  model_path=str(model_path.relative_to(model_out)))
                    np.savez_compressed(input_path, Y=Y, Y_norm=Z, source_indices=ix,
                        channels=np.asarray(channels, dtype=str), bin_times=task.bin_times,
                        mean=task.spikes_mu, std=task.spikes_sd, metadata=json.dumps(record))
                    np.savez_compressed(model_path,
                        components=fit.fa.components_, private_variances=fit.fa.noise_variance_,
                        shared_spectrum=spectral, retained_rank=fit.d_shared,
                        metadata=json.dumps(record))
                    fits.append(dict(spectrum=spectral, private=fit.fa.noise_variance_.copy(),
                                     rank=int(fit.d_shared), indices=ix, channels=record['channels'],
                                     model_path=str(model_path.relative_to(model_out))))
                    records.append(record)
                common_fits[pair_id] = fits
                common_metadata[pair_id] = records
                log(f'{pair_id}: {len(channels)} matched units; retained ranks {[f["rank"] for f in fits]}; '
                    f'generating D={2*(min(f["rank"] for f in fits)//2)}.')
        (out / 'inputs/common_units.json').write_text(json.dumps(common_metadata, indent=2))

    # Keep the original spectra and geometry when N already matches. When N
    # changes (or common-unit refitting changes D), rebuild the bases using
    # Jay's geometry recipe. In common-unit mode the new fitted spectrum and
    # every matched unit's private variance are used directly in channel order.
    # Legacy count-only mode retains its original full-session parameter recipe.
    generating = {}
    pair_metadata = {}
    for original_a, original_b in selected:
        a, b = copy.deepcopy((original_a, original_b))
        n = common[tuple(sorted((a['session'], b['session'])))]
        d, index, rep, case = a['D'], a['pair_index'], a['replicate'], a['case']
        if args.common_unit_cache:
            d = 2 * (min(f['rank'] for f in common_fits[a['pair_id']]) // 2)
        if d < 2 or 3*d > 2*n:
            raise ValueError(f'{a["pair_id"]}: N={n} cannot support this D={d} geometry.')
        if n != a['N'] or d != a['D']:
            rng = np.random.default_rng(np.random.SeedSequence([args.geometry_seed, index, rep]))
            x, y, z = random_bases(n, d, rng)
            rng.integers(2**32)  # Preserve the original permutation RNG positions.
            permutations = [rng.permutation(d), rng.permutation(d)]
            if case == 1:
                bases = [np.column_stack([x, z]), np.column_stack([y, z])]
            elif case == 2:
                bases = [np.column_stack([z, x]), np.column_stack([z, y])]
            else:
                bases = [np.column_stack([z, x])[:, permutations[0]],
                         np.column_stack([z, y])[:, permutations[1]]]
        else:
            bases = [a['basis'], b['basis']]
        private_rng = np.random.default_rng(np.random.SeedSequence([args.geometry_seed, index, 987]))
        for side, p in enumerate((a, b)):
            if args.common_unit_cache:
                fit = common_fits[p['pair_id']][side]
                indices, private = fit['indices'], fit['private']
                p.update(shared_variances=fit['spectrum'][:d].copy(),
                         common_channels=fit['channels'], common_fit_path=fit['model_path'])
            else:
                model = models[p['session']]['fa_thres']['fa']
                indices = np.sort(private_rng.choice(model.n_features_in_, n, replace=False))
                private = model.noise_variance_[indices]
                # For unchanged N the model and pickle must describe the same inputs.
                if n == p['N']:
                    np.testing.assert_allclose(private, p['private_variances'], atol=1e-12)
            entropy = [args.seed, index, rep, side]
            p.update(source_N=p['N'], source_D=p['D'], source_T=p['T'], N=n, D=d, T=args.samples,
                     basis=bases[side], private_variances=private.copy(),
                     private_variance_indices=indices, mean=np.zeros(n), center=True,
                     noise_seed=None, seed_entropy=entropy,
                     rng_state=np.random.default_rng(np.random.SeedSequence(entropy)).bit_generator.state,
                     geometry_seed=args.geometry_seed, N_source='empirical_common_count',
                     parameter_source=parameter_source)
            np.testing.assert_allclose(p['basis'].T @ p['basis'], np.eye(d), atol=1e-12)
        key = (a['task'], b['task'], a['patient'])
        generating.setdefault(key, []).append((a, b))
        pair_metadata[a['pair_id']] = dict(pair=a['pair_id'], patient=a['patient'],
            task_pair='/'.join(sorted((a['task'], b['task']))), session_a=a['session'], session_b=b['session'],
            N=n, source_N=a['source_N'], source_D=a['source_D'], D=d, generating_K=d//2, T=args.samples,
            N_source='empirical_common_count', parameter_source=parameter_source)
        if args.common_unit_cache:
            pair_metadata[a['pair_id']].update(
                common_M1=common_fits[a['pair_id']][0]['rank'],
                common_M2=common_fits[a['pair_id']][1]['rank'])
    with (model_out / 'generating_params.pkl').open('wb') as stream:
        pickle.dump(generating, stream, protocol=pickle.HIGHEST_PROTOCOL)
    config['generating_params_sha256'] = empirical.sha256(model_out / 'generating_params.pkl')
    config.update(n_pairs=len(pair_ids), n_patients=len({a['patient'] for a, b in selected}),
                  n_comparisons=len(selected))
    (provenance / 'config.json').write_text(json.dumps(config, indent=2))
    pd.DataFrame(pair_metadata.values()).to_csv(tables / 'pairs.csv', index=False)
    log(f'{len(pair_ids)} pairs; {args.replicates} repeats x 3 cases; {len(selected)} comparisons; T={args.samples}.')
    log('Empirical PAGDVE: z-score each task, fit full FA, retain 95% shared variance; tol=1e-4, max_iter=10000.')

    fractions, comparisons, angles, diagnostics = [], [], [], []
    completed = 0
    with threadpool_limits(limits=1):
        for records in generating.values():
            for a, b in records:
                identity = dict(pair=a['pair_id'], patient=a['patient'],
                                task_pair=pair_metadata[a['pair_id']]['task_pair'],
                                replicate=a['replicate'], case=a['case'], N=a['N'], D=a['D'])
                Y1, Y2 = generate_activity(a), generate_activity(b)
                input_hashes = [hashlib.sha256(Y.tobytes()).hexdigest() for Y in (Y1, Y2)]
                try:
                    result = empirical.pagdve(Y1, Y2, pair_id=a['pair_id'], **analysis)
                except Exception as exc:
                    # Preserve the failed realization; never replace it with a luckier draw.
                    failure = dict(identity, error=repr(exc), input_sha256=input_hashes)
                    (provenance / 'failure.json').write_text(json.dumps(failure, indent=2))
                    config.update(status='failed', completed=completed, failure=failure)
                    (provenance / 'config.json').write_text(json.dumps(config, indent=2))
                    raise
                ve, ranks = result['data_ve'], result['ranks']
                d, k = int(result['d']), int(result['K'])
                assert d == min(ranks) and k == d//2
                assert np.isfinite(ve).all() and np.all(ve[:6] >= -1e-12) and np.all(ve[:6] <= 1+1e-12)
                np.testing.assert_allclose(ve[6:], ve[[0, 3]]-ve[[1, 4]], atol=1e-12)
                np.testing.assert_allclose(result['data_gd'], np.linalg.norm(result['data_pa']), atol=1e-12)
                assert np.all(np.diff(result['data_pa']) >= -1e-10)
                assert all(fit['converged'] for fit in result['metadata']['fits'])
                for side in range(2):
                    grouped = ve[3*side:3*side+2].sum()
                    assert grouped <= 1+1e-12
                    if ranks[side] == 2*k:
                        np.testing.assert_allclose(grouped, 1, atol=1e-12)
                meta = dict(result['metadata'], **identity, input_sha256=input_hashes,
                            seed_entropy=[a['seed_entropy'], b['seed_entropy']])
                dest = out / 'comparisons' / a['pair_id'] / f'case_{a["case"]}'
                dest.mkdir(parents=True, exist_ok=True)
                fitted_path = model_out / 'estimated_bases' / a['pair_id'] / f'case_{a["case"]}' / f'repeat_{a["replicate"]:03d}.npz'
                fitted_path.parent.mkdir(parents=True, exist_ok=True)
                meta['model_path'] = str(fitted_path.relative_to(model_out))
                numeric = {key: value for key, value in result.items() if key not in ('metadata', 'reference')}
                reference = result['reference']
                np.savez_compressed(fitted_path, L1=reference['L1'], L2=reference['L2'],
                                    ranks=result['ranks'], metadata=json.dumps(meta))
                np.savez_compressed(dest / f'repeat_{a["replicate"]:03d}.npz',
                                    **numeric, C1=reference['C1'], C2=reference['C2'], metadata=json.dumps(meta))
                comparisons.append(dict(identity, d=d, K=k, M1=int(ranks[0]), M2=int(ranks[1]),
                    GD=float(result['data_gd']), **dict(zip(empirical.VE_COLUMNS, ve))))
                for i, value in enumerate(result['data_pa']):
                    angles.append(dict(identity, d=d, K=k, angle_index=i+1, angle_degrees=value))
                for side, p in enumerate((a, b)):
                    diagnostics.append(dict(identity, session=p['session'], task=p['task'],
                                            **result['metadata']['fits'][side]))
                    for category, value in zip(('more', 'less', 'top'), ve[side*3:side*3+3]):
                        fractions.append(dict(identity, session=p['session'], task=p['task'],
                            method='fitted_data_total', category=category, fraction=value,
                            retained_rank=int(ranks[side]), d=d, K=k))
                completed += 1
                if completed % 30 == 0 or completed == len(selected):
                    log(f'{completed}/{len(selected)} comparisons; {time.monotonic()-started:.1f}s; '
                        f'{a["pair_id"]}, repeat {a["replicate"]}, case {a["case"]}.')
    pd.DataFrame(fractions).to_csv(tables / 'results.csv', index=False)
    pd.DataFrame(comparisons).to_csv(tables / 'comparisons.csv', index=False)
    pd.DataFrame(angles).to_csv(tables / 'principal_angles.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(tables / 'fit_diagnostics.csv', index=False)
    assert completed == len(expected)
    validation = (f'PASS: {len(pair_ids)} pairs x {args.replicates} repeats x 3 cases = {completed} comparisons; '
                  f'{2*completed} converged FA fits.\n'
                  'PA sorted; GD equals PA norm; VE fractions within [0,1]; delta identities verified.\n'
                  'More+less <= 1, with equality when own retained rank equals 2K.\n'
                  'An odd final direction or unmatched extra dimensions remain in the denominator.\n')
    (provenance / 'validation.txt').write_text(validation)
    config.update(status='complete', completed=completed, elapsed_seconds=time.monotonic()-started)
    (provenance / 'config.json').write_text(json.dumps(config, indent=2))
    manifest = dict(layout_version=2, dataset='empirical_pagdve', directories=config['directories'],
                    config='provenance/config.json', generating_parameters='generating_params.pkl',
                    note='Paths in directories are relative to this data directory; model paths are relative to its models directory.')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    log(validation.rstrip())


if __name__ == '__main__':
    main()
