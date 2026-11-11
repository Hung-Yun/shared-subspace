#!/usr/bin/env python3
"""Fig. 6: principal angles, geodesic distance, and directional variance.

Sequential laptop computation and finalization. The scientific calculation is pagdve(); the
remaining code saves inputs, checkpoints repeats, and collects results.
Saved-result analysis and plotting cells are in the root subspace_comparison.py. Nothing runs on
import. See README.md for commands and definitions.

Within-task controls deliberately preserve the real pair's retained ranks
(M1, M2). Their main VE comparisons use task-1 half 1 and task-2 half 2;
collected results expose four named statistics for each of these comparisons.
Original eight-statistic repeat checkpoints remain intact for reproducibility.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import warnings

import numpy as np
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

# Direct execution works from scripts/ and from a frozen run directory.
HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / 'decomposition.py').exists() else HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import decomposition
from decomposition import FAResults
DECOMPOSITION_SOURCE = Path(decomposition.__file__).resolve()


CONDITIONS = ('rand', 'shuf_task1', 'shuf_task2')
# Eight SUMMARY statistics, independent of the number of principal angles.
# This is the RAW pagdve/checkpoint layout. Sides 1/2 mean real tasks for
# data/random and halves of the source task for shuffled controls. Collection
# selects only the rank-matched half's VE; PA/GD still require both halves.
VE_COLUMNS = ('more_1', 'less_1', 'top_1', 'more_2', 'less_2', 'top_2', 'delta_1', 'delta_2')
VE_MAIN_COLUMNS = ('more-aligned VE', 'less-aligned VE', 'top-factor VE', 'ΔVE')
VE_MAIN_INDICES = {1: [0, 1, 2, 6], 2: [3, 4, 5, 7]}
DEFAULT_PARAMS = dict(n_rands=500, n_shufs=500, seed=0, shared_var_thresh=.95,
                      max_iter=10_000, tol=1e-4, svd_method='lapack', max_attempts=3)


#%% Principal angles, geodesic distance, and variance explained

def pagdve(Y1, Y2, *, n_rands=500, n_shufs=500, seed=0, pair_id='',
           retained_ranks=None, shared_var_thresh=.95, max_iter=10_000,
           tol=1e-4, svd_method='lapack', max_attempts=3,
           _reference=None, _jobs=None):
    """Calculate observed PA/GD/VE and matched random / within-task controls.

    Y1/Y2 are samples x neurons, with the SAME ordered neurons. Normalization,
    full FA fitting, and retention follow FAResults / the existing Fig. 6.
    Defaults select each observed rank using 95% modeled shared variance.
    retained_ranks optionally fixes the retained (not fitted) observed ranks.

    Return RAW flat arrays: data_pa (d,), data_gd scalar, data_ve (8,); each of
    rand/shuf_task1/shuf_task2 has *_pa (N,d), *_gd (N,), *_ve (N,8), and
    *_index (N,). VE_COLUMNS specifies the eight columns, as fractions.
    Within shuf_task1 or shuf_task2, sides 1/2 mean two complementary halves
    of THAT task, with retained ranks (M1,M2) in both within-task controls.
    Call ve_main_view(result) for the four-column analysis format also used
    in collected result files: data_task{i}_ve, rand_task{i}_ve, and
    shuf_task{i}_ve all follow VE_MAIN_COLUMNS. Only the raw return and
    checkpoints contain complementary shuffled-half VE; the collected
    shuf_task1_ve/shuf_task2_ve contain H1/H2 respectively (ΔVE is column 3).

    M1/M2 count retained FA dimensions, selected from the full real tasks;
    they do not count samples. Each half is fitted at the full neuron count,
    then half 1 retains M1 and half 2 retains M2, regardless of source task.
    This intentionally matches the observed pair's two ranks, PA count, and
    VE group size. It is an asymmetric within-task control when M1 != M2.

    For a source task with T time-bin rows, every repeat uses floor(T/2) rows
    in half 1 and ceil(T/2) in half 2. Counts stay fixed across repeats;
    neuron order is fixed and whole population rows are permuted together.
    Each half is normalized separately AFTER splitting. These half-length
    samples do not match the full observed recordings' sample counts, and
    task 1 and task 2 can have different recording lengths.

    K=d//2; more=[:K], less=[K:2*K]. An odd final angle is excluded from VE
    groups but included in full GD. GD is norm(PA in degrees), as before.
    Denominators include the entire own retained subspace. Empirical covariance
    (ddof=0) computes exactly the variance of projected normalized activity;
    this is not variance of the FA modeled shared covariance.

    In the RAW eight-column layout, compare data_ve[6] against
    shuf_task1_ve[:,6] (task-1 half 1, retaining M1) and data_ve[7] against
    shuf_task2_ve[:,7] (task-2 half 2, retaining M2). Each retained side uses
    the other half as its alignment partner. VE is evaluated using that
    side's own normalized activity and own full retained-space denominator;
    the denominator definition/rank matches the real task, not its value.
    Complementary half results remain in raw checkpoints only. If M1 == M2,
    both halves have the same rank convention; an odd T still gives one
    extra row in half 2. The two half statistics from one split are paired.

    This benchmark asks whether variance-alignment enrichment also arises
    in separately estimated subspaces from the same task at the matched
    rank pair. It is not a null the data must exceed, nor a mathematical
    bound. Fits, direction selection, and VE evaluation reuse each half;
    this is not held-out evaluation. Random row splitting does not remove
    temporal dependence or make the repeats independent patient samples.

    Random PA compares R1/R2 generated by the existing Gaussian-loading,
    FA-simulate/refit procedure. Random VE instead selects directions in real
    L1 against R2 and in real L2 against R1, evaluated on real covariance.
    rand_task1_pa / rand_task2_pa record these different comparisons.

    Metadata includes fit diagnostics, seeds/attempts, and split permutations.
    _reference and _jobs are internal hooks for the sequential runner, which
    freezes real fits once and resumes indexed repeats.
    No surrogate is silently lost: a failed job raises after bounded retries.
    Convergence failures raise immediately, preserving the realization for
    diagnosis rather than selecting a different draw because it fits faster.
    """
    Y1, Y2 = np.asarray(Y1, dtype=float), np.asarray(Y2, dtype=float)
    for Y in (Y1, Y2):
        if Y.ndim != 2 or min(Y.shape) < 2 or not np.isfinite(Y).all():
            raise ValueError('Inputs must be finite sample x neuron matrices, at least 2 x 2.')
    if Y1.shape[1] != Y2.shape[1]:
        raise ValueError('Inputs must have the same ordered neuron columns.')
    for name, value, minimum in (('n_rands', n_rands, 0), ('n_shufs', n_shufs, 0),
                                 ('seed', seed, 0), ('max_iter', max_iter, 1),
                                 ('max_attempts', max_attempts, 1)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}.')
    if not 0 < shared_var_thresh <= 1 or not np.isfinite(tol) or tol <= 0:
        raise ValueError('Require 0 < shared_var_thresh <= 1 and finite tol > 0.')
    if svd_method not in ('lapack', 'randomized'):
        raise ValueError('svd_method must be lapack or randomized.')
    if retained_ranks is not None:
        if len(retained_ranks) != 2 or any(isinstance(m, bool) or not isinstance(m, (int, np.integer))
                or not 1 <= m <= Y1.shape[1] for m in retained_ranks):
            raise ValueError('retained_ranks must contain two valid integer ranks.')
    fit_log, repeat_log = [], []

    # Small shared operations keep observed and surrogate definitions identical.
    def fit(Y, label, retain=None, fitted_rank=None):
        with np.errstate(divide='ignore', invalid='ignore'):
            Z = np.nan_to_num((Y - Y.mean(axis=0)) / Y.std(axis=0),
                              nan=0., posinf=0., neginf=0.)
        if min(Z.shape) < (fitted_rank or Z.shape[1]):
            raise ValueError(f'{label}: too few samples for the requested FA fit.')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model = FAResults().fa_fit(Z, n_components=fitted_rank,
                shared_var_thresh=shared_var_thresh, max_iter=max_iter,
                tol=tol, svd_method=svd_method)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        fit_log.append(dict(label=label, n_iter=int(model.fa.n_iter_), converged=converged,
                            fitted_rank=int(fitted_rank or Z.shape[1]),
                            retained_rank=int(model.d_shared if retain is None else retain),
                            warnings=[str(w.message) for w in caught]))
        if not converged:
            raise RuntimeError(f'{label}: FA did not converge in {max_iter} iterations. '
                               'This repeat is not counted; its inputs and seed are reproducible.')
        rank = int(model.d_shared if retain is None else retain)
        L = model.fa.components_.T[:, :rank]
        if L.shape[1] != rank or np.linalg.matrix_rank(L) != rank:
            raise np.linalg.LinAlgError(f'{label}: retained loading matrix lacks rank {rank}.')
        basis = np.linalg.svd(L, full_matrices=False)[0]
        # Center once more for exact agreement with np.var(Z @ direction, ddof=0).
        centered = Z - Z.mean(axis=0)
        covariance = centered.T @ centered / len(Z)
        return basis, covariance

    def directions(A, B):
        # Same QR/SVD as utils.principal_angles, including the QR coordinate map.
        Qa, _ = np.linalg.qr(A)
        Qb, _ = np.linalg.qr(B)
        U, singular, Vh = np.linalg.svd(Qa.T @ Qb, full_matrices=False)
        angles = np.rad2deg(np.arccos(np.clip(singular, -1., 1.)))
        return angles, Qa @ U, Qb @ Vh.T

    def variance(C, basis, principal):
        own = np.diag(basis.T @ C @ basis)
        denominator = own.sum()
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError('Own-subspace variance denominator must be positive and finite.')
        projected = np.diag(principal.T @ C @ principal)
        return np.array([projected[:K].sum(), projected[K:2*K].sum(), own[:K].sum()]) / denominator

    if _reference is None:
        rank1, rank2 = retained_ranks if retained_ranks is not None else (None, None)
        L1, C1 = fit(Y1, 'data_1', rank1)
        L2, C2 = fit(Y2, 'data_2', rank2)
    else:
        L1, L2, C1, C2 = (_reference[k] for k in ('L1', 'L2', 'C1', 'C2'))
    M1, M2 = L1.shape[1], L2.shape[1]
    d, K = min(M1, M2), min(M1, M2) // 2
    if K < 1:
        raise ValueError('VE grouping requires at least two paired dimensions (d >= 2).')
    angles, P1, P2 = directions(L1, L2)
    v1, v2 = variance(C1, L1, P1), variance(C2, L2, P2)
    result = dict(data_pa=angles, data_gd=np.linalg.norm(angles),
                  data_ve=np.r_[v1, v2, v1[0]-v1[1], v2[0]-v2[1]],
                  ranks=np.array([M1, M2]), d=d, K=K,
                  denominators=np.array([np.trace(L1.T @ C1 @ L1), np.trace(L2.T @ C2 @ L2)]),
                  reference=dict(L1=L1, L2=L2, C1=C1, C2=C2))
    jobs = ([(c, i) for c in CONDITIONS for i in range(n_rands if c == 'rand' else n_shufs)]
            if _jobs is None else list(_jobs))
    if len(set(jobs)) != len(jobs):
        raise ValueError('Duplicate repeat identities.')
    for condition, index in jobs:
        if condition not in CONDITIONS or not isinstance(index, (int, np.integer)) or not (
                0 <= index < (n_rands if condition == 'rand' else n_shufs)):
            raise ValueError(f'Invalid repeat identity: {(condition, index)}')
    if any(c.startswith('shuf') for c, _ in jobs) and min(len(Y1)//2, len(Y2)//2) < Y1.shape[1]:
        raise ValueError('Each half needs at least as many samples as the full fitted FA rank (neurons).')
    buffers = {c: dict(pa=[], gd=[], ve=[], index=[], task1_pa=[], task2_pa=[]) for c in CONDITIONS}
    pair_words = np.frombuffer(hashlib.sha256(str(pair_id).encode()).digest(), dtype='<u4').tolist()
    for condition, index in jobs:
        attempt, failures = 0, []
        while attempt < max_attempts:
            # Address each random stream directly; never search a list of seeds.
            entropy = [int(seed), *pair_words, CONDITIONS.index(condition), int(index), attempt]
            rng = np.random.default_rng(np.random.SeedSequence(entropy))
            record = dict(condition=condition, index=int(index), attempt=attempt, entropy=entropy,
                          rejected_attempts=failures)
            try:
                if condition == 'rand':
                    random_bases = []
                    for side, (Y, rank) in enumerate(((Y1, M1), (Y2, M2)), 1):
                        loading = rng.standard_normal((rank, Y.shape[1]))
                        simulated = rng.standard_normal((len(Y), rank)) @ loading
                        simulated += rng.normal(scale=np.sqrt(1e-4), size=Y.shape)
                        R, _ = fit(simulated, f'rand_{index}_{side}_attempt{attempt}',
                                   retain=rank, fitted_rank=rank)
                        random_bases.append(R)
                    R1, R2 = random_bases
                    pa, _, _ = directions(R1, R2)
                    task1_pa, P1, _ = directions(L1, R2)
                    task2_pa, _, P2 = directions(R1, L2)
                    a, b = variance(C1, L1, P1), variance(C2, L2, P2)
                    # Neither variance call uses a random basis or simulated covariance.
                else:
                    Y = Y1 if condition == 'shuf_task1' else Y2
                    ix = rng.permutation(len(Y))
                    cut = len(ix) // 2
                    half1, half2 = ix[:cut], ix[cut:]
                    # Deliberate rank pair (M1,M2), not a separate rank choice
                    # for each half. This holds d/K and comparison geometry
                    # fixed; M1/M2 are dimensions, not half sample counts.
                    # fit() separately normalizes each half after the split.
                    A, Ca = fit(Y[half1], f'{condition}_{index}_half1_attempt{attempt}', retain=M1)
                    B, Cb = fit(Y[half2], f'{condition}_{index}_half2_attempt{attempt}', retain=M2)
                    pa, Pa, Pb = directions(A, B)
                    # Both halves contribute to PA/principal directions, and
                    # both have their own VE. The main task-1 comparison uses
                    # a; the main task-2 comparison uses b of shuf_task2.
                    a, b = variance(Ca, A, Pa), variance(Cb, B, Pb)
                    record.update(half1_indices=half1.tolist(), half2_indices=half2.tolist())
                if pa.shape != (d,) or not np.isfinite(pa).all():
                    raise np.linalg.LinAlgError(f'Expected {d} finite principal angles; got {pa.shape}.')
                ve = np.r_[a, b, a[0]-a[1], b[0]-b[1]]
                if not np.isfinite(ve).all():
                    raise np.linalg.LinAlgError('Nonfinite variance result.')
            except np.linalg.LinAlgError as exc:
                failures.append(dict(attempt=attempt, error=str(exc)))
                attempt += 1
                continue
            buf = buffers[condition]
            for field, value in (('pa', pa), ('gd', np.linalg.norm(pa)), ('ve', ve), ('index', index)):
                buf[field].append(value)
            if condition == 'rand':
                buf['task1_pa'].append(task1_pa)
                buf['task2_pa'].append(task2_pa)
            repeat_log.append(record)
            break
        else:
            raise RuntimeError(f'{condition} repeat {index} failed after {max_attempts} attempts: {failures}')
    for condition, buf in buffers.items():
        for field, width in (('pa', d), ('ve', 8)):
            result[f'{condition}_{field}'] = np.asarray(buf[field], dtype=float).reshape(-1, width)
        result[f'{condition}_gd'] = np.asarray(buf['gd'], dtype=float)
        result[f'{condition}_index'] = np.asarray(buf['index'], dtype=int)
    for side in (1, 2):
        result[f'rand_task{side}_pa'] = np.asarray(buffers['rand'][f'task{side}_pa']).reshape(-1, d)
    result['metadata'] = dict(pair_id=str(pair_id), seed=int(seed), ve_columns=VE_COLUMNS,
                              fits=fit_log, repeats=repeat_log)
    return result


#%% Saving, preparation, and sequential execution
def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def document_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def versions():
    return dict(numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                threadpoolctl=version('threadpoolctl'))


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            if path.suffix == '.json':
                stream.write((json.dumps(value, indent=2, allow_nan=False) + '\n').encode())
            else:
                np.savez_compressed(stream, **value)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def ve_main_view(result):
    """Return named, four-column VE views without refitting or changing PA/GD.

    Accept raw pagdve/legacy collected output or this same format (schema 2).
    Task-1 shuffle keeps H1 at M1; task-2 shuffle keeps H2 at M2. Both real
    tasks survive for data/random, with separate task-specific arrays. Their
    combined eight-column data_ve/rand_ve are retained for compatibility and
    use paired_ve_columns, NOT the four labels in ve_columns.
    """
    output = dict(result)
    metadata = result['metadata']
    is_dict = isinstance(metadata, dict)
    meta = dict(metadata) if is_dict else json.loads(str(metadata))
    schema = meta.get('ve_schema', 1)
    if schema not in (1, 2):
        raise ValueError(f'Unsupported VE schema: {schema}')
    expected_columns = VE_COLUMNS if schema == 1 else VE_MAIN_COLUMNS
    if tuple(meta['ve_columns']) != expected_columns:
        raise ValueError('VE column labels do not match the saved schema.')
    if schema == 2 and (tuple(result.get('ve_columns', ())) != VE_MAIN_COLUMNS
                       or tuple(meta.get('paired_ve_columns', ())) != VE_COLUMNS):
        raise ValueError('Missing or inconsistent VE column labels.')
    for key, ndim in (('data_ve', 1), ('rand_ve', 2),
                      ('shuf_task1_ve', 2), ('shuf_task2_ve', 2)):
        values = np.asarray(result[key])
        width = 4 if schema == 2 and key.startswith('shuf') else 8
        if values.ndim != ndim or values.shape[-1] != width or not np.isfinite(values).all():
            raise ValueError(f'Invalid VE shape or values: {key}')
        delta, difference = ((values[..., 3], values[..., 0] - values[..., 1]) if width == 4
                            else (values[..., 6:], values[..., [0, 3]] - values[..., [1, 4]]))
        if not np.allclose(delta, difference):
            raise ValueError(f'Inconsistent ΔVE: {key}')
    for side, indices in VE_MAIN_INDICES.items():
        for condition in ('data', 'rand'):
            key = f'{condition}_task{side}_ve'
            selected = np.asarray(result[f'{condition}_ve'])[..., indices]
            if schema == 2 and (key not in result or not np.array_equal(result[key], selected)):
                raise ValueError(f'Inconsistent task-specific VE: {key}')
            output[key] = selected
        key = f'shuf_task{side}_ve'
        if schema == 1:
            output[key] = np.asarray(result[key])[:, indices]
    selection = {f'shuf_task{side}_ve': dict(source_task=side, evaluated_half=side,
                    retained_rank=f'M{side}', raw_columns=indices)
                 for side, indices in VE_MAIN_INDICES.items()}
    if schema == 2 and meta.get('ve_selection') != selection:
        raise ValueError('Inconsistent shuffled VE half selection metadata.')
    output['ve_columns'] = np.asarray(VE_MAIN_COLUMNS)
    meta.update(ve_schema=2, ve_columns=list(VE_MAIN_COLUMNS),
                paired_ve_columns=list(VE_COLUMNS), ve_selection=selection)
    output['metadata'] = meta if is_dict else np.array(json.dumps(meta, allow_nan=False))
    return output


def update_ve_results(bundle):
    """Update every collected pair to four-column VE views; preserve raw files.

    This storage-only operation accepts historical frozen source versions.
    It checks run/pair identities and complete repeat counts before writing
    any files. Individual writes are atomic; interrupted migrations can be
    resumed because ve_main_view accepts either format. Checkpoints, inputs,
    frozen source, manifest, result index, and plots are never written here.
    """
    bundle = Path(bundle)
    if (bundle / 'RUNNING.json').exists():
        raise RuntimeError('Wait for the running fits before updating collected VE views.')
    finalizing = bundle / 'FINALIZING.json'
    if finalizing.exists():
        try:
            owner = json.loads(finalizing.read_text()).get('pid')
        except (ValueError, TypeError, AttributeError):
            owner = None
        if owner != os.getpid():
            raise RuntimeError('Wait for finalization before updating collected VE views.')
    manifest = json.loads((bundle / 'manifest.json').read_text())
    index = json.loads((bundle / 'results/index.json').read_text())
    if manifest['run_id'] != document_hash({k: v for k, v in manifest.items() if k != 'run_id'}):
        raise ValueError('Manifest changed; cannot identify the source run.')
    if index['run_id'] != manifest['run_id']:
        raise ValueError('Saved result index and manifest do not match.')
    identities = [(p['key'], p['pair_id']) for p in manifest['pairs']]
    if (len(set(identities)) != len(identities) or
            sorted(identities) != sorted((p['key'], p['pair_id']) for p in index['pairs'])):
        raise ValueError('Result index must contain every manifest pair exactly once.')
    report_path = bundle / 'results/ve_views.json'
    previous = json.loads(report_path.read_text()) if report_path.exists() else {}
    if previous and previous['run_id'] != manifest['run_id']:
        raise ValueError('VE view audit belongs to a different run.')
    pending, audit = [], []
    for pair in index['pairs']:
        path = bundle / 'results' / pair['path']
        if path.resolve().parent != (bundle / 'results').resolve():
            raise ValueError(f'Result path is outside the results directory: {path}')
        saved = load_npz(path)
        meta = json.loads(str(saved['metadata']))
        if (meta['run_id'], meta['pair_id'], meta['key']) != (
                manifest['run_id'], pair['pair_id'], pair['key']):
            raise ValueError(f'Saved pair identity mismatch: {path}')
        selected = ve_main_view(saved)
        d = int(saved['d'])
        if saved['data_pa'].shape != (d,):
            raise ValueError(f'Invalid observed PA shape: {path}')
        for condition in CONDITIONS:
            count = manifest['params']['n_rands' if condition == 'rand' else 'n_shufs']
            if (saved[f'{condition}_pa'].shape != (count, d)
                    or saved[f'{condition}_gd'].shape != (count,)
                    or saved[f'{condition}_ve'].shape[0] != count
                    or not np.array_equal(saved[f'{condition}_index'], np.arange(count))):
                raise ValueError(f'Incomplete or inconsistent repeat counts: {path}/{condition}')
        before = sha256(path)
        pending.append((path, selected, meta.get('ve_schema') != 2))
        audit.append(dict(key=pair['key'], path=pair['path'], before_sha256=before))
    for (path, selected, changed), entry in zip(pending, audit):
        if changed:
            atomic_save(path, selected)
        entry['after_sha256'] = sha256(path)
    # Repeated execution on the same files is a true no-op, including the audit.
    if previous and previous.get('files') and {
            p['path']: p['after_sha256'] for p in previous['files']} == {
            p['path']: p['after_sha256'] for p in audit}:
        return previous
    report = dict(run_id=manifest['run_id'], ve_schema=2,
                  updated_utc=datetime.now(timezone.utc).isoformat(),
                  ve_columns=list(VE_MAIN_COLUMNS),
                  selection={f'shuf_task{s}_ve': ix for s, ix in VE_MAIN_INDICES.items()},
                  raw_checkpoints_preserved=True, scientific_recomputation=False,
                  files=audit)
    atomic_save(report_path, report)
    print(f'Updated VE views for {len(audit)} saved pairs; no FA fits run.', flush=True)
    return report


def prepare_bundle(pairs, bundle, **params):
    """Freeze iterable of {pair_id, Y1, Y2, metadata}; also usable from Python."""
    bundle = Path(bundle)
    settings = dict(DEFAULT_PARAMS, **params)
    unknown = set(settings) - set(DEFAULT_PARAMS) - {'retained_ranks'}
    if unknown:
        raise ValueError(f'Unknown settings: {unknown}')
    for name in ('n_rands', 'n_shufs', 'seed', 'max_iter', 'max_attempts'):
        minimum = 1 if name in ('max_iter', 'max_attempts') else 0
        if isinstance(settings[name], bool) or not isinstance(settings[name], int) or settings[name] < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}.')
    if bundle.exists():
        raise FileExistsError(f'Refusing to replace existing bundle: {bundle}')
    bundle.mkdir(parents=True)
    (bundle / 'inputs').mkdir()
    manifest = dict(schema=1, created_utc=datetime.now(timezone.utc).isoformat(),
                    params=settings, versions=versions(), python=sys.version,
                    ve_columns=VE_COLUMNS, pairs=[], sources={})
    seen = set()
    for index, entry in enumerate(pairs):
        pair_id = str(entry['pair_id'])
        if pair_id in seen:
            raise ValueError(f'Duplicate pair ID: {pair_id}')
        seen.add(pair_id)
        record = dict(pair_id=pair_id, key=f'pair_{index:03d}', metadata=entry.get('metadata', {}), inputs=[], times=[])
        arrays = [np.asarray(entry[k], dtype=float) for k in ('Y1', 'Y2')]
        if any(a.ndim != 2 or not np.isfinite(a).all() for a in arrays) or arrays[0].shape[1] != arrays[1].shape[1]:
            raise ValueError(f'{pair_id}: invalid input matrices.')
        for side, array in enumerate(arrays, 1):
            name = f'inputs/{record["key"]}_Y{side}.npy'
            np.save(bundle / name, array, allow_pickle=False)
            record['inputs'].append(dict(path=name, sha256=sha256(bundle / name), shape=list(array.shape)))
            if f'times{side}' in entry:
                times = np.asarray(entry[f'times{side}'])
                if times.shape != (len(array),):
                    raise ValueError(f'{pair_id}: timestamp rows do not match input rows.')
                name = f'inputs/{record["key"]}_times{side}.npy'
                np.save(bundle / name, times, allow_pickle=False)
                record['times'].append(dict(path=name, sha256=sha256(bundle / name)))
        manifest['pairs'].append(record)
    if not manifest['pairs']:
        raise ValueError('No task pairs to prepare.')
    for name, source in (('subspace_alignment.py', Path(__file__)), ('decomposition.py', DECOMPOSITION_SOURCE)):
        shutil.copy2(source, bundle / name)
        manifest['sources'][name] = sha256(bundle / name)
    (bundle / 'requirements.txt').write_text(
        f'numpy=={np.__version__}\nscipy=={scipy.__version__}\nscikit-learn=={sklearn.__version__}\n'
        f'threadpoolctl=={version("threadpoolctl")}\n')
    manifest['run_id'] = document_hash(manifest)
    atomic_save(bundle / 'manifest.json', manifest)
    print(f'Prepared {len(manifest["pairs"])} pairs in {bundle}; no FA fits run.', flush=True)
    return manifest


def load_manifest(bundle, *, check_inputs=False):
    bundle = Path(bundle)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if manifest['schema'] != 1 or manifest['run_id'] != document_hash({k: v for k, v in manifest.items() if k != 'run_id'}):
        raise ValueError('Manifest changed; create a new bundle for different settings.')
    if manifest['versions'] != versions():
        raise ValueError('Dependency versions differ; install bundle/requirements.txt.')
    for name, expected in manifest['sources'].items():
        source = DECOMPOSITION_SOURCE if name == 'decomposition.py' else HERE / name
        if not source.exists() or sha256(source) != expected:
            raise ValueError(f'{name} differs from frozen source; run the script inside the bundle.')
    if check_inputs:
        for pair in manifest['pairs']:
            for item in pair['inputs'] + pair.get('times', []):
                if sha256(bundle / item['path']) != item['sha256']:
                    raise ValueError(f'Input changed: {item["path"]}')
    return manifest


def checkpoint_path(bundle, key, condition, index):
    return Path(bundle) / 'checkpoints' / key / condition / f'{index:04d}.npz'


def read_checkpoint(path, manifest, pair, condition, index, d=None):
    saved = load_npz(path)
    meta = json.loads(str(saved['metadata']))
    if (meta['run_id'], meta['pair_id'], meta['condition'], meta['index']) != (
            manifest['run_id'], pair['pair_id'], condition, index):
        raise ValueError(f'Checkpoint identity mismatch: {path}')
    if saved['pa'].shape != (int(saved['d']),) or (d is not None and int(saved['d']) != d):
        raise ValueError(f'Checkpoint angle dimensions mismatch: {path}')
    if saved['ve'].shape != (8,) or not all(np.isfinite(saved[k]).all() for k in ('pa', 'gd', 've')):
        raise ValueError(f'Invalid checkpoint values: {path}')
    if not np.isclose(saved['gd'], np.linalg.norm(saved['pa'])) or not np.allclose(
            saved['ve'][6:], saved['ve'][[0, 3]] - saved['ve'][[1, 4]]):
        raise ValueError(f'Inconsistent checkpoint metrics: {path}')
    if condition != 'data':
        reference_path = Path(path).parents[1] / 'data' / '0000.npz'
        if not reference_path.exists() or meta['reference_sha256'] != sha256(reference_path):
            raise ValueError(f'Checkpoint refers to different observed fits: {path}')
    return saved


def run_job(job):
    """Compute and atomically save one observed comparison or surrogate repeat."""
    bundle, manifest, pair, condition, index = job
    path = checkpoint_path(bundle, pair['key'], condition, index)
    if path.exists():
        read_checkpoint(path, manifest, pair, condition, index)
        return f'{pair["key"]}/{condition}/{index}: already saved', True
    started = time.monotonic()
    try:
        arrays = [np.load(Path(bundle) / item['path'], mmap_mode='r', allow_pickle=False) for item in pair['inputs']]
        reference = None
        if condition != 'data':
            observed = read_checkpoint(checkpoint_path(bundle, pair['key'], 'data', 0),
                                       manifest, pair, 'data', 0)
            reference = {k: observed[k] for k in ('L1', 'L2', 'C1', 'C2')}
        with threadpool_limits(limits=1):
            result = pagdve(*arrays, pair_id=pair['pair_id'], **manifest['params'],
                            _reference=reference, _jobs=[] if condition == 'data' else [(condition, index)])
        meta = dict(result['metadata'], run_id=manifest['run_id'], condition=condition, index=index,
                    host=socket.gethostname(), seconds=time.monotonic()-started, versions=versions())
        if condition != 'data':
            meta['reference_sha256'] = sha256(checkpoint_path(bundle, pair['key'], 'data', 0))
        saved = dict(pa=result['data_pa'] if condition == 'data' else result[f'{condition}_pa'][0],
                     gd=result['data_gd'] if condition == 'data' else result[f'{condition}_gd'][0],
                     ve=result['data_ve'] if condition == 'data' else result[f'{condition}_ve'][0],
                     d=result['d'], K=result['K'], ranks=result['ranks'],
                     metadata=np.array(json.dumps(meta, allow_nan=False)))
        if condition == 'data':
            saved.update(result['reference'], denominators=result['denominators'])
        elif condition == 'rand':
            saved.update(task1_pa=result['rand_task1_pa'][0], task2_pa=result['rand_task2_pa'][0])
        atomic_save(path, saved)
        path.with_suffix('.failure.json').unlink(missing_ok=True)
        return f'{pair["key"]}/{condition}/{index}: saved ({meta["seconds"]:.1f}s)', True
    except Exception:
        # Catch only at the job boundary, record the full failure, and return failure status.
        atomic_save(path.with_suffix('.failure.json'), dict(run_id=manifest['run_id'],
                    pair_id=pair['pair_id'], condition=condition, index=index,
                    seed=manifest['params']['seed'], traceback=traceback.format_exc()))
        return f'{pair["key"]}/{condition}/{index}: FAILED; see {path.with_suffix(".failure.json")}', False


def run_bundle(bundle, *, pair_keys=None):
    bundle = Path(bundle).resolve()
    manifest = load_manifest(bundle, check_inputs=True)
    pairs = [p for p in manifest['pairs'] if pair_keys is None or p['key'] in pair_keys]
    if not pairs or (pair_keys is not None and set(pair_keys) - {p['key'] for p in pairs}):
        raise ValueError('Unknown or empty pair selection.')
    # Prevent two accidental launches from writing the same run.
    lock = bundle / 'RUNNING.json'
    try:
        stream = lock.open('x')
    except FileExistsError:
        raise RuntimeError(f'{lock} exists. Check its host/PID before removing a stale marker.') from None
    with stream:
        json.dump(dict(host=socket.gethostname(), pid=os.getpid()), stream)
    failures = 0
    try:
        for phase in ('data', 'repeats'):
            jobs = []
            for pair in pairs:
                reference_path = checkpoint_path(bundle, pair['key'], 'data', 0)
                if phase == 'repeats' and not reference_path.exists():
                    continue
                identities = [('data', 0)]
                if phase == 'repeats':
                    identities = []
                    for condition in CONDITIONS:
                        count = manifest['params']['n_rands' if condition == 'rand' else 'n_shufs']
                        identities.extend((condition, i) for i in range(count))
                for condition, index in identities:
                    path = checkpoint_path(bundle, pair['key'], condition, index)
                    if path.exists():
                        read_checkpoint(path, manifest, pair, condition, index)
                    else:
                        jobs.append((str(bundle), manifest, pair, condition, index))
            # Visit each pair/control at a repeat index before advancing the index.
            jobs.sort(key=lambda job: (job[4], job[2]['key'], job[3]))
            print(f'{phase}: {len(jobs)} pending calculations; sequential execution.', flush=True)
            for job in jobs:
                message, ok = run_job(job)
                print(message, flush=True)
                failures += not ok
    finally:
        lock.unlink(missing_ok=True)
    if failures:
        raise RuntimeError(f'{failures} jobs failed; valid checkpoints retained. Inspect *.failure.json.')


def collect_bundle(bundle, *, status_only=False):
    bundle = Path(bundle)
    manifest = load_manifest(bundle)
    coverage, collected = [], []
    for pair in manifest['pairs']:
        output, complete = {}, True
        d = None
        for condition in ('data', *CONDITIONS):
            count = 1 if condition == 'data' else manifest['params']['n_rands' if condition == 'rand' else 'n_shufs']
            records, missing, failed = [], [], []
            for index in range(count):
                path = checkpoint_path(bundle, pair['key'], condition, index)
                if not path.exists():
                    missing.append(index)
                    if path.with_suffix('.failure.json').exists():
                        failed.append(index)
                    continue
                record = read_checkpoint(path, manifest, pair, condition, index, d=d)
                d = int(record['d'])
                records.append(record)
            complete &= not missing
            coverage.append(dict(key=pair['key'], pair_id=pair['pair_id'], condition=condition,
                                 saved=len(records), planned=count, missing=missing, failed=failed))
            if not missing and not status_only and d is not None:
                for metric, width in (('pa', d), ('gd', None), ('ve', 8)):
                    values = np.asarray([r[metric] for r in records])
                    if condition == 'data':
                        output[f'data_{metric}'] = values[0]
                    else:
                        output[f'{condition}_{metric}'] = values.reshape((count,) if width is None else (count, width))
                if condition == 'data':
                    output.update({k: records[0][k] for k in ('ranks', 'd', 'K', 'denominators')})
                else:
                    output[f'{condition}_index'] = np.arange(count)
                if condition == 'rand':
                    for side in (1, 2):
                        output[f'rand_task{side}_pa'] = np.asarray([r[f'task{side}_pa'] for r in records]).reshape(count, d)
        if complete and not status_only:
            output['metadata'] = np.array(json.dumps(dict(pair, run_id=manifest['run_id'],
                    ve_columns=VE_COLUMNS, params=manifest['params']), allow_nan=False))
            collected.append((pair['key'], ve_main_view(output)))
    if status_only:
        for row in coverage:
            print(f'{row["key"]} {row["condition"]}: {row["saved"]}/{row["planned"]}, {len(row["failed"])} failed')
        return coverage
    if any(row['missing'] for row in coverage):
        raise RuntimeError('Incomplete results; run status to see missing/failed repeats. No final arrays written.')
    for key, output in collected:
        atomic_save(bundle / 'results' / f'{key}.npz', output)
    atomic_save(bundle / 'results' / 'index.json', dict(run_id=manifest['run_id'],
                pairs=[dict(key=p['key'], pair_id=p['pair_id'], path=f'{p["key"]}.npz',
                            metadata=p['metadata']) for p in manifest['pairs']]))
    print(f'Collected {len(collected)} complete pairs into {bundle / "results"}.', flush=True)
    return coverage


def frozen_runner_path(bundle):
    """Validate the saved source before dispatching an old or new bundle."""
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if manifest['schema'] != 1 or manifest['run_id'] != document_hash(
            {k: v for k, v in manifest.items() if k != 'run_id'}):
        raise ValueError('Manifest changed; create a new bundle for different settings.')
    name = next((candidate for candidate in ('subspace_alignment.py', 'pagdve.py', 'fig6.py')
                 if candidate in manifest['sources']), None)
    if name not in manifest['sources']:
        raise ValueError('Manifest does not identify a frozen PAGDVE runner.')
    for source, expected in manifest['sources'].items():
        path = bundle / source
        if path.resolve().parent != bundle or not path.is_file() or sha256(path) != expected:
            raise ValueError(f'Frozen source changed: {source}')
    return bundle / name


def finalize_bundle(bundle, *, wait=False):
    """Collect and audit saved fits with the frozen runner; never refit or plot.

    Historical fig6.py/pagdve.py bundles and current subspace_alignment.py
    bundles keep their original source/manifest/checkpoint identity.
    The current storage adapter presents the same four-column VE views for both.
    """
    bundle = Path(bundle).resolve()
    source = frozen_runner_path(bundle)
    spec = importlib.util.spec_from_file_location('_fig6_frozen_runner', source)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    manifest = engine.load_manifest(bundle, check_inputs=True)
    marker = bundle / 'FINALIZING.json'
    with marker.open('x') as stream:
        json.dump(dict(pid=os.getpid(), run_id=manifest['run_id']), stream)
    try:
        engine.atomic_save(bundle / 'completion.json', dict(status='waiting_for_fits', run_id=manifest['run_id']))
        if (bundle / 'RUNNING.json').exists() and not wait:
            raise RuntimeError('Fits are still running; use --wait to finish after them.')
        while (bundle / 'RUNNING.json').exists():
            time.sleep(5)
        # An interrupted or failed run cannot pass collection, so partial
        # results cannot be marked as the complete comparison.
        engine.atomic_save(bundle / 'completion.json', dict(status='validating', run_id=manifest['run_id']))
        coverage = engine.collect_bundle(bundle)
        update_ve_results(bundle)
        fit_count, max_iterations, rejected_attempts, checked_splits = 0, 0, 0, 0
        for pair in manifest['pairs']:
            for condition in ('data', *engine.CONDITIONS):
                count = 1 if condition == 'data' else manifest['params']['n_rands' if condition == 'rand' else 'n_shufs']
                for index in range(count):
                    saved = engine.load_npz(engine.checkpoint_path(bundle, pair['key'], condition, index))
                    metadata = json.loads(str(saved['metadata']))
                    for fit in metadata['fits']:
                        if not fit['converged'] or fit['n_iter'] > manifest['params']['max_iter']:
                            raise ValueError(f'Unconverged fit: {pair["key"]}/{condition}/{index}')
                        fit_count += 1
                        max_iterations = max(max_iterations, fit['n_iter'])
                    for repeat in metadata['repeats']:
                        rejected_attempts += len(repeat['rejected_attempts'])
                        if condition == 'rand':
                            continue
                        side = 0 if condition == 'shuf_task1' else 1
                        n = pair['inputs'][side]['shape'][0]
                        first, second = repeat['half1_indices'], repeat['half2_indices']
                        if (len(first) != n // 2 or len(second) != n - n // 2
                                or not np.array_equal(np.sort(first + second), np.arange(n))):
                            raise ValueError(f'Invalid complementary split: {pair["key"]}/{condition}/{index}')
                        checked_splits += 1
        report = dict(run_id=manifest['run_id'], params=manifest['params'], coverage=coverage,
                      converged_fits=fit_count, maximum_fa_iterations=max_iterations,
                      rejected_attempts=rejected_attempts, checked_complementary_splits=checked_splits)
        engine.atomic_save(bundle / 'verification.json', report)
        engine.atomic_save(bundle / 'completion.json', dict(status='complete', run_id=manifest['run_id'],
                        completed_utc=datetime.now(timezone.utc).isoformat(),
                        collected_pairs=len(manifest['pairs']), analysis_script='subspace_comparison.py'))
        print(f'Complete: {len(manifest["pairs"])} pairs, {fit_count} converged fits. '
              f'Results: {bundle / "results"}. Analyze with the root subspace_comparison.py.', flush=True)
        return report
    except KeyboardInterrupt:
        engine.atomic_save(bundle / 'completion.json', dict(status='interrupted', run_id=manifest['run_id']))
        raise
    except Exception:
        engine.atomic_save(bundle / 'completion.json', dict(status='failed', run_id=manifest['run_id'],
                        traceback=traceback.format_exc()))
        raise
    finally:
        marker.unlink(missing_ok=True)


#%% Command-line entry point

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare', help='Freeze current session cache and matched channels; no fitting.')
    prepare.add_argument('--cache', type=Path, default=ROOT / 'data/temp/df_all.h5')
    prepare.add_argument('--bundle', type=Path, required=True)
    for name in ('n-rands', 'n-shufs', 'seed', 'max-iter', 'max-attempts'):
        prepare.add_argument('--' + name, type=int, default=DEFAULT_PARAMS[name.replace('-', '_')])
    prepare.add_argument('--tol', type=float, default=DEFAULT_PARAMS['tol'])
    prepare.add_argument('--shared-var-thresh', type=float, default=.95)
    prepare.add_argument('--svd-method', choices=('lapack', 'randomized'), default='lapack')
    run = sub.add_parser('run', help='Sequentially compute all missing repeats.')
    run.add_argument('--bundle', type=Path, default=HERE)
    run.add_argument('--pair', action='append', help='Optional pair key, e.g. pair_000; repeat for multiple pairs.')
    views = sub.add_parser('ve-views', help='Update all collected VE views without fitting; keep raw checkpoints.')
    views.add_argument('--bundle', type=Path, required=True)
    for name in ('status', 'collect'):
        p = sub.add_parser(name)
        p.add_argument('--bundle', type=Path, default=HERE)
    finalize = sub.add_parser('finalize', help='Collect and audit a finished run; plotting is in root subspace_comparison.py.')
    finalize.add_argument('--bundle', type=Path, required=True)
    finalize.add_argument('--wait', action='store_true', help='Wait for RUNNING.json to disappear.')
    args = parser.parse_args(argv)
    if args.command in ('run', 'status', 'collect'):
        # Resume historical runs with their exact frozen scientific code.
        frozen = frozen_runner_path(args.bundle)
        if frozen != Path(__file__).resolve():
            command = [sys.executable, '-u', str(frozen), args.command,
                       '--bundle', str(args.bundle.resolve())]
            if args.command == 'run':
                for key in args.pair or []:
                    command.extend(['--pair', key])
            subprocess.run(command, check=True)
            if args.command == 'collect':
                update_ve_results(args.bundle)
            return
    if args.command == 'prepare':
        # Heavy manuscript/session imports occur only during local input export.
        import pandas as pd
        from utils import get_paired_df
        cache = args.cache.resolve()
        df_pairs = get_paired_df(pd.read_hdf(cache, key='all'))
        cache_hash = sha256(cache)
        entries = []
        for row in df_pairs.itertuples():
            t1, t2 = row.task_1, row.task_2
            if t1.matfile_type != t2.matfile_type:
                raise ValueError('Cannot align unlike channel-identifier types.')
            field = 'filename' if t1.matfile_type == 'mat' else 'seq_num'
            channels = sorted(row.common)
            if len(channels) < 2:
                raise ValueError(f'{t1.session_key}/{t2.session_key}: fewer than 2 common neurons.')
            arrays, source_indices = [], []
            for task in (t1, t2):
                labels = task.df_chan[field].tolist()
                if len(set(labels)) != len(labels):
                    raise ValueError(f'{task.session_key}: duplicate channel identifiers.')
                positions = {label: i for i, label in enumerate(labels)}
                ix = [positions[channel] for channel in channels]
                arrays.append(task.spikes_per_bin[:, ix])
                source_indices.append(ix)
            metadata = dict(patient=str(row.subj), task_pair=row.task_pair,
                sessions=[t1.session_key, t2.session_key], tasks=[t1.task, t2.task],
                channel_field=field, channels=[str(c) for c in channels], source_indices=source_indices,
                bin_size=[getattr(t, 'bin_size', None) for t in (t1, t2)],
                event_mode=[getattr(t, 'mode', None) for t in (t1, t2)],
                max_gap_dur_secs=[getattr(t, 'max_gap_dur_secs', None) for t in (t1, t2)],
                cache_path=str(cache), cache_sha256=cache_hash)
            entries.append(dict(pair_id=f'{row.subj}:{t1.session_key}:{t2.session_key}',
                                Y1=arrays[0], Y2=arrays[1], metadata=metadata,
                                times1=t1.bin_times, times2=t2.bin_times))
        settings = {k: getattr(args, k) for k in DEFAULT_PARAMS}
        prepare_bundle(entries, args.bundle, **settings)
    elif args.command == 'run':
        run_bundle(args.bundle, pair_keys=args.pair)
    elif args.command == 'finalize':
        finalize_bundle(args.bundle, wait=args.wait)
    elif args.command == 've-views':
        update_ve_results(args.bundle)
    else:
        collect_bundle(args.bundle, status_only=args.command == 'status')


if __name__ == '__main__':
    main()
