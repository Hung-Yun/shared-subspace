#!/usr/bin/env python3
"""Prepare, run, inspect, and collect fixed-rank ACC/HPC population shifts.

Only prepare reads the old CV bundle. Workers perform one FA fit per task,
never CV. The final pickle is session -> shifted_acc/shifted_hpc -> index -> fit.
Load only trusted research bundles: joblib/pickle can execute Python code.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import sys
import tempfile
import time
import traceback
import warnings

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning, InconsistentVersionWarning
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
REPO = HERE if (HERE / 'decomposition.py').exists() else HERE.parent
sys.path.insert(0, str(HERE if (HERE / 'decomposition.py').exists() else REPO))
from decomposition import FAResults

CONDITIONS = ('shifted_hpc', 'shifted_acc')
FIT_FIELDS = ('fa', 'n_components', 'd_shared', 'shared_var_per_unit',
              'explained_variance_ratio_', 'subspace', 'first_d_components')
DEFAULT_PARAMS = dict(max_iter=10000, tol=1e-4, svd_method='lapack')


#%% File and validation helpers

def now():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    print(f'[{now()}] {message}', flush=True)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def document_hash(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, allow_nan=False).encode()).hexdigest()


def versions():
    return dict(numpy=np.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, joblib=joblib.__version__)


def read_pickle(path):
    with warnings.catch_warnings():
        warnings.simplefilter('error', InconsistentVersionWarning)
        return joblib.load(path)


def atomic_save(path, value, *, as_json=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    os.close(fd)
    try:
        if as_json:
            Path(temporary).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        else:
            joblib.dump(value, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_manifest(bundle, *, check_inputs=False):
    bundle = Path(bundle)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    payload = {k: v for k, v in manifest.items() if k != 'run_id'}
    if manifest['schema'] != 1 or document_hash(payload) != manifest['run_id']:
        raise ValueError('Manifest changed. Prepare a separate run for different inputs/settings.')
    if manifest['versions'] != versions():
        raise ValueError('Dependency versions differ. Install this bundle\'s requirements.txt.')
    if sha256(sys.modules['decomposition'].__file__) != manifest['source_sha256']['decomposition.py']:
        raise ValueError('FAResults differs from the prepared implementation.')
    if check_inputs:
        checks = {'original_cache.pkl': manifest['baseline_sha256']}
        for session in manifest['sessions']:
            checks[session['array']] = session['array_sha256']
            checks[session['bin_times']] = session['bin_times_sha256']
        for name, expected in checks.items():
            if sha256(bundle / name) != expected:
                raise ValueError(f'Input changed: {name}')
        log(f'Validated {len(checks)} input files; no FA fits performed.')
    return manifest


def jobs(manifest):
    # No round barrier: idle workers immediately take the next available job.
    return [(s['key'], condition, index)
            for index in range(manifest['n_shifts'])
            for s in manifest['sessions'] for condition in CONDITIONS]


def paths_for(bundle, job):
    key, condition, index = job
    stem = Path(bundle) / 'results' / key / condition / f'{index:03d}'
    return stem.with_suffix('.joblib'), stem.with_suffix('.json')


def validate_fit(fit, session, condition, index, params):
    model, rank = fit['fa'], session['rank']
    actual = model.get_params()
    if (fit['n_components'] != rank or model.n_components != rank
            or fit['d_shared'] != rank or model.components_.shape != (rank, session['n_units'])
            or fit['offset_bins'] != session['offsets'][condition][index]
            or actual['tol'] != params['tol'] or actual['svd_method'] != params['svd_method']
            or actual['max_iter'] < params['max_iter']
            or fit['params'] != {k: actual[k] for k in params}
            or fit['n_iter'] != model.n_iter_ or not 1 <= model.n_iter_ <= actual['max_iter']
            or np.shape(fit['shared_var_per_unit']) != (session['n_units'],)
            or np.shape(fit['subspace']) != (rank, session['n_units'])):
        raise ValueError('Fitted rank, offset, shape, or settings differ from the planned task.')
    shared = np.sum(model.components_ ** 2, axis=0)
    noise = model.noise_variance_
    if (not np.isfinite(shared).all() or not np.isfinite(noise).all() or np.any(noise <= 0)
            or not np.isfinite(model.mean_).all() or not np.isfinite(fit['subspace']).all()
            or not np.isfinite(model.loglike_).all()):
        raise ValueError('Invalid fitted covariance or likelihood.')
    np.testing.assert_allclose(fit['shared_var_per_unit'], shared / (shared + noise), rtol=1e-10, atol=1e-12)
    capped = any(w['category'] == 'ConvergenceWarning' for w in fit['warnings'])
    if type(fit['converged']) is not bool or fit['converged'] == capped:
        raise ValueError('Inconsistent convergence flag.')


def read_result(bundle, manifest, session, job):
    filename, _ = paths_for(bundle, job)
    saved = read_pickle(filename)
    if saved['run_id'] != manifest['run_id'] or tuple(saved['job']) != tuple(job):
        raise ValueError(f'Result belongs to a different run/task: {filename}')
    validate_fit(saved['fit'], session, job[1], job[2], manifest['params'])
    return saved['fit']


#%% Prepare inputs

def prepare(args):
    source, bundle = args.source.resolve(), args.bundle.resolve()
    if bundle.exists():
        raise FileExistsError(f'{bundle} already exists; preparation never overwrites a run.')
    if (args.n_shifts < 1 or args.seed < 0 or not np.isfinite(args.min_shift_seconds)
            or args.min_shift_seconds <= 0):
        raise ValueError('Use positive shifts/minimum lag and a nonnegative seed.')
    old = json.loads((source / 'manifest.json').read_text())
    if old['versions'] != versions():
        raise ValueError('Use the archived bundle\'s pinned dependency versions for preparation.')
    baseline_hash = sha256(args.original_cache)
    baseline = read_pickle(args.original_cache)
    old_entries = {s['key']: s for s in old['sessions']}
    sessions, excluded, prepared = [], [], []
    with threadpool_limits(limits=1):
        for key, original in baseline.items():
            if not original.get('fa_acc') or not original.get('fa_hpc'):
                excluded.append(dict(key=key, reason='Both ACC and HPC are required'))
                log(f'{key}: excluded; both regions are required.')
                continue
            if not re.fullmatch(r'[A-Za-z0-9_-]+', key):
                raise ValueError(f'Unsafe session key: {key}')
            entry = old_entries[key]
            for field in ('array', 'record'):
                if sha256(source / entry[field]) != entry[field + '_sha256']:
                    raise ValueError(f'Archived {field} changed: {key}')
            record = read_pickle(source / entry['record'])
            matrix = np.load(source / entry['array'], allow_pickle=False)
            acc, hpc = np.asarray(record['acc_ix']), np.asarray(record['hpc_ix'])
            np.testing.assert_array_equal(np.sort(np.r_[acc, hpc]), np.arange(matrix.shape[1]))
            if not len(acc) or not len(hpc) or not np.isfinite(matrix).all():
                raise ValueError(f'{key}: invalid region indices or matrix.')
            for name, columns in (('fa_cv', np.arange(matrix.shape[1])), ('fa_acc', acc), ('fa_hpc', hpc)):
                fit = original[name]
                model = fit['fa']
                if model.components_.shape[1] != len(columns):
                    raise ValueError(f'{key}/{name}: channel count changed.')
                score = model.score(matrix[:, columns])
                if not np.isclose(score, model.loglike_[-1] / len(matrix), rtol=1e-9, atol=1e-7):
                    raise ValueError(f'{key}/{name}: input does not reproduce the original training likelihood.')
                shared = (model.components_ ** 2).sum(axis=0)
                np.testing.assert_allclose(fit['shared_var_per_unit'], shared / (shared + model.noise_variance_), atol=1e-10)
            rank = int(original['fa_cv']['n_components'])
            if rank != original['fa_cv']['fa'].n_components or not 0 <= rank < matrix.shape[1]:
                raise ValueError(f'{key}: invalid original joint rank.')
            meta = record['metadata']
            minimum = max(1, int(np.ceil(args.min_shift_seconds * 1000 / meta['bin_size_ms'])))
            allowed = np.arange(minimum, len(matrix) - minimum + 1)
            if len(allowed) < args.n_shifts:
                raise ValueError(f'{key}: not enough distinct allowed shifts.')
            offsets = {}
            for condition in CONDITIONS:
                seed = int(hashlib.sha256(f'{args.seed}:{key}:{condition}'.encode()).hexdigest()[:16], 16)
                candidates = np.random.default_rng(seed).permutation(allowed).tolist()
                prior = list(map(int, record['offsets'])) if condition == 'shifted_hpc' else []
                prior = list(dict.fromkeys(x for x in prior if minimum <= x <= len(matrix) - minimum))
                offsets[condition] = (prior + [x for x in candidates if x not in prior])[:args.n_shifts]
            session = dict(key=key, patient=meta['patient'], task=meta['task'], rank=rank,
                           n_bins=len(matrix), n_units=matrix.shape[1], acc_ix=acc.tolist(), hpc_ix=hpc.tolist(),
                           channel_indices=np.asarray(meta['channel_indices']).tolist(),
                           bin_size_ms=meta['bin_size_ms'], n_time_gaps=meta['n_time_gaps'],
                           shift_axis='retained_bin_order', event_mode=meta['event_mode'],
                           max_gap_dur_secs=meta['max_gap_dur_secs'], min_fr=meta['min_fr'],
                           offsets=offsets, array=f'inputs/{key}.npy', array_sha256=entry['array_sha256'],
                           bin_times=f'inputs/{key}_times.npy')
            times = np.asarray(meta['bin_times_ms'])
            if len(times) != len(matrix) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise ValueError(f'{key}: invalid retained-bin times.')
            sessions.append(session)
            prepared.append((source / entry['array'], times))
            log(f'{key}: verified all three original models; fixed joint rank {rank}; {len(acc)} ACC + {len(hpc)} HPC units.')
    if not sessions:
        raise ValueError('No eligible dual-region sessions.')
    (bundle / 'inputs').mkdir(parents=True)
    for session, (array, times) in zip(sessions, prepared):
        shutil.copyfile(array, bundle / session['array'])
        np.save(bundle / session['bin_times'], times, allow_pickle=False)
        session['bin_times_sha256'] = sha256(bundle / session['bin_times'])
    shutil.copyfile(args.original_cache, bundle / 'original_cache.pkl')
    if sha256(bundle / 'original_cache.pkl') != baseline_hash:
        raise ValueError('Original cache changed during preparation; do not use this partial bundle.')
    shutil.copyfile(__file__, bundle / 'shift_controls.py')
    shutil.copyfile(sys.modules['decomposition'].__file__, bundle / 'decomposition.py')
    shutil.copyfile(REPO / 'requirements.txt', bundle / 'requirements.txt')
    shutil.copyfile(REPO / 'README.md', bundle / 'README.md')
    manifest = dict(schema=1, created_utc=now(), n_shifts=args.n_shifts, seed=args.seed,
                    min_shift_seconds=args.min_shift_seconds, params=DEFAULT_PARAMS.copy(),
                    versions=versions(), sessions=sessions, excluded_sessions=excluded,
                    baseline_sha256=baseline_hash,
                    source_sha256={name: sha256(bundle / name) for name in ('shift_controls.py', 'decomposition.py')})
    manifest['run_id'] = document_hash(manifest)
    atomic_save(bundle / 'manifest.json', manifest, as_json=True)
    log(f'READY: {len(sessions)} sessions x {args.n_shifts} shifts x 2 regions = {len(jobs(manifest))} fresh fits. No old shifted fits reused.')


#%% Fit regional shift controls and save checkpoints

def initialize_worker(bundle, manifest):
    global WORK_BUNDLE, WORK_MANIFEST, WORK_SESSIONS, ARRAYS
    WORK_BUNDLE, WORK_MANIFEST = Path(bundle), manifest
    WORK_SESSIONS = {s['key']: s for s in manifest['sessions']}
    ARRAYS = {}
    if mp.current_process().name != 'MainProcess':
        signal.signal(signal.SIGINT, signal.SIG_IGN)


def run_task(argument):
    job, retry_max_iter = argument
    key, condition, index = job
    session = WORK_SESSIONS[key]
    result_path, state_path = paths_for(WORK_BUNDLE, job)
    label = f'{key}/{condition}/{index:03d}'
    previous = read_result(WORK_BUNDLE, WORK_MANIFEST, session, job) if result_path.exists() else None
    if previous is not None and (previous['converged'] or retry_max_iter is None):
        return label, 'cached'
    params = WORK_MANIFEST['params'].copy()
    if previous is not None:
        params['max_iter'] = retry_max_iter
    state = dict(run_id=WORK_MANIFEST['run_id'], job=list(job), status='running',
                 pid=os.getpid(), host=socket.gethostname(), started_utc=now(),
                 rank=session['rank'], offset_bins=session['offsets'][condition][index], params=params)
    atomic_save(state_path, state, as_json=True)
    try:
        log(f'{label}: START rank={session["rank"]}, offset={state["offset_bins"]}, max_iter={params["max_iter"]}, tol={params["tol"]}; no CV')
        if key not in ARRAYS:
            ARRAYS[key] = np.load(WORK_BUNDLE / session['array'], mmap_mode='r', allow_pickle=False)
        matrix = np.array(ARRAYS[key], copy=True)
        indices = session['hpc_ix'] if condition == 'shifted_hpc' else session['acc_ix']
        matrix[:, indices] = np.roll(matrix[:, indices], state['offset_bins'], axis=0)
        started = time.monotonic()
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = FAResults().fa_fit(matrix, n_components=session['rank'], cv_components=None, **params)
        fit = {name: getattr(result, name) for name in FIT_FIELDS}
        fit.update(offset_bins=state['offset_bins'], params=params,
                   converged=not any(issubclass(w.category, ConvergenceWarning) for w in caught),
                   warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught],
                   n_iter=int(result.fa.n_iter_), seconds=time.monotonic() - started,
                   completed_utc=now(), attempt=1 if previous is None else previous['attempt'] + 1)
        validate_fit(fit, session, condition, index, WORK_MANIFEST['params'])
        atomic_save(result_path, dict(run_id=WORK_MANIFEST['run_id'], job=list(job), fit=fit))
        state.update(status='complete' if fit['converged'] else 'flagged', updated_utc=now(),
                     n_iter=fit['n_iter'], seconds=fit['seconds'])
        atomic_save(state_path, state, as_json=True)
        log(f'{label}: SAVED {state["status"]}; iterations={fit["n_iter"]}, seconds={fit["seconds"]:.2f}')
        return label, state['status']
    except Exception:
        state.update(status='failed', updated_utc=now(), error=traceback.format_exc())
        atomic_save(state_path, state, as_json=True)
        log(f'{label}: FAILED; other checkpoints retained.\n{state["error"]}')
        return label, 'failed'


def process_alive(owner):
    if owner.get('host') != socket.gethostname():
        return None
    try:
        os.kill(owner['pid'], 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return None


@contextmanager
def launcher_lock(bundle, clear_stale=False):
    # Exclusive creation works without flock on the server's shared filesystem.
    path = Path(bundle) / 'RUNNING.json'
    if path.exists() and clear_stale:
        owner = json.loads(path.read_text())
        if process_alive(owner) is not False:
            raise RuntimeError('Cannot clear a live or unverified launcher lock.')
        for state_path in (Path(bundle) / 'results').glob('*/*/*.json'):
            state = json.loads(state_path.read_text())
            if state.get('status') == 'running' and process_alive(state) is not False:
                raise RuntimeError('A previous worker is still alive or unverified; stop it before clearing the lock.')
        path.unlink()
    try:
        stream = path.open('x')
    except FileExistsError:
        raise RuntimeError('A launcher owns this bundle. If it was killed, use --clear-stale-lock on the same host.') from None
    try:
        with stream:
            json.dump(dict(pid=os.getpid(), host=socket.gethostname(), started_utc=now()), stream)
        yield
    finally:
        path.unlink(missing_ok=True)


def run(args):
    bundle = args.bundle.resolve()
    manifest = load_manifest(bundle, check_inputs=True)
    if args.workers < 1:
        raise ValueError('workers must be positive.')
    if args.retry_max_iter is not None and not args.retry_flagged:
        raise ValueError('--retry-max-iter requires --retry-flagged.')
    cap = manifest['params']['max_iter'] if args.retry_max_iter is None else args.retry_max_iter
    if cap < manifest['params']['max_iter']:
        raise ValueError('Retry cap must be at least the initial cap.')
    sessions = {s['key']: s for s in manifest['sessions']}
    with launcher_lock(bundle, args.clear_stale_lock):
        queue = []
        for job in jobs(manifest):
            result_path, _ = paths_for(bundle, job)
            if result_path.exists():
                fit = read_result(bundle, manifest, sessions[job[0]], job)
                if fit['converged'] or not args.retry_flagged:
                    continue
            queue.append((job, cap if args.retry_flagged else None))
        log(f'START {len(queue)}/{len(jobs(manifest))} jobs; {args.workers} workers x 1 BLAS thread; round-robin across sessions and regions.')
        if not queue:
            return 0
        failures = 0
        if args.workers == 1:
            initialize_worker(bundle, manifest)
            for number, argument in enumerate(queue, 1):
                label, outcome = run_task(argument)
                failures += outcome == 'failed'
                log(f'[{number}/{len(queue)}] {label}: {outcome}')
        else:
            with mp.get_context('spawn').Pool(args.workers, initializer=initialize_worker, initargs=(bundle, manifest)) as pool:
                iterator = pool.imap_unordered(run_task, queue, chunksize=1)
                completed = 0
                while completed < len(queue):
                    try:
                        label, outcome = iterator.next(timeout=30)
                    except mp.TimeoutError:
                        log(f'WAITING: {completed}/{len(queue)} dispatched jobs finished; current fits are still running.')
                        continue
                    completed += 1
                    failures += outcome == 'failed'
                    log(f'[{completed}/{len(queue)}] {label}: {outcome}')
        log('Run finished. Inspect status for flagged or failed fits before collection.')
        return 1 if failures else 0


#%% Status and result collection

def snapshot(bundle, manifest):
    rows = []
    sessions = {s['key']: s for s in manifest['sessions']}
    for job in jobs(manifest):
        result_path, state_path = paths_for(bundle, job)
        row = dict(session=job[0], condition=job[1], shift_index=job[2], status='pending', saved=result_path.exists())
        try:
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            if state and (state['run_id'] != manifest['run_id'] or tuple(state['job']) != job):
                raise ValueError('Status belongs to another run/task.')
            alive = process_alive(state) if state.get('status') == 'running' else False
            if state.get('status') == 'running' and alive is not False:
                row['status'] = 'active' if alive is True else 'unknown'
            elif row['saved']:
                if state.get('status') in ('complete', 'flagged'):
                    row['status'] = state['status']
                else:
                    fit = read_result(bundle, manifest, sessions[job[0]], job)
                    row['status'] = 'complete' if fit['converged'] else 'flagged'
            elif state.get('status') == 'running':
                row['status'] = 'interrupted'
            elif state:
                row['status'] = 'failed' if state.get('status') == 'failed' else 'interrupted'
            if 'error' in state:
                row['error'] = state['error']
        except Exception as error:
            row.update(status='unreadable', error=str(error))
        rows.append(row)
    return rows


def status(args):
    manifest = load_manifest(args.bundle)
    rows = snapshot(args.bundle, manifest)
    counts = {name: sum(r['status'] == name for r in rows)
              for name in ('complete', 'flagged', 'active', 'interrupted', 'failed', 'pending', 'unknown', 'unreadable')}
    print(f'Read-only snapshot: {now()}')
    print(json.dumps(dict(tasks=counts, total=len(rows), saved=sum(r['saved'] for r in rows)), indent=2))
    print('\nSession      Converged  Flagged  Active  Failed  Saved/total')
    for s in manifest['sessions']:
        subset = [r for r in rows if r['session'] == s['key']]
        count = lambda name: sum(r['status'] == name for r in subset)
        print(f'{s["key"]:<12} {count("complete"):>8} {count("flagged"):>8} {count("active"):>7} {count("failed"):>7}  {sum(r["saved"] for r in subset)}/{len(subset)}')
    if args.failed:
        for row in rows:
            if row['status'] in ('failed', 'flagged', 'unreadable'):
                print(f'{row["session"]}/{row["condition"]}/{row["shift_index"]:03d}: {row["status"]}\n{row.get("error", "")}')
    print('Flags are saved nonconverged fits, not successful fits. No files modified.')


# ---------- 4. Collect the flat, integer-indexed analysis pickle ----------

def collect(args):
    bundle = args.bundle.resolve()
    manifest = load_manifest(bundle, check_inputs=True)
    sessions = {s['key']: s for s in manifest['sessions']}
    output = {key: {condition: {} for condition in CONDITIONS} for key in sessions}
    missing, flagged = [], []
    for job in jobs(manifest):
        path, _ = paths_for(bundle, job)
        if not path.exists():
            missing.append(list(job))
            continue
        fit = read_result(bundle, manifest, sessions[job[0]], job)
        output[job[0]][job[1]][job[2]] = fit
        if not fit['converged']:
            flagged.append(list(job))
    if missing and not args.allow_partial:
        raise ValueError(f'{len(missing)} fits have no checkpoint. Use --allow-partial for an explicitly incomplete preview.')
    # No run IDs, fingerprints, or extra dictionary levels in this pickle.
    destination = bundle / 'shifted_data.pkl'
    atomic_save(destination, output)
    coverage = dict(created_utc=now(), run_id=manifest['run_id'], planned=len(jobs(manifest)),
                    saved=len(jobs(manifest)) - len(missing), missing=missing, flagged=flagged,
                    output_sha256=sha256(destination))
    atomic_save(bundle / 'coverage.json', coverage, as_json=True)
    log(f'SAVED {destination}: {coverage["saved"]}/{coverage["planned"]} fits, {len(flagged)} flagged. Missing indices are never renumbered.')


#%% Command-line entry point

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare', help='Validate/copy old inputs and original ranks; no fitting')
    p.add_argument('--source', type=Path, default=REPO / 'archive/fig4e/data')
    p.add_argument('--original-cache', type=Path, default=REPO / 'models/fa_thres_cv_regions.pkl')
    p.add_argument('--bundle', type=Path, default=REPO / 'data/shift_controls')
    p.add_argument('--n-shifts', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--min-shift-seconds', type=float, default=10.)
    p.set_defaults(function=prepare)
    for command, function in (('run', run), ('status', status), ('collect', collect)):
        p = commands.add_parser(command)
        p.add_argument('--bundle', type=Path, default=HERE if (HERE / 'manifest.json').exists() else REPO / 'data/shift_controls')
        p.set_defaults(function=function)
        if command == 'run':
            p.add_argument('--workers', type=int, default=16)
            p.add_argument('--retry-flagged', action='store_true')
            p.add_argument('--retry-max-iter', type=int)
            p.add_argument('--clear-stale-lock', action='store_true')
        elif command == 'status':
            p.add_argument('--failed', action='store_true')
        else:
            p.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args(argv)
    try:
        return args.function(args) or 0
    except KeyboardInterrupt:
        log('Stopped. Saved fits remain intact; rerun the same command to resume.')
        return 130


if __name__ == '__main__':
    sys.exit(main())
