#!/usr/bin/env python3
"""Empirically parameterized Fig. 6 simulations; see README.md for definitions."""
import argparse
import itertools
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy.optimize import minimize
from sklearn.decomposition import FactorAnalysis
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

TASKS = {'C': 'Podcast', 'S': 'Pursuit', 'R': 'Rotations'}
CASES = {1: 'Shared low variance', 2: 'Shared high variance', 3: 'Random variance assignment'}
DEFAULT_SAMPLES = 14300
ROOT = Path(__file__).resolve().parents[2]


def spectrum(model):
    """Shared covariance eigenvalues, ordered by decreasing variance."""
    return np.linalg.svd(model.components_, compute_uv=False) ** 2


def task_code(session):
    return session.split('-')[1][0]


def pairs(models):
    for a, b in itertools.combinations(sorted(models), 2):
        if a.split('-')[0] == b.split('-')[0] and task_code(a) != task_code(b):
            yield a, b


def common_neuron_counts(manifest_path):
    """Optional empirical pair counts; session order does not affect lookup."""
    if manifest_path is None:
        return None
    manifest = json.loads(Path(manifest_path).read_text())
    counts = {}
    for pair in manifest['pairs']:
        sessions = tuple(sorted(pair['metadata']['sessions']))
        shapes = [item['shape'] for item in pair['inputs']]
        n = len(pair['metadata']['channels'])
        if len(sessions) != 2 or len(shapes) != 2 or n < 2 or any(s[1] != n for s in shapes):
            raise ValueError(f'Invalid common-neuron metadata for {sessions}.')
        if sessions in counts:
            raise ValueError(f'Duplicate empirical pair: {sessions}.')
        counts[sessions] = n
    if not counts:
        raise ValueError('The empirical manifest contains no pairs.')
    return counts


def random_bases(n, d, rng):
    """Z random; X and Y independently random in Z's orthogonal complement.

    Each task basis [Z,X] or [Z,Y] is orthonormal, so specified eigenvalues
    are its actual shared variance spectrum. X and Y are NOT mutually orthogonal.
    """
    k = d // 2
    z = np.linalg.qr(rng.normal(size=(n, k)), mode='reduced')[0]
    def specific():
        x = rng.normal(size=(n, k))
        x -= z @ (z.T @ x)
        return np.linalg.qr(x, mode='reduced')[0]
    return specific(), specific(), z


def principal_bases(a, b):
    """Inputs and outputs: N x D orthonormal bases; angles ascending."""
    u, s, vh = np.linalg.svd(a.T @ b, full_matrices=False)
    return a @ u, b @ vh.T, np.rad2deg(np.arccos(np.clip(s, 0, 1)))


def fractions(cov, basis, principal):
    v = np.einsum('ij,ij->j', basis, cov @ basis)
    p = np.einsum('ij,ij->j', principal, cov @ principal)
    k = basis.shape[1] // 2
    return dict(top=v[:k].sum()/v.sum(), more=p[:k].sum()/v.sum(), less=p[k:].sum()/v.sum())


def sample_data(basis, v, psi, t, rng):
    return (rng.standard_normal((t, len(v))) * np.sqrt(v)) @ basis.T + rng.standard_normal((t, len(psi))) * np.sqrt(psi)


def fit_covariance(cov, d, t, tol=1e-6, max_iter=100000, warm_start_iterations=1000):
    """Exact Gaussian-FA sufficient-statistic reduction, retaining T in tolerance.

    Compact observations have zero mean and exactly the original sample
    covariance (ddof=0). sklearn's likelihood is multiplied by row count, so
    scaling tol by (N+1)/T preserves its stopping criterion. The sequence of
    covariance estimates is the same as fitting the full data, up to roundoff.
    """
    n = len(cov)
    ev, evec = np.linalg.eigh(cov)
    compact = np.sqrt(n+1) * helmert(n+1).T @ (np.sqrt(np.maximum(ev, 0))[:, None] * evec.T)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model = FactorAnalysis(n_components=d, svd_method='lapack', tol=tol*(n+1)/t, max_iter=min(max_iter, warm_start_iterations)).fit(compact)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    model.solver_ = 'sklearn'
    if not converged:
        # Profile likelihood over diagonal Psi: the optimal loading covariance
        # conditional on Psi follows from the leading whitened eigenvectors.
        # This accelerates flat/near-boundary solutions of the same FA model.
        def objective(logpsi, return_loadings=False):
            psi = np.exp(logpsi)
            root = np.sqrt(psi)
            whitened = cov / root[:, None] / root[None, :]
            ev, evec = np.linalg.eigh(whitened)
            selected = np.arange(n-d,n)
            selected = selected[ev[selected] > 1]
            lam, vectors = ev[selected], evec[:, selected]
            loading = (root[:,None]*vectors) * np.sqrt(lam-1)
            value = logpsi.sum() + np.log(lam).sum() + ev.sum() - lam.sum() + len(lam)
            inv = np.diag(1/psi) - ((vectors * (1-1/lam)) @ vectors.T) / root[:,None] / root[None,:]
            gradient = (np.diag(inv) - np.einsum('ij,ji->i', inv@cov, inv)) * psi
            if return_loadings: return loading
            return value, gradient
        result = minimize(objective, np.log(np.maximum(model.noise_variance_,1e-12)), jac=True, method='L-BFGS-B',
                          bounds=list(zip(np.full(n,np.log(1e-12)),np.log(np.maximum(np.diag(cov),1e-12)))),
                          options=dict(maxiter=max_iter,ftol=1e-13,gtol=1e-8,maxls=50))
        model.noise_variance_ = np.exp(result.x)
        loading = objective(result.x, return_loadings=True)
        # Pad zero factors if the optimum has fewer than D positive factors.
        model.components_ = np.pad(loading.T, ((0,d-loading.shape[1]),(0,0)))
        model.n_iter_ += result.nit
        model.solver_ = 'sklearn+profile_lbfgs'
        model.optimizer_message_ = str(result.message)
        converged = bool(result.success)
        if not converged:
            # A line-search failure can occur at a nearly flat optimum. Verify
            # and finish with the original iterative FA updates from that point.
            with warnings.catch_warnings(record=True) as retry_warnings:
                warnings.simplefilter('always', ConvergenceWarning)
                retry = FactorAnalysis(n_components=d, svd_method='lapack', tol=tol*(n+1)/t,
                                       max_iter=max_iter, noise_variance_init=model.noise_variance_).fit(compact)
            converged = not any(issubclass(w.category, ConvergenceWarning) for w in retry_warnings)
            retry.n_iter_ += model.n_iter_
            retry.solver_ = 'sklearn+profile_lbfgs+sklearn'
            model = retry
    basis = np.linalg.svd(model.components_.T, full_matrices=False)[0][:, :d]
    return model, basis, converged


def run(args):
    out = Path(args.output)
    tables, provenance = out / 'tables', out / 'provenance'
    if (tables / 'results.csv').exists():
        raise FileExistsError(f'Preserving existing results in {out}; choose a new data destination.')
    tables.mkdir(parents=True, exist_ok=True)
    provenance.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        models = joblib.load(args.models)
    all_pairs = list(pairs(models))
    common_counts = common_neuron_counts(getattr(args, 'pair_manifest', None))
    if common_counts is not None:
        missing = set(common_counts) - set(all_pairs)
        if missing:
            raise ValueError(f'Empirical pairs are missing from the supplied models: {sorted(missing)}')
    if args.limit_pairs:
        all_pairs = all_pairs[:args.limit_pairs]
    # Keep the original model-pair indices (and hence seeds) when filtering.
    selected_pairs = [(i, pair) for i, pair in enumerate(all_pairs)
                      if (getattr(args, 'pair_index', -1) < 0 or i == args.pair_index)
                      and (common_counts is None or pair in common_counts)]
    if not selected_pairs:
        raise ValueError('No pairs match the requested empirical cohort and pair selection.')
    metadata, rows, diagnostics = [], [], []
    start = time.time()
    for completed, (pair_index, (sa, sb)) in enumerate(selected_pairs, 1):
        fa, fb = models[sa]['fa_thres'], models[sb]['fa_thres']
        model_n = min(fa['fa'].n_features_in_, fb['fa'].n_features_in_)
        n = model_n if common_counts is None else common_counts[(sa, sb)]
        d = 2 * (min(int(fa['d_shared']), int(fb['d_shared'])) // 2)
        if n > model_n:
            raise ValueError(f'{sa}/{sb}: common-neuron count exceeds the supplied model sizes.')
        if 3 * d > 2 * n:
            raise ValueError(f'{sa}/{sb}: N={n} cannot support exactly D/2 shared directions at D={d}.')
        if d < 2:
            metadata.append(dict(session_a=sa, session_b=sb, N=n, D=d, status='skipped D<2'))
            continue
        va, vb = spectrum(fa['fa'])[:d], spectrum(fb['fa'])[:d]
        pair_rng = np.random.default_rng(np.random.SeedSequence([args.seed, pair_index, 987]))
        ia = np.sort(pair_rng.choice(fa['fa'].n_features_in_, n, replace=False))
        ib = np.sort(pair_rng.choice(fb['fa'].n_features_in_, n, replace=False))
        psis = [fa['fa'].noise_variance_[ia], fb['fa'].noise_variance_[ib]]
        task_pair = '/'.join(sorted([TASKS[task_code(sa)], TASKS[task_code(sb)]]))
        pair_id = sa+'__'+sb
        metadata.append(dict(pair=pair_id, session_a=sa, session_b=sb, task_pair=task_pair, patient=sa.split('-')[0],
                             N=n, D=d, K=d//2, T=args.samples, N_a=fa['fa'].n_features_in_, N_b=fb['fa'].n_features_in_,
                             N_source='minimum_model_count' if common_counts is None else 'empirical_common_count',
                             psi_indices_a=json.dumps(ia.tolist()), psi_indices_b=json.dumps(ib.tolist()),
                             eigenvalues_a=json.dumps(va.tolist()), eigenvalues_b=json.dumps(vb.tolist()), status='included'))
        for rep in range(args.replicates):
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, pair_index, rep]))
            x, y, z = random_bases(n, d, rng)
            noise_seed = int(rng.integers(2**32))
            permutations = [rng.permutation(d), rng.permutation(d)]
            for case in CASES:
                if case == 1:
                    bases = [np.column_stack([x,z]), np.column_stack([y,z])]
                elif case == 2:
                    bases = [np.column_stack([z,x]), np.column_stack([z,y])]
                else:
                    bases = [np.column_stack([z,x])[:, permutations[0]], np.column_stack([z,y])[:, permutations[1]]]
                # Common latent/noise draws across cases reduce Monte Carlo differences.
                data_rng = np.random.default_rng(noise_seed)
                covs, shared_covs, generating_covs, fitted_covs, estimates = [], [], [], [], []
                for j, (basis, v, psi) in enumerate(zip(bases, [va,vb], psis)):
                    data = sample_data(basis, v, psi, args.samples, data_rng)
                    data -= data.mean(axis=0)
                    cov = data.T @ data / args.samples
                    fit, estimate, converged = fit_covariance(cov, d, args.samples, args.tol, args.max_iter)
                    shared = (basis*v) @ basis.T
                    fitted_shared = fit.components_.T @ fit.components_
                    covs.append(cov); shared_covs.append(shared)
                    generating_covs.append(shared+np.diag(psi)); fitted_covs.append(fitted_shared); estimates.append(estimate)
                    diagnostics.append(dict(pair=pair_id, replicate=rep, case=case, task=j, iterations=fit.n_iter_, converged=converged, solver=fit.solver_,
                                            relative_shared_cov_error=np.linalg.norm(fitted_shared-shared)/np.linalg.norm(shared),
                                            signal_subspace_overlap=np.linalg.norm(basis.T@estimate,'fro')**2/d))
                pa, pb, angles = principal_bases(*estimates)
                oa, ob, true_angles = principal_bases(*bases)
                for j, (session, basis, estimate, p, oracle) in enumerate(zip([sa,sb], bases, estimates, [pa,pb], [oa,ob])):
                    # Sort oracle basis by variance so 'top' means largest eigenvalues.
                    sorted_basis = basis[:, np.argsort(-[va,vb][j])]
                    configurations = [
                        ('fitted_data_total', covs[j], estimate, p),
                        ('fitted_fa_shared', fitted_covs[j], estimate, p),
                        ('fitted_true_shared', shared_covs[j], estimate, p),
                        ('oracle_population_total', generating_covs[j], sorted_basis, oracle),
                        ('oracle_population_shared', shared_covs[j], sorted_basis, oracle),
                        ('oracle_sample_total', covs[j], sorted_basis, oracle),
                    ]
                    for method, cov, bas, principal in configurations:
                        vals = fractions(cov, bas, principal)
                        for category, value in vals.items():
                            rows.append(dict(pair=pair_id, patient=sa.split('-')[0], task_pair=task_pair,
                                             session=session, task=TASKS[task_code(session)], replicate=rep, case=case,
                                             method=method, category=category, fraction=value, N=n, D=d))
                if rep == 0:
                    print(f'{pair_id} case {case}: angles estimated {angles[:d//2].mean():.1f}/{angles[d//2:].mean():.1f}; true {true_angles[:d//2].mean():.1f}/{true_angles[d//2:].mean():.1f}', flush=True)
        pd.DataFrame(rows).to_csv(tables/'results.csv',index=False)
        pd.DataFrame(metadata).to_csv(tables/'pairs.csv',index=False)
        pd.DataFrame(diagnostics).to_csv(tables/'fit_diagnostics.csv',index=False)
        print(f'Completed {completed}/{len(selected_pairs)} pairs; {time.time()-start:.1f}s',flush=True)
    config = vars(args).copy(); config['elapsed_seconds'] = time.time()-start
    import sklearn, scipy
    config['versions'] = dict(numpy=np.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__)
    (provenance/'config.json').write_text(json.dumps(config,indent=2))
    (out/'manifest.json').write_text(json.dumps(dict(layout_version=2, dataset='fixed_D_simulation',
        config='provenance/config.json'),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',required=True)
    p.add_argument('--output',default=str(ROOT / 'data/subspace_simulation'))
    p.add_argument('--replicates',type=int,default=100)
    p.add_argument('--samples',type=int,default=DEFAULT_SAMPLES)
    p.add_argument('--pair-manifest',default=None,
                   help='Empirical PAGDVE manifest: use its common-neuron counts and included pairs.')
    p.add_argument('--seed',type=int,default=20261023)
    p.add_argument('--tol',type=float,default=1e-6)
    p.add_argument('--max-iter',type=int,default=100000)
    p.add_argument('--limit-pairs',type=int,default=0)
    p.add_argument('--pair-index',type=int,default=-1,help='Optional zero-based pair index; preserves pair-specific random seeds')
    args=p.parse_args()
    if args.samples < 2 or args.replicates < 1: p.error('samples>=2 and replicates>=1 required')
    with threadpool_limits(limits=1): run(args)

if __name__=='__main__': main()
