"""Regenerate the continuous Gaussian activity used in the Fig. 6 simulations.

Requirements: Python >=3.9 and NumPy (1.26+ recommended for loading the pickle).

    import pickle
    from scripts.subspace_simulation.generate_activity import generate_activity

    with open('models/subspace_simulation/simulation_generating_params.pkl', 'rb') as f:
        params = pickle.load(f)
    key = ('Podcast', 'Rotations', 'A')
    p1, p2 = params[key][0]
    Y1 = generate_activity(p1)  # shape (10000, p1['N'])
    Y2 = generate_activity(p2)
    D = p1['D']               # identical to p2['D']

The pickle is one plain dictionary:
    (task1_name, task2_name, patient_ID) -> [(params1, params2), ...]

All three cases, all 24 original session pairs and all 100 repeats are included.
Each task parameter dict contains case (1,2,3), replicate (0..99), session,
pair_id, pair_index, task, patient, N, T, D, basis, shared_variances,
private_variances, mean, center, rng_state, noise_seed and task_index.

Order within a key: original session pairs in their original order, then
replicate 0..99, then cases 1,2,3. Multiple session pairs belonging to the same
patient/task pair are appended to that key's list. Filter on params1['case']
to select a case; use pair_id/session to distinguish repeated recordings.

For example, to select all Case 2 parameter pairs for one key:
    case2 = [(a, b) for a, b in params[key] if a['case'] == 2]

For each task, the population covariance before sample centering is:
    (basis * shared_variances) @ basis.T + diag(private_variances)
The mean is zero. basis is an orthonormal N x D matrix, shared_variances has
length D, and private_variances has length N. Per-neuron mus/sds alone would
not capture the cross-neuron shared covariance.

Default generation restores the exact PCG64 random state from the original
simulation, then centers each neuron across time, matching the original FA
input. No FA fitting or original model pickle is needed. Output is float64
continuous activity (can be negative); it is not integer spike counts. The
original simulation did not use a Poisson or another count observation model.
Reproduction uses the same Gaussian draws; last-bit matrix multiplication
results can depend on NumPy/BLAS and threading. Original NumPy: 2.3.5.

Use generate_activity(p, seed=123) for a new independent realization, and
center=False for uncentered Gaussian observations. No rounding/clipping or
per-neuron variance normalization is applied.
"""
import copy
import numpy as np


def generate_activity(params, *, seed=None, T=None, center=None):
    """Return a T x N Gaussian activity matrix from one task parameter dict.

    seed=None restores the saved random state. An explicit seed generates a
    new sample. T=None and center=None use the values stored in params.
    This function does not change params or advance any shared/global RNG.
    """
    n, d = int(params['N']), int(params['D'])
    t = int(params['T'] if T is None else T)
    if t <= 0:
        raise ValueError('T must be positive')
    basis = np.asarray(params['basis'], dtype=np.float64)
    v = np.asarray(params['shared_variances'], dtype=np.float64)
    psi = np.asarray(params['private_variances'], dtype=np.float64)
    mean = np.asarray(params['mean'], dtype=np.float64)
    if basis.shape != (n, d) or v.shape != (d,) or psi.shape != (n,) or mean.shape != (n,):
        raise ValueError('Inconsistent N, D, or parameter array shapes')
    if np.any(v < 0) or np.any(psi < 0):
        raise ValueError('Variance parameters must be nonnegative')
    if seed is None:
        rng = np.random.Generator(np.random.PCG64())
        rng.bit_generator.state = copy.deepcopy(params['rng_state'])
    else:
        rng = np.random.default_rng(seed)
    activity = (rng.standard_normal((t, d)) * np.sqrt(v)) @ basis.T + rng.standard_normal((t, n)) * np.sqrt(psi)
    activity += mean
    should_center = params['center'] if center is None else center
    if should_center:
        activity -= activity.mean(axis=0)
    return activity
