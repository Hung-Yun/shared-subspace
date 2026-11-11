# Subspace-alignment simulations

These scripts generate neural activity, fit subspaces, and evaluate alignment and variance explained for Fig. 6 and Fig. S13. Plotting cells are in [subspace_comparison.py](../../subspace_comparison.py).

Use the environment described in the repository's [installation instructions](../../README.md#installation).

## Storage

Default output locations are:

- [models/subspace_simulation](../../models/subspace_simulation): generating parameters, common-unit FA parameters, and fitted simulation bases.
- [data/subspace_simulation](../../data/subspace_simulation): numerical results, input activity, metadata, logs, and provenance. `manifest.json` identifies the data.
- [plots/subspace_simulation](../../plots/subspace_simulation): exported PNG/PDF/SVG figures.

The common-unit dataset uses FA fits to units matched across tasks: 22 session pairs, 100 repeats per case, three cases, and 14,300 samples per task.

## Generate activity

```python
import pickle
from scripts.subspace_simulation.generate_activity import generate_activity

with open('models/subspace_simulation/simulation_generating_params.pkl', 'rb') as f:
    params = pickle.load(f)
p1, p2 = params[('Podcast', 'Rotations', 'A')][0]
Y1, Y2 = generate_activity(p1), generate_activity(p2)
```

The pickle has 18 `(task1, task2, patient)` keys containing 7,200 parameter pairs: 24 session pairs x 100 repeats x 3 cases. Each entry stores basis, shared/private variances, N, D, T, case/repeat/session identities, and RNG state. A key can contain multiple recording pairs; use `pair_id` to distinguish them. Outputs are centered continuous Gaussian activity, not integer spike counts. The generator itself performs no per-neuron z-scoring or FA fitting.

Default generation replays the saved 10,000-sample draws. When changing T, use separate task seeds: the saved task-2 state follows the task-1 10,000-sample draw, so enlarging both matrices with those states would reuse random draws across tasks. `simulate_subspace_recovery.py` records separate task streams using SeedSequence([seed, original_pair_index, replicate, task_index]), with common random draws across cases. Its default sample count is 14,300.

## Empirical-pipeline computation

`simulate_subspace_recovery.py` generates each activity pair and calls `scripts/subspace_alignment.py:pagdve`. It is sequential, with one numerical-library thread. Z-scoring, full FA, 95% shared-variance retention, principal vectors, K, PA/GD, and projected activity VE follow the empirical pipeline. FA uses LAPACK, max_iter=10,000, and tol=1e-4. No nested random or split-half controls are run inside a simulation comparison.

`--common-unit-cache data/temp/df_all.h5` reads cached df_all and calls `get_paired_df`, then fits each task to **all actual common units in matching column order**. The simulation uses those fits' spectra and each matched unit's private variance. Generating D is the smaller retained rank rounded down to an even number. Without this option, the parameterization matches neuron counts using full-session models instead of refitting matched units.

In `subspace_comparison.py`, the `Subspace alignment analysis hyperparams` cell sets the inputs and simulation settings. Set `RERUN=True`, then run the `Subspace alignment simulation` cell to compute all task pairs with 100 repeats per case and 14,300 samples per task. Set `TASK_PAIR` to a specific pair for a smaller run. Progress is streamed into the cell and saved in `provenance/run.log`; verification runs after completion. The launcher refuses to overwrite fitted models or result data. If it finds results explicitly marked `jay_prelim`, it archives their data and plots in sibling `jay_prelim` folders before starting.

The runner's default destinations are the shared data/model paths above; `--output` (alias `--data-dir`) and `--model-dir` can override them. It refuses to overwrite existing result data or fitted models. `--task-pair all` selects every cached session pair covered by the generating parameters; output tables retain the actual task-pair names for plotting.

The empirical-pipeline outputs are:

- `data/.../inputs/<pair>/task_<side>.npz`: raw and normalized activity, channel identities/positions, timestamps, and normalization statistics.
- `models/.../common_unit_fits/<pair>/task_<side>.npz`: FA loadings, private variances, shared spectra, and retained ranks.
- `models/.../generating_params.pkl`: actual generating parameters and RNG states.
- `models/.../estimated_bases/<pair>/case_<case>/repeat_<index>.npz`: fitted bases.
- `data/.../comparisons/<pair>/case_<case>/repeat_<index>.npz`: PA/GD/VE, evaluation covariance matrices, diagnostics, and a reference to the matching model file.
- `data/.../tables/`: repeat-level and summary CSV tables.
- `data/.../provenance/`: configuration, logs, input/source hashes, verification, and source snapshots.

`check_subspace_recovery.py --data-dir <data-directory>` verifies a common-unit run, including channel identities, source fits, separated model/data references, coverage, and replay. `subspace_comparison.py` reads the result tables from the selected data directory for plotting.

## Fixed-rank simulation

`simulate_fixed_rank.py` fits a specified rank D. `run_fixed_rank_parallel.py` can run session pairs in parallel. Results and configuration are saved in `tables/` and `provenance/` within the data destination.

The fixed-rank results use the smaller full-session neuron count for N. D is min(d_shared_1,d_shared_2), rounded down to even; K=D/2. Each source task supplies its top D absolute shared-covariance eigenvalues. Private variances are sampled from its full-session model when reducing N. These are not parameters refitted to actual common units.

Z is a random N x K orthonormal basis. X and Y are sampled independently outside Z and orthonormalized separately; X and Y are not mutually orthogonal.

1. Shared low variance: bases [X,Z] and [Y,Z], with decreasing eigenvalues.
2. Shared high variance: bases [Z,X] and [Z,Y].
3. Random variance assignment: independently permute columns before assigning the decreasing eigenvalues.

Gaussian factors plus diagonal Gaussian private noise generate the activity. The analysis fits exactly D factors with LAPACK, tol=1e-6, and a profile-likelihood fallback after 1,000 iterative FA steps, up to 100,000 steps. A compact matrix with identical covariance accelerates the FA fitting; the likelihood tolerance is scaled to preserve the full-sample criterion.

The primary `fitted_data_total` measure is sample variance projected into fitted principal directions, divided by sample variance in the entire retained fitted subspace. The even-D grouping gives more+less=1. The other five saved methods evaluate fitted/generating shared covariance or known generating subspaces. The empirical-pipeline grouping can instead leave an odd direction or extra unmatched dimensions outside both groups while retaining them in the denominator. No equivalence of those two rank-selection procedures is implied.

Validate the saved fixed-rank results:

```bash
python scripts/subspace_simulation/check_fixed_rank_results.py --data-dir /path/to/fixed_rank_results
```

`check_fixed_rank_math.py` separately checks geometry, principal-vector sign invariance, and agreement between compact and full-data FA on a small numerical example. Each run records its execution settings and software versions in `provenance/config.json` within its output directory.

## Plotting

In `subspace_comparison.py`, run the simulation plotting cells. `RESULTS_DIR` points to `data/subspace_simulation`, `TABLE_DIR` to its `tables` subfolder, and `PLOT_DIR` to `plots/subspace_simulation`. `PAIRS` selects the task pairs to plot; the plotting loop makes one figure per task pair and simulation case. Each figure calls `finish_plot`, which handles layout, display, and PNG/SVG/PDF exports. `SHOW` and `SAVEFIG` control display and saving. The common-unit simulation plots show estimated-subspace VE.

Small dots are session-pair averages over repeats. Large dots average session pairs equally. Multiple pairs may come from one patient; simulation repeats are not additional independent patients. No inferential significance stars are added.

The plotting loop creates `df` with `session` (vA/vB in the selected task-pair order), `alignment` (top/more/less), and `var_exp`, retaining pair and patient IDs. It uses the Fig. 6C stripplot and black-edged mean-marker format. Statistics match the empirical Fig. 6C analysis: average repeats within each session pair, then use a two-factor independent-group ANOVA and unpaired post-hoc tests with BH-FDR correction, without averaging patients. Overall alignment contrasts pool both task sides. This design does not model pairing or dependence among session pairs from the same patient.

The three overall alignment comparisons are corrected together within each case/task pair; the six task-specific alignment comparisons form a separate correction family. Correction is not pooled across panels. The full post-hoc and ANOVA outputs, identified by task pair and case, are saved respectively to `stats/subspace_simulation.csv` and `stats/subspace_simulation_anova.csv`.
