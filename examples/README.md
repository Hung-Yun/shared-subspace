# Running the example

Install the environment using the [repository instructions](../README.md), then
run these commands from the repository root:

```bash
conda activate subspace
cd examples
python -m jupyterlab reviewer_example.ipynb
```

Select a Python kernel from the `subspace` environment and run all cells in
order. The working directory must be `examples/`. The example uses the supplied
data and saved control results; access to the restricted clinical-data folder
is not required. Figures are displayed in the notebook. Set `SAVEFIG = True`
to also write them to `examples/plots/`.

## Expected outputs and runtime

The notebook prints numerical summaries and produces plots of shared variance,
regional comparisons, principal angles, and paired firing rates. Selected
reference outputs are:

| Output | Expected value |
| --- | --- |
| Pursuit data shape (time bins, units) | `(13981, 33)` |
| Rotations data shape (time bins, units) | `(7401, 39)` |
| Mean held-out task-decoding ROC AUC | Approximately `0.9690` |
| Shared dimensions using all units | Pursuit: `9`; Rotations: `10` |
| More-aligned subspace distance | Approximately `25.19` degrees |

All 18 code cells completed successfully on macOS 15.7.7 (arm64), Python 3.13.5,
using the analysis dependencies pinned in `../requirements.txt`. A headless
execution took approximately 9 seconds on 11 November 2026. This measurement
excludes installation, notebook-server/kernel startup, and interactive figure
display. The example loads saved shuffle, shift-control, and dimensionality
results rather than recomputing those full analyses.

The example task-identity decoder uses 10 folds; the main analysis in `decode.py`
uses 5 folds for Fig. 2I. The example demonstrates the workflow and is not an
exact reproduction of the complete manuscript dataset.

## Using your own data

Prepare NumPy `.npz` files with the fields described below. For the decoding and
factor-analysis sections, use nonnegative spike counts in 100 ms bins, with
time bins as rows and units as columns. Use matching `unit_ids` for units
recorded in both tasks. Supply each unit's `regions` label (`HPC` or `ACC`) and
its `full_session_mean_rate_hz` for the regional summaries. Replace the two
input filenames in the first code cell and run the decoding and factor-analysis
sections on the new recordings.

The later sections that load saved control results require corresponding
results for the new recordings. Do not reuse the bundled reference results
for different data. The repository's `scripts/shift_controls.py`,
`scripts/subspace_alignment.py`, and the dimensionality-comparison section of
`main_analysis.py` implement those calculations. The main README maps analysis
scripts to manuscript figures; those scripts additionally require the study's
session metadata and data layout.

# Example files

`reviewer_example.ipynb` runs the example analyses. All `.npz` files are in `data_for_reviewers/` and contain data or saved results for two anonymized sessions from one participant doing a Pursuit task and a Rotations task.

**`example_pursuit.npz` and `example_rotations.npz`** — On-task activity for Pursuit and Rotations.

| Field | Meaning |
| --- | --- |
| `spike_counts` | Counts in retained 100 ms bins × units: Pursuit `(13981, 33)`, Rotations `(7401, 39)`. |
| `unit_ids` | Anonymous IDs in column order; matching IDs identify the same unit across tasks. |
| `regions` | HPC or ACC for each unit. |
| `full_session_mean_rate_hz` | Each unit's mean firing rate over the full session, in Hz. |

**`example_pursuit_shifts.npz` and `example_rotations_shifts.npz`** — Saved regional FA and HPC-shift control results for each task.

| Field | Meaning |
| --- | --- |
| `unit_ids` | Unit IDs aligned with the corresponding activity file. |
| `joint_shared_var_per_unit` | Shared-variance fractions from the joint HPC/ACC model. |
| `region_only_shared_var_per_unit` | Shared-variance fractions from each unit's region-only model. |
| `shifted_shared_var_per_unit` | Shared-variance fractions for 200 shifted fits × units. |
| `shift_converged` | Convergence flag for each shifted fit. |
| `shift_offsets_bins` | Circular offsets in retained 100 ms bins. |
| `shifted_region` | Shifted population: HPC. |
| `joint_rank`, `hpc_rank`, `acc_rank` | Dimensions of the joint and region-only models. |
| `n_shifts_planned` | Planned number of shifts: 200. |

**`example_principal_angles.npz`** — Reference principal angles and controls for the Pursuit/Rotations pair.

| Field | Meaning |
| --- | --- |
| `data_pa` | Eight observed principal angles, in degrees. |
| `rand_pa` | Random-control angles: 500 repeats × 8 angles. |
| `shuf_task1_pa`, `shuf_task2_pa` | Within-Pursuit and within-Rotations shuffled angles, each `(500, 8)`. |
| `unit_ids` | The 33 shared unit IDs in the comparison's column order. |
| `task_names` | Task order: Pursuit, Rotations. |
| `ranks` | Retained subspace dimensions in task order. |
| `shared_var_thresh` | Shared-variance retention threshold: 0.95. |
| `fa_max_iter`, `fa_tol` | FA iteration limit and convergence tolerance. |
| `seed` | Random seed used for the controls. |

**`example_pursuit_dimensionality.npz` and `example_rotations_dimensionality.npz`** — Saved FA results for comparing dimensionality estimates.

| Field | Meaning |
| --- | --- |
| `n_units` | Number of units fitted. |
| `shared_var_thresh` | Shared-variance retention threshold: 0.95. |
| `explained_variance_ratio` | Fraction of shared variance explained by each factor. |
| `cv_components` | Candidate dimensions tested by cross-validation. |
| `cv_scores` | Mean held-out log-likelihood for each candidate dimension. |
