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
