"""Run the reviewer analyses on the two supplied anonymized sessions.

Run from the repository root with: uv run python example_analysis.py
Numerical summaries are printed; seven PDF figures are saved in examples/results/
and opened in the default PDF viewer when available.
The calculations and settings follow examples/reviewer_example.ipynb.
"""

from pathlib import Path
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")  # Save figures without requiring a display or opening windows.
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import normalize
from decomposition import FAResults
from utils import AREA_CLR, TASK_CLR, TASK_PAIRS_CLR, fig_set, finish_plot, principal_angles, section

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "examples" / "data_for_reviewers"
RESULTS_DIR = ROOT / "examples" / "results"
fig_set()

# %% Load the example data
# Counts use 100 ms on-task bins. Matching unit IDs identify the same unit across tasks.
section("Load the example data")

pursuit = np.load(DATA_DIR / "example_pursuit.npz")
rotations = np.load(DATA_DIR / "example_rotations.npz")

print(pursuit.files)

print("Pursuit (bins, units):", pursuit["spike_counts"].shape)
print("Rotations (bins, units):", rotations["spike_counts"].shape)

# %% Task A/B decoding — Fig. 2I
# Use 33 matched units, training-fold standardization, and 10 stratified folds (seed 0).
section("Task A/B decoding — Fig. 2I")

shared_units, pursuit_idx, rotations_idx = np.intersect1d(
    pursuit["unit_ids"], rotations["unit_ids"], return_indices=True
)
# Convert counts in 100 ms bins to Hz.
X_pursuit = pursuit["spike_counts"][:, pursuit_idx] * 10.0
X_rotations = rotations["spike_counts"][:, rotations_idx] * 10.0
X = np.vstack([X_pursuit, X_rotations])
y = np.r_[np.zeros(len(X_pursuit)), np.ones(len(X_rotations))]

decoder = make_pipeline(
    StandardScaler(),
    LogisticRegression(solver="liblinear", random_state=0),
)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
auc = cross_val_score(decoder, X, y, cv=cv, scoring="roc_auc")
print(f"Pursuit vs Rotations: mean ROC AUC = {auc.mean():.4f}")

# %% Factor analysis — Fig. 3B and 3F
# Fit each task independently with all its units after within-task normalization.
# Shared dimensionality retains at least 95% of shared variance.
section("Factor analysis — Fig. 3B and 3F")

rates_pursuit = pursuit["spike_counts"] * 10.0
rates_rotations = rotations["spike_counts"] * 10.0
fa_pursuit = FAResults().fa_fit(normalize(rates_pursuit))
fa_rotations = FAResults().fa_fit(normalize(rates_rotations))
max_units = max(rates_pursuit.shape[1], rates_rotations.shape[1])

print(f"Pursuit: shared dimensions = {fa_pursuit.d_shared}; units = {rates_pursuit.shape[1]}")
print(f"Rotations: shared dimensions = {fa_rotations.d_shared}; units = {rates_rotations.shape[1]}")

# %% Cumulative shared variance — Fig. 3B
section("Cumulative shared variance — Fig. 3B")

plt.figure(figsize=(2.8, 2.8), dpi=300)
for task, fa in [("pursuit", fa_pursuit), ("rotations", fa_rotations)]:
    dimensions = np.arange(1, len(fa.explained_variance_ratio_) + 1)
    cumulative = np.cumsum(fa.explained_variance_ratio_) * 100
    plt.plot(dimensions, cumulative, "o-", markersize=2,
             color=TASK_CLR[task], label=task.capitalize())
    plt.axvline(fa.d_shared, ls="--", lw=0.8, color=TASK_CLR[task])

plt.axhline(95, ls="--", lw=0.8, color="k")
plt.xlabel("Factor dimension")
plt.ylabel("Cumulative shared\nvariance explained (%)")
plt.ylim(0, 105)
plt.xlim(0.5, max_units + 0.5)
plt.xticks([1, 10, 20, 30, max_units])
plt.legend(frameon=False)
finish_plot(filename="cumulative_shared_variance", save_dir=RESULTS_DIR, show=False, exts=("pdf",))

# %% Shared variance per unit — Fig. 3F
# Sort each task independently; the x-axis is within-task rank.
section("Shared variance per unit — Fig. 3F")

plt.figure(figsize=(2.8, 2.8), dpi=300)
for task, fa in [("pursuit", fa_pursuit), ("rotations", fa_rotations)]:
    shared_variance = np.sort(fa.shared_var_per_unit) * 100
    unit_numbers = np.arange(1, len(shared_variance) + 1)
    plt.plot(unit_numbers, shared_variance, "o-", markersize=2,
             color=TASK_CLR[task], label=task.capitalize())

plt.xlabel("Unit rank (within task)")
plt.ylabel("Shared variance (%)")
plt.ylim(-5, 100)
plt.xlim(0.5, max_units + 0.5)
plt.xticks([1, 10, 20, 30, max_units])
plt.legend(frameon=False)
finish_plot(filename="shared_variance_per_unit", save_dir=RESULTS_DIR, show=False, exts=("pdf",))

# %% Regional comparisons — Fig. 4A–B
# Use all units. Shared variance uses on-task FA; supplied mean firing rates cover the full session.
section("Regional comparisons — Fig. 4A–B")

regional_data = {"pursuit": pursuit, "rotations": rotations}
regional_rates = {"pursuit": rates_pursuit, "rotations": rates_rotations}
regional_fa = {"pursuit": fa_pursuit, "rotations": fa_rotations}

for task, data in regional_data.items():
    for region in ["HPC", "ACC"]:
        selected = data["regions"] == region
        shared = regional_fa[task].shared_var_per_unit[selected] * 100
        rates = data["full_session_mean_rate_hz"][selected]
        sv_q1, sv_median, sv_q3 = np.percentile(shared, [25, 50, 75])
        fr_q1, fr_median, fr_q3 = np.percentile(rates, [25, 50, 75])
        print(f"{task.capitalize()} {region} (n={selected.sum()}): "
              f"shared variance {sv_median:.1f}% (IQR {sv_q1:.1f}–{sv_q3:.1f}%); "
              f"firing rate {fr_median:.2f} Hz (IQR {fr_q1:.2f}–{fr_q3:.2f})")

fig, axes = plt.subplots(2, 2, figsize=(5.6, 5.0), dpi=300, sharex="col", sharey="col")
for row, (task, data) in enumerate(regional_data.items()):
    for region in ["HPC", "ACC"]:
        selected = data["regions"] == region
        shared = regional_fa[task].shared_var_per_unit[selected] * 100
        log_rates = np.log10(data["full_session_mean_rate_hz"][selected])
        axes[row, 0].hist(shared, bins=np.linspace(0, 100, 21), density=True,
                          histtype="step", color=AREA_CLR[region], lw=0.8, label=region)
        axes[row, 1].hist(log_rates, bins=np.linspace(-0.5, 2, 26), density=True,
                          histtype="step", color=AREA_CLR[region], lw=0.8, label=region)
    for ax in axes[row]:
        ax.set_title(task.capitalize())
        ax.set_ylabel("Density")
        ax.legend(frameon=False)

axes[-1, 0].set_xlabel("Shared variance (%)")
axes[-1, 1].set_xlabel("Firing rate (Hz)")
axes[-1, 1].set_xticks([-0.5, 0, 0.5, 1, 1.5, 2],
                       [r"$10^{-0.5}$", r"$10^0$", r"$10^{0.5}$", r"$10^1$", r"$10^{1.5}$", r"$10^2$"])
finish_plot(filename="regional_shared_variance_and_rates", save_dir=RESULTS_DIR, show=False, exts=("pdf",))

# %% HPC contributions — Fig. 4D
# Balance HPC/ACC counts with ten resamples (seeds 0–9); use identical histogram axes.
# Repeated resamples are not independent observations.
section("HPC contributions — Fig. 4D")

hpc_contributions = {}
for task, data in regional_data.items():
    hpc_indices = np.flatnonzero(data["regions"] == "HPC")
    acc_indices = np.flatnonzero(data["regions"] == "ACC")
    n_per_region = min(len(hpc_indices), len(acc_indices))
    contributions = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        hpc_selected = np.sort(rng.choice(hpc_indices, n_per_region, replace=False))
        acc_selected = np.sort(rng.choice(acc_indices, n_per_region, replace=False))
        selected = np.r_[hpc_selected, acc_selected]  # HPC columns come first.
        fa_balanced = FAResults().fa_fit(
            normalize(regional_rates[task][:, selected]), shared_var_thresh=0.95, max_iter=10000
        )
        contributions.extend((fa_balanced.subspace[:, :n_per_region] ** 2).sum(axis=1))
    hpc_contributions[task] = np.asarray(contributions)

contribution_bins = np.linspace(0, 1, 21)
max_density = max(np.histogram(values, bins=contribution_bins, density=True)[0].max()
                  for values in hpc_contributions.values())
for task, values in hpc_contributions.items():
    plt.figure(figsize=(2.8, 2.8), dpi=300)
    plt.hist(values, bins=contribution_bins, density=True,
             histtype="step", color=AREA_CLR["HPC"], lw=0.8)
    plt.axvline(0.5, ls="--", color="k", lw=0.8)
    plt.xlim(0, 1)
    plt.ylim(0, max_density * 1.1)
    plt.xlabel("HPC contribution")
    plt.ylabel("Density")
    plt.title(task.capitalize())
    finish_plot(filename=f"hpc_contribution_{task}", save_dir=RESULTS_DIR, show=False, exts=("pdf",))

# %% Units exceeding the shifted threshold — Fig. 4F
# Calculate p-values from saved HPC-shift fits at the original CV-selected ranks.
# As in regions.py, p is the fraction of converged shifted changes strictly greater than the real change.
section("Units exceeding the shifted threshold — Fig. 4F")

shift_pvalues = {}
for task, data in regional_data.items():
    with np.load(DATA_DIR / f"example_{task}_shifts.npz") as results:
        assert np.array_equal(results["unit_ids"], data["unit_ids"])
        converged = results["shift_converged"]
        alone = results["region_only_shared_var_per_unit"]
        real_delta = 100 * (results["joint_shared_var_per_unit"] - alone)
        shifted_delta = 100 * (results["shifted_shared_var_per_unit"][converged] - alone)
        p_values = (shifted_delta > real_delta[None, :]).mean(axis=0)
        shift_pvalues[task] = p_values
        print(f"{task.capitalize()} ({converged.sum()}/{int(results['n_shifts_planned'])} converged HPC shifts):")
        for region in ["HPC", "ACC"]:
            selected = data["regions"] == region
            n_pass = np.count_nonzero(p_values[selected] < 0.05)
            n_units = selected.sum()
            print(f"  {region}: {100 * n_pass / n_units:.1f}% ({n_pass}/{n_units} units)")

# %% Principal angles — Fig. 5
# Refit FA on matched units in the saved column order and verify against reference angles.
# Load 500 random controls and 500 within-task shuffled controls per task.
section("Principal angles — Fig. 5")

angle_results = np.load(DATA_DIR / "example_principal_angles.npz")
pa_fits = []
for data in [pursuit, rotations]:
    columns = [np.flatnonzero(data["unit_ids"] == unit)[0] for unit in angle_results["unit_ids"]]
    rates = data["spike_counts"][:, columns] * 10.0
    fa = FAResults().fa_fit(
        normalize(rates), shared_var_thresh=float(angle_results["shared_var_thresh"]),
        max_iter=int(angle_results["fa_max_iter"]), tol=float(angle_results["fa_tol"]),
        svd_method="lapack",
    )
    pa_fits.append(fa)

angles = principal_angles(pa_fits[0].subspace.T, pa_fits[1].subspace.T)
np.testing.assert_allclose(angles, angle_results["data_pa"], rtol=0, atol=1e-8)
random_angles = angle_results["rand_pa"]
shuffled_angles = np.concatenate([angle_results["shuf_task1_pa"], angle_results["shuf_task2_pa"]])

fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
angle_indices = np.arange(1, len(angles) + 1)
color = TASK_PAIRS_CLR["RS"]
ax.scatter(angle_indices, angles, color=color, s=5, lw=0.8)
ax.plot(angle_indices, angles, color=color, lw=0.8, label="Data")
for values, label, linestyle in [(random_angles, "Random", "-"),
                                 (shuffled_angles, "Shuffled", "--")]:
    mean, std = values.mean(axis=0), values.std(axis=0)
    ax.scatter(angle_indices, mean, edgecolors="gray", facecolors="none", s=5, lw=0.8)
    ax.plot(angle_indices, mean, color="gray", ls=linestyle, lw=0.8, label=label)
    ax.fill_between(angle_indices, mean - std, mean + std, color="gray", alpha=0.3, lw=0)
ax.set(ylim=(-0.5, 90.5), yticks=np.linspace(0, 90, 4), xticks=angle_indices,
       xlabel="Principal angle index", ylabel="Principal angles (deg)", title="Pursuit / Rotations")
ax.legend(frameon=False)
finish_plot(filename="principal_angles", save_dir=RESULTS_DIR, show=False, exts=("pdf",))

# %% More-aligned subspace distance — Fig. 5C
# Use the Euclidean norm of the smallest four principal angles, in degrees.
section("More-aligned subspace distance — Fig. 5C")

n_more_aligned = len(angles) // 2
more_aligned_distance = np.linalg.norm(angles[:n_more_aligned])
random_distances = np.linalg.norm(random_angles[:, :n_more_aligned], axis=1)
shuffled_distances = np.linalg.norm(shuffled_angles[:, :n_more_aligned], axis=1)

print(f"More-aligned dimensions: {n_more_aligned}")
print(f"Data subspace distance: {more_aligned_distance:.2f}°")
print(f"Shuffled: {shuffled_distances.mean():.2f} ± {shuffled_distances.std():.2f}° (mean ± SD)")
print(f"Random: {random_distances.mean():.2f} ± {random_distances.std():.2f}° (mean ± SD)")

# %% Proportional variance explained — Fig. 6C
# Divide each four-dimensional category by variance across the entire retained task subspace.
section("Proportional variance explained — Fig. 6C")

_, U, Vh = principal_angles(pa_fits[0].subspace.T, pa_fits[1].subspace.T, return_vectors=True)
Q_pursuit, _ = np.linalg.qr(pa_fits[0].subspace.T)
Q_rotations, _ = np.linalg.qr(pa_fits[1].subspace.T)
principal_directions = [Q_pursuit @ U, Q_rotations @ Vh.T]
n_per_group = len(angles) // 2

variance_fractions = {}
print(f"{'Task':<12} {'Top factors':>12} {'More aligned':>14} {'Less aligned':>14}")
for task, fa, directions in zip(["pursuit", "rotations"], pa_fits, principal_directions):
    factor_variance = np.var(fa.data @ fa.subspace.T, axis=0)
    principal_variance = np.var(fa.data @ directions, axis=0)
    total_variance = factor_variance.sum()
    fractions = np.array([
        factor_variance[:n_per_group].sum(),
        principal_variance[:n_per_group].sum(),
        principal_variance[n_per_group:2 * n_per_group].sum(),
    ]) / total_variance
    variance_fractions[task] = fractions  # Top factors, more aligned, less aligned.
    print(f"{task.capitalize():<12} {fractions[0]:12.4f} {fractions[1]:14.4f} {fractions[2]:14.4f}")

# %% Paired mean firing rates — Fig. S6C
# Connect the same 33 units across tasks; show on-task mean rates with 5th–95th percentile whiskers.
section("Paired mean firing rates — Fig. S6C")

paired_mean_rates = np.column_stack([X_pursuit.mean(axis=0), X_rotations.mean(axis=0)])
log_mean_rates = np.log10(paired_mean_rates)

fig, ax = plt.subplots(figsize=(2.8, 2.8), dpi=300)
sns.boxplot(data=log_mean_rates, ax=ax, width=0.5, fill=False, color="k",
            whis=(5, 95), showfliers=False)
jitter_offsets = np.random.default_rng(0).uniform(-0.05, 0.05, len(log_mean_rates))
for values, jitter in zip(log_mean_rates, jitter_offsets):
    paired_x = np.array([0, 1]) + jitter
    ax.plot(paired_x, values, color=TASK_PAIRS_CLR["RS"], lw=0.5, alpha=0.5, zorder=1)
    ax.scatter(paired_x, values, facecolors="none", edgecolors=TASK_PAIRS_CLR["RS"],
               linewidths=1, s=5, zorder=2)
ax.set(xticks=[0, 1], xticklabels=["Pursuit", "Rotations"], xlim=(-0.5, 1.5),
       xlabel="", ylabel=r"$\log_{10}$ mean firing rate (Hz)")
finish_plot(filename="paired_mean_firing_rates", save_dir=RESULTS_DIR, show=False, exts=("pdf",))

# %% Shared dimensionality by two methods — Fig. S7
# Use saved 200 ms CV scores and shared-variance spectra; no CV rerun is needed.
section("Shared dimensionality by two methods — Fig. S7")

print(f"{'Task':<12} {'CV':>7} {'95% threshold':>15}")
for task in ["pursuit", "rotations"]:
    with np.load(DATA_DIR / f"example_{task}_dimensionality.npz") as result:
        d_cv = int(result["cv_components"][np.argmax(result["cv_scores"])])
        d_threshold = np.searchsorted(np.cumsum(result["explained_variance_ratio"]),
                                      float(result["shared_var_thresh"])) + 1
        print(f"{task.capitalize():<12} {d_cv:7d} {d_threshold:15d}")

pursuit.close()
rotations.close()
angle_results.close()

print(f"\nFinished. Seven PDF figures saved in {RESULTS_DIR}")

pdf_files = [RESULTS_DIR / f"{name}.pdf" for name in (
    "cumulative_shared_variance", "shared_variance_per_unit",
    "regional_shared_variance_and_rates", "hpc_contribution_pursuit",
    "hpc_contribution_rotations", "principal_angles", "paired_mean_firing_rates",
)]
try:
    if sys.platform == "darwin":
        subprocess.run(["open", *map(str, pdf_files)], check=True)
    elif sys.platform == "win32":
        for pdf_file in pdf_files:
            os.startfile(pdf_file)
    else:
        for pdf_file in pdf_files:
            subprocess.run(["xdg-open", str(pdf_file)], check=True)
except (OSError, subprocess.CalledProcessError):
    print(f"Could not open a PDF viewer. The PDFs are available in {RESULTS_DIR}")
