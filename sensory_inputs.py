#%% Imports

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, ttest_ind

import data_loader as io
from decomposition import FAResults
from utils import (
    LW,
    principal_angles,
    EXCLUDED_SESSIONS,
    TASK_CLR,
    TASK_KEY,
    TASK_NAMES,
    TASK_PAIRS_CLR,
    fig_set,
    finish_plot,
    get_paired_df,
    read_all_sessions,
    read_energy_files,
)

#%% Settings and paths

# Session preparation and sensory-control thresholds.
BIN_SIZE_MS = 100
MAX_GAP_DUR_SECS = 1
MOTION_PERCENTILE = 50
ACOUSTIC_PERCENTILE = 95

# Historical control-experiment fitting settings.
N_RAND = 100
N_SHUF = 500
SHUF_KEY = 2
FA_MAX_ITER = 100
FA_TOL = 1e-2
FA_SVD_METHOD = 'randomized'

# Cached results are used unless recomputation is requested.
# SAVEFIG controls figure files; result caches and statistics are still saved.
FORCE_RECOMPUTE = False
SAVEFIG = False

# Data roots follow data_loader; outputs remain anchored to the repository.
DATA_DIR = Path(io.PREFIX) / 'sensory_inputs' / 'control_exp'
RESULTS_FILE = DATA_DIR / 'principal_angles.pkl'
CHECKPOINT_FILE = DATA_DIR / 'principal_angles_checkpoint.pkl'
GEODIST_FILE = DATA_DIR / 'geodesic_distances.pkl'
STATS_FILE = DATA_DIR / 'geodesic_stats.csv'
PLOT_DIR = io.REPO_ROOT / 'plots' / 'control_exp'
PAIR_PLOT_DIR = PLOT_DIR / 'pairs'
SUMMARY_PLOT_DIR = PLOT_DIR / 'summary'
SUMMARY_3PANEL_DIR = PLOT_DIR / 'summary_3panel'
SUMMARY_STATS_FILE = SUMMARY_3PANEL_DIR / 'stats.csv'
ENERGY_PLOT_DIR = PLOT_DIR / 'energy'



#%% Sensory comparisons and summary statistics

COMPARISONS_BY_PAIR = {
    'CS': [('C1S', 'C1', 'S'), ('C2S', 'C2', 'S'), ('CS1', 'C', 'S1')],
    'CR': [('C1R', 'C1', 'R'), ('C2R', 'C2', 'R'), ('CR1', 'C', 'R1')],
    'RS': [('R1S', 'R1', 'S'), ('RS1', 'R', 'S1')],
}
COMPARISON_ORDER = ['C1S', 'C2S', 'C1R', 'C2R', 'CS1', 'CR1', 'R1S', 'RS1']
CONDITION_TITLES = {
    'C1': 'Podcast (low motion)', 'C2': 'Podcast (high motion)', 'C': 'Original Podcast',
    'S1': 'Pursuit (low noise)', 'R1': 'Rotations (low noise)',
    'S': 'Original Pursuit', 'R': 'Original Rotations',
}
COMPARISON_TITLES = {
    comparison: f'{CONDITION_TITLES[a]} vs {CONDITION_TITLES[b]}'
    for comparisons in COMPARISONS_BY_PAIR.values()
    for comparison, a, b in comparisons
}
PANEL_TITLES = {'more': 'More aligned', 'less': 'Less aligned', 'full': 'Full dimensions'}
GROUP_LABELS = {'data': 'Data', 'rand': 'Random', 'shuf': 'Shuffled'}
CONTRASTS = [('data', 'rand'), ('data', 'shuf'), ('rand', 'shuf')]

def calc_principal_angles(Y1, Y2, n_components=None,
                          calc_rand=True, calc_shuf=True,
                          n_rands=100, n_shufs=50,
                          shuf_key: int = 0,
                          max_iter=int(1e3), tol=1e-2,
                          svd_method='randomized'):
    """Fit two FA subspaces and calculate data, random, and shuffled angles."""
    def _sim(N, L, psi, rng):
        K = L.shape[0]
        z = rng.standard_normal((N, K))
        eps = rng.normal(scale=np.sqrt(psi), size=(N, L.shape[1]))
        return z @ L + eps

    def _get_subspace(y, n_components=n_components):
        result = FAResults()
        return result.fa_fit(
            io.normalize(y),
            n_components=n_components,
            **fa_params,
        )

    def _get_angle(FA1, FA2):
        return principal_angles(FA1.subspace.T, FA2.subspace.T)

    fa_params = {'max_iter': max_iter, 'tol': tol, 'svd_method': svd_method}

    FA1, FA2 = _get_subspace(Y1), _get_subspace(Y2)
    ang = _get_angle(FA1, FA2)
    dim1 = FA1.d_shared
    dim2 = FA2.d_shared

    rand_angles = []
    if calc_rand:
        print('== Random principal angles ==')
        for it in range(n_rands):
            if it % 10 == 0:
                print(f' - {it} iterations')
            L1 = np.random.randn(*FA1.subspace.shape)
            L2 = np.random.randn(*FA2.subspace.shape)
            Y1_sim = _sim(Y1.shape[0], L1, 1e-4, rng=np.random.default_rng(it))
            Y2_sim = _sim(Y2.shape[0], L2, 1e-4, rng=np.random.default_rng(it))
            L1_sim = _get_subspace(Y1_sim, n_components=dim1)
            L2_sim = _get_subspace(Y2_sim, n_components=dim2)
            rand_angles.append(_get_angle(L1_sim, L2_sim))

    shuf_angles = []
    shuf_angles_2 = []
    if calc_shuf:
        print('== Shuffled principal angles ==')
        if shuf_key == 0:
            shuf_angles.append([0 for _ in range(min(dim1, dim2))])
        elif shuf_key == 1:
            for it in range(n_shufs):
                if it % 10 == 0:
                    print(f' - {it} iterations')
                L = np.random.randn(*np.min((FA1.subspace.shape, FA2.subspace.shape), axis=0))
                Y1_sim = _sim(Y1.shape[0], L, 1e-4, rng=np.random.default_rng(it))
                Y2_sim = _sim(Y2.shape[0], L, 1e-4, rng=np.random.default_rng(it**2 + it))
                L1_sim = _get_subspace(Y1_sim, n_components=dim1)
                L2_sim = _get_subspace(Y2_sim, n_components=dim2)
                shuf_angles.append(_get_angle(L1_sim, L2_sim))
        elif shuf_key == 2:
            rng = np.random.default_rng(it)
            d = min(dim1, dim2)
            for it in range(n_shufs):
                try:
                    if it % 5 == 0:
                        print(f' - {it} iterations')
                    ix = rng.permutation(np.arange(Y1.shape[0]))
                    L1_1 = _get_subspace(Y1[ix[len(ix)//2:]], n_components=d)
                    L1_2 = _get_subspace(Y1[:ix[len(ix)//2]], n_components=d)
                    angs = _get_angle(L1_1, L1_2)
                    if len(angs) == d:
                        shuf_angles.append(angs)
                    ix = rng.permutation(np.arange(Y2.shape[0]))
                    L2_1 = _get_subspace(Y2[ix[len(ix)//2:]], n_components=d)
                    L2_2 = _get_subspace(Y2[:ix[len(ix)//2]], n_components=d)
                    angs = _get_angle(L2_1, L2_2)
                    if len(angs) == d:
                        shuf_angles_2.append(angs)
                except Exception:
                    continue
        else:
            raise ValueError('shuf-key unsupported.')

    print('== Done ==')
    return {
        'data': ang,
        'rand': rand_angles,
        'shuf': shuf_angles,
        'shuf2': shuf_angles_2,
    }


def plot_principal_angle(result: dict, ax, color):
    """Plot observed principal angles with random and shuffled controls."""
    ang = result['data']
    rand = result['rand']
    shuf = result['shuf']
    if 'shuf2' in result and len(result['shuf2']):
        shuf = np.concatenate([shuf, result['shuf2']], axis=0)

    def _plot_angles(metric, label, color='k', linestyle='--', alpha=0.3):
        mean = np.mean(metric, axis=0)
        std = np.std(metric, axis=0)
        ax.scatter(range(len(mean)), mean, ec=color, marker='o', s=5, fc='none', lw=LW)
        ax.plot(range(len(mean)), mean, c=color, ls=linestyle, lw=LW, label=label)
        ax.fill_between(range(len(mean)), mean-std, mean+std,
                        color=color, alpha=alpha, lw=0)

    ax.scatter(range(len(ang)), ang, c=color, marker='o', s=5, lw=LW)
    ax.plot(range(len(ang)), ang, c=color, lw=LW, label='Data')
    if len(rand) > 0:
        _plot_angles(rand, label='Random', linestyle='-', color='gray')
    if len(shuf) > 0:
        _plot_angles(shuf, label='Shuffled', linestyle='--', color='gray')
    ax.set_ylim([-0.5, 90.5])
    ax.set_yticks(np.linspace(0, 90, 4))
    ax.set_xticks(np.arange(len(ang)), np.arange(len(ang)) + 1)


def _p_latex(p):
    # Retain small p-values without rounding them to a misleading p=0.000.
    if 0 < p < 0.001:
        mantissa, exponent = f'{p:.2e}'.split('e')
        return rf'$p={mantissa}\times 10^{{{int(exponent)}}}$'
    if p == 0:
        return r'$p<0.001$'
    return f'$p={p:.3f}$'


def summary_statistics(distances):
    """Return 24 ANOVAs and 72 two-sided, equal-variance independent t-tests.

    Preserve the existing figures' analysis: one observation per session pair
    per group, using the already averaged ``rand`` and ``shuf`` columns. The
    three contrasts within each panel form a Bonferroni family. Export all
    contrasts even when the omnibus ANOVA is nonsignificant. T is A minus B.
    No correction is applied across panels or figures, and these are not
    repeated-measures tests despite the connected observations in the plots.
    """
    keys = ['comparison', 'style', 'session_1', 'session_2']
    required = set(keys + ['subj', 'task_pair', *GROUP_LABELS])
    missing = required - set(distances.columns)
    if missing:
        raise ValueError(f'Missing distance columns: {sorted(missing)}')
    if distances[keys + ['subj', 'task_pair']].isna().any().any():
        raise ValueError('Missing session, comparison, or panel identifiers.')
    if distances.duplicated(keys).any():
        raise ValueError('Duplicate session pairs within a comparison/panel.')
    expected = {(c, s) for c in COMPARISON_ORDER for s in PANEL_TITLES}
    actual = set(distances[['comparison', 'style']].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError(f'Expected all 24 panels; missing={sorted(expected - actual)}, '
                         f'unexpected={sorted(actual - expected)}')
    if not np.isfinite(distances[list(GROUP_LABELS)].to_numpy(dtype=float)).all():
        raise ValueError('Distances must be finite; no observations are silently dropped.')

    rows = []
    for comparison in COMPARISON_ORDER:
        reference_pairs = None
        for panel, (style, title) in enumerate(PANEL_TITLES.items(), start=1):
            subset = distances.loc[(distances.comparison == comparison) & (distances['style'] == style)]
            pairs = set(subset[['session_1', 'session_2']].itertuples(index=False, name=None))
            if reference_pairs is not None and pairs != reference_pairs:
                raise ValueError(f'{comparison}: session pairs differ between panels.')
            reference_pairs = pairs
            n = len(subset)
            if n < 2:
                raise ValueError(f'{comparison}/{style}: need at least two session pairs.')
            groups = {key: subset[key].to_numpy(dtype=float) for key in GROUP_LABELS}
            common = dict(
                figure=f'geodist_3panel_{comparison}', comparison=comparison,
                comparison_title=COMPARISON_TITLES[comparison], task_pair=subset.task_pair.iloc[0],
                panel=panel, style=style, panel_title=title,
                n_session_pairs=n, n_subjects=subset.subj.nunique(),
                n_data=n, n_rand=n, n_shuf=n,
                mean_data=groups['data'].mean(), mean_rand=groups['rand'].mean(),
                mean_shuf=groups['shuf'].mean(), shuffle_source='shuf',
            )
            anova = f_oneway(*groups.values())
            if np.isnan(anova.statistic) or not np.isfinite(anova.pvalue):
                raise ValueError(f'{comparison}/{style}: ANOVA is undefined.')
            anova_row = dict(common, test='anova', Source='type', ddof1=2, ddof2=3*n-3,
                             F=anova.statistic, **{'p-unc': anova.pvalue, 'p-adjust': 'none'})
            pairwise = []
            for a, b in CONTRASTS:
                test = ttest_ind(groups[a], groups[b], equal_var=True)
                if np.isnan(test.statistic) or not np.isfinite(test.pvalue):
                    raise ValueError(f'{comparison}/{style}: {a} vs {b} t-test is undefined.')
                p_corr = min(test.pvalue * len(CONTRASTS), 1.0)
                report = (f"``{GROUP_LABELS[a]}'' and ``{GROUP_LABELS[b]}'': "
                          f'$T({2*n-2})={test.statistic:.2f}$, {_p_latex(p_corr)}')
                pairwise.append(dict(
                    common, test='pairwise', Contrast='type', A=a, B=b,
                    Paired=False, Parametric=True, equal_var=True, alternative='two-sided',
                    T=test.statistic, dof=2*n-2, n_comparisons=len(CONTRASTS),
                    **{'p-unc': test.pvalue, 'p-corr': p_corr, 'p-adjust': 'bonf'},
                    manuscript_latex=report,
                ))
            anova_row['manuscript_latex'] = '; '.join([
                f'$F(2,{3*n-3})={anova.statistic:.2f}$, {_p_latex(anova.pvalue)}',
                pairwise[0]['manuscript_latex'], pairwise[1]['manuscript_latex'],
            ])
            rows.extend([anova_row, *pairwise])
    return pd.DataFrame(rows)


def write_summary_stats(distances, output=SUMMARY_STATS_FILE):
    """Overwrite the CSV deterministically, independently of figure-save settings."""
    stats = summary_statistics(distances)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Nullable dtypes preserve integral degrees of freedom and blank N/A cells.
    for column in ['ddof1', 'ddof2', 'dof', 'n_comparisons']:
        stats[column] = stats[column].astype('Int64')
    for column in ['Paired', 'Parametric', 'equal_var']:
        stats[column] = stats[column].astype('boolean')
    stats.to_csv(output, index=False)
    return stats

#%% Optional export of cached statistics only

if __name__ == '__main__':
    import sys
    if '--stats-only' in sys.argv:
        import argparse
        parser = argparse.ArgumentParser(description='Export cached sensory-control statistics.')
        parser.add_argument('--stats-only', action='store_true')
        parser.add_argument('--input', type=Path, default=GEODIST_FILE)
        parser.add_argument('--output', type=Path, default=SUMMARY_STATS_FILE)
        args = parser.parse_args()
        exported = write_summary_stats(pd.read_pickle(args.input), args.output)
        print(f'Saved {len(exported)} tests: {args.output}')
        raise SystemExit(0)

fig_set()

#%% compute and store principal-angle data

if FORCE_RECOMPUTE or not RESULTS_FILE.exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_all = read_all_sessions(max_gap_dur_secs=MAX_GAP_DUR_SECS, bin_size=BIN_SIZE_MS, get_subspace=False, event_mode='all')
    df_all = df_all[(df_all.session_class != 'ignore') & ~(df_all.filename.isin(EXCLUDED_SESSIONS))].reset_index(drop=True)

    for i in range(len(df_all)):
        s = df_all.iloc[i].session_class
        s.choose_channels().find_y().get_spikes_mu_sd()
        read_energy_files(s)

    df_paired = get_paired_df(df_all)
    results = []
    if CHECKPOINT_FILE.exists() and not FORCE_RECOMPUTE:
        results = pd.read_pickle(CHECKPOINT_FILE).to_dict('records')
        print(f'Resuming from {len(results)} stored comparisons.')
    completed = {(row['comparison'], row['session_1'], row['session_2']) for row in results}

    for i in range(len(df_paired)):
        row = df_paired.iloc[i]
        sessions = {TASK_KEY[row.task_1.task]: row.task_1, TASK_KEY[row.task_2.task]: row.task_2}
        common = sorted(row.common)
        if not common:
            print(f'{row.subj} {row.task_pair}: no common channels; skipping.')
            continue

        conditions = {}
        cutoffs = {}
        for task_code, s in sessions.items():
            event_bins = np.asarray(s.on_task, dtype=bool)
            if len(s.event_times) == 0:
                event_bins = np.ones(len(s.spikes_per_bin), dtype=bool)
            conditions[task_code] = event_bins

            if task_code == 'C' and s.motion_energy is not None:
                valid = event_bins & np.isfinite(s.motion_energy)
                if np.any(valid):
                    cutoff = np.nanpercentile(s.motion_energy[valid], MOTION_PERCENTILE)
                    cutoffs['C'] = cutoff
                    conditions['C1'] = valid & (s.motion_energy < cutoff)
                    conditions['C2'] = valid & (s.motion_energy >= cutoff)

            if task_code in ['S', 'R'] and s.acoustic_energy is not None:
                valid = event_bins & np.isfinite(s.acoustic_energy)
                if np.any(valid):
                    cutoff = np.nanpercentile(s.acoustic_energy[valid], ACOUSTIC_PERCENTILE)
                    cutoffs[task_code] = cutoff
                    conditions[f'{task_code}1'] = valid & (s.acoustic_energy < cutoff)

        for comparison, condition_1, condition_2 in COMPARISONS_BY_PAIR[row.task_pair]:
            s1 = sessions[condition_1[0]]
            s2 = sessions[condition_2[0]]
            result_key = (comparison, s1.session_key, s2.session_key)
            if result_key in completed:
                continue
            if condition_1 not in conditions or condition_2 not in conditions:
                print(f'{s1.session_key}, {s2.session_key}: missing energy for {comparison}; skipping.')
                continue

            channel_col = 'filename' if s1.matfile_type == 'mat' else 'seq_num'
            ix1 = [s1.df_chan.index[s1.df_chan[channel_col] == chan][0] for chan in common]
            ix2 = [s2.df_chan.index[s2.df_chan[channel_col] == chan][0] for chan in common]
            Y1 = s1.spikes_per_bin[conditions[condition_1]][:, ix1]
            Y2 = s2.spikes_per_bin[conditions[condition_2]][:, ix2]

            print(f'[{i + 1}/{len(df_paired)}] {comparison}: {s1.session_key} ({len(Y1)} bins) vs {s2.session_key} ({len(Y2)} bins), {len(common)} channels')
            angles = calc_principal_angles(
                Y1, Y2, calc_rand=True, calc_shuf=True, n_rands=N_RAND, n_shufs=N_SHUF,
                shuf_key=SHUF_KEY, max_iter=FA_MAX_ITER, tol=FA_TOL, svd_method=FA_SVD_METHOD,
            )
            results.append({
                'subj': row.subj, 'task_pair': row.task_pair, 'comparison': comparison,
                'condition_1': condition_1, 'condition_2': condition_2,
                'session_1': s1.session_key, 'session_2': s2.session_key,
                'n_bins_1': len(Y1), 'n_bins_2': len(Y2), 'n_common': len(common),
                'cutoff_1': cutoffs.get(condition_1[0], np.nan), 'cutoff_2': cutoffs.get(condition_2[0], np.nan),
                **angles,
            })
            completed.add(result_key)
            pd.DataFrame(results).to_pickle(CHECKPOINT_FILE)

    df_results = pd.DataFrame(results)
    df_results.to_pickle(RESULTS_FILE)
    print(f'Saved computed principal angles: {RESULTS_FILE}')


#%% read stored data and plot each session-pair comparison

df_results = pd.read_pickle(RESULTS_FILE)
print(f'Read {len(df_results)} computed comparisons from {RESULTS_FILE}')

for _, row in df_results.iterrows():
    fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
    plot_principal_angle(row[['data', 'rand', 'shuf', 'shuf2']].to_dict(), ax, color=TASK_PAIRS_CLR[row.task_pair])
    ax.set_xlabel('Principal angle index')
    ax.set_ylabel('Principal angles (deg)')
    ax.set_title(f'{row.comparison}: {row.session_1}, {row.session_2}')
    ax.legend(frameon=False)
    filename = f'prinAng_{row.comparison}_{row.session_1}_{row.session_2}'
    finish_plot(filename=filename, save_dir=PAIR_PLOT_DIR, savefig=SAVEFIG, show=False)


#%% store geodesic distances and make 24 summary figures

geodist_rows = []
for _, row in df_results.iterrows():
    for style in ['more', 'less', 'full']:
        if style == 'more':
            select = lambda x: x[:len(x)//2]
        elif style == 'less':
            select = lambda x: x[len(x)//2:]
        else:
            select = lambda x: x

        rand = [np.linalg.norm(select(x)) for x in row['rand']]
        shuf = [np.linalg.norm(select(x)) for x in row['shuf']]
        geodist_rows.append({
            'subj': row.subj, 'task_pair': row.task_pair, 'comparison': row.comparison,
            'session_1': row.session_1, 'session_2': row.session_2,
            'style': style, 'data': np.linalg.norm(select(row['data'])),
            'rand': np.mean(rand), 'rand_std': np.std(rand),
            'shuf': np.mean(shuf), 'shuf_std': np.std(shuf),
        })

df_geodist = pd.DataFrame(geodist_rows)
df_geodist.to_pickle(GEODIST_FILE)
print(f'Saved geodesic distances: {GEODIST_FILE}')

ttests = []
for comparison in COMPARISON_ORDER:
    for style in ['more', 'less', 'full']:
        sub_geodists = df_geodist[(df_geodist.comparison == comparison) & (df_geodist['style'] == style)].copy()
        if sub_geodists.empty:
            print(f'No stored comparisons for {comparison}, {style}; skipping plot.')
            continue

        df_long = pd.melt(sub_geodists, id_vars=['comparison'], value_vars=['shuf', 'data', 'rand'], var_name='type', value_name='geodist')
        groups = {name: df_long[df_long['type'] == name].geodist.to_numpy() for name in ['shuf', 'data', 'rand']}
        anova = f_oneway(groups['shuf'], groups['data'], groups['rand'])
        if anova.pvalue < 0.05:
            for name_1, name_2 in [('shuf', 'data'), ('shuf', 'rand'), ('data', 'rand')]:
                test = ttest_ind(groups[name_1], groups[name_2], equal_var=True)
                ttests.append({
                    'comparison': comparison, 'style': style, 'A': name_1, 'B': name_2,
                    'T': test.statistic, 'dof': len(groups[name_1]) + len(groups[name_2]) - 2,
                    'p-unc': test.pvalue, 'p-corr': min(test.pvalue * 3, 1), 'p-adjust': 'bonf',
                })

        plt.figure(figsize=(2, 2.5), dpi=300)
        color = TASK_PAIRS_CLR[sub_geodists.task_pair.iloc[0]]
        for _, result in sub_geodists.iterrows():
            plt.plot([0, 1, 2], [result['shuf'], result['data'], result['rand']], c=color, marker='o', markersize=3)
            plt.errorbar(0, result['shuf'], yerr=result['shuf_std'], c=color, capsize=3)
            plt.errorbar(2, result['rand'], yerr=result['rand_std'], c=color, capsize=3)
        plt.xticks([0, 1, 2], ['Shuffled', 'Data', 'Random'], rotation=45)
        plt.xlim([-0.5, 2.5])
        plt.yticks(np.linspace(0, 180, 5))
        plt.ylabel('Subspace distance (deg)')
        plt.title(f'{comparison}, {style}, p={anova.pvalue:.2e}')
        finish_plot(filename=f'geodist_{comparison}_{style}', save_dir=SUMMARY_PLOT_DIR, savefig=SAVEFIG, show=False)

if ttests:
    df_stats = pd.DataFrame(ttests)
    df_stats.to_csv(STATS_FILE, index=False)
    print(f'Saved geodesic statistics: {STATS_FILE}')


#%% three-panel geodesic summaries

df_geodist = pd.read_pickle(GEODIST_FILE)
df_summary_stats = write_summary_stats(df_geodist, SUMMARY_STATS_FILE)
print(f'Saved all three-panel statistics: {SUMMARY_STATS_FILE}')
#%% plot three-panel geodesic summaries

for comparison in COMPARISON_ORDER:
    comparison_geodists = df_geodist[df_geodist.comparison == comparison].copy()
    n_samples = len(comparison_geodists[['session_1', 'session_2']].drop_duplicates())
    fig, axes = plt.subplots(1, 3, figsize=(5, 2.5), dpi=300, sharey=True)

    for ax, style in zip(axes, PANEL_TITLES):
        sub_geodists = comparison_geodists[comparison_geodists['style'] == style].copy()
        if sub_geodists.empty:
            ax.axis('off')
            continue

        color = TASK_PAIRS_CLR[sub_geodists.task_pair.iloc[0]]
        for _, result in sub_geodists.iterrows():
            ax.plot([0, 1, 2], [result['shuf'], result['data'], result['rand']], c=color, marker='o', markersize=3)
            ax.errorbar(0, result['shuf'], yerr=result['shuf_std'], c=color, capsize=3)
            ax.errorbar(2, result['rand'], yerr=result['rand_std'], c=color, capsize=3)

        panel_stats = df_summary_stats[(df_summary_stats.comparison == comparison)
                                       & (df_summary_stats['style'] == style)]
        for name_1, name_2, label in [
            ('data', 'shuf', 'Shuffled vs Data'),
            ('data', 'rand', 'Data vs Random'),
        ]:
            # Use the exported tests for the figure annotations as well.
            p_corr = panel_stats.loc[(panel_stats.A == name_1)
                                    & (panel_stats.B == name_2), 'p-corr'].iloc[0]
            if p_corr < 0.001:
                sig = '***'
            elif p_corr < 0.01:
                sig = '**'
            elif p_corr < 0.05:
                sig = '*'
            else:
                sig = 'n.s.'
            ax.plot([], [], linestyle='none', label=f'{label}: {sig}, p={p_corr:.2g}')

        panel_title = PANEL_TITLES[style]
        ax.set_title(panel_title)
        ax.set_xticks([0, 1, 2], ['Shuffled', 'Data', 'Random'], rotation=0)
        ax.set_xlim([-0.5, 2.5])
        ax.set_yticks(np.linspace(0, 180, 5))
        ax.legend(handlelength=0, handletextpad=0, loc='best', fontsize=6, frameon=False)

    axes[0].set_ylabel('Subspace distance (deg)')
    fig.suptitle(f'{COMPARISON_TITLES[comparison]}, N={n_samples}', y=0.95)
    finish_plot(filename=f'geodist_3panel_{comparison}', save_dir=SUMMARY_3PANEL_DIR, savefig=SAVEFIG, show=True)

#%% Energy distribution

# Cached principal angles bypass session loading above. Load the energy inputs
# here as well so running the complete script from a fresh process still works.
if 'df_all' not in globals():
    df_all = read_all_sessions(max_gap_dur_secs=MAX_GAP_DUR_SECS, bin_size=BIN_SIZE_MS, get_subspace=False, event_mode='all')
    df_all = df_all[(df_all.session_class != 'ignore') & ~(df_all.filename.isin(EXCLUDED_SESSIONS))].reset_index(drop=True)
    for s in df_all.session_class:
        read_energy_files(s)

motions = defaultdict(list)
for i in range(len(df_all)):
    s = df_all.iloc[i].session_class
    if s.motion_energy is None:
        continue
    motions[s.task].extend(np.log10(s.motion_energy))

acoustics = defaultdict(list)
for i in range(len(df_all)):
    s = df_all.iloc[i].session_class
    if s.acoustic_energy is None:
        continue
    acoustics[s.task].extend(np.log10(s.acoustic_energy))

# Motion energy
fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=300)
for task in TASK_NAMES:
    ax.hist(motions[task], bins=np.arange(4, 5.5, 0.05), color=TASK_CLR[task], histtype='step', density=True)
ax.set_xlabel('Log motion energy')
ax.set_ylabel('Density')
finish_plot(filename='log_motion_energy_hist_byTask', save_dir=ENERGY_PLOT_DIR, savefig=SAVEFIG, show=True, exts=('png', 'svg', 'pdf'))

# Acoustic amplitude
fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=300)
for task in TASK_NAMES:
    ax.hist(acoustics[task], bins=np.arange(3.7, 6.3, 0.1), color=TASK_CLR[task], histtype='step', density=True)
ax.set_xlabel('Log acoustic amplitude')
ax.set_ylabel('Density')
finish_plot(filename='log_acoustic_amplitude_hist_byTask', save_dir=ENERGY_PLOT_DIR, savefig=SAVEFIG, show=True, exts=('png', 'svg', 'pdf'))

# %%
