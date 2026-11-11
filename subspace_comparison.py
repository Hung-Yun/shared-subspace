#%% Imports
from pathlib import Path
import json
import subprocess
import sys
import utils
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt

from utils import *
ROOT = Path(utils.__file__).resolve().parent
SAVEFIG = False
SAVESTATS = False
SHOW = True

#%% Read data

df_all = read_all_sessions(max_gap_dur_secs=1, bin_size=100, get_subspace=True)
df_all = df_all[(df_all.session_class!='ignore') & ~(df_all.filename.isin(EXCLUDED_SESSIONS))].reset_index(drop=True)

#%% Principal angles and geodesic distance functions
def document_hash(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}

def load_results(bundle):
    """Return (results_by_pair_key, lightweight_df_pairs), without refitting.

    Read-only loading also supports historical bundles with a different frozen
    execution script. It verifies run/pair identities rather than requiring the
    current source file to match the historical source used to calculate them.
    """
    from scripts.subspace_alignment import ve_main_view

    bundle = Path(bundle)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    index = json.loads((bundle / 'results/index.json').read_text())
    if (index['run_id'] != manifest['run_id'] or manifest['run_id'] != document_hash(
            {k: v for k, v in manifest.items() if k != 'run_id'})):
        raise ValueError('Saved result index and manifest do not match.')
    results, rows = {}, []
    for item in index['pairs']:
        path = bundle / 'results' / item['path']
        # Validate current four-column VE views, or select them in memory for
        # historical eight-column files. This read never changes saved files.
        result = ve_main_view(load_npz(path))
        meta = json.loads(str(result['metadata']))
        if (meta['run_id'], meta['pair_id'], meta['key']) != (index['run_id'], item['pair_id'], item['key']):
            raise ValueError(f'Saved pair identity mismatch: {path}')
        labels = item['metadata']
        sessions = labels.get('sessions', [item['pair_id'], ''])
        results[item['key']] = result
        rows.append(dict(key=item['key'], pair_id=item['pair_id'],
                         patient=labels.get('patient', ''), task_pair=labels.get('task_pair', ''),
                         session_1=sessions[0], session_2=sessions[1],
                         n_channels=len(labels.get('channels', [])),
                         d=int(result['d']), K=int(result['K']), result_path=str(path)))
    # Keep the manuscript panel order; pair keys remain stable storage identities.
    return results, pd.DataFrame(rows).sort_values('task_pair', ascending=False).reset_index(drop=True)


def plot_principal_angle(result, ax, color, linewidth=.8):
    """Plot current PAGDVE arrays, pooling the two named within-task controls."""
    ang, rand = result['data_pa'], result['rand_pa']
    shuffled = np.concatenate([result['shuf_task1_pa'], result['shuf_task2_pa']], axis=0)
    indices = np.arange(len(ang))
    ax.scatter(indices, ang, c=color, s=5, lw=linewidth)
    ax.plot(indices, ang, c=color, lw=linewidth, label='Data')
    for metric, label, linestyle in ((rand, 'Random', '-'), (shuffled, 'Shuffled', '--')):
        if len(metric):
            mean, std = np.mean(metric, axis=0), np.std(metric, axis=0)
            ax.scatter(indices, mean, ec='gray', fc='none', s=5, lw=linewidth)
            ax.plot(indices, mean, c='gray', ls=linestyle, lw=linewidth, label=label)
            ax.fill_between(indices, mean-std, mean+std, color='gray', alpha=.3, lw=0)
    ax.set(ylim=(-.5, 90.5), yticks=np.linspace(0, 90, 4), xticks=indices)
    ax.set_xticklabels(indices + 1)


def geodesic_summary(results, df_pairs, style='full', shuffle='both'):
    """One row per pair, preserving the old GD slices and mean/std convention.

    GD 'less' is [d//2:], INCLUDING an odd final angle, as in the manuscript
    plot. This differs intentionally from VE's [K:2*K]. Default shuffle='both'
    pools the two source-task controls, each calculated at ranks (M1,M2).
    'task1' or 'task2' selects one source task. PA plots pool both too.
    This pools PA vectors/GD values, not the half-specific VE columns.
    """
    if style not in ('more', 'less', 'full') or shuffle not in ('task1', 'task2', 'both'):
        raise ValueError('Invalid GD style or shuffle selection.')
    rows = []
    for row in df_pairs.itertuples():
        result = results[row.key]
        d = len(result['data_pa'])
        selection = {'more': slice(None, d//2), 'less': slice(d//2, None), 'full': slice(None)}[style]
        sources = ('shuf_task1_pa', 'shuf_task2_pa') if shuffle == 'both' else (f'shuf_{shuffle}_pa',)
        shuffled = np.concatenate([result[key] for key in sources], axis=0)
        rand_gd = np.linalg.norm(result['rand_pa'][:, selection], axis=1)
        shuf_gd = np.linalg.norm(shuffled[:, selection], axis=1)
        if not len(rand_gd) or not len(shuf_gd):
            raise ValueError('GD control plots require at least one repeat of each selected control.')
        rows.append(dict(key=row.key, pair_id=row.pair_id, patient=row.patient, task_pair=row.task_pair,
                         style=style, shuffle_source=shuffle,
                         data=float(np.linalg.norm(result['data_pa'][selection])),
                         rand=float(rand_gd.mean()), rand_std=float(rand_gd.std()),
                         shuf=float(shuf_gd.mean()), shuf_std=float(shuf_gd.std()),
                         n_rand=len(rand_gd), n_shuf=len(shuf_gd)))
    return pd.DataFrame(rows)


#%% Principal angles (Fig 5A-B, compare)

## Load saved results
FIG6_BUNDLE = ROOT / 'data/subspace_alignment/seed0_n500'
FIG6_PLOT_DIR = ROOT / 'plots/principal_angles'
gd_shuffle = 'both'

save_dir = FIG6_PLOT_DIR
tables, anovas, ttests = [], [], []
results, df_pairs = load_results(FIG6_BUNDLE)

fig_set(font_size=8, linewidth=LW)
fig, axes = plt.subplots(int(np.ceil(len(df_pairs)/6)), 6, figsize=(6, 6),
                            sharey=True, squeeze=False, dpi=300)
for ax, row in zip(axes.flat, df_pairs.itertuples()):
    plot_principal_angle(results[row.key], ax, TASK_PAIRS_CLR[row.task_pair], linewidth=LW)
    # ax.tick_params(axis='x', labelsize=4)
    ax.set_title(f'{row.session_1}, {row.session_2}', fontsize=6)
for ax in list(axes.flat)[len(df_pairs):]:
    ax.set_visible(False)
counts = {(len(r['rand_pa']), len(r['shuf_task1_pa']), len(r['shuf_task2_pa'])) for r in results.values()}
if len(counts) == 1:
    n_rand, n_1, n_2 = counts.pop()
    fig.suptitle(f'{n_rand} randomized; {n_1} + {n_2} within-task shuffled (pooled)', y=.98, fontsize=6)
finish_plot(filename='prinAng_all_pairs', save_dir=save_dir, savefig=SAVEFIG, show=SHOW)

## Enlarged example
example_index = 18
overview = df_pairs.drop(index=example_index, errors='ignore')
fig, axes = plt.subplots(max(1, int(np.ceil(len(overview)/7))), 7, figsize=(7, 3), dpi=300, squeeze=False)
for ax, row in zip(axes.flat, overview.itertuples()):
    plot_principal_angle(results[row.key], ax, TASK_PAIRS_CLR[row.task_pair], linewidth=LW)
    sns.despine(ax=ax, trim=True)
    ax.set(xticks=[], yticks=[])
for ax in list(axes.flat)[len(overview):]:
    ax.axis('off')
finish_plot(filename=f'prinAng_example_all_less{example_index}', save_dir=save_dir, savefig=SAVEFIG, show=SHOW)

## Remaining examples 
if len(df_pairs) > example_index:
    row = df_pairs.iloc[example_index]
    fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
    plot_principal_angle(results[row.key], ax, TASK_PAIRS_CLR[row.task_pair], linewidth=LW)
    ax.set(xlabel='Principal angle index', ylabel='Principal angles (deg)')
    ax.set_title(f'{row.session_1}, {row.session_2}', fontsize=6)
    ax.legend(frameon=False)
    finish_plot(filename=f'prinAng_example_{example_index}', save_dir=save_dir, savefig=SAVEFIG, show=SHOW)

#%% Geodesic distance (Fig 5C and Fig S10, compare and sup-geodist)
for style in ('more', 'less', 'full'):
    table = geodesic_summary(results, df_pairs, style, gd_shuffle)
    tables.append(table)
    for task_pair in TASK_PAIRS:
        subset = table.loc[table.task_pair == task_pair]
        if subset.empty:
            continue
        long = subset.melt(id_vars=['key', 'task_pair'], value_vars=['shuf', 'data', 'rand'],
                            var_name='type', value_name='geodist')
        stat = pg.anova(data=long, dv='geodist', between='type').iloc[0]
        anovas.append(dict(stat, task_pair=task_pair, style=style, shuffle_source=gd_shuffle))
        if stat['p-unc'] < .05:
            test = pg.pairwise_tests(dv='geodist', between='type', data=long, padjust='bonf')
            ttests.append(test.assign(task_pair=task_pair, style=style, shuffle_source=gd_shuffle))
        fig, ax = plt.subplots(figsize=(2, 2.5), dpi=300)
        name = f'geodist_perTaskPair_{style}_{task_pair}_{gd_shuffle}'
        for row in subset.itertuples():
            color = TASK_PAIRS_CLR[task_pair]
            ax.plot([0, 1, 2], [row.shuf, row.data, row.rand], c=color, marker='o', markersize=3)
            ax.errorbar(0, row.shuf, yerr=row.shuf_std, c=color, capsize=3)
            ax.errorbar(2, row.rand, yerr=row.rand_std, c=color, capsize=3)
        ax.set(xticks=[0, 1, 2], xticklabels=['Shuffle', 'Data', 'Random'], xlim=(-.5, 2.5),
                yticks=np.linspace(0, 180, 5), ylabel='Subspace distance (deg)',
                title=f'{task_pair}, p={stat["p-unc"]:.2e}')
        finish_plot(filename=name, save_dir=save_dir, savefig=SAVEFIG, show=SHOW)

## Stats
summary = pd.concat(tables, ignore_index=True)
anova = pd.DataFrame(anovas)
pairwise = pd.concat(ttests, ignore_index=True) if ttests else pd.DataFrame()

if SAVESTATS:
    for name, frame in (('geodists', summary), ('geodist_anova', anova), ('geodist_pairwise', pairwise)):
        frame.to_csv(Path(save_dir) / f'{name}_{gd_shuffle}.csv', index=False)


#%% Subspace alignment analysis hyperparams

TASK_PAIR = 'all'  # Or, for example, 'Pursuit/Rotations'.
N_REPEATS = 100  # Per session pair and per case; supplied geometry supports up to 100.
N_SAMPLES = 14300  # Per task in each simulated comparison.
SEED = 20261026
GEOMETRY_SEED = 20261023
VERIFY = True
CACHE = ROOT / 'data/temp/df_all.h5'
RESULTS_DIR = ROOT / 'data/subspace_simulation'
MODEL_DIR = ROOT / 'models/subspace_simulation'
PLOT_DIR = ROOT / 'plots/subspace_simulation'
PARAMS = MODEL_DIR / 'simulation_generating_params.pkl'
RERUN = False

#%% Subspace alignment simulation (Fig 6B and Fig S13, align and sup-align)
if RERUN: 
    for sim_input in (CACHE, PARAMS):
        if not sim_input.is_file():
            raise FileNotFoundError(sim_input)
    if not 1 <= N_REPEATS <= 100 or N_SAMPLES < 2:
        raise ValueError('Use 1–100 repeats and at least two samples per task.')
    if min(SEED, GEOMETRY_SEED) < 0:
        raise ValueError('Seeds must be nonnegative.')
    if TASK_PAIR != 'all' and TASK_PAIR not in (
            'Podcast/Pursuit', 'Podcast/Rotations', 'Pursuit/Rotations'):
        raise ValueError('Choose all, Podcast/Pursuit, Podcast/Rotations, or Pursuit/Rotations.')
    if any(p.name != PARAMS.name for p in MODEL_DIR.iterdir()):
        raise FileExistsError(f'Existing fitted models are preserved: {MODEL_DIR}')

    if RESULTS_DIR.exists() and any(RESULTS_DIR.iterdir()):
        sim_manifest_path = RESULTS_DIR / 'manifest.json'
        sim_manifest = json.loads(sim_manifest_path.read_text()) if sim_manifest_path.is_file() else {}
        if sim_manifest.get('dataset') != 'jay_prelim':
            raise FileExistsError(f'Existing simulation results are preserved: {RESULTS_DIR}')
        sim_jay_data = RESULTS_DIR.with_name('jay_prelim')
        sim_jay_plots = PLOT_DIR.with_name('jay_prelim')
        if sim_jay_data.exists() or sim_jay_plots.exists():
            raise FileExistsError('jay_prelim already exists; preserving both datasets.')
        RESULTS_DIR.rename(sim_jay_data)
        if PLOT_DIR.exists():
            PLOT_DIR.rename(sim_jay_plots)
        # Keep the archived manifest's figure link and file inventory usable.
        sim_manifest['directories']['plots'] = os.path.relpath(sim_jay_plots, sim_jay_data)
        for sim_file in sim_manifest.get('files', []):
            for sim_prefix in ('data', 'plots'):
                sim_file['path'] = sim_file['path'].replace(
                    f'{sim_prefix}/subspace_simulation/', f'{sim_prefix}/jay_prelim/', 1)
        (sim_jay_data / 'manifest.json').write_text(json.dumps(sim_manifest, indent=2))
        print(f'Preserved Jay preliminary outputs: {sim_jay_data}\n{sim_jay_plots}', flush=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    sim_commands = [[
        sys.executable, '-u', str(ROOT / 'scripts/subspace_simulation/simulate_subspace_recovery.py'),
        '--common-unit-cache', str(CACHE), '--params', str(PARAMS),
        '--data-dir', str(RESULTS_DIR), '--model-dir', str(MODEL_DIR),
        '--task-pair', TASK_PAIR, '--replicates', str(N_REPEATS),
        '--samples', str(N_SAMPLES), '--seed', str(SEED),
        '--geometry-seed', str(GEOMETRY_SEED),
    ]]
    if VERIFY:
        sim_commands.append([
            sys.executable, '-u', str(ROOT / 'scripts/subspace_simulation/check_subspace_recovery.py'),
            '--data-dir', str(RESULTS_DIR),
        ])
    for sim_command in sim_commands:
        print(f'Running {Path(sim_command[2]).name} with {sys.executable}', flush=True)
        with subprocess.Popen(sim_command, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1) as sim_process:
            try:
                for sim_line in sim_process.stdout:
                    print(sim_line, end='', flush=True)
                if sim_process.wait():
                    raise subprocess.CalledProcessError(sim_process.returncode, sim_command)
            except KeyboardInterrupt:
                sim_process.terminate()
                try:
                    sim_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    sim_process.kill()
                    sim_process.wait()
                raise
    print(f'Done. Results: {RESULTS_DIR}\nModels: {MODEL_DIR}', flush=True)

RESULTS_DIR = ROOT / 'data/subspace_simulation'
TABLE_DIR = RESULTS_DIR / 'tables'
CASES = {1: 'Shared low-var dims', 2: 'Shared high-var dims', 3: 'Random var assignment'}
ORDER = ['top', 'more', 'less']
LABELS = ['Top factors', 'More aligned', 'Less aligned']
PAIRS = ['Pursuit/Rotations', 'Podcast/Pursuit', 'Podcast/Rotations']

sim_config = json.loads((RESULTS_DIR / 'provenance/config.json').read_text())
if sim_config['status'] != 'complete':
    raise RuntimeError('The simulation is not complete yet; run this cell after it finishes.')
sim_raw = pd.read_csv(TABLE_DIR / 'results.csv')
sim_fitted = (sim_raw.loc[sim_raw.method == 'fitted_data_total']
              .groupby(['case', 'task_pair', 'pair', 'patient', 'task', 'category'],
                       as_index=False).fraction.mean()) # Take average

fig_set(font_size=10, linewidth=0.8)
stats = []
anovas = []
for task_pair in PAIRS:
    for sim_case in range(1,4):
        session_tasks = dict(zip(['vA', 'vB'], task_pair.split('/')))
        palette = {'vA': "#EF4B6A", 'vB': '#1E5EAA'}
        # Same long-format columns as Fig. 6C, retaining pair and patient identifiers.
        # vA/vB refer to the first/second task in task_pair, not two new subjects.
        # sim_fitted already averages the 100 repeats within each session pair.
        df = (sim_fitted.loc[(sim_fitted.case == sim_case)
                            & (sim_fitted.task_pair == task_pair)]
            .rename(columns={'category': 'alignment', 'fraction': 'var_exp'}).copy())
        df['session'] = df.task.map({task: sess for sess, task in session_tasks.items()})

        # Match empirical Fig. 6C: independent-group tests on session-pair means,
        # with no patient averaging. Overall alignment contrasts pool vA and vB.
        # These tests do not model pairing or dependence among pairs sharing patients.
        # BH-FDR corrects the three overall alignment comparisons within this panel.
        df_anova = pg.anova(dv='var_exp', between=['session', 'alignment'], data=df)
        df_stats = pg.pairwise_tests(dv='var_exp', between=['session', 'alignment'], data=df, padjust='fdr_bh')
        stats.append(df_stats.assign(task_pair=task_pair, case=sim_case))
        anovas.append(df_anova.assign(task_pair=task_pair, case=sim_case))

        fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=300)
        sns.stripplot(data=df, x='alignment', y='var_exp', hue='session', ax=ax,
                    order=ORDER, hue_order=['vA', 'vB'], palette=palette,
                    size=3, linewidth=0.8, dodge=True, jitter=False, legend=False, alpha=0.7)
        for sess in ['vA', 'vB']:
            ys = df.loc[df.session == sess].groupby('alignment').var_exp.mean().reindex(ORDER)
            ax.plot([0, 1, 2], ys, lw=0.8, c=palette[sess], zorder=4)
            ax.scatter([0, 1, 2], ys, s=30, fc=palette[sess], ec='k', linewidth=0.8, zorder=5)
        ax.set(xticks=[0, 1, 2], xticklabels=['Top\nfactors', 'More\naligned', 'Less\naligned'], ylim=(0.25,0.75),
            xlim=(-.5, 2.5), xlabel='', ylabel='Proportional variance explained')
        ax.set_title(f'Case {sim_case}: {CASES[sim_case]}\n{task_pair}')
        finish_plot(filename=f'case_{sim_case}_{task_pair.replace("/", "_")}', save_dir=PLOT_DIR, savefig=SAVEFIG, show=SHOW)

if SAVESTATS:
    pd.concat(stats, ignore_index=True).to_csv('stats/subspace_simulation.csv', index=False)
    pd.concat(anovas, ignore_index=True).to_csv('stats/subspace_simulation_anova.csv', index=False)




#%% Subspace alignment (Fig 6C, align)

def calc_var_exp(row, n_in_low_group=None):
    """Calculate the variance fractions for one session pair.

    Inputs: paired-session row with task_1, task_2 and common unit filenames;
    optional number of dimensions per alignment group (default: half the
    number of principal angles, rounded down).
    Outputs: dict of vA/vB_top, _more and _less fractions. Each denominator is
    the projected variance across that task's entire retained FA subspace.
    Refits FA and updates both Session objects in memory. When the number of
    principal angles is odd, the final direction is excluded from both groups.
    Principal vectors are mapped through the QR bases, as in PAGDVE. This
    corrects directional VE while preserving PA, K, and the denominators.
    """
    t1, t2, common = row.task_1, row.task_2, row.common

    t1.find_y(np.where(t1.df_chan.filename.isin(common))[0]).find_subspace()
    t2.find_y(np.where(t2.df_chan.filename.isin(common))[0]).find_subspace()

    A = t1.FA.subspace
    B = t2.FA.subspace

    # Same QR/SVD as principal_angles and scripts/subspace_alignment.py. U and Vh
    # rotate the QR bases; applying them to A.T/B.T would omit QR sign changes.
    Qa, _ = np.linalg.qr(A.T)
    Qb, _ = np.linalg.qr(B.T)
    U, singular, Vh = np.linalg.svd(Qa.T @ Qb, full_matrices=False)
    angles = np.rad2deg(np.arccos(np.clip(singular, -1., 1.)))
    principal_A = Qa @ U
    principal_B = Qb @ Vh.T

    varA   = np.var(t1.Y_norm @ A.T, axis=0)
    varA_p = np.var(t1.Y_norm @ principal_A, axis=0)
    varB   = np.var(t2.Y_norm @ B.T, axis=0)
    varB_p = np.var(t2.Y_norm @ principal_B, axis=0)

    n_in_low_group = n_in_low_group or len(angles) // 2

    # Trim arrays if needed
    total = 2 * n_in_low_group
    if total < len(angles):
        varA_p = varA_p[:total]
        varB_p = varB_p[:total]
        assert len(varA_p) == total

    return {
        'vA_top': np.sum(varA[:n_in_low_group]) / np.sum(varA),
        'vA_more': np.sum(varA_p[:n_in_low_group]) / np.sum(varA),
        'vA_less': np.sum(varA_p[n_in_low_group:]) / np.sum(varA),
        'vB_top': np.sum(varB[:n_in_low_group]) / np.sum(varB),
        'vB_more': np.sum(varB_p[:n_in_low_group]) / np.sum(varB),
        'vB_less': np.sum(varB_p[n_in_low_group:]) / np.sum(varB)
    }

stats = []
anovas = []
df_pairs = get_paired_df(df_all)
for j in range(3):
    df_pair = df_pairs[df_pairs.task_pair == TASK_PAIRS[j]]
    results = df_pair.reset_index(drop=True).apply(calc_var_exp, axis=1)
    df_pair = pd.concat([df_pair.reset_index(drop=True), results.apply(pd.Series)], axis=1)
    df = pd.melt(df_pair, id_vars=['task_pair'],
                value_vars=['vA_top', 'vA_more', 'vA_less', 'vB_top', 'vB_more', 'vB_less'],
                var_name='type', value_name='var_exp')
    df['session'] = df['type'].str.split('_').str[0]
    df['alignment'] = df['type'].str.split('_').str[1]
    df_anova = pg.anova(dv='var_exp', between=['session', 'alignment'], data=df)
    df_stats = pg.pairwise_tests(dv='var_exp', between=['session', 'alignment'], data=df, padjust='fdr_bh')

    stats.append(df_stats.assign(task_pair=TASK_PAIRS[j]))
    anovas.append(df_anova.assign(task_pair=TASK_PAIRS[j]))


    # BH-FDR correction across the three alignment comparisons within this task pair.
    df_ = df_stats.query('Contrast=="alignment"')
    hl = df_[(df_.A=='less')&(df_.B=='more')]['p-corr'].iloc[0]
    mh = df_[(df_.A=='less')&(df_.B=='top')]['p-corr'].iloc[0]
    ml = df_[(df_.A=='more')&(df_.B=='top')]['p-corr'].iloc[0]

    fig, ax = plt.subplots(figsize=(2.5,2.5), dpi=300)
    sns.stripplot(data=df, x='alignment', y='var_exp', hue='session', ax=ax,
                palette=[TASK_CLR[df_pair.iloc[0].task_1.task], TASK_CLR[df_pair.iloc[0].task_2.task]],
                size=3, linewidth=1, dodge=True, jitter=False, legend=None, alpha=1)

    for sess in ['vA', 'vB']:
        df_ = df[df.session==sess]
        ys = [df_[df_.alignment=='top']['var_exp'].mean(),
              df_[df_.alignment=='more']['var_exp'].mean(),
              df_[df_.alignment=='less']['var_exp'].mean()]
        c = TASK_CLR[df_pair.iloc[0].task_1.task] if sess == 'vA' else TASK_CLR[df_pair.iloc[0].task_2.task]
        plt.plot([0,1,2], ys, lw=0.8, c=c, zorder=4)
        plt.scatter([0,1,2], ys, s=30, fc=c, ec='k', linewidth=0.8, zorder=5)

    plt.xticks([0,1,2], ['Top\nfactors', 'More\naligned', 'Less\naligned'])
    plt.xlim([-0.5,2.5])
    plt.ylim([0.25, 0.75])
    plt.xlabel('')
    plt.ylabel('Proportional variance explained')
    plt.title(f'Task pair: {TASK_PAIRS[j]}\nEmpirical')
    save_dir = os.path.join('plots', 'principal_angles')
    finish_plot(filename=f'var_explained_perTaskPair_{TASK_PAIRS[j]}', save_dir=save_dir, savefig=SAVEFIG)

if SAVESTATS:
    pd.concat(stats, ignore_index=True).to_csv('stats/subspace_alignment.csv', index=False)
    pd.concat(anovas, ignore_index=True).to_csv('stats/subspace_alignment_anova.csv', index=False)

