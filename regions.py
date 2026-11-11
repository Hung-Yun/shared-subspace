#%% Imports
import io
import json
import joblib
import numpy as np
import pingouin as pg
import collections
import matplotlib.pyplot as plt
from diptest import diptest
from pathlib import Path
from scripts.shift_controls import load_manifest, sha256

from utils import *
ROOT = Path(io.__file__).resolve().parent
BUNDLE = ROOT / 'data' / 'shift_controls'
SAVE_DIR = ROOT / 'plots' / 'shift_controls'
SAVEFIG = False
COMPLETE_ONLY = False  # Preview available converged shifts; True requires all 200.
MIN_SHIFTS = 1

## Load data
manifest = load_manifest(BUNDLE)
if not (BUNDLE / 'shifted_data.pkl').exists():
    raise FileNotFoundError('Collect the server results first: python scripts/shift_controls.py collect --bundle data/shift_controls')
coverage = json.loads((BUNDLE / 'coverage.json').read_text())
if coverage['run_id'] != manifest['run_id'] or sha256(BUNDLE / 'shifted_data.pkl') != coverage['output_sha256']:
    raise ValueError('Collected data and metadata do not match; collect/copy them together.')
if sha256(BUNDLE / 'original_cache.pkl') != manifest['baseline_sha256']:
    raise ValueError('Original model cache changed.')
print(f'{coverage["saved"]}/{coverage["planned"]} fits saved; {len(coverage["flagged"])} nonconverged fits flagged across both shift conditions. Only converged HPC-shifted fits are used below.')

UNSHIFTED_DATA = joblib.load(BUNDLE / 'original_cache.pkl')
SHIFTED_DATA = joblib.load(BUNDLE / 'shifted_data.pkl')
SESSIONS = {s['key']: s for s in manifest['sessions']}
SESSION_KEYS = list(SHIFTED_DATA.keys())

def collect_delta_svpu_per_region(session_key, region):
    """
    Return real values, converged HPC-shifted values, and empirical p-values.
    The same whole-fit convergence filter applies to both regional readouts.

    USAGE
        ix = 1
        acc_real, acc_shifted, p_acc = collect_delta_svpu_per_region(session_key=SESSION_KEYS[ix], region='acc')
        hpc_real, hpc_shifted, p_hpc = collect_delta_svpu_per_region(session_key=SESSION_KEYS[ix], region='hpc')
    """
    if region not in ['hpc','acc']: raise ValueError('Region not supported')
    
    ix = np.asarray(SESSIONS[session_key][f'{region}_ix']) # Get the channel index for that region in that session

    alone = UNSHIFTED_DATA[session_key][f'fa_{region}']['shared_var_per_unit'] # svpu when fitted alone
    real = UNSHIFTED_DATA[session_key]['fa_cv']['shared_var_per_unit'][ix] # svpu when including the other region
    diff_real = 100 * (real - alone) # ∆svpu - the original metric reported

    # Keep only converged fits; available shift indices may have gaps.
    diff_shifted = []
    for fit in SHIFTED_DATA[session_key]['shifted_hpc'].values():
        if not fit['converged']:
            continue
        shifted = fit['shared_var_per_unit'][ix]
        diff_shifted.append(100 * (shifted - alone))
    if not diff_shifted:
        raise ValueError(f'{session_key}: no converged HPC-shifted fits available.')
    diff_shifted = np.array(diff_shifted)

    # Collecting p values
    pvals = (diff_real[None,:] < diff_shifted).sum(axis=0) / diff_shifted.shape[0]

    return diff_real, diff_shifted, pvals

#%% Read data

df_all = read_all_sessions(max_gap_dur_secs=1, bin_size=100, get_subspace=True)
df_all = df_all[(df_all.session_class!='ignore') & ~(df_all.filename.isin(EXCLUDED_SESSIONS))].reset_index(drop=True)

#%% Regional difference - simple characterization (Fig 4A-B, region)

hpc, acc = [], []
for i in range(len(df_all)):
    s = df_all.iloc[i].session_class
    n_hpc = np.sum(s.chosen_chan['region'] == 'HPC')
    n_acc = np.sum(s.chosen_chan['region'] == 'ACC')
    hpc.append(n_hpc)
    acc.append(n_acc)

# Unused
plt.figure(figsize=(2.5,2.5), dpi=300)
plt.scatter(hpc, acc, ec='k', fc='none', s=5, lw=LW)
plt.plot([0,32],[0,32], c='k',ls='--', lw=LW)
plt.xticks(np.linspace(0,32,5))
plt.yticks(np.linspace(0,32,5))
plt.xlabel('# HPC channels')
plt.ylabel('# ACC channels')
plt.title(f'Avg ACC {np.mean(acc):.2f}, HPC {np.mean(hpc):.2f} channels', fontsize=6)
save_dir = os.path.join('plots', 'regional_difference')
finish_plot(filename='n_chan_perRegion', save_dir=save_dir, savefig=SAVEFIG)

# 1. nfiring rates of all subjects & all sessions
hpc, acc = [], []
for i in range(len(df_all)):
    s = df_all.iloc[i].session_class
    hpc.extend(s.chosen_chan[s.chosen_chan['region']=='HPC']['fr'])
    acc.extend(s.chosen_chan[s.chosen_chan['region']=='ACC']['fr'])

res = pg.mwu(hpc,acc)
stats = res.iloc[0]['U-val']
pval = res.iloc[0]['p-val']
lghpc = np.log10(hpc)
lgacc = np.log10(acc)

# Firing rate plot with linear scale (Unused)
plt.figure(figsize=(2.5,2.5), dpi=300)
h1 = plt.hist(hpc, bins=np.linspace(0,100,21), edgecolor=AREA_CLR['HPC'], density=True, label='HPC', histtype='step', linewidth=LW)
h2 = plt.hist(acc, bins=np.linspace(0,100,21), edgecolor=AREA_CLR['ACC'], density=True, label='ACC', histtype='step', linewidth=LW)
disp = np.max([np.max(h1[0]), np.max(h2[0])])*1.1
plt.scatter(np.median(hpc), disp, color=AREA_CLR['HPC'], marker='v', s=5)
plt.scatter(np.median(acc), disp, color=AREA_CLR['ACC'], marker='v', s=5)
plt.title(f'Mann-Whitney U, U={stats:.2f}, p={pval:.2e}', fontsize=6)
plt.xlabel('Firing rate (Hz)')
plt.ylabel('Density')
plt.xticks(np.linspace(0,100,6, dtype=int))
plt.subplots_adjust(hspace=0.1,left=0.2,right=0.9,bottom=0.2)
plt.legend(frameon=False)
save_dir = os.path.join('plots', 'regional_difference')
finish_plot(filename='firing_rate_perRegion', save_dir=save_dir, savefig=SAVEFIG)

# Firing rate plot with log scale
plt.figure(figsize=(2.5,2.75), dpi=300)
h1 = plt.hist(lghpc, bins=np.linspace(-0.5,2,26), edgecolor=AREA_CLR['HPC'], density=True, label='HPC', histtype='step', linewidth=LW)
h2 = plt.hist(lgacc, bins=np.linspace(-0.5,2,26), edgecolor=AREA_CLR['ACC'], density=True, label='ACC', histtype='step', linewidth=LW)
disp = np.max([np.max(h1[0]), np.max(h2[0])])*1.1
plt.scatter(np.log10(np.median(hpc)), disp, color=AREA_CLR['HPC'], marker='v', s=5)
plt.scatter(np.log10(np.median(acc)), disp, color=AREA_CLR['ACC'], marker='v', s=5)
plt.title(f'Mann-Whitney U, U={stats:.2f}, p={pval:.2e}', fontsize=6)
plt.xlabel('Firing rate (Hz)')
plt.ylabel('Density')
plt.xticks(np.arange(-0.5, 2.1, 0.5), [f'$10^{{{i}}}$' for i in np.arange(-0.5, 2.1, 0.5)], rotation=45)
plt.subplots_adjust(hspace=0.1,left=0.2,right=0.9,bottom=0.2)
plt.legend(frameon=False)
save_dir = os.path.join('plots', 'regional_difference')
finish_plot(filename='log_firing_rate_perRegion', save_dir=save_dir, savefig=SAVEFIG)

# 2. shared var per unit of all subjects & all sessions
hpc, acc = [], []
for i in range(len(df_all)):
    s = df_all.iloc[i].session_class
    hpc_ix = s.chosen_chan[s.chosen_chan['region'] == 'HPC'].seq_index
    acc_ix = s.chosen_chan[s.chosen_chan['region'] == 'ACC'].seq_index
    s.find_y().get_spikes_mu_sd().find_subspace()
    hpc.extend(s.FA.shared_var_per_unit[hpc_ix])
    acc.extend(s.FA.shared_var_per_unit[acc_ix])

hpc, acc = np.array(hpc)*100, np.array(acc)*100

res = pg.mwu(hpc,acc)
stats = res.iloc[0]['U-val']
pval = res.iloc[0]['p-val']

plt.figure(figsize=(2.5,2.5), dpi=300)
h2 = plt.hist(acc, bins=np.linspace(0,101,21), edgecolor=AREA_CLR['ACC'], histtype='step', density=True, label='ACC', linewidth=LW)
h1 = plt.hist(hpc, bins=np.linspace(0,101,21), edgecolor=AREA_CLR['HPC'], histtype='step', density=True, label='HPC', linewidth=LW)
disp = np.max([np.max(h1[0]), np.max(h2[0])])*1.1
plt.scatter(np.median(hpc), disp, color=AREA_CLR['HPC'], marker='v', s=5)
plt.scatter(np.median(acc), disp, color=AREA_CLR['ACC'], marker='v', s=5)
plt.title(f'Mann-Whitney U, U={stats:.2f}, p={pval:.2e}', fontsize=6)
plt.xlabel('% shared variance')
plt.ylabel('Density')
plt.legend(frameon=False)
save_dir = os.path.join('plots', 'regional_difference')
finish_plot(filename='pct_shared_var_perRegion', save_dir=save_dir, savefig=SAVEFIG)

# Unused
for j in range(len(TASK_NAMES)):
    plt.figure(figsize=(2.5,2.5),dpi=300)
    dfs = get_single_task_df(df_all[(df_all['Patient']!='YEY')&(df_all['Patient']!='YEZ')], TASK_NAMES[j])
    hpc, acc = [], []
    for i in range(len(dfs)):
        s = dfs.iloc[i].session_class
        hpc_ix = s.chosen_chan[s.chosen_chan['region'] == 'HPC'].seq_index
        acc_ix = s.chosen_chan[s.chosen_chan['region'] == 'ACC'].seq_index
        hpc.extend(s.FA.shared_var_per_unit[hpc_ix])
        acc.extend(s.FA.shared_var_per_unit[acc_ix])

    res = pg.ttest(hpc,acc, paired=False, alternative='greater')
    stats = res.iloc[0]['T']
    pval = res.iloc[0]['p-val']
    dof = res.iloc[0]['dof']

    plt.hist(hpc, bins=np.linspace(0,1,21), color=AREA_CLR['HPC'], alpha=0.5, edgecolor='k', density=True, label='HPC')
    plt.hist(acc, bins=np.linspace(0,1,21), color=AREA_CLR['ACC'], alpha=0.5, edgecolor='k', density=True, label='ACC')
    plt.axvline(np.median(hpc), ls='--',c=AREA_CLR['HPC'])
    plt.axvline(np.median(acc), ls='--',c=AREA_CLR['ACC'])
    plt.title(f'{TASK_NAMES[j]}\nT({dof:.1f})={stats:.2f}, p={pval:.2e}', fontdict={'size': 6})
    plt.xlabel('Shared variance per unit')
    plt.ylabel('Density')
    plt.legend(frameon=False)
    plt.tight_layout()
    sns.despine(trim=True)
    save_dir = os.path.join('plots', 'regional_difference')
    finish_plot(filename=f'pct_shared_var_perRegion_{TASK_NAMES[j]}', save_dir=save_dir, savefig=SAVEFIG)
    plt.show()

#%% Regional difference - hpv sv density, resampled (Fig 4D)

def summarize_hpc_power(hpc_power):
    x = np.asarray(hpc_power, dtype=float)
    dip, pval = diptest(x)
    return {'n': len(x),'median': np.median(x),'mean': np.mean(x),'iqr': np.percentile(x, 75) - np.percentile(x, 25),
            'pct_center': np.mean((x >= 0.33) & (x <= 0.67)) * 100,'pct_edge': np.mean((x <= 0.2) | (x >= 0.8)) * 100,
            'dip': dip,'dip_p': pval,}

hpc_powers = collections.defaultdict(list)
for i in range(len(df_all)):
    s = df_all.iloc[i].session_class; print(s)
    results = FAResults()
    for j in range(10):
        print(f' - resample {j}')
        Y_norm = s.resample_y(random_state=j).get_spikes_mu_sd().Y_norm
        fa_subspace = results.fa_fit(Y_norm, shared_var_thresh=0.95, max_iter=int(1e4)).subspace
        total_var_per_unit_per_dim = fa_subspace**2 # (d, N)
        hpc_powers[s.task].extend(total_var_per_unit_per_dim[:, :s.Y.shape[1]//2].sum(axis=1)) # First half is HPC

hpc_power_stats = []
for j in range(len(TASK_NAMES)):
    task = TASK_NAMES[j]
    hpc_power = np.asarray(hpc_powers[task])
    stats = summarize_hpc_power(hpc_power)
    hpc_power_stats.append({'task': task, **stats})

    plt.figure(figsize=(2.5,2.75), dpi=300)
    h = plt.hist(hpc_power, bins=np.linspace(0,1,21), density=True, edgecolor='purple', histtype='step', linewidth=LW)
    plt.axvline(0.5, ls='--', c='k', lw=LW)
    plt.yticks(np.arange(0, max(1, np.ceil(np.max(h[0]))), dtype=int))
    plt.ylabel('Density')
    plt.xlabel('HPC contribution')
    plt.title(
        f'{task}\n'
        f'dip p={stats["dip_p"]:.2e}, med={stats["median"]:.2f}\n'
        f'center={stats["pct_center"]:.1f}%, edge={stats["pct_edge"]:.1f}%',
        fontsize=6
    )
    save_dir = os.path.join('plots', 'regional_difference')
    finish_plot(filename=f'hpc_power_dist_resampled_{task}', save_dir=save_dir, savefig=SAVEFIG)

hpc_power_stats = pd.DataFrame(hpc_power_stats)
print(hpc_power_stats)

#%% Representative unit - violin plot (Fig 4E, region)

fig_set(font_size=10, linewidth=LW)
fig, ax = plt.subplots(1,2,figsize=(3,2.5), dpi=300)
for j, (region, key, ix) in enumerate([('hpc', 'B-S1', 12), ('acc', 'W-S1', 11)]):
    real, shifted, pvals = collect_delta_svpu_per_region(session_key=key, region=region)

    violin = ax[j].violinplot(shifted[:,ix], positions=[0], widths=0.7,
                          showmeans=False, showmedians=False, showextrema=False)
    for body in violin['bodies']:
        body.set_facecolor('grey')
        body.set_edgecolor('grey')
        body.set_alpha(0.5)
    ax[j].scatter(0, real[ix], color=AREA_CLR[region.upper()], marker='o', s=15, zorder=3)
    ax[j].scatter(0, shifted[:,ix].mean(), color='k', marker='o', s=15, zorder=3)
    ax[j].set(xticks=[0], xticklabels=[f'Example {region.upper()} unit'], xlim=(-0.6,0.6),
           ylabel='∆ shared variance (%)')
finish_plot(filename='shifted_REPRESENTATIVE_VIOLIN', save_dir=SAVE_DIR, savefig=SAVEFIG)


#%% Histograms of the difference between real and shuffled ∆svpu (Fig S8D, sup-region)

beats = dict()
for task in TASK_NAMES:
    p_accs, p_hpcs = [], []
    real_accs, real_hpcs = [], []
    shifted_accs, shifted_hpcs = [], [] 
    for i in range(len(SESSION_KEYS)):
        key = SESSION_KEYS[i]
        if SESSIONS[key]['task'] == task:
            real_acc, shifted_acc, p_acc = collect_delta_svpu_per_region(session_key=key, region='acc')
            real_hpc, shifted_hpc, p_hpc = collect_delta_svpu_per_region(session_key=key, region='hpc')
            p_accs.extend(p_acc)
            p_hpcs.extend(p_hpc)
            real_accs.extend(real_acc)
            real_hpcs.extend(real_hpc)
            shifted_accs.extend(shifted_acc.mean(0))
            shifted_hpcs.extend(shifted_hpc.mean(0))

    p_accs = np.array(p_accs)
    p_hpcs = np.array(p_hpcs)
    real_accs = np.array(real_accs)
    real_hpcs = np.array(real_hpcs)
    shifted_accs = np.array(shifted_accs)
    shifted_hpcs = np.array(shifted_hpcs)

    hpc_beat = (p_hpcs < 0.05).sum()/len(p_hpcs)*100
    acc_beat = (p_accs < 0.05).sum()/len(p_accs)*100
    beats[task] = [hpc_beat, acc_beat]

    diff_hpc = real_hpcs-shifted_hpcs
    title = f'{task.capitalize()}\n'
    res = pg.ttest(diff_hpc, 0, alternative='greater').iloc[0]
    stats, pval, dof = res['T'], res['p-val'], res['dof']
    title += f'HPC, T({dof:.1f})={stats:.2f}, p={pval:.2e}\n'

    diff_acc = real_accs-shifted_accs
    res = pg.ttest(diff_acc, 0, alternative='greater').iloc[0]
    stats, pval, dof = res['T'], res['p-val'], res['dof']
    title += f'ACC, T({dof:.1f})={stats:.2f}, p={pval:.2e}'

    plt.figure(figsize=(2.5,2.5), dpi=300)
    h1 = plt.hist(diff_hpc, bins=np.arange(-40, 41, 2), histtype='step', color=AREA_CLR['HPC'], lw=1, label=f'HPC: {(p_hpcs < 0.05).sum()/len(p_hpcs)*100:.2f}%', density=True)
    h2 = plt.hist(diff_acc, bins=np.arange(-40, 41, 2), histtype='step', color=AREA_CLR['ACC'], lw=1, label=f'ACC: {(p_accs < 0.05).sum()/len(p_accs)*100:.2f}%', density=True)
    plt.scatter(np.mean(diff_hpc), max(np.concatenate((h1[0], h2[0]))) + 0.01, color=AREA_CLR['HPC'], marker='v', s=10)
    plt.scatter(np.mean(diff_acc), max(np.concatenate((h1[0], h2[0]))) + 0.01, color=AREA_CLR['ACC'], marker='v', s=10)
    plt.axvline(0, color='grey', ls='--', lw=1)
    plt.ylabel('Density')
    plt.xlabel('∆ shared variance (Real-Shuffled) (%)')
    plt.title(title)
    finish_plot(filename=f'{task}_shifted_paired_histogram', save_dir=SAVE_DIR, savefig=SAVEFIG)

#%% Percent units beating shuffled (Fig 4F, region)
fig_set(font_size=10)
fig, ax = plt.subplots(1,1,figsize=(2.5,2.5), dpi=300)
for j, (task, (hpc, acc)) in enumerate(beats.items()):
    bar_hpc = ax.bar(j-0.2, hpc, color=AREA_CLR['HPC'], width=0.4)
    bar_acc = ax.bar(j+0.2, acc, color=AREA_CLR['ACC'], width=0.4)
    
    ax.bar_label(bar_hpc, padding=2, fmt='%.1f', fontsize=8)
    ax.bar_label(bar_acc, padding=2, fmt='%.1f', fontsize=8)

plt.xticks([0,1,2], [t.capitalize() for t in TASK_NAMES])
plt.ylabel('Percentage (%)')

finish_plot(filename=f'summary_percent_units_beats_shuffle', save_dir=SAVE_DIR, savefig=SAVEFIG)
