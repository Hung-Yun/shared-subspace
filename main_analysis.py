#%% imports

import os
import glob
import data_loader as io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import pingouin as pg
import scipy
import seaborn as sns
import collections
from decomposition import FAResults
import statsmodels.formula.api as sm

import warnings
warnings.filterwarnings('ignore')

from utils import *

plt.figure()
plt.close()
fig_set(font_size=10, linewidth=0.8)

try:
    df_all = pd.read_hdf(os.path.join(io.PREFIX, 'temp', 'df_all.h5'), 'all')
except FileNotFoundError:
    df_all = read_all_sessions(max_gap_dur_secs=1, bin_size=100, get_subspace=True)
    df_all = df_all[(df_all.session_class!='ignore') & ~(df_all.filename.isin(EXCLUDED_SESSIONS))].reset_index(drop=True)
    if not os.path.exists(os.path.join(io.PREFIX, 'temp')):
        os.makedirs(os.path.join(io.PREFIX, 'temp'))
    df_all.to_hdf(os.path.join(io.PREFIX,'temp','df_all.h5'), key='all')

BDI = {'W': 1, 'X': 2, 'Y': 17, 'Z': 8, 'A': 25, 'B': np.nan, 'C': 1, 'D': 37, 'F': np.nan, 'I': 10, 'J': 24}
df_all['BDI'] = df_all['Patient'].apply(lambda x: BDI[x[-1]])

#%% Time-resolved firing-rate example (Fig 2D-F, Fig S4)
plot_trfr_all(df_all, subject='YFD', unit_id='times_mRT2bHb02_2318', savefig=SAVEFIG)

#%% Plots the variance explained by each FA dimension, use subject A as example (Fig 3B, subspace)

df = get_single_subj_df(df_all, 'A')
s1 = df.iloc[0].session_class
s2 = df.iloc[1].session_class
s3 = df.iloc[2].session_class
s1.choose_channels(min_fr=MIN_FR, must_have_cluster=True)
s2.choose_channels(min_fr=MIN_FR, must_have_cluster=True)
s3.choose_channels(min_fr=MIN_FR, must_have_cluster=True)
common = list(set(s1.chosen_chan.filename) & 
              set(s2.chosen_chan.filename) & 
              set(s3.chosen_chan.filename))

plt.figure(figsize=(2.5,2.5), dpi=300)
for i in range(len(df)):
    s = df.iloc[i].session_class
    s.find_y(np.where(s.df_chan.filename.isin(common))[0]).find_subspace()
    data = np.cumsum(s.FA.explained_variance_ratio_)*100
    plt.scatter(range(len(data)), data, s=5, c=TASK_CLR[s.task])
    plt.plot(range(len(data)), data, lw=LW, ls='-', c=TASK_CLR[s.task], label=s.task)
    # vertical line where cumulative variance explained reaches 95%
    plt.axvline(np.where(data>=95)[0][0], ls='--', lw=LW, c=TASK_CLR[s.task])

plt.axhline(95, ls='--', lw=LW, c='k')
plt.ylabel('Cumulative shared\nvariance explained (%)')
plt.xlabel('Factor dimension')
plt.title('Subject A')
plt.xticks(np.arange(0,33,8))
plt.legend(frameon=False)
save_dir = os.path.join('plots', 'subspace')
finish_plot(filename='cumulative_var_explained', save_dir=save_dir, savefig=SAVEFIG)

#%% Plot shared dimensionality and number of units (Fig 3C-E, subspace)
for i in range(len(df_all)):
    s = df_all.iloc[i].session_class
    s.choose_channels().find_y().find_subspace()

n_units = df_all['session_class'].apply(lambda s: s.FA.n_components)
d_shared = df_all['session_class'].apply(lambda s: s.FA.d_shared)

res = scipy.stats.linregress(n_units, d_shared)
pearsonr = res.rvalue
n_sample = len(n_units)
x_range = np.array([np.min(n_units)-2, np.max(n_units)+2])

plt.figure(figsize=(2.5,2.5),dpi=300)
for pair, value in collections.Counter(list(zip(n_units, d_shared))).items():
    plt.scatter(*pair, s=(value-1)*15+5, ec='k',fc='none', lw=LW)

plt.plot(x_range, x_range*res.slope+res.intercept,ls='--', c='k', lw=LW)
plt.ylim([0,20])
plt.yticks(np.linspace(0,20,5))
plt.xticks(np.arange(10,60,10))
plt.ylabel('Shared dimensionality')
plt.xlabel('Number of units')
plt.title(f'Pearson R = {pearsonr:.3f}, N={n_sample}', fontsize=6)
save_dir = os.path.join('plots', 'subspace')
finish_plot(filename='d_shared_n_units_scatter', save_dir=save_dir, savefig=SAVEFIG)

dfs = df_all[['Patient','Task','session_class']]
dfs['d_shared'] = df_all['session_class'].apply(lambda s: s.FA.d_shared)
dfs['Task'] = dfs['Task'].str.lower()
result = (
    dfs.groupby('Patient')['d_shared']
      .mean()             # compute average per patient
      .reset_index()      # turn index back into a column
      .sort_values('d_shared', ascending=False)  # sort by average value
).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(2.5,2.5), dpi=300)
sns.stripplot(data=dfs, x='d_shared',y='Patient',hue='Task', orient='h',
              jitter=False, order=result['Patient'], edgecolor='k', linewidth=LW,
              s=5, palette=TASK_CLR, ax=ax)
ax.legend(frameon=False, loc='upper left', fontsize=10)
ax.set_xlim([0,20])
ax.set_xticks(np.linspace(0,20,6))
ax.set_xticklabels(np.linspace(0,20,6, dtype=int))
ax.set_xlabel('Shared dimensionality')
yticks = ax.get_yticks()
ax.set_yticks(yticks)
ax.set_yticklabels(result['Patient'].apply(lambda x: x[-1]))
save_dir = os.path.join('plots', 'subspace')
finish_plot(filename='d_shared_perPatient_sorted', save_dir=save_dir, savefig=SAVEFIG)

df = pd.DataFrame({'ratio': d_shared/n_units, 'task': df_all.Task.str.lower()})
res = scipy.stats.kruskal(
    df[df.task=='podcast'].ratio,
    df[df.task=='pursuit'].ratio,
    df[df.task=='rotations'].ratio)

fig,ax = plt.subplots(figsize=(2.5,2.5), dpi=300)
sns.barplot(data=df, x='task',y='ratio', capsize=0.1, ax=ax, palette=TASK_CLR, err_kws={'linewidth':LW})
plt.ylabel('Ratio of shared dim. to # units')
plt.title(f'KW test, stats: {res.statistic:.3f}, pval: {res.pvalue:.3f}', fontsize=6)
plt.xlabel('')
save_dir = os.path.join('plots', 'subspace')
finish_plot(filename='d_shared_n_units_ratio', save_dir=save_dir, savefig=SAVEFIG)

#%% Plot shared variance per unit - use subject A as example. Other examples are XZD (Fig 3F, subspace)

df = get_single_subj_df(df_all, 'A')
s1 = df.iloc[0].session_class
s2 = df.iloc[1].session_class
s3 = df.iloc[2].session_class
s1.choose_channels(min_fr=MIN_FR, must_have_cluster=True)
s2.choose_channels(min_fr=MIN_FR, must_have_cluster=True)
s3.choose_channels(min_fr=MIN_FR, must_have_cluster=True)
common = list(set(s1.chosen_chan.filename) & 
              set(s2.chosen_chan.filename) & 
              set(s3.chosen_chan.filename))

plt.figure(figsize=(2.5,2.5), dpi=300)
ix = None
for i in range(len(df)):
    s = df.iloc[i].session_class
    s.find_y(np.where(s.df_chan.filename.isin(common))[0]).find_subspace()
    data = s.FA.shared_var_per_unit*100
    if ix is None: ix = np.argsort(data)
    plt.scatter(range(len(data)), data[ix], s=5, c=TASK_CLR[s.task])
    plt.plot(range(len(data)), data[ix], lw=LW, ls='-', c=TASK_CLR[s.task], label=s.task)
plt.xlabel('Unit number')
plt.ylabel('% shared variance')
plt.title('Subject A')
plt.ylim([-5,100])
plt.xlim([-1,32])
plt.xticks(np.arange(0,33,8))
plt.legend(frameon=False)
save_dir = os.path.join('plots', 'subspace')
finish_plot(filename='pct_shared_var_YFA', save_dir=save_dir, savefig=SAVEFIG)

#%% Shared variance per unit, one data point for each patient (Fig 3G, subspace)

df_pairs = get_paired_df(df_all)
xs, ys = [], []
for i in range(len(df_pairs)):
    s1,s2 = df_pairs.iloc[i].task_1,df_pairs.iloc[i].task_2
    s1_common = s1.df_chan.filename.isin(df_pairs.iloc[i].common)
    s2_common = s2.df_chan.filename.isin(df_pairs.iloc[i].common)
    s1_id = s1.df_chan[s1_common].index.tolist()
    s2_id = s2.df_chan[s2_common].index.tolist()
    s1.find_y(s1_id).find_subspace(); s2.find_y(s2_id).find_subspace()

    x = np.mean(s1.FA.shared_var_per_unit) * 100
    y = np.mean(s2.FA.shared_var_per_unit) * 100
    xs.append(x); ys.append(y)

df_pairs['task1'] = xs
df_pairs['task2'] = ys

for j in range(len(TASK_PAIRS)):
    df_pair = df_pairs[df_pairs.task_pair == TASK_PAIRS[j]]
    res = pg.wilcoxon(df_pair.task1, df_pair.task2)
    wval = res.iloc[0]['W-val']
    pval = res.iloc[0]['p-val']

    df_res = pd.melt(df_pair, id_vars=['task_pair'], value_vars=['task1','task2'], value_name='pct_shared_var', var_name='task')
    task1, task2 = df_pair.task_1.iloc[0].task, df_pair.task_2.iloc[0].task
    d = {'task1': task1, 'task2': task2}
    df_res.task = df_res.task.apply(lambda row: d[row])

    fig, ax = plt.subplots(figsize=(2,2.5), dpi=300)
    sns.boxplot(data=df_res, y='pct_shared_var', x='task', fill=False, ax=ax,
                 linewidth=LW, legend=None, color='k', whis=(0, 100))
    sns.stripplot(data=df_res, y='pct_shared_var', x='task', hue='task',
                  dodge=False, s=5, edgecolor='k', linewidth=LW, palette=TASK_CLR, ax=ax)
    ax.set_ylabel('Avg % shared variance')
    ax.set_xlabel('')
    plt.ylim([0,25])
    plt.yticks(np.linspace(0,25,6, dtype=int))
    plt.title(f'{TASK_PAIRS[j]}, W={wval}, p={pval:.3e}', fontsize=6)
    save_dir = os.path.join('plots', 'subspace')
    finish_plot(filename=f'pct_shared_var_perPatient_{TASK_PAIRS[j]}', save_dir=save_dir, savefig=SAVEFIG)

#%% Time of session relative to first session (Fig S1, sup-sessions)

fig, ax = plt.subplots(figsize=(2.5,2.5), dpi=300)
for j in range(len(SUBJS)):
    df_sub = get_single_subj_df(df_all,SUBJS[j])
    times = df_sub['Datetime']
    times -= min(times)
    times = times.dt.total_seconds().to_numpy() / 3600 # To hour

    for i in range(len(times)):
        ax.scatter(times[i], j, c=TASK_CLR[df_sub.Task.str.lower().iloc[i]], marker='|', s=15)
ax.set_xlabel('Hours')
ax.set_ylabel('Patients')
ax.set_yticks(np.arange(len(SUBJS)))
ax.set_yticklabels(SUBJS)
ax.set_xticks(np.arange(0,81,20))
ax.axvline(30, ls='--', lw=LW, c='k')
save_dir = os.path.join('plots','general_plots')
finish_plot(filename=f'session_time', save_dir=save_dir, savefig=SAVEFIG)

#%% Raw ephys trace (Fig S2A, sup-raw-trace)

def plot_raw_ephys_with_spikes(session_key: str, channel_number: int, start_time_s=None,
                               duration_s: float = 0.5, ns5_path=None):
    """Plot an NS5 channel and the spike times assigned to it by ``io.Session``."""
    session = get_session_by_key(df_all, session_key)
    channel_rows = session.df_chan[session.df_chan.raw_num == channel_number]
    if len(channel_rows) != 1:
        available = sorted(session.df_chan.raw_num.astype(int).tolist())
        raise ValueError(f'Expected one raw channel {channel_number} in {session_key}; '
                         f'found {len(channel_rows)}. Available channels: {available}')

    channel_idx = channel_rows.index[0]
    channel_file = channel_rows.filename.iloc[0]
    electrode_name = os.path.splitext(channel_file)[0].removeprefix('times_').rsplit('_', 1)[0]
    ns5_channel_name = f'{electrode_name}-{channel_number:03d}'

    if ns5_path is None:
        emu = session.filename.split('_')[1]
        pattern = os.path.join(io.PREFIX, 'ns5', f'EMU-{emu}_subj-{session.subj}_*.ns5')
        ns5_matches = sorted(glob.glob(pattern))
        if len(ns5_matches) != 1:
            raise ValueError(f'Expected one NS5 file matching {pattern}; found {len(ns5_matches)}.')
        ns5_path = ns5_matches[0]

    recording = io.load_blackrock_recording(ns5_path)
    ns5_channel_id = io.get_channel_id(recording, ns5_channel_name)
    ns5_duration_s = recording.get_total_duration()
    session_duration_s = session.spikes.shape[1] / 1000
    duration_difference_ms = (ns5_duration_s - session_duration_s) * 1000
    if duration_s <= 0 or duration_s > ns5_duration_s:
        raise ValueError(f'duration_s must be between 0 and {ns5_duration_s:.3f} s.')

    spike_row = session.spikes.getrow(channel_idx)
    spike_times_s = np.repeat(spike_row.indices, spike_row.data.astype(int)) / 1000
    spike_times_s = spike_times_s[spike_times_s < ns5_duration_s]
    if len(spike_times_s) == 0:
        raise ValueError(f'No spikes found for channel {channel_number} in {session_key}.')

    if start_time_s is None:
        possible_starts = np.clip(spike_times_s - duration_s / 2, 0,
                                  ns5_duration_s - duration_s)
        spike_counts = (np.searchsorted(spike_times_s, possible_starts + duration_s)
                        - np.searchsorted(spike_times_s, possible_starts))
        start_time_s = possible_starts[np.argmax(spike_counts)]

    times_s, trace_uV, raw_fs = io.get_audio_timeseries(
        recording, channel_name=ns5_channel_name, start_time_s=start_time_s,
        duration_s=duration_s, demean=True)
    stop_time_s = times_s[-1] + 1 / raw_fs
    window_spikes_s = spike_times_s[(spike_times_s >= start_time_s)
                                    & (spike_times_s < stop_time_s)]

    fig, (ax_trace, ax_spikes) = plt.subplots(
        2, 1, figsize=(5, 2.5), dpi=300, sharex=True,
        gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.08})
    ax_trace.plot(times_s, trace_uV, color='k', lw=0.4)
    ax_trace.vlines(window_spikes_s, trace_uV.min(), trace_uV.max(),
                    color=TASK_CLR[session.task], lw=0.5, alpha=0.35)
    ax_spikes.eventplot(window_spikes_s, colors=TASK_CLR[session.task], lineoffsets=0,
                        linelengths=0.8, linewidths=0.8)
    ax_trace.set_ylabel(r'Raw voltage ($\mu$V)')
    ax_spikes.set(xlabel='Time from recording start (s)', ylabel='Spikes', yticks=[])
    ax_trace.set_title(f'{session_key}: io raw channel {channel_number} ({channel_file})\n'
                       f'NS5 channel {ns5_channel_name} (ID {ns5_channel_id}); '
                       f'duration difference = {duration_difference_ms:.1f} ms', fontsize=6)
    finish_plot(filename=f'raw_ephys_{session_key}_channel_{channel_number}_start_{start_time_s:.3f}_dur_{duration_s:.3f}',
                save_dir=os.path.join('plots', 'raw_ephys'), savefig=SAVEFIG, show=True)

    return {'session': session, 'channel_file': channel_file, 'channel_idx': channel_idx,
            'ns5_path': ns5_path, 'ns5_channel_name': ns5_channel_name,
            'ns5_channel_id': ns5_channel_id, 'times_s': times_s, 'trace_uV': trace_uV,
            'spike_times_s': window_spikes_s, 'sampling_frequency': raw_fs,
            'session_duration_s': session_duration_s, 'ns5_duration_s': ns5_duration_s,
            'duration_difference_ms': duration_difference_ms}

raw_ephys_example = plot_raw_ephys_with_spikes(
    session_key='B-S1', channel_number=11, start_time_s=116.5, duration_s=1)
raw_ephys_example = plot_raw_ephys_with_spikes(
    session_key='B-S1', channel_number=11, start_time_s=266.3, duration_s=1)

#%% From individual spike times to binned firing rate (Fig S2B, sup-raw-trace)

def plot_spikes_and_firing_rate(session_key: str, channel_number: int, start_time_s: float,
                                duration_s: float = 15):
    """Show spike times in temporal bins and the resulting firing rate."""
    session = get_session_by_key(df_all, session_key)
    channel_rows = session.df_chan[session.df_chan.raw_num == channel_number]
    if len(channel_rows) != 1:
        available = sorted(session.df_chan.raw_num.astype(int).tolist())
        raise ValueError(f'Expected one raw channel {channel_number} in {session_key}; '
                         f'found {len(channel_rows)}. Available channels: {available}')
    if duration_s <= 0:
        raise ValueError('duration_s must be positive.')

    channel_idx = channel_rows.index[0]
    start_ms, stop_ms = start_time_s * 1000, (start_time_s + duration_s) * 1000
    bin_indices = np.flatnonzero((session.bin_times >= start_ms) & (session.bin_times < stop_ms))
    if len(bin_indices) == 0:
        raise ValueError(f'No retained {session.bin_size} ms bins occur between '
                         f'{start_time_s:g} and {start_time_s + duration_s:g} s.')
    selected_bin_times = session.bin_times[bin_indices].astype(int)

    reconstructed_counts = np.array([
        session.spikes[channel_idx, t:t + session.bin_size].sum()
        for t in selected_bin_times])
    reconstructed_rates = reconstructed_counts * (1000 / session.bin_size)
    stored_rates = session.spikes_per_bin[bin_indices, channel_idx]
    max_error = float(np.max(np.abs(reconstructed_rates - stored_rates)))
    if not np.allclose(reconstructed_rates, stored_rates):
        raise AssertionError(f'Reconstructed and stored firing-rate matrices differ by up to '
                             f'{max_error:g} Hz.')

    spike_row = session.spikes.getrow(channel_idx)
    spike_times_ms = np.repeat(spike_row.indices, spike_row.data.astype(int))
    window_spikes_s = spike_times_ms[(spike_times_ms >= start_ms)
                                     & (spike_times_ms < stop_ms)] / 1000
    bin_centers_s = (selected_bin_times + session.bin_size / 2) / 1000

    fig, (ax_bins, ax_rate) = plt.subplots(
        2, 1, figsize=(5,2.5), dpi=300, sharex=True,
        gridspec_kw={'height_ratios': [1, 1.4], 'hspace': 0.18})
    for i, bin_start_ms in enumerate(selected_bin_times):
        bin_start_s, bin_stop_s = bin_start_ms / 1000, (bin_start_ms + session.bin_size) / 1000
        ax_bins.axvspan(bin_start_s, bin_stop_s, color='0.92' if i % 2 == 0 else '0.82')
    ax_bins.vlines(window_spikes_s, 0, 0.85, color=TASK_CLR[session.task], lw=0.7)
    ax_bins.set(xlim=(start_time_s, start_time_s + duration_s), ylim=(0, 1), yticks=[],
                ylabel='Spikes')
    ax_bins.set_title(f'{session_key}, channel {channel_number}: spikes in '
                      f'{session.bin_size} ms bins', fontsize=8)

    contiguous_runs = np.split(np.arange(len(selected_bin_times)),
                               np.flatnonzero(np.diff(selected_bin_times) != session.bin_size) + 1)
    for run in contiguous_runs:
        ax_rate.plot(bin_centers_s[run], reconstructed_rates[run],
                     color=TASK_CLR[session.task], lw=0.8)
    ax_rate.scatter(bin_centers_s, reconstructed_rates, color=TASK_CLR[session.task],
                    edgecolor='k', linewidth=LW, s=10, zorder=3)
    ax_rate.set(xlim=(start_time_s, start_time_s + duration_s),
                xlabel='Time from recording start (s)', ylabel='Firing rate (Hz)')


    finish_plot(filename=f'spikes_and_firing_rate_{session_key}_channel_{channel_number}',
                save_dir=os.path.join('plots', 'raw_ephys'), savefig=SAVEFIG, show=True)
    return {'session': session, 'channel_idx': channel_idx,
            'bin_indices': bin_indices, 'bin_times_ms': selected_bin_times,
            'reconstructed_counts': reconstructed_counts,
            'reconstructed_rates': reconstructed_rates, 'stored_rates': stored_rates,
            'max_reconstruction_error_hz': max_error}

spike_rate_example = plot_spikes_and_firing_rate(
    session_key='B-S1', channel_number=11, start_time_s=250, duration_s=10)

#%% Time-resolved firing-rate example, additional event alignments (Fig S4)

# Same example as Fig 2D-F: YFD_times_mRT2bHb02_2318_x3tasks_smooth_15ms.
plot_trfr_all(
    df_all, subject='YFD', unit_id='times_mRT2bHb02_2318', savefig=SAVEFIG)

#%% Firing rate cohen D of one neuron (Fig S6A, sup-unchanged-grand-fr)
df_pairs = get_paired_df(df_all)

# df_pairs iloc[1] and common i=1 for the example in the manuscript.
i=1 
row = df_pairs.iloc[1]
task1, task2, common = row.task_1, row.task_2, row.common   
ix1 = np.where(task1.df_chan.filename.isin(common[i:i+1]))[0]
ix2 = np.where(task2.df_chan.filename.isin(common[i:i+1]))[0]
Y1 = task1.find_y(ix1).Y
Y2 = task2.find_y(ix2).Y
region = task1.df_chan['region'].iloc[ix1].to_list()
task_pair = row.task_pair
row_ix = row.name
res = pg.ttest(Y1.flatten(), Y2.flatten(), paired=False).iloc[0]

plt.figure(figsize=(2.5,2.5), dpi=300)
plt.hist(Y1, bins=np.arange(0,101,10), histtype='step', color=TASK_CLR[task1.task], cumulative=False, density=True, linewidth=LW, label=task1.task)
plt.hist(Y2, bins=np.arange(0,101,10), histtype='step', color=TASK_CLR[task2.task], cumulative=False, density=True, linewidth=LW, label=task2.task)
plt.axvline(np.mean(Y1), color=TASK_CLR[task1.task], ls='--', lw=LW)
plt.axvline(np.mean(Y2), color=TASK_CLR[task2.task], ls='--', lw=LW)
plt.title(f"cohen={res['cohen-d']:.3f}")
plt.xlabel('Firing rate (Hz)')
plt.ylabel('Density')
plt.xticks(np.linspace(0,100,6))
plt.legend(frameon=False)
save_dir = os.path.join('plots', 'general_plots')
finish_plot(filename=f'cohend_example', save_dir=save_dir, savefig=SAVEFIG)

#%% Fig S6B-C - shared firing-rate statistics (computation)

df_firing_rates = []
df_pairs = get_paired_df(df_all)
for i in range(len(df_pairs)):
    row = df_pairs.iloc[i]
    task1, task2, common = row.task_1, row.task_2, row.common   
    ix1 = np.where(task1.df_chan.filename.isin(common))[0]
    ix2 = np.where(task2.df_chan.filename.isin(common))[0]
    Y1 = task1.find_y(ix1).Y
    Y2 = task2.find_y(ix2).Y
    region = task1.df_chan['region'].iloc[ix1].to_list()
    task_pair = row.task_pair
    row_ix = row.name

    Y1_m, Y2_m = np.mean(Y1, 0), np.mean(Y2, 0)
    ps = []
    ds = []
    for j in range(Y1.shape[1]):
        res = pg.ttest(Y1[:,j], Y2[:,j], paired=False).iloc[0]
        ps.append(res['p-val'])
        ds.append(res['cohen-d'])

    df_firing_rates.append(pd.DataFrame({'y1': Y1_m, 'y2': Y2_m, 'p': ps, 'cohen': ds, 'region': region, 'pairs': task_pair, 'ix': row_ix }))
df_firing_rates = pd.concat(df_firing_rates, ignore_index=True)

#%% Cohen D distribution across all pairs (Fig S6B, sup-unchanged-grand-fr)

fig, ax = plt.subplots(figsize=(2.5,2.5), dpi=300)
for i in range(3):
    df = df_firing_rates[df_firing_rates.pairs == TASK_PAIRS[i]]
    plt.hist(df.cohen, histtype='step', bins=np.arange(0,2,0.1), color=TASK_PAIRS_CLR[TASK_PAIRS[i]], density=True, label=TASK_PAIRS[i])
plt.xticks(np.arange(0,2.1,0.5))
plt.xlabel("Cohen's d")
plt.ylabel('Density')
plt.legend(frameon=False)
plt.axvline(0.5, lw=LW, ls='--', c='k')
save_dir = os.path.join('plots', 'general_plots')
finish_plot(filename=f'cohend_dist', save_dir=save_dir, savefig=SAVEFIG)

#%% Plot firing rates, separating brain regions (Fig S6B inset, sup-unchanged-grand-fr)

plt.figure(figsize=(1.5,1.25), dpi=300)
plt.bar([0,1], np.array([np.sum(df_firing_rates.cohen<0.5),np.sum(df_firing_rates.cohen>0.5)])/len(df_firing_rates)*100, facecolor='none', edgecolor='k', linewidth=LW)
plt.xticks([0,1], ['d<0.5', 'd>0.5'])
plt.yticks([0,30,60,90])
plt.ylabel('Percentage (%)')
save_dir = os.path.join('plots', 'general_plots')
finish_plot(filename=f'cohend_bar', save_dir=save_dir, savefig=SAVEFIG)

#%% Firing rate - population (Fig S6C, sup-unchanged-grand-fr)

for i in range(3):
    fig, ax = plt.subplots(1,1,figsize=(2.5,2.5), dpi=300)
    df = df_firing_rates[df_firing_rates.pairs == TASK_PAIRS[i]][['y1', 'y2']].reset_index(drop=True).melt(var_name='source', value_name='firing_rate', ignore_index=False)
    df['firing_rate'] = np.log10(df['firing_rate'])
    sns.boxplot(data=df, x='source', y='firing_rate', ax=ax, width=0.5, fill=False, color='k', whis=(5,95), showfliers=False)
    jitter_rng = np.random.default_rng(0)
    jitter_offsets = jitter_rng.uniform(-0.05, 0.05, df.index.nunique())
    for jitter, (_, pair) in zip(jitter_offsets, df.groupby(level=0, sort=False)):
        paired_rates = pair.set_index('source').loc[['y1', 'y2'], 'firing_rate']
        paired_x = np.array([0, 1]) + jitter
        ax.plot(paired_x, paired_rates, color=TASK_PAIRS_CLR[TASK_PAIRS[i]], linewidth=0.5, alpha=0.5, zorder=1)
        ax.scatter(paired_x, paired_rates, facecolors='none', edgecolors=TASK_PAIRS_CLR[TASK_PAIRS[i]], linewidths=1, s=5, zorder=2)
    ax.set(xticks=[0, 1], xlabel='', xlim=(-0.5, 1.5), 
           xticklabels=[TASK_KEY_R[TASK_PAIRS[i][0]].capitalize(), TASK_KEY_R[TASK_PAIRS[i][1]].capitalize()],
           ylabel='Log firing rate (Hz)')
    finish_plot(filename=f'firing_rate_between_tasks_{TASK_PAIRS[i]}', save_dir=save_dir, savefig=SAVEFIG)

#%% compare d_shared obtained by two different methods (computation for Fig S7)
# df_all = read_all_sessions(max_gap_dur_secs=1, bin_size=200, get_subspace=True)

# Restore all selected units after the matched-unit panels above.
for s in df_all.session_class:
    s.choose_channels().find_y()

SAVEDATA = True
filepath = os.path.join(MODEL_DIR, 'fa_thres_cv_regions_200ms.pkl')
data = {}
if os.path.exists(filepath):
    data = joblib.load(filepath)
else:   
    # Check the d_shared obtained by two different methods
    for i in range(len(df_all)):
        s = df_all.iloc[i].session_class; print(s)
        hpc_ix = s.chosen_chan[s.chosen_chan['region'] == 'HPC'].seq_index
        acc_ix = s.chosen_chan[s.chosen_chan['region'] == 'ACC'].seq_index
        
        Y_norm = s.find_y().get_spikes_mu_sd().Y_norm
        results = FAResults()
        print(' - FA through 95% threshold')
        fa_thres = results.fa_fit(Y_norm, shared_var_thresh=0.95, max_iter=int(1e5)).fa
        print(f' - FA for all channels through CV, {Y_norm.shape[1]} components')
        fa_cv = results.fa_fit(Y_norm, cv_components=np.arange(Y_norm.shape[1]), max_iter=int(1e5)).fa
        print(f' - FA for HPC channels, {len(hpc_ix)} components')
        fa_hpc = results.fa_fit(Y_norm[:, hpc_ix], cv_components=np.arange(len(hpc_ix)), max_iter=int(1e5)).fa

        if len(acc_ix) == 0: 
            print(' - No ACC channels')
            fa_acc = None
        else:
            print(f' - FA for ACC channels, {len(acc_ix)} components')
            fa_acc = results.fa_fit(Y_norm[:, acc_ix], cv_components=np.arange(len(acc_ix)), max_iter=int(1e5)).fa

        data[s.session_key] = {
            'fa_thres': fa_thres,
            'fa_cv': fa_cv,
            'fa_hpc': fa_hpc,
            'fa_acc': fa_acc}
    if SAVEDATA:
        joblib.dump(data, filepath)
    else:
        print('Data not saved.')

#%% Plot d-shared obtained from CV vs threshold (Fig S7, sup-dshared)

thres_component,cv_component = [], []
for i in range(len(df_all)):
        key = df_all.iloc[i].session_key
        cv_component.append(data[key]['fa_cv']['d_shared'])
        thres_component.append(data[key]['fa_thres']['d_shared'])   

plt.figure(figsize=(2.5,2.5),dpi=300)
plt.scatter(thres_component, cv_component, s=5, lw=LW, ec='k', fc='none')
plt.plot([0,30],[0,30], c='k', ls='--', lw=LW)
plt.xticks(np.arange(0,31,10))
plt.yticks(np.arange(0,31,10))
plt.xlabel('Shared dimensionality\nby 95% explained variance')
plt.ylabel('Shared dimensionality\nby cross-validation')
plt.legend(frameon=False)
save_dir = os.path.join('plots', 'cross_val')
finish_plot(filename='d_shared_twoway', save_dir=save_dir, savefig=SAVEFIG)

n_units, d_cv, d_thres = np.zeros((len(df_all),)), np.zeros((len(df_all),)),  np.zeros((len(df_all),))
for i in range(len(df_all)):
    s = df_all.iloc[i].session_class
    n_units[i] = s.Y.shape[1]
    d_cv[i] = data[s.session_key]['fa_cv']['d_shared']
    d_thres[i] = data[s.session_key]['fa_thres']['d_shared']

df = pd.DataFrame({'ratio': n_units/d_cv, 'task': df_all.Task.str.lower()})

fig,ax = plt.subplots(figsize=(2.5,2.5),dpi=300)
sns.barplot(data=df, x='task',y='ratio', capsize=0.1, ax=ax, palette=TASK_CLR, err_kws={'lw':0.5})
plt.ylabel('Ratio of # units to d-shared-cv')
plt.legend(frameon=False)
save_dir = os.path.join('plots', 'cross_val')
finish_plot(filename='d_shared_ratio', save_dir=save_dir, savefig=SAVEFIG)

plt.figure(figsize=(2.5,2.5),dpi=300)
for pair, value in collections.Counter(list(zip(n_units, d_cv))).items():
    plt.scatter(*pair, s=value*3, ec='k',fc='none', lw=LW)
res = scipy.stats.linregress(n_units, d_cv)
pearsonr = res.rvalue
n_sample = len(n_units)
x_range = np.array([np.min(n_units)-2, np.max(n_units)+2])
plt.plot(x_range, x_range*res.slope+res.intercept, ls='--', c='k', lw=LW, label=f'R={pearsonr:.3f}, N={n_sample}')
plt.xlabel('n_units')
plt.ylabel('d_cv')
plt.legend(frameon=False, fontsize=6)
save_dir = os.path.join('plots', 'cross_val')
finish_plot(filename='d_shared_cv', save_dir=save_dir, savefig=SAVEFIG)

plt.figure(figsize=(2.5,2.5),dpi=300)
for pair, value in collections.Counter(list(zip(n_units, d_thres))).items():
    plt.scatter(*pair, s=value*3, ec='k',fc='none', lw=LW)
res = scipy.stats.linregress(n_units, d_thres)
pearsonr = res.rvalue
n_sample = len(n_units)
x_range = np.array([np.min(n_units)-2, np.max(n_units)+2])
plt.plot(x_range, x_range*res.slope+res.intercept, ls='--', c='k', lw=LW, label=f'R={pearsonr:.3f}, N={n_sample}')
plt.xlabel('n_units')
plt.ylabel('d_thres')
plt.legend(frameon=False, fontsize=6)
save_dir = os.path.join('plots', 'cross_val')
finish_plot(filename='d_shared_thres', save_dir=save_dir, savefig=SAVEFIG)

#%% Plot factor loading of the first subspace dimension, AXZD finished all 3 tasks (Fig S9A, sup-firstdim)

for subj in 'AXZD':
    df = get_single_subj_df(df_all, subj)
    sets = [ set(row.session_class.chosen_chan.filename) for _, row in df.iterrows() ]
    st = set.intersection(*sets)
    plt.figure(figsize=(2.5,2.5), dpi=300)
    ix = None
    for i in range(len(df)):
        s = df.iloc[i].session_class
        s.find_y(np.where(s.df_chan.filename.isin(st))[0]).find_subspace()
        data = s.FA.subspace[0]
        if ix is None: ix = np.argsort(data)
        plt.scatter(range(len(data[ix])), data[ix], s=5, c=TASK_CLR[s.task])
        plt.plot(range(len(data[ix])), data[ix], lw=LW, ls='-', c=TASK_CLR[s.task], label=s.task)
    plt.xlabel('Unit number')
    plt.ylabel('Factor loading')
    plt.title(f'Subject {subj}')
    plt.legend(frameon=False)
    save_dir = os.path.join('plots', 'subspace')
    finish_plot(filename=f'first_dim_loading_subj_{subj}', save_dir=save_dir, savefig=SAVEFIG)

#%% Using subject Z as example, do subspace distance (Fig S9B, sup-firstdim)

def _sim(N, L, psi, rng):
    """
    Generate N samples from FA model:
        x = L z + eps
    """
    K = L.shape[0]
    z = rng.standard_normal((N, K))
    eps = rng.normal(scale=np.sqrt(psi), size=(N, L.shape[1]))
    return z @ L + eps

def calc_ang(v1, v2):
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2), -1, 1)))

def _get_subspace(y):
    result = FAResults()
    return result.fa_fit(io.normalize(y))

df = get_paired_df(get_single_subj_df(df_all, 'Z'))
row = df.iloc[1] # Z-C1 and Z-R1
task1, task2, common = row.task_1, row.task_2, row.common
task1.find_y(np.where(task1.df_chan.filename.isin(common))[0]).get_spikes_mu_sd().find_subspace()
task2.find_y(np.where(task2.df_chan.filename.isin(common))[0]).get_spikes_mu_sd().find_subspace()
Y1, Y2 = task1.Y_norm, task2.Y_norm

data = calc_ang(task1.FA.subspace[0], task2.FA.subspace[0])
rand_angles = []
for i in range(1000):
    L1, L2 = np.random.randn(*task1.FA.subspace.shape), np.random.randn(*task2.FA.subspace.shape)
    
    Y1_sim = _sim(Y1.shape[0], L1, 0, rng=np.random.default_rng(i))
    Y2_sim = _sim(Y2.shape[0], L2, 0, rng=np.random.default_rng(i))

    # Obtain L1_sim and L2_sim through FA and then compute angles
    L1_sim, L2_sim = _get_subspace(Y1_sim).subspace, _get_subspace(Y2_sim).subspace
    rand_angles.append(calc_ang(L1_sim[0], L2_sim[0]))
                                   
rng = np.random.default_rng(0)
shuf_angles = []
for it in range(1000):
    for Y in [Y1, Y2]:
        ix = rng.permutation(np.arange(Y.shape[0]))
        L1 = _get_subspace(Y[ix[len(ix)//2:]]).subspace
        L2 = _get_subspace(Y[ix[:len(ix)//2]]).subspace
        angs = calc_ang(L1[0], L2[0])
        shuf_angles.append(angs)

# Should use permutation test
test1 = pg.ttest(rand_angles, data).iloc[0]
test2 = pg.ttest(shuf_angles, data).iloc[0]

plt.figure(figsize=(2.5,2.5),dpi=300)
plt.axvline(data, ls='--', lw=LW, c='k', label=f'Data, {data:.2f}')
h1 = plt.hist(rand_angles,  bins=np.arange(0,181,4), color='green', linewidth=LW, density=True, label=f'Random, D={test1["cohen-d"]:.2f}')
h2 = plt.hist(shuf_angles, bins=np.arange(0,181,4), color='blue', linewidth=LW, density=True, label=f'Shuffle, D={test2["cohen-d"]:.2f}')
plt.scatter(np.mean(rand_angles),np.max(h1[0])+0.01,color='green', marker='v', s=3, label=f'{np.mean(rand_angles):.2f}, {np.std(rand_angles):.2f}')
plt.scatter(np.mean(shuf_angles),np.max(h2[0])+0.01,color='blue', marker='v', s=3, label=f'{np.mean(shuf_angles):.2f}, {np.std(shuf_angles):.2f}')
plt.xlim([-10,180])
plt.xticks(np.arange(0,181,45))
plt.xlabel('Angles of 1st subspace\ndimensions (deg)')
plt.ylabel('Density')
plt.legend(frameon=False)
save_dir = os.path.join('plots', 'subspace')
finish_plot(filename='prin_angle_example', save_dir=save_dir, savefig=SAVEFIG)

# ========================================================================
# Additional manuscript statistics
# ========================================================================

#%% Print number of units per session (stats)
df = df_all.apply(lambda row: len(row.session_class.chosen_chan), axis=1)
print(f'Mean: {df.mean():.2f}, SD: {df.std():.2f}, Min: {df.min()}, Max: {df.max()}')

#%% Stats of paired results (stats)
df_pairs = get_paired_df(df_all)
for j in range(len(df_pairs)):
    s1,s2 = df_pairs.iloc[j].task_1,df_pairs.iloc[j].task_2
    s1.choose_channels(); s2.choose_channels()
    s1.find_y(); s2.find_y()
    s1.find_subspace(); s2.find_subspace()
tests = ['mean_FR', 'shared_ratio','shared_var_per_unit']
test_fx = dict(zip(tests,(
    lambda s: s.spikes_mu,
    lambda s: s.FA.d_shared / s.FA.n_components,
    lambda s: s.FA.shared_var_per_unit
)))

stats_fx = dict(zip(tests,(
    lambda x,y: pg.ttest(x, y, paired=True, alternative='two-sided'),
    lambda x,y: pg.wilcoxon(x, y, alternative='two-sided', nan_policy='omit'),
    lambda x,y: pg.ttest(x, y, paired=True, alternative='two-sided')
)))

df_stats = []
for pair_name in TASK_PAIRS:

    df_pair = df_pairs[df_pairs.task_pair == pair_name]
    for test in tests:
        task_1_res = np.hstack(df_pair['task_1'].apply(test_fx[test]))
        task_2_res = np.hstack(df_pair['task_2'].apply(test_fx[test]))
        res = stats_fx[test](task_1_res, task_2_res)

        match res.index[0]:
            case 'Wilcoxon': metric = 'W-val'
            case 'T-test': metric = 'T'

        df_stats.append({'pair_name': pair_name,'test': test,'n_samples': np.min((len(task_1_res), len(task_2_res))),
                         'statistic': res[metric].iloc[0],'pvalue': res['p-val'].iloc[0]})
        
df_stats = pd.DataFrame(df_stats)
save_dir = os.path.join('stats', 'general_stats')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
df_stats.to_csv(os.path.join(save_dir, 'paired_stats.csv'), index=False)

#%% Check manuscript-reported channel and time-bin counts (stats)

paper_counts = []
for task_name in ['Podcast', 'Pursuit', 'Rotations']:
    df_task = get_single_task_df(df_all, task_name)
    for i in range(len(df_task)):
        s = df_task.iloc[i].session_class
        s.choose_channels(brain_areas=['HPC','ACC'], min_fr=MIN_FR, must_have_cluster=True)
        chan_col = 'filename' if 'filename' in s.chosen_chan.columns else 'seq_num'
        paper_counts.append({
            'patient': df_task.iloc[i].Patient,
            'task': task_name,
            'session_key': df_task.iloc[i].session_key,
            'n_channels': len(s.chosen_chan),
            'n_timebins': s.spikes_per_bin.shape[0],
            'n_events': len(s.event_times) if s.event_times is not None else 0,
            'channels': list(map(str, s.chosen_chan[chan_col].tolist())),
        })

paper_counts = pd.DataFrame(paper_counts)
print('\nPer-session counts')
print(paper_counts.drop(columns='channels').sort_values(['task','patient','session_key']).to_string(index=False))

print('\nTime bins per task')
print(paper_counts.groupby('task')['n_timebins'].agg(['count','mean','std']).round(3).to_string())

patient_channels = []
for patient, df_patient in paper_counts.groupby('patient'):
    patient_channels.append({
        'patient': patient,
        'n_channels': len(set(sum(df_patient.channels.tolist(), []))),
    })
patient_channels = pd.DataFrame(patient_channels).sort_values('patient')

print('\nChannels per patient')
print(patient_channels.to_string(index=False))
print(patient_channels['n_channels'].agg(['count','mean','std']).round(3).to_string())
