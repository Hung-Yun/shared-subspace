#%% IMPORTS

import os
import numpy as np
import matplotlib.pyplot as plt
from utils import *
plt.figure()
plt.close()
fig_set()

#%% READ DATA

df_all = read_all_sessions(max_gap_dur_secs=1, bin_size=100, get_subspace=False, event_mode='all')
df_all = df_all[(df_all.session_class!='ignore') & ~(df_all.filename.isin(EXCLUDED_SESSIONS))].reset_index(drop=True)

for i in range(len(df_all)):
    s = df_all.iloc[i].session_class
    s.choose_channels().find_y().get_spikes_mu_sd()

#%% PURSUIT ANALYSIS

PURSUIT_RERUN = False
df_pursuit = get_single_task_df(df_all, task_name='pursuit').reset_index(drop=True)
df_spatial_tuning = get_spatial_tuning_df(df_pursuit, rerun=PURSUIT_RERUN)
df_pursuit_event, _ = get_pursuit_event_modulation_df(df_pursuit, rerun=PURSUIT_RERUN)

pursuit_tuning = df_spatial_tuning.merge(
    df_pursuit_event[['neuron', 'session_name', 'event_modulation']],
    on=['neuron', 'session_name'], how='outer', validate='one_to_one',
)
# Missing results mean untested, not a negative tuning result.
pursuit_tuning[['spatial_tuning', 'event_modulation']] = pursuit_tuning[['spatial_tuning', 'event_modulation']].astype('boolean')

#%% PODCAST ANALYSIS

PODCAST_RERUN = False
category_settings = dict(min_occurrences=20, min_words=5, n_permutations=10000, random_state=0, alpha=0.05, min_shift=100)
df_podcast = get_single_task_df(df_all, task_name='podcast').reset_index(drop=True)
df_category_tuning = get_semantic_category_df(df_podcast, rerun=PODCAST_RERUN, **category_settings)
df_podcast_event, _ = get_podcast_event_modulation_df(df_podcast, rerun=PODCAST_RERUN)

podcast_tuning = df_category_tuning.merge(
    df_podcast_event[['neuron', 'session_name', 'event_modulation']],
    on=['neuron', 'session_name'], how='outer', validate='one_to_one',
)
podcast_tuning[['category_tuning', 'event_modulation']] = podcast_tuning[['category_tuning', 'event_modulation']].astype('boolean')

#%% ROTATIONS ANALYSIS

ROTATIONS_RERUN = False
rotations_settings = dict(min_occurrences=10, min_cell_occurrences=5, n_permutations=10000, random_state=0, alpha=0.05)
df_rotations = get_single_task_df(df_all, task_name='rotations').reset_index(drop=True)
rotations_tuning = get_rotations_tuning_df(df_rotations, rerun=ROTATIONS_RERUN, **rotations_settings)

#%% TASK RESPONSIVENESS AND VENN DIAGRAMS

# Match by patient + neuron filename, as in tuning_old.py. This operational
# identity does not independently establish unit stability across recordings.
# Keep repeated-session unknowns unless some session establishes responsiveness.

task_tables = [('C', podcast_tuning, 'category_tuning'), ('S', pursuit_tuning, 'spatial_tuning'),
               ('R', rotations_tuning, 'task_specific_tuning')]
task_names = {'C': 'Podcast', 'S': 'Pursuit', 'R': 'Rotations'}
variable_labels = {'C': 'Category tuning', 'S': 'Spatial tuning', 'R': 'Task-specific tuning'}
session_lookup = df_all[['Patient', 'filename']].drop_duplicates().rename(columns={'filename': 'session_name'})
task_rows = []
for code, table, variable in task_tables:
    flags = table[['event_modulation', variable]].astype('boolean')
    task_rows.append(table[['neuron', 'session_name']].assign(task_code=code,
        event_only=flags.event_modulation, task_variable=flags[variable], task_responsive=flags.event_modulation | flags[variable]))
task_responsiveness = pd.concat(task_rows, ignore_index=True).merge(session_lookup, on='session_name', validate='many_to_one', how='left')
if task_responsiveness.Patient.isna().any():
    raise ValueError('Every tuning session must have a patient identifier.')
response_columns = ['event_only', 'task_variable', 'task_responsive']
task_responsiveness = task_responsiveness.groupby(['Patient', 'neuron', 'task_code'], as_index=False)[response_columns].agg(lambda v: v.any(skipna=False))
task_responsiveness[response_columns] = task_responsiveness[response_columns].astype('boolean')
response_matrices = {name: task_responsiveness.pivot(index=['Patient', 'neuron'], columns='task_code', values=column).reindex(columns=['C', 'S', 'R'])
                     for name, column in [('broad', 'task_responsive'), ('event_only', 'event_only')]}

# Task totals use all available patients; each Venn uses its own complete cohort.
summary_rows = []
for definition, column in [('broad', 'task_responsive'), ('event_only', 'event_only')]:
    for code, task in task_names.items():
        values = task_responsiveness.loc[task_responsiveness.task_code.eq(code), column]
        valid = values.dropna()
        summary_rows.append(dict(definition=definition, task=task, n_units=len(valid), n_unknown=int(values.isna().sum()),
            n_responsive=int(valid.sum()), n_not_responsive=int((~valid).sum()), pct_responsive=100 * valid.mean() if len(valid) else np.nan))
task_summary = pd.DataFrame(summary_rows)
print(task_summary.to_string(index=False))
#%%

def plot_task_venn(responses, labels, title, ax, colors=None, n_excluded=0, label_mode='percent'):
    """Percentage-area Venn for two/three sets of matched, assessed units.

    Rows must have a unique unit index and complete Boolean results. Return exact
    region counts, including neither/none. Three-circle areas are approximate;
    numeric counts are exact. Unknown units are excluded, never made negative.
    Region areas use the full assessed denominator, including neither/none.
    label_mode ('percent', 'count', 'both') changes text only. Give compared panels
    identical axis limits and sizes to preserve their common percentage scale.
    """
    from matplotlib_venn import venn2, venn3
    from matplotlib_venn.layout.venn2 import DefaultLayoutAlgorithm as Layout2
    from matplotlib_venn.layout.venn3 import DefaultLayoutAlgorithm as Layout3
    if responses.isna().any().any() or not responses.index.is_unique:
        raise ValueError('Venn rows must be unique units with no missing results.')
    n_sets = responses.shape[1]
    regions = {2: ['10', '01', '11', '00'], 3: ['100', '010', '110', '001', '101', '011', '111', '000']}[n_sets]
    membership = responses.astype(int).astype(str).agg(''.join, axis=1) if len(responses) else pd.Series(dtype=str)
    counts = membership.value_counts().reindex(regions, fill_value=0).rename('n_units')
    counts.index.name = 'region'
    if counts.iloc[:-1].sum():
        colors = colors or ['#4C78A8', '#F58518', '#54A24B'][:n_sets]
        fractions = counts.iloc[:-1] / len(responses)
        layout = (Layout2 if n_sets == 2 else Layout3)(normalize_to=fractions.sum())
        out = (venn2 if n_sets == 2 else venn3)(fractions.to_numpy(), set_labels=labels, set_colors=colors, ax=ax, layout_algorithm=layout)
        for region, fraction in fractions.items():
            label = out.get_label_by_id(region)
            if label is not None:
                label.set_text({'percent': f'{100 * fraction:.1f}%', 'count': str(counts[region]),
                                'both': f'{counts[region]}\n({100 * fraction:.1f}%)'}[label_mode])
        for label in list(out.set_labels or []) + list(out.subset_labels):
            if label is not None:
                label.set_fontsize(12)
        if n_sets == 2 and out.set_labels:
            for x, label in zip([.2, .8], out.set_labels):
                if label is not None:
                    label.set_transform(ax.transAxes)
                    label.set_position((x, 0.1))
                    label.set_ha('center')
                    label.set_va('top')
                    label.set_fontsize(12)
    else:
        ax.text(.5, .5, 'No assessed units' if responses.empty else 'No responsive units', ha='center', transform=ax.transAxes)
    outside = 'Neither' if n_sets == 2 else 'None'
    ax.text(.5, -.10 if labels is None else -.15, f'{outside}: {counts.iloc[-1]/len(responses)*100:.1f}%\nTotal: {len(responses)}',
            ha='center', transform=ax.transAxes, fontsize=12)
    ax.set_title(f'{title}', fontsize=12)
    ax.axis('off')
    return counts


VENN_LABEL_MODE = 'percent'  # 'count', 'percent', or 'both'; labels never change the areas.
fig, axes = plt.subplots(1, 3, figsize=(9, 5), dpi=300)
within_task_rows = []
component_rows = []
for ax, (code, task) in zip(axes, task_names.items()):
    available = task_responsiveness.loc[task_responsiveness.task_code.eq(code)].set_index(['Patient', 'neuron'])[['event_only', 'task_variable']]
    matched = available.dropna().astype(bool)
    excluded = len(available) - len(matched)
    counts = plot_task_venn(matched, None, task, ax, n_excluded=excluded, label_mode=VENN_LABEL_MODE)
    detected = available.eq(True).fillna(False)
    n_event, n_variable = int(detected.event_only.sum()), int(detected.task_variable.sum())
    n_both = int((detected.event_only & detected.task_variable).sum())
    component_rows.append(dict(task=task, n_units=len(available), n_event_assessed=int(available.event_only.count()),
        n_variable_classified=int(available.task_variable.count()), n_event_modulated=n_event,
        n_task_specific=n_variable, n_both=n_both, n_venn=len(matched), n_excluded=excluded))
    within_task_rows.append(counts.reset_index().assign(task=task, n_total=len(matched), n_excluded=excluded,
        pct_units=100 * counts.to_numpy() / len(matched) if len(matched) else np.nan))
# Prevent per-panel autoscaling from enlarging diagrams with smaller percentages.
xlim = (min(ax.get_xlim()[0] for ax in axes), max(ax.get_xlim()[1] for ax in axes))
ylim = (min(ax.get_ylim()[0] for ax in axes), max(ax.get_ylim()[1] for ax in axes))
for ax in axes:
    ax.set(xlim=xlim, ylim=ylim, aspect='equal')
finish_plot(filename='within_task_responsiveness_venn', save_dir='plots/tuning', savefig=SAVEFIG)
within_task_counts = pd.concat(within_task_rows, ignore_index=True)
component_summary = pd.DataFrame(component_rows)
print(component_summary.to_string(index=False))
#%%

# Pairwise analyses require only their two tasks. Both definition panels use
# the identical cohort; missing results in the third task do not exclude units.
comparisons = {'CS': ['C', 'S'], 'CR': ['C', 'R'], 'SR': ['S', 'R'], 'CSR': ['C', 'S', 'R']}
venn_rows = []
for comparison, codes in comparisons.items():
    complete = response_matrices['broad'][codes].notna().all(axis=1) & response_matrices['event_only'][codes].notna().all(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(6,3), dpi=200)
    for ax, (definition, matrix) in zip(axes, response_matrices.items()):
        matched = matrix.loc[complete, codes].astype(bool)
        labels = [f'{task_names[c]}' for c in codes]
        title = ''#'Broad task responsiveness' if definition == 'broad' else 'Event-only responsiveness'
        counts = plot_task_venn(matched, labels, title, ax, colors=[TASK_CLR[task_names[c].lower()] for c in codes], n_excluded=int((~complete).sum()), label_mode=VENN_LABEL_MODE)
        venn_rows.append(counts.reset_index().assign(definition=definition, comparison=comparison, n_total=len(matched),
            n_excluded=int((~complete).sum()), pct_units=100 * counts.to_numpy() / len(matched) if len(matched) else np.nan))
    xlim = (min(ax.get_xlim()[0] for ax in axes), max(ax.get_xlim()[1] for ax in axes))
    ylim = (min(ax.get_ylim()[0] for ax in axes), max(ax.get_ylim()[1] for ax in axes))
    for ax in axes:
        ax.set(xlim=xlim, ylim=ylim, aspect='equal')
    finish_plot(filename=f'task_responsiveness_venn_{comparison}', save_dir='plots/tuning', savefig=SAVEFIG)
venn_counts = pd.concat(venn_rows, ignore_index=True)

overlap_output_dir = Path('stats/task_responsiveness/category_rotations')
overlap_output_dir.mkdir(parents=True, exist_ok=True)
for name, table in [('task_summary', task_summary), ('component_summary', component_summary), ('venn_counts', venn_counts), ('within_task_counts', within_task_counts)]:
    table.to_csv(overlap_output_dir / f'{name}.csv', index=False)
pd.concat(response_matrices, axis=1).to_csv(overlap_output_dir / 'unit_task_responsiveness.csv')

# %%
