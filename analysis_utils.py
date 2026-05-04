
import os
import data_loader as io
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

MIN_FR = 0.5
SUBJS = 'ABCDFIJWXYZ'
SAVEFIG = False
VERBOSE = False
TASK_NAMES = ['podcast','pursuit','rotations']
TASK_KEY = dict(zip(TASK_NAMES, ['C','S','R']))
TASK_KEY_R = dict(zip(['C','S','R'], TASK_NAMES))
TASK_CLR = dict(zip(TASK_NAMES, ['#F24B69','#24A5F5','#FFBA36']))
TASK_SHAPE = dict(zip(TASK_NAMES, ['s','o','^']))
TASK_PAIRS = ['CS','CR','RS']
TASK_PAIRS_CLR = dict(zip(TASK_PAIRS, ["#66c5cc","#f6cf71","#f89c74"]))
SUBJ_CLR = dict(zip(SUBJS, [plt.cm.rainbow(i/len(SUBJS)) for i in range(len(SUBJS))]))
AREA_CLR = {'HPC': 'purple', 'ACC': 'green'}
MODEL_DIR = os.path.join('models')
EXCLUDED_SESSIONS = ['YFJ_0045_rotations', 'YEY_0071_rotations'] # Noisy sessions

LW = 0.8
def fig_set(font_size=8, linewidth=LW):
    sns.set(style="ticks", context="paper",
            font="sans-serif",
            rc={"font.size": font_size,
                "figure.titlesize": font_size,
                "figure.labelweight": font_size,
                "axes.titlesize": font_size,
                "axes.labelsize": font_size,
                "axes.linewidth": linewidth,
                "lines.linewidth": 1,
                "lines.markersize": 3,
                "xtick.labelsize": font_size,
                "ytick.labelsize": font_size,
                "savefig.transparent": True,
                "xtick.major.size": 2.5,
                "ytick.major.size": 2.5,
                "xtick.major.width": linewidth,
                "ytick.major.width": linewidth,
                "xtick.minor.size": 2,
                "ytick.minor.size": 2,
                "xtick.minor.width": linewidth,
                "ytick.minor.width": linewidth,
                'legend.fontsize': font_size,
                'legend.title_fontsize': font_size,
                'legend.frameon': False,
                })
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['backend'] = 'QtAgg'

def make_session_class(row, 
        max_gap_dur_secs: float = 1.0,
        bin_size: int = 50,
        brain_areas_for_analysis: tuple[str]|list[str] = ('HPC', 'ACC'),
        strict_brain_areas: bool = False,
        event_mode: str = 'events',
        min_fr=MIN_FR,
        get_subspace: bool = False):
    S = io.Session(row) # Modify initiation of Session

    if not S.fileexists: return 'ignore'
    try: # Read the spikes data
        S.get_spikes(binsize=bin_size)
    except ValueError as e:
        print(f'Skipping {S.session_key} get_spikes due to error: {e}')

    try: # Read the channel data
        S.get_channels(brain_areas_for_analysis, strict_brain_areas=strict_brain_areas)
    except ValueError as e:
        print(f'Skipping {S.session_key} get_channels due to error: {e}')

    # Sanity check
    assert S.spikes_per_bin.shape[1] == S.df_chan.shape[0], ValueError(f'Channel number mismatched: {S.spikes_per_bin.shape[1]}, {S.df_chan.shape[0]}')

    S.get_events(mode=event_mode, max_gap_dur_secs=max_gap_dur_secs)

    if get_subspace: # Do factor analysis
        S.choose_channels( # We first choose channels based on some criteria
            brain_areas=['HPC','ACC'],
            min_fr=min_fr, 
            must_have_cluster=True
        ).find_y().find_subspace()

    return S

def read_all_sessions(max_gap_dur_secs: float = 1.0,
                      bin_size: int = 50,
                      brain_areas_for_analysis: tuple[str]|list[str] = ('HPC', 'ACC'),
                      strict_brain_areas: bool = False,
                      event_mode: str = 'events',
                      min_fr=MIN_FR,
                      get_subspace: bool = False):
    
    df = io.load_sessions_csv() # The df is built on load_sessions_csv.
    
    df['session_class'] = df.apply(make_session_class, axis=1, args=(
        max_gap_dur_secs,
        bin_size, 
        brain_areas_for_analysis, 
        strict_brain_areas, 
        event_mode, 
        min_fr, 
        get_subspace)
    )
    df.reset_index(drop=True, inplace=True)
    return df

def save_plot(
    filename: str,
    save_dir: str,
    exts=('png', 'svg'),
    savefig=True,
    use_date_subfolder=False,
    include_time=False,
    extra_folder=None):
    """
    Save matplotlib plots into structured directories with optional date-based
    folders, experiment names, and auto-incremented versioning.
    """
    if not savefig:
        return

    # Base folder — with date or timestamp
    if use_date_subfolder:
        date_str = datetime.now().strftime('%Y-%m-%d' if not include_time else '%Y-%m-%d_%H-%M-%S')
        save_dir = os.path.join(save_dir, date_str)
    else:
        save_dir = save_dir

    # extra_folder
    if extra_folder:
        save_dir = os.path.join(save_dir, extra_folder)

    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.gcf()
    fig_patch_visible = fig.patch.get_visible()
    ax_patch_visible = [ax.patch.get_visible() for ax in fig.axes]

    # Prevent Matplotlib from exporting transparent background rectangles
    # as editable SVG objects in vector editors like Inkscape.
    fig.patch.set_visible(False)
    for ax in fig.axes:
        ax.patch.set_visible(False)

    # Save in all desired formats
    try:
        for ext in exts:
            path = os.path.join(save_dir, f"{filename}.{ext}")
            plt.savefig(path, bbox_inches='tight')
            print(f"Saved: {path}")
    finally:
        fig.patch.set_visible(fig_patch_visible)
        for ax, visible in zip(fig.axes, ax_patch_visible):
            ax.patch.set_visible(visible)


get_session_by_key = lambda df, key: df[df.session_key == key].session_class.values[0]


def get_single_task_df(df_all, task_name: str):

    if np.all(pd.isna(df_all.session_class)):
        print('Session class not read yet.')
        return

    print(f'Analyzing {task_name} task sessions...')
    df_single = df_all[(df_all.Task==task_name.capitalize()) & ~pd.isna(df_all.session_class)]
    j = []
    for i in range(len(df_single)):
        S = df_single.iloc[i]['session_class']
        if np.all(S.spikes_sd == 0):
            print(f'Skipping {S.session_key} due to zero spikes_sd')
        else:
            j.append(i)
    print(f'Found {len(df_single)} analyzable {task_name} sessions over {len(np.unique(df_single.Patient))} patients. Found spikes in {len(j)} sessions.')
    df_single = df_single.iloc[j]
    return df_single

def get_single_subj_df(df_all, patient_name: str, verbose=False):
    if len(patient_name) > 1:
        patient_name = patient_name[-1] # Only taking the last character
    if patient_name not in SUBJS:
        raise ValueError('Patient name not listed')

    df_single = df_all[(df_all.Patient.apply(lambda x: x[-1]) == patient_name) & ~pd.isna(df_all.session_class)]

    j = []
    for i in range(len(df_single)):
        S = df_single.iloc[i]['session_class']
        if np.all(S.spikes_sd == 0) and verbose:
            print(f'Skipping {S.session_key} due to zero spikes_sd')
        else:
            j.append(i)

    if len(df_single) == 0:
        if verbose:
            print(f'No recording for patient {patient_name}')
        return
    else:
        if verbose:
            print(f'Found {len(df_single)} analyzable sessions in patient {patient_name}. Found spikes in {len(j)} sessions.')
        return df_single
    
def get_paired_df(df_all, choose_channels: bool = True, brain_areas=['HPC','ACC']):

    df_pairs = []

    # make sure fields are valid.
    fields = ['Patient','Task','Datetime','filepath','session_class']
    if not all([f for f in fields if f in df_all.keys()]):
        raise ValueError('Fields should be in {list(df_all.keys())}')

    # First find df_single for each subj
    for subj in SUBJS:
        df_single = df_all[(df_all.Patient.apply(lambda x: x[-1]) == subj) & ~pd.isna(df_all.session_class)]

        for i in range(len(df_single)):
            for j in range(i, len(df_single)):
                task_1 = df_single.iloc[i][fields]
                task_2 = df_single.iloc[j][fields]
                if np.abs(task_1.Datetime-task_2.Datetime) > pd.Timedelta(hours=30):
                    continue
                if task_1.Task == task_2.Task:
                    continue

                # Return 'CS', 'RS', or 'CR' exclusively.
                task_pair = ''.join(sorted(list(map(lambda x: TASK_KEY[x.lower()], [task_1.Task,task_2.Task]))))

                pair = {'subj': subj, 
                        'task_pair': task_pair,
                        'task_1': task_1.session_class,
                        'task_2': task_2.session_class}
                
                df_pairs.append(pair)

    df_pairs = pd.DataFrame(df_pairs)

    # Find common channels in these tasks
    if choose_channels:
        commons, unions = [], []
        for i in range(len(df_pairs)):
            s1 = df_pairs.iloc[i].task_1
            s2 = df_pairs.iloc[i].task_2
            s1.choose_channels(brain_areas=brain_areas, min_fr=MIN_FR, must_have_cluster=True)
            s2.choose_channels(brain_areas=brain_areas, min_fr=MIN_FR, must_have_cluster=True)
            if s1.matfile_type == 'mat':
                common = list(set(s1.chosen_chan.filename) & set(s2.chosen_chan.filename))
                union  = list(set(s1.chosen_chan.filename) | set(s2.chosen_chan.filename))
            elif s1.matfile_type == 'hdf5':
                common = list(set(s1.chosen_chan.seq_num) & set(s2.chosen_chan.seq_num))
                union  = list(set(s1.chosen_chan.seq_num) | set(s2.chosen_chan.seq_num))

            commons.append(common)
            unions.append(union)

        df_pairs['common'] = commons
        df_pairs['union'] = unions
        df_pairs['n_common_union'] = list(zip(df_pairs.common.apply(len), df_pairs.union.apply(len)))
    else:
        for i in range(len(df_pairs)):
            s1 = df_pairs.iloc[i].task_1
            s2 = df_pairs.iloc[i].task_2
            s1.choose_channels(brain_areas=['HPC','ACC'], min_fr=None, must_have_cluster=False)
            s2.choose_channels(brain_areas=['HPC','ACC'], min_fr=None, must_have_cluster=False)

        df_pairs['common'] = {}
        df_pairs['union'] = {}
        df_pairs['n_common_union'] = {}
    return df_pairs

def make_gaussian_kernel(sigma_ms, fs):
    sigma_samples = sigma_ms / 1000 * fs
    window = int(4 * sigma_samples)
    t = np.arange(-window, window+1)
    kernel = np.exp(-0.5 * (t / sigma_samples)**2)
    kernel /= kernel.sum()  # normalize to preserve spike count
    return kernel

def smooth_spikes(spike_trains, fs=1000, sigma_ms=25):
    """
    spike_trains: ndarray (n_trials, n_samples), 0/1 spikes
    fs: sampling frequency, e.g., 1000 Hz
    sigma_ms: Gaussian kernel width
    """
    kernel = make_gaussian_kernel(sigma_ms, fs)
    
    # convolve each trial
    smoothed = convolve(spike_trains, kernel[None, :], mode='same')
    
    return smoothed

def get_on_off(s: io.Session, max_gap: float = 1.0):
    try: 
        y = io.ix_during_events(s.event_times, s.bin_times, max_gap_dur_secs=max_gap)
        return y
    except Exception as e:
        print(f'Error in {s}: {e}')
        return
