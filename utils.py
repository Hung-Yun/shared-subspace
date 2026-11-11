
import os
import glob
import json
from pathlib import Path
import data_loader as io
import pandas as pd
import numpy as np
import h5py
import scipy
from scipy.io import whosmat
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import wilcoxon
from scipy.sparse import csr_matrix
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.multitest import multipletests
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from decomposition import FAResults


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
MODEL_DIR = io.REPO_ROOT / 'models'
EXCLUDED_SESSIONS = ['YFJ_0045_rotations', 'YEY_0071_rotations'] # Noisy sessions

import warnings
warnings.filterwarnings("ignore")

#%% 

##########################
# == GENERAL PRINTING == #
##########################

def section(title: str):
    print("\n" + "=" * (len(title)+5))
    print(title)
    print("=" * (len(title)+5))

def announce(title):
    print("\n" + "-"  * (len(title)+5))
    print(title)
    print("-"  * (len(title)+5))

######################
# == READING DATA == #
######################

def make_session_class(row, 
        max_gap_dur_secs: float = 1.0,
        bin_size: int = 100,
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
                      bin_size: int = 100,
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


def regenerate_bin_times(session_path: str | Path, bin_size_ms: int = 100) -> np.ndarray:
    """Regenerate neural bin starts for sensory-input processing (Fig. S11–S12).

    These are the arrays saved in ``scripts/sensory_inputs/bin_times/`` and
    used by ``get_acoustic_energy.py`` to calculate one microphone RMS value
    per neural bin. They cover the full session, before on/off-task filtering.

    Parameters
    ----------
    session_path : str or Path
        Neural session MAT file containing ``spikes`` with neurons as rows and
        1 ms samples as columns, as in ``data/sessions/perChannel/<task>/``.
        Supports ordinary MAT files and MATLAB v7.3 sparse CSC storage.
    bin_size_ms : int, default 100
        Positive integer bin width in milliseconds. Sensory inputs use 100 ms.

    Returns
    -------
    numpy.ndarray
        One-dimensional int64 array of bin starts in milliseconds, beginning
        at zero. The final incomplete bin is discarded, matching
        ``Session.get_spikes()``: ``bin_size_ms * arange(n_samples // bin_size_ms)``.
        Only the spike-matrix dimensions are read; spike values are not loaded.
        Existing bin-time files are not read or overwritten.

    Examples
    --------
    Generate an array and explicitly save it for acoustic extraction:

    >>> session_path = Path('data/sessions/perChannel/podcast/YFB_0042_podcast.mat')
    >>> bin_times = regenerate_bin_times(session_path)
    >>> output_dir = Path('scripts/sensory_inputs/bin_times')
    >>> output_dir.mkdir(parents=True, exist_ok=True)
    >>> np.save(output_dir / f'{session_path.stem}_bin_times.npy', bin_times)
    """
    if (isinstance(bin_size_ms, (bool, np.bool_))
            or not isinstance(bin_size_ms, (int, np.integer))
            or bin_size_ms <= 0):
        raise ValueError('bin_size_ms must be a positive integer.')
    bin_size_ms = int(bin_size_ms)

    if h5py.is_hdf5(session_path):
        with h5py.File(session_path, 'r') as recording:
            # A CSC column-pointer array has one entry per time sample plus one.
            n_samples = recording['spikes']['jc'].shape[0] - 1
    else:
        shapes = {name: shape for name, shape, _ in whosmat(session_path)}
        if 'spikes' not in shapes or len(shapes['spikes']) != 2:
            raise ValueError(f'Expected a two-dimensional spikes matrix in {session_path}.')
        n_samples = shapes['spikes'][1]

    return bin_size_ms * np.arange(n_samples // bin_size_ms, dtype=np.int64)


def read_energy_files(s: io.Session):
    """Read acoustic and motion energy arrays and align them to a session."""
    if s.Y is None:
        s.choose_channels().find_y().get_spikes_mu_sd()

    acoustic_path = glob.glob(os.path.join(io.PREFIX, 'sensory_inputs', 'acoustic-energy', f'{s.filename}*'))
    motion_path = glob.glob(os.path.join(io.PREFIX, 'sensory_inputs', 'motion-energy', f'{s.filename}*'))

    if acoustic_path:
        s.acoustic_energy = np.load(acoustic_path[0]).flatten()
        if len(acoustic_path) > 1:
            print(f'{s.filename}: multiple acoustic-energy files; used the first one.')
    else:
        s.acoustic_energy = None
        print(f'{s.filename}: no acoustic energy.')

    if motion_path:
        motion_energy = pd.read_csv(motion_path[0])
        if len(motion_path) > 1:
            print(f'{s.filename}: multiple motion-energy files; used the first one.')
        downsample_factor = round(s.bin_size / (1000 / 30))
        motion_energy = motion_energy.groupby(
            np.arange(len(motion_energy)) // downsample_factor
        ).mean().reset_index(drop=True)
        s.motion_energy = motion_energy.values.flatten()
    else:
        s.motion_energy = None
        print(f'{s.filename}: no motion energy.')

    for field in ['acoustic_energy', 'motion_energy']:
        energy = getattr(s, field)
        if energy is None:
            continue
        n_missing = s.Y.shape[0] - len(energy)
        if n_missing > 0:
            energy = np.pad(energy, (0, n_missing), constant_values=np.nan)
        elif n_missing < 0:
            energy = energy[:s.Y.shape[0]]
        setattr(s, field, energy)

    s.has_acoustic_energy = s.acoustic_energy is not None
    s.has_motion_energy = s.motion_energy is not None
    return s

get_session_by_key = lambda df, key: df[df.session_key == key].session_class.values[0]
get_session_by_session_name = lambda df, name: df[df.filename.str.lower() == name.lower()].session_class.values[0]

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


#########################
# == PROCESSING DATA == #
#########################

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
    smoothed = np.convolve(spike_trains, kernel[None, :], mode='same')
    
    return smoothed

def get_on_off(s: io.Session, max_gap: float = 1.0):
    try: 
        y = io.ix_during_events(s.event_times, s.bin_times, max_gap_dur_secs=max_gap)
        return y
    except Exception as e:
        print(f'Error in {s}: {e}')
        return


##################
# == PLOTTING == #
##################

LW = 0.8

def _get_plot_modules():
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    return matplotlib, plt, sns


def fig_set(font_size=8, linewidth=LW):
    matplotlib, _, sns = _get_plot_modules()
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


def finish_plot(
    filename=None,
    save_dir=None,
    savefig=True,
    show=True,
    exts=('png', 'svg', 'pdf'),
):
    _, plt, sns = _get_plot_modules()
    sns.despine(trim=False)
    plt.tight_layout()

    if savefig and save_dir is not None and filename is not None:
        os.makedirs(save_dir, exist_ok=True)

        fig = plt.gcf()
        fig_patch_visible = fig.patch.get_visible()
        ax_patch_visible = [ax.patch.get_visible() for ax in fig.axes]

        fig.patch.set_visible(False)
        for ax in fig.axes:
            ax.patch.set_visible(False)

        try:
            for ext in exts:
                path = os.path.join(save_dir, f'{filename}.{ext}')
                plt.savefig(path, bbox_inches='tight')
                print(f'Saved: {path}')
        finally:
            fig.patch.set_visible(fig_patch_visible)
            for ax, visible in zip(fig.axes, ax_patch_visible):
                ax.patch.set_visible(visible)

    if show:
        plt.show()
    else:
        plt.close()



##########################
# == PRINCIPAL ANGLES == #
##########################


def principal_angles(A, B, return_vectors=False):
    """
    compute the principal angles (degrees) between subspaces A and B, in ascending order
    
    inputs:
        - A: (M, N) array_like
        - B: (M, K) array_like
    returns:
        - angles: ndarray, shape (min(N, K),)
        - if return_vectors=True: (angles, U, Vh), with rotations relative
          to Qa/Qb from np.linalg.qr(A/B). The principal directions in neuron
          coordinates are Qa @ U and Qb @ Vh.T. Use those QR bases when
          projecting data; even orthonormal inputs can change signs under QR.

    note: identical to np.rad2deg(scipy.linalg.subspace_angles(A,B)[::-1]) but faster
    """
    # Orthogonalize columns of A and B (QR decomposition)
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    
    # Compute the singular values of the overlap matrix
    Ua, S, Ub = np.linalg.svd(Qa.T @ Qb, full_matrices=False)
    
    # Compute the principal angles (in radians)
    angles = np.arccos(np.clip(S, -1, 1))  # Clip to avoid numerical issues
    angles = np.rad2deg(angles)

    # Return angles or angles plus rotations in the QR coordinate systems.
    return angles if not return_vectors else (angles, Ua, Ub)


###################################
# == TIME-RESOLVED FIRING RATE == #
###################################


def align_trials(task: io.Session, sp_full: np.ndarray, anchor: str,
                             pre_ms: int, post_ms: int, fs: int = 1000, sigma_ms: int = 25):

    def smooth_trials(spike_trials, fs, sigma_ms):
        """
        - spike_trials: (n_trials, n_samples) 0/1
        - gaussian_filter1d accepts sigma in samples apply along axis=1 (time) for each trial
        """
        sigma_samples = sigma_ms / 1000.0 * fs
        return gaussian_filter1d(spike_trials.astype(float), sigma=sigma_samples, axis=-1, mode='reflect')
    
    peri_trials = []

    if task.task == 'podcast':
        if type(task.events) != list:
            raise ValueError('Not a podcast experiment')
        if anchor not in ['onset', 'offset']:
            raise ValueError('Wrong anchor points for podcast')
        for ev in task.events:
            t = int(ev[anchor][0][0][0])
            start = t - pre_ms
            end = t + post_ms
            if start < 0 or end > len(sp_full):
                continue
            peri_trials.append(sp_full[start:end])

        peri_trials = np.array(peri_trials)
        result = smooth_trials(peri_trials, fs=fs, sigma_ms=sigma_ms)
    elif task.task == 'pursuit':
        if anchor not in task.events.columns:
            raise ValueError(f'Event {anchor} not found in task events')
        for k in range(len(task.events)):
            t = int(task.events.iloc[k][anchor])
            start = t - pre_ms
            end = t + post_ms
            if start < 0 or end > len(sp_full):
                continue
            peri_trials.append(sp_full[start:end])

        peri_trials = np.array(peri_trials)
        result = smooth_trials(peri_trials, fs=fs, sigma_ms=sigma_ms)
    elif task.task == 'rotations':
        if len(task.event_times) == 0:
            print('No event times found for this rotations task')
            return
        if anchor not in ['onset', 'offset']:
            raise ValueError('Wrong anchor points for rotations')
        full_smoothed = smooth_trials(sp_full, fs=fs, sigma_ms=sigma_ms)
        event_idx = 0 if anchor == 'onset' else 1
        for event_time in task.event_times:
            t = int(event_time[event_idx])
            start = t - pre_ms
            end = t + post_ms
            if start < 0 or end > len(full_smoothed):
                continue
            peri_trials.append(full_smoothed[start:end])

        result = np.array(peri_trials)
    else:
        raise ValueError('Unknown task type')

    if result is None or result.size == 0:
        return None

    return result

def stats_trft(smoothed_trials: np.ndarray, grand_mean: float,
                                   fs: int = 1000, alpha: float = 0.05,
                                   correction_method: str = 'fdr_bh'):
    if smoothed_trials.ndim != 2:
        raise ValueError('smoothed_trials must be 2D: (n_trials, n_time)')

    n_trials = smoothed_trials.shape[0]
    if n_trials == 0:
        raise ValueError('No valid event-aligned trials found')

    mean_trace = smoothed_trials.mean(axis=0)
    psth = mean_trace * fs
    sem = smoothed_trials.std(axis=0, ddof=1) / np.sqrt(n_trials) * fs

    with np.errstate(divide='ignore', invalid='ignore'):
        test = scipy.stats.ttest_1samp(
            smoothed_trials, popmean=grand_mean, axis=0, nan_policy='omit'
        )
    p_values = np.asarray(test.pvalue, dtype=float)
    t_values = np.asarray(test.statistic, dtype=float)

    # scipy returns NaN when a time bin is constant and equals the null mean.
    # Treat that no-difference case as explicitly non-significant.
    constant = np.all(smoothed_trials == smoothed_trials[:1], axis=0)
    constant_at_mean = constant & np.isclose(smoothed_trials[0], grand_mean)
    p_values[constant_at_mean] = 1.0
    t_values[constant_at_mean] = 0.0

    valid = np.isfinite(p_values)
    p_values_fdr = np.full(p_values.shape, np.nan)
    significant = np.zeros(p_values.shape, dtype=bool)
    if valid.any():
        reject, pvals_corr, _, _ = multipletests(p_values[valid], alpha=alpha, method=correction_method)
        p_values_fdr[valid] = pvals_corr
        significant[valid] = reject

    return {
        'n_trials': n_trials,
        'psth': psth,
        'sem': sem,
        'grand_mean_hz': grand_mean * fs,
        'p_values': p_values,
        'p_values_fdr': p_values_fdr,
        'significant': significant,
        't_values': t_values,
    }

def plot_trft(stats, pre_ms: int, post_ms: int, 
            label: str, ax=None, title=None, finish=True, *, task, neuron=None, savefig=SAVEFIG):
    """Plot an event-aligned trace using explicit session and optional unit context."""
    t_axis = np.arange(-pre_ms, post_ms)

    if ax is None:
        _, ax = plt.subplots(figsize=(2.5, 2.5), dpi=300)
    ax.axhline(stats['grand_mean_hz'], color='k', linestyle=':', linewidth=LW)# , label='Grand mean')
    ax.plot(t_axis, stats['psth'], color=TASK_CLR[task.task])# , label=label)
    ax.fill_between(t_axis, stats['psth'] - stats['sem'], stats['psth'] + stats['sem'],
                    alpha=0.3, color=TASK_CLR[task.task], linewidth=0)
    ax.axvline(0, color='k', linestyle='--', linewidth=LW)

    sig_mask = stats['significant']
    if np.any(sig_mask):
        ax.scatter(t_axis[sig_mask], np.repeat(stats['grand_mean_hz'], np.sum(sig_mask)), s=3, color='k', marker='s')# , label='FDR < 0.05')

    ax.set_xlim(-pre_ms, post_ms)
    ax.set_xlabel(f'Time to {label.lower()} (ms)')
    ax.set_ylabel('Firing rate (Hz)')
    ax.legend(frameon=False, fontsize=6)
    if title is not None:
        ax.set_title(title, fontsize=6)
    save_dir = os.path.join('plots', 'single_neuron_tuning')
    if finish:
        finish_plot(
            filename=f'{task.session_key}_{Path(neuron).stem}' if neuron else task.session_key,
            save_dir=save_dir,
            savefig=savefig,
            show=True,
        )



def plot_trfr_all(df_all, subject: str | None = None, unit_id: str | None = None,
                  *, savefig: bool = SAVEFIG, show: bool = True):
    """Plot the three-task firing-rate example used in Fig. 2D-F and Fig. S4.

    Inputs
    ------
    df_all : pandas.DataFrame
        Session table containing Patient, Task, and loaded session_class objects
        with full-resolution spikes and event times for all three tasks.
    subject : str or None
        Participant ID, e.g. 'YFD' or 'D'. None uses the original example
        participants (YFA, YFD, YEX, YEZ) present in df_all.
    unit_id : str or None
        Exact unit filename or stem, e.g. 'times_mRT2bHb02_2318.mat' or
        'times_mRT2bHb02_2318'. None plots every unit selected in all three tasks.
    savefig, show : bool
        Save PNG/SVG/PDF outputs and display the figures, respectively.

    Outputs
    -------
    list of matplotlib.figure.Figure
        One nine-panel figure per selected participant/unit, using the existing
        +/-200 ms windows, 15 ms Gaussian smoothing, and time-bin statistics.
        The first eligible session of each task is used, as in the original
        examples. An unavailable requested subject or shared unit raises an error.
    """

    ANCHOR_CANDIDATES = {
        'podcast': ['onset','offset'],
        'pursuit': ['wait_start', 'chase_start', 'chase_end', 'feedback_start', 'trial_end'],
        'rotations': ['onset','offset']}

    ANCHOR_PAIRS = [[key, value] for key, values in ANCHOR_CANDIDATES.items() for value in values]

    available_subjects = df_all.Patient.dropna().unique()
    if subject is None:
        pt_three_tasks = [patient for patient in ['YFA', 'YFD', 'YEX', 'YEZ']
                          if patient in available_subjects]
    else:
        requested_subject = subject.strip().upper()
        pt_three_tasks = [patient for patient in available_subjects
                          if patient == requested_subject or
                          (len(requested_subject) == 1 and patient.endswith(requested_subject))]
        if len(pt_three_tasks) != 1:
            raise ValueError(f'Expected one subject matching {subject!r}; found {pt_three_tasks}.')
    requested_unit = Path(unit_id).stem if unit_id is not None else None
    PRE_MS, POST_MS = 200,200
    SIGMA_MS = 15
    fs = 1000
    figures = []
    unit_found = False

    for patient in pt_three_tasks:
        one_pt = df_all.loc[df_all.Patient == patient]
        task_rows = {name: get_single_task_df(one_pt, name) for name in ANCHOR_CANDIDATES}
        if any(rows is None or rows.empty for rows in task_rows.values()):
            if subject is not None:
                raise ValueError(f'{patient} needs an eligible session for each of the three tasks.')
            continue
        tasks = {name: rows.iloc[0].session_class for name, rows in task_rows.items()}
        if any(len(task.event_times) == 0 for task in tasks.values()):
            if subject is not None:
                raise ValueError(f'{patient} needs event times for all three tasks.')
            continue

        for task in tasks.values():
            task.choose_channels(min_fr=MIN_FR, must_have_cluster=True)
        common = sorted(set.intersection(*(set(task.chosen_chan.filename) for task in tasks.values())))
        if requested_unit is not None:
            common = [candidate for candidate in common if Path(candidate).stem == requested_unit]
            unit_found = unit_found or bool(common)

        for candidate in common:
            file_name = f'{patient}_{os.path.splitext(candidate)[0]}_x3tasks_smooth_{SIGMA_MS}ms'
            dir_name = os.path.join('plots', 'single_neuron_tuning', 'across_three_tasks')
            if savefig and not show and os.path.exists(os.path.join(dir_name, f'{file_name}.png')):
                print(f'File {file_name}.png already exists. Skipping.')
                continue

            # The actual plotting
            fig, ax = plt.subplots(3,3, figsize=(7.5, 7.5), dpi=300, sharey=False)
            ax = ax.flatten()
            for ax_ix, (task_name, anchor) in enumerate(ANCHOR_PAIRS):
                task = tasks[task_name]
                ch = np.where(task.df_chan.filename == candidate)[0][0]
                sp_full = task.spikes[ch].toarray().squeeze()
                smoothed_trials = align_trials(task=task, sp_full=sp_full, anchor=anchor, pre_ms=PRE_MS, post_ms=POST_MS, fs=fs, sigma_ms=SIGMA_MS)
                stats = stats_trft(smoothed_trials=smoothed_trials, grand_mean=float(np.mean(sp_full)), fs=fs, alpha=0.05)
                plot_trft(stats, pre_ms=PRE_MS, post_ms=POST_MS, label=anchor, ax=ax[ax_ix], finish=False, task=task)
            plt.suptitle(f'Subject {patient}: {candidate}', fontsize=8, y=1)
            finish_plot(filename=file_name, save_dir=dir_name, savefig=savefig, show=show)
            figures.append(fig)

    if requested_unit is not None and not unit_found:
        raise ValueError(f'Unit {unit_id!r} is not selected in all three tasks for {pt_three_tasks}.')
    return figures


#######################
# == NEURAL TUNING == #
#######################


def _unwrap_matlab_scalar(value):
    """
    Unwrap nested MATLAB-style arrays until a scalar/string is reached.
    
    INPUTS:
        value: A potentially nested np.ndarray or scalar.

    OUTPUTS:
        The unwrapped scalar or string, or None if the input was an empty array.    
    """
    while isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.flat[0]
    return value


###########################
##### PURSUIT RELATED #####
###########################

PURSUIT_ARENA_BOUNDS = (0.0, 1872.0, 0.0, 1032.0)
PURSUIT_EVENTS = ('wait_start', 'chase_start', 'chase_end', 'feedback_start', 'trial_end')

def make_chericoni_glm_data(s, n_spatial_bins=6):
    """Recreate the frame-level design matrix used by MAIN_neural_fit.m."""
    if s.task != 'pursuit':
        raise ValueError('A pursuit sessionis required.')
    if s.trials is None or s.locations is None:
        print(f'Not processing {s} session due to lack of trials or locations.')
        return

    trials = io.pursuit_trials_to_dataframe(s)
    if len(trials) != len(s.locations):
        raise ValueError(f'{s.filename}: {len(trials)} trials but {len(s.locations)} location records.')

    if s.chosen_chan is None:
        channel_indices = np.arange(s.spikes.shape[0])
        channel_names = [f'unit_{i}' for i in channel_indices]
    else:
        channel_indices = s.chosen_chan.index.to_numpy(dtype=int)
        channel_names = s.chosen_chan['filename'].astype(str).tolist() if 'filename' in s.chosen_chan else [f'unit_{i}' for i in channel_indices]

    frame_tables = []
    spike_blocks = []

    for trial_index, trial in trials.iterrows():
        if int(trial['preys_num']) != 2 or float(trial['reward_val']) == 0:
            continue

        loc = s.locations[trial_index]
        time = np.asarray(loc['time'], dtype=float).ravel()
        self_x = np.asarray(loc['x'], dtype=float).ravel()
        self_y = np.asarray(loc['y'], dtype=float).ravel()
        prey_x = _as_time_by_agent(loc['x_prey'], len(time), 'x_prey')
        prey_y = _as_time_by_agent(loc['y_prey'], len(time), 'y_prey')
        if len(time) < 2 or prey_x.shape[1] != 2 or prey_y.shape[1] != 2:
            continue

        chosen_index = 0 if np.isclose(float(trial['reward_val']), float(trial['prey1_val'])) else 1
        unchosen_index = 1 - chosen_index

        frame_edges = np.ceil(time).astype(int)
        if np.any(np.diff(frame_edges) <= 0) or frame_edges[0] < 0 or frame_edges[-1] > s.spikes.shape[1]:
            raise ValueError(f'{s.filename} trial {trial_index}: invalid behavioral frame times.')

        raw_trial_spikes = s.spikes[channel_indices, frame_edges[0]:frame_edges[-1]].toarray()
        interval_starts = frame_edges[:-1] - frame_edges[0]
        interval_counts = np.add.reduceat(raw_trial_spikes, interval_starts, axis=1).T
        frame_counts = np.vstack([interval_counts[1:], np.zeros((1, len(channel_indices)), dtype=interval_counts.dtype)])

        frame_tables.append(pd.DataFrame({
            'trial_index': trial_index,
            'trial_num': int(trial['trial_num']),
            'time_ms': time[1:],
            'self_x': self_x[1:],
            'self_y': self_y[1:],
            'chosen_x': prey_x[1:, chosen_index],
            'chosen_y': prey_y[1:, chosen_index],
            'unchosen_x': prey_x[1:, unchosen_index],
            'unchosen_y': prey_y[1:, unchosen_index],
        }))
        spike_blocks.append(frame_counts)

    if not frame_tables:
        raise ValueError(f'{s.filename}: no rewarded two-prey trials with location data.')

    frames = pd.concat(frame_tables, ignore_index=True)
    spike_counts = np.vstack(spike_blocks).astype(int)
    design_matrices = {}
    bin_edges = {}

    for agent in ['self', 'chosen', 'unchosen']:
        x = frames[f'{agent}_x'].to_numpy()
        y = frames[f'{agent}_y'].to_numpy()
        x_edges = np.histogram_bin_edges(x, bins=n_spatial_bins)
        y_edges = np.histogram_bin_edges(y, bins=n_spatial_bins)
        x_bin = np.searchsorted(x_edges, x, side='right') - 1
        y_bin = np.searchsorted(y_edges, y, side='right') - 1
        x_bin = np.clip(x_bin, 0, n_spatial_bins - 1)
        y_bin = np.clip(y_bin, 0, n_spatial_bins - 1)
        location = y_bin * n_spatial_bins + x_bin
        design_matrices[agent] = np.eye(n_spatial_bins ** 2)[location]
        frames[f'{agent}_x_bin'] = x_bin
        frames[f'{agent}_y_bin'] = y_bin
        bin_edges[agent] = {'x': x_edges, 'y': y_edges}

    if len(frames) != len(spike_counts):
        raise RuntimeError('Behavioral frames and frame-level spike counts are not aligned.')

    return {'X': design_matrices, 'spike_counts': spike_counts, 'frames': frames, 'bin_edges': bin_edges, 'channel_indices': channel_indices, 'channel_names': channel_names, 'n_spatial_bins': n_spatial_bins}

def fit_chericoni_glm(glm_data, n_splits=10, p_threshold=0.05, regularization=10.0, random_state=0, neuron_indices=None, verbose=True):
    """Run Chericoni's smoothness-regularized forward GLM selection."""

    if glm_data is None: return {'selection': pd.DataFrame([]), 'fold_scores': pd.DataFrame([]), 'parameters': {}, 'model_masks': []}

    agents = ['self', 'chosen', 'unchosen']
    spike_counts = np.asarray(glm_data['spike_counts'])
    n_spatial_bins = int(glm_data['n_spatial_bins'])
    n_locations = n_spatial_bins ** 2
    model_masks = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    model_ids = {mask: i + 1 for i, mask in enumerate(model_masks)}
    neuron_indices = np.arange(spike_counts.shape[1]) if neuron_indices is None else np.asarray(neuron_indices, dtype=int)

    difference = np.diff(np.eye(n_spatial_bins), axis=0)
    one_dimensional_penalty = difference.T @ difference
    spatial_penalty = np.kron(np.eye(n_spatial_bins), one_dimensional_penalty) + np.kron(one_dimensional_penalty, np.eye(n_spatial_bins))

    def signed_rank_greater(first, second=None):
        first = np.asarray(first, dtype=float)
        difference_values = first if second is None else first - np.asarray(second, dtype=float)
        difference_values = difference_values[np.isfinite(difference_values)]
        if len(difference_values) == 0 or np.allclose(difference_values, 0):
            return 1.0
        return float(wilcoxon(difference_values, alternative='greater', zero_method='wilcox').pvalue)

    def evaluate_model(mask, response, splits):
        selected_agents = [agent for agent, include in zip(agents, mask) if include]
        design = np.hstack([glm_data['X'][agent] for agent in selected_agents])
        penalty = regularization * np.kron(np.eye(len(selected_agents)), spatial_penalty)
        fold_scores = np.full(len(splits), np.nan)
        fold_parameters = np.full((len(splits), design.shape[1]), np.nan)
        converged = np.zeros(len(splits), dtype=bool)
        initial = np.full(design.shape[1], 1e-3)

        for fold, (train_index, test_index) in enumerate(splits):
            train_design = design[train_index]
            train_response = response[train_index]

            def objective(parameters):
                linear_prediction = train_design @ parameters
                expected_counts = np.exp(linear_prediction)
                value = np.sum(expected_counts - train_response * linear_prediction) + 0.5 * parameters @ penalty @ parameters
                gradient = train_design.T @ (expected_counts - train_response) + penalty @ parameters
                return value, gradient

            optimization = minimize(objective, initial, method='L-BFGS-B', jac=True, options={'maxiter': 500})
            parameters = optimization.x
            initial = parameters
            converged[fold] = optimization.success
            fold_parameters[fold] = parameters

            test_response = response[test_index]
            test_spikes = test_response.sum()
            if test_spikes == 0:
                continue

            predicted_counts = np.exp(design[test_index] @ parameters)
            mean_test_count = test_response.mean()
            model_nll = np.sum(predicted_counts - test_response * np.log(predicted_counts) + gammaln(test_response + 1)) / test_spikes
            null_nll = np.sum(mean_test_count - test_response * np.log(mean_test_count) + gammaln(test_response + 1)) / test_spikes
            fold_scores[fold] = np.log(2) * (-model_nll + null_nll)

        return {'scores': fold_scores, 'parameters': np.nanmean(fold_parameters, axis=0), 'converged': converged}

    selection_rows = []
    fold_rows = []
    parameters = {}

    for neuron_index in neuron_indices:
        response = spike_counts[:, neuron_index]
        neuron_name = glm_data['channel_names'][neuron_index]
        n_positive = np.count_nonzero(response > 0)
        n_zero = np.count_nonzero(response == 0)
        actual_splits = min(n_splits, n_positive, n_zero)
        if verbose:
            print(f'Fitting neuron {neuron_index}: {neuron_name}')

        if response.sum() == 0 or actual_splits < 2:
            selection_rows.append({
                'neuron_index': neuron_index, 'neuron': neuron_name, 'selected_model': np.nan, 
                'predictors': None, 'SelfPos': False, 'ChosenPreyPos': False, 'UnchosenPreyPos': False, 
                'p_vs_null': np.nan, 'mean_llh_increase': np.nan, 'n_folds': actual_splits, 'n_spikes': int(response.sum())})
            continue

        splitter = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
        splits = list(splitter.split(np.zeros(len(response)), response))
        fitted = {}

        for mask in model_masks[:3]:
            fitted[mask] = evaluate_model(mask, response, splits)

        selected_mask = max(model_masks[:3], key=lambda mask: np.nanmean(fitted[mask]['scores']))

        for model_size in [2, 3]:
            candidates = [mask for mask in model_masks if sum(mask) == model_size and all(not selected_mask[i] or mask[i] for i in range(3))]
            for mask in candidates:
                fitted[mask] = evaluate_model(mask, response, splits)
            candidate_mask = max(candidates, key=lambda mask: np.nanmean(fitted[mask]['scores']))
            improvement_p = signed_rank_greater(fitted[candidate_mask]['scores'], fitted[selected_mask]['scores'])
            if improvement_p < p_threshold:
                selected_mask = candidate_mask
            else:
                break

        p_vs_null = signed_rank_greater(fitted[selected_mask]['scores'])
        significant_mask = selected_mask if p_vs_null < p_threshold else None
        predictors = None if significant_mask is None else tuple(agent for agent, include in zip(agents, significant_mask) if include)
        selected_flags = (0, 0, 0) if significant_mask is None else significant_mask
        selection_rows.append({
            'neuron_index': neuron_index, 'neuron': neuron_name, 
            'selected_model': np.nan if significant_mask is None else model_ids[significant_mask], 
            'predictors': predictors, 
            'SelfPos': bool(selected_flags[0]), 
            'ChosenPreyPos': bool(selected_flags[1]), 
            'UnchosenPreyPos': bool(selected_flags[2]), 
            'p_vs_null': p_vs_null, 
            'mean_llh_increase': np.nanmean(fitted[selected_mask]['scores']), 
            'n_folds': actual_splits, 'n_spikes': int(response.sum())})

        for mask, result in fitted.items():
            model_id = model_ids[mask]
            model_predictors = tuple(agent for agent, include in zip(agents, mask) if include)
            parameters[(neuron_index, model_id)] = result['parameters']
            for fold, score in enumerate(result['scores']):
                fold_rows.append({'neuron_index': neuron_index, 'neuron': neuron_name, 'model': model_id, 'predictors': model_predictors, 'fold': fold, 'llh_increase': score, 'converged': bool(result['converged'][fold])})

    return {'selection': pd.DataFrame(selection_rows), 'fold_scores': pd.DataFrame(fold_rows), 'parameters': parameters, 'model_masks': model_masks}

def make_event_modulation_data(s, event_times,
                               pre_window_ms=(-250, -50), post_window_ms=(0, 250)):
    """Collect paired pre/post firing rates for a table of event times.

    ``event_times`` has one row per occurrence or trial and one column per
    event. Rates have shape (occurrences, events, neurons). Windows outside the
    neural recording are left as NaN and excluded from statistical tests.
    """
    if s.spikes is None:
        raise RuntimeError('Call get_spikes() before making event modulation data.')
    if not isinstance(event_times, pd.DataFrame):
        raise TypeError('event_times must be a pandas DataFrame.')

    event_times = event_times.copy()
    event_names = tuple(event_times.columns)
    pre_window_ms = tuple(pre_window_ms)
    post_window_ms = tuple(post_window_ms)
    for name, window in [('pre_window_ms', pre_window_ms), ('post_window_ms', post_window_ms)]:
        if len(window) != 2 or window[0] >= window[1]:
            raise ValueError(f'{name} must contain increasing start and stop times.')
    if pre_window_ms[1] > 0 or post_window_ms[0] < 0:
        raise ValueError('Pre/post windows must remain on their respective sides of the event.')

    if s.chosen_chan is None:
        channel_indices = np.arange(s.spikes.shape[0])
        channel_names = [f'unit_{i}' for i in channel_indices]
    else:
        channel_indices = s.chosen_chan.index.to_numpy(dtype=int)
        channel_names = (
            s.chosen_chan['filename'].astype(str).tolist()
            if 'filename' in s.chosen_chan else [f'unit_{i}' for i in channel_indices]
        )

    event_time_values = event_times.to_numpy(dtype=float)
    n_occurrences, n_events = event_time_values.shape
    n_neurons = len(channel_indices)
    pre_rates = np.full((n_occurrences, n_events, n_neurons), np.nan)
    post_rates = np.full_like(pre_rates, np.nan)
    n_timepoints = s.spikes.shape[1]

    def window_bounds(window):
        starts = np.floor(event_time_values + window[0])
        stops = np.ceil(event_time_values + window[1])
        finite = np.isfinite(starts) & np.isfinite(stops)
        valid = (
            finite & (starts >= 0) & (stops <= n_timepoints) & (starts < stops)
        )
        starts = np.where(finite, starts, 0).astype(int)
        stops = np.where(finite, stops, 0).astype(int)
        return starts, stops, valid

    pre_starts, pre_stops, pre_valid = window_bounds(pre_window_ms)
    post_starts, post_stops, post_valid = window_bounds(post_window_ms)
    valid_occurrences = pre_valid & post_valid

    for neuron_position, channel_index in enumerate(channel_indices):
        spike_row = s.spikes.getrow(channel_index)
        spike_times = spike_row.indices
        cumulative_counts = np.r_[0, np.cumsum(spike_row.data)]

        pre_left = np.searchsorted(spike_times, pre_starts[valid_occurrences], side='left')
        pre_right = np.searchsorted(spike_times, pre_stops[valid_occurrences], side='left')
        post_left = np.searchsorted(spike_times, post_starts[valid_occurrences], side='left')
        post_right = np.searchsorted(spike_times, post_stops[valid_occurrences], side='left')

        pre_counts = cumulative_counts[pre_right] - cumulative_counts[pre_left]
        post_counts = cumulative_counts[post_right] - cumulative_counts[post_left]
        pre_rates[:, :, neuron_position][valid_occurrences] = (
            pre_counts * 1000.0
            / (pre_stops[valid_occurrences] - pre_starts[valid_occurrences])
        )
        post_rates[:, :, neuron_position][valid_occurrences] = (
            post_counts * 1000.0
            / (post_stops[valid_occurrences] - post_starts[valid_occurrences])
        )

    return {
        'pre_rates': pre_rates,
        'post_rates': post_rates,
        'event_times': event_times,
        'valid_occurrences': valid_occurrences,
        'event_names': event_names,
        'channel_indices': channel_indices,
        'channel_names': channel_names,
        'pre_window_ms': pre_window_ms,
        'post_window_ms': post_window_ms,
    }

def make_pursuit_event_data(s, event_names=PURSUIT_EVENTS,
                            pre_window_ms=(-250, -50), post_window_ms=(0, 250),
                            successful_only=True, two_prey_only=True):
    """Build event-modulation data for selected pursuit trials."""
    if s.task != 'pursuit':
        raise ValueError('A pursuit session is required.')
    if s.trials is None:
        raise ValueError(f'{s.filename} has no pursuit trials.')

    event_names = tuple(event_names)
    trials = io.pursuit_trials_to_dataframe(s)
    missing_events = set(event_names) - set(trials.columns)
    if missing_events:
        raise KeyError(f'Missing pursuit events: {sorted(missing_events)}')
    if successful_only:
        trials = trials[trials['successful']]
    if two_prey_only:
        trials = trials[trials['preys_num'] == 2]
    trials = trials.copy()

    event_data = make_event_modulation_data(
        s,
        trials.loc[:, event_names],
        pre_window_ms=pre_window_ms,
        post_window_ms=post_window_ms,
    )
    event_data['trials'] = trials
    return event_data

def fit_event_modulation(event_data, alpha=0.05,
                         correction_method='holm', min_occurrences=10):
    """Test paired pre/post-event rates with Wilcoxon signed-rank tests.

    Multiple-event correction is performed separately for each neuron. The
    returned selection table contains one binary event_modulation flag per
    neuron; event_results contains the event-level statistics.
    """
    pre_rates = np.asarray(event_data['pre_rates'], dtype=float)
    post_rates = np.asarray(event_data['post_rates'], dtype=float)
    event_names = tuple(event_data['event_names'])
    channel_names = list(event_data['channel_names'])

    expected_shape = (pre_rates.shape[0], len(event_names), len(channel_names))
    if pre_rates.shape != expected_shape or post_rates.shape != expected_shape:
        raise ValueError(f'Expected pre/post rate arrays with shape {expected_shape}.')

    def paired_wilcoxon(pre, post):
        valid = np.isfinite(pre) & np.isfinite(post)
        pre = pre[valid]
        post = post[valid]
        differences = post - pre
        if len(differences) < min_occurrences:
            return np.nan, np.nan, pre, post, differences
        if np.allclose(differences, 0):
            return 0.0, 1.0, pre, post, differences
        result = wilcoxon(differences, alternative='two-sided', zero_method='wilcox', method='auto')
        return float(result.statistic), float(result.pvalue), pre, post, differences

    event_rows = []
    selection_rows = []
    for neuron_index, neuron_name in enumerate(channel_names):
        neuron_rows = []
        for event_index, event_name in enumerate(event_names):
            statistic, p_value, pre, post, differences = paired_wilcoxon(
                pre_rates[:, event_index, neuron_index],
                post_rates[:, event_index, neuron_index],
            )
            neuron_rows.append({
                'neuron_index': neuron_index,
                'neuron': neuron_name,
                'event': event_name,
                'n_occurrences': len(differences),
                'n_nonzero_differences': int(np.count_nonzero(differences)),
                'pre_rate_hz': float(np.mean(pre)) if len(pre) else np.nan,
                'post_rate_hz': float(np.mean(post)) if len(post) else np.nan,
                'mean_delta_hz': float(np.mean(differences)) if len(differences) else np.nan,
                'median_delta_hz': float(np.median(differences)) if len(differences) else np.nan,
                'statistic': statistic,
                'p_value': p_value,
            })

        p_values = np.array([row['p_value'] for row in neuron_rows])
        valid = np.isfinite(p_values)
        p_values_corrected = np.full(len(neuron_rows), np.nan)
        significant = np.zeros(len(neuron_rows), dtype=bool)
        if valid.any():
            reject, corrected, _, _ = multipletests(
                p_values[valid], alpha=alpha, method=correction_method
            )
            p_values_corrected[valid] = corrected
            significant[valid] = reject

        for row, corrected, reject in zip(neuron_rows, p_values_corrected, significant):
            row['p_value_corrected'] = corrected
            row['significant'] = bool(reject)
        event_rows.extend(neuron_rows)

        significant_events = tuple(
            row['event'] for row in neuron_rows if row['significant']
        )
        selection_rows.append({
            'neuron_index': neuron_index,
            'neuron': neuron_name,
            'event_modulation': bool(significant_events),
            'significant_events': significant_events,
            'n_significant_events': len(significant_events),
            'min_p_value_corrected': (
                float(np.nanmin(p_values_corrected)) if valid.any() else np.nan
            ),
        })

    return {
        'selection': pd.DataFrame(selection_rows),
        'event_results': pd.DataFrame(event_rows),
        'parameters': {
            'alpha': alpha,
            'correction_method': correction_method,
            'min_occurrences': min_occurrences,
            'pre_window_ms': event_data['pre_window_ms'],
            'post_window_ms': event_data['post_window_ms'],
        },
    }

def _as_time_by_agent(values, n_time, name):
    """Return a location field as an (n_time, n_agents) float array."""
    values = np.asarray(values, dtype=float).squeeze()
    if values.ndim == 1:
        values = values[:, None]
    elif values.ndim != 2:
        raise ValueError(f'{name} must be one- or two-dimensional; got {values.shape}.')

    if values.shape[0] != n_time and values.shape[1] == n_time:
        values = values.T
    if values.shape[0] != n_time:
        raise ValueError(f'{name} has {values.shape[0]} samples but time has {n_time}.')
    return values

def _captured_prey_index(self_x, self_y, prey_x, prey_y):
    """Infer the captured prey as the prey nearest self at chase termination."""
    final_distance = np.hypot(prey_x[-1] - self_x[-1], prey_y[-1] - self_y[-1])
    if not np.isfinite(final_distance).any():
        raise ValueError('No finite self-to-prey distance at chase termination.')
    return int(np.nanargmin(final_distance))

def get_pursuit_spatial_samples(s, successful_only=True):
    """Align pursuit locations and firing rates at one row per neural bin.

    Returns
    -------
    samples : pandas.DataFrame
        DataFrame with one row per neural bin, containing trial number, time, and self/prey locations.
    firing_rates : numpy.ndarray
        Array of shape (n_bins, n_units) containing firing rates for each neural bin.
        firing_rates[i, j] corresponds to the firing rate of unit j during the neural bin represented by samples.iloc[i].
    """
    if s.task != 'pursuit':
        raise ValueError('Expected a pursuit Session.')
    if s.trials is None:
        raise ValueError(f'{s.filename} has no pursuit events.')
    if s.locations is None:
        raise ValueError(f'{s.filename} has no pursuit locations.')
    if s.bin_times is None or not hasattr(s, 'spikes_per_bin'):
        raise RuntimeError('Call get_spikes() and get_events() before spatial alignment.')

    s.trials_df = io.pursuit_trials_to_dataframe(s)
    if len(s.trials_df) != len(s.locations):
        raise ValueError(
            f'{s.filename}: {len(s.trials_df)} trials but {len(s.locations)} location records.'
        )

    sample_tables = []
    rate_blocks = []
    location_fields = set(s.locations.dtype.names or ())

    for trial_index, trial in s.trials_df.iterrows():
        if successful_only and not trial['successful']:
            continue

        loc = s.locations[trial_index]
        time = np.asarray(loc['time'], dtype=float).ravel()
        self_x = np.asarray(loc['x'], dtype=float).ravel()
        self_y = np.asarray(loc['y'], dtype=float).ravel()
        if not (len(time) == len(self_x) == len(self_y)):
            raise ValueError(f'{s.filename} trial {trial_index}: self trajectory length mismatch.')

        ## Prey locations may be missing or have multiple agents.
        prey_x = _as_time_by_agent(loc['x_prey'], len(time), 'x_prey')
        prey_y = _as_time_by_agent(loc['y_prey'], len(time), 'y_prey')
        if prey_x.shape != prey_y.shape:
            raise ValueError(f'{s.filename} trial {trial_index}: prey trajectory shape mismatch.')

        ## chosen or not
        chosen_index = _captured_prey_index(self_x, self_y, prey_x, prey_y)
        unchosen_index = 1 - chosen_index if prey_x.shape[1] == 2 else None

        ## turn time into bins
        neural_bin = np.searchsorted(s.bin_times, time, side='right') - 1

        # rejects timestamps before the first neural bin. Without this check, index -1 would accidentally refer to the final bin.
        valid = neural_bin >= 0

        # verifies that the timestamp actually falls before that bin’s end.
        # This matters when bin_times contains gaps after event filtering.
        valid[valid] &= (time[valid] < s.bin_times[neural_bin[valid]] + s.bin_size)
        if not valid.any():
            continue

        ## compile data for this trial
        frame_data = {
            'neural_bin': neural_bin[valid],
            'self_x': self_x[valid],
            'self_y': self_y[valid],
            'chosen_x': prey_x[valid, chosen_index],
            'chosen_y': prey_y[valid, chosen_index],
            'unchosen_x': (
                prey_x[valid, unchosen_index]
                if unchosen_index is not None else np.full(valid.sum(), np.nan)
            ),
            'unchosen_y': (
                prey_y[valid, unchosen_index]
                if unchosen_index is not None else np.full(valid.sum(), np.nan)
            ),
        }
        if {'x_pred', 'y_pred'}.issubset(location_fields):
            raw_pred_x = np.asarray(loc['x_pred'])
            raw_pred_y = np.asarray(loc['y_pred'])
            if raw_pred_x.size and raw_pred_y.size:
                pred_x = _as_time_by_agent(raw_pred_x, len(time), 'x_pred')[:, 0]
                pred_y = _as_time_by_agent(raw_pred_y, len(time), 'y_pred')[:, 0]
            else:
                pred_x = np.full(len(time), np.nan)
                pred_y = np.full(len(time), np.nan)
            frame_data['predator_x'] = pred_x[valid]
            frame_data['predator_y'] = pred_y[valid]

        ## each row is the average position of self/prey at each neural bin
        per_bin = pd.DataFrame(frame_data).groupby('neural_bin', sort=True).mean()
        bin_index = per_bin.index.to_numpy(dtype=int)
        per_bin.insert(0, 'time_ms', s.bin_times[bin_index] + s.bin_size / 2)
        per_bin.insert(0, 'trial_num', int(trial['trial_num']))
        per_bin.insert(0, 'trial_index', int(trial_index))
        per_bin['reward_val'] = float(trial['reward_val'])
        per_bin['preys_num'] = int(trial['preys_num'])
        per_bin['chosen_prey_index'] = chosen_index
        sample_tables.append(per_bin.reset_index())
        rate_blocks.append(s.spikes_per_bin[bin_index][:, s.chosen_chan.index.to_numpy(dtype=int)])

    if not sample_tables:
        n_units = len(s.chosen_chan) if s.chosen_chan is not None else s.spikes_per_bin.shape[1]
        return pd.DataFrame(), np.empty((0, n_units))

    samples = pd.concat(sample_tables, ignore_index=True)
    firing_rates = np.vstack(rate_blocks)
    if len(samples) != len(firing_rates):
        raise RuntimeError('Spatial samples and firing rates are not aligned.')

    return samples, firing_rates

def make_spatial_rate_maps(samples, firing_rates, x_col, y_col, bin_size_ms, n_spatial_bins=6, arena_bounds=PURSUIT_ARENA_BOUNDS, min_occupancy_s=0.0):
    """Average each neuron's firing rate within each spatial location.

    The returned sample_bins table has one row per row of samples and records
    its x/y spatial-bin assignment for later statistical tests.
    """
    if len(samples) != len(firing_rates):
        raise ValueError('samples and firing_rates must have the same number of rows.')
    if x_col not in samples or y_col not in samples:
        raise KeyError(f'Missing position columns: {x_col}, {y_col}.')

    x_min, x_max, y_min, y_max = arena_bounds
    x_edges = np.linspace(x_min, x_max, n_spatial_bins + 1)
    y_edges = np.linspace(y_min, y_max, n_spatial_bins + 1)

    sample_bins = samples[['trial_index', 'trial_num', 'neural_bin']].copy()
    sample_bins['x_bin'] = pd.cut(samples[x_col], bins=x_edges, labels=False, include_lowest=True)
    sample_bins['y_bin'] = pd.cut(samples[y_col], bins=y_edges, labels=False, include_lowest=True)
    sample_bins['valid_position'] = sample_bins[['x_bin', 'y_bin']].notna().all(axis=1)

    valid_bins = sample_bins[sample_bins['valid_position']].copy()
    valid_bins[['x_bin', 'y_bin']] = valid_bins[['x_bin', 'y_bin']].astype(int)

    sample_counts = np.zeros((n_spatial_bins, n_spatial_bins), dtype=int)
    counts = valid_bins.groupby(['y_bin', 'x_bin']).size()
    for (y_bin, x_bin), count in counts.items():
        sample_counts[y_bin, x_bin] = count

    occupancy_s = sample_counts * (bin_size_ms / 1000)
    support = (occupancy_s > 0) & (occupancy_s >= min_occupancy_s)

    rates_df = pd.DataFrame(firing_rates)
    rates_df['x_bin'] = sample_bins['x_bin'].to_numpy()
    rates_df['y_bin'] = sample_bins['y_bin'].to_numpy()
    mean_rates = rates_df[sample_bins['valid_position'].to_numpy()].groupby(['y_bin', 'x_bin']).mean()

    n_units = firing_rates.shape[1]
    rate_maps = np.full((n_units, n_spatial_bins, n_spatial_bins), np.nan, dtype=float)
    for (y_bin, x_bin), rates_at_location in mean_rates.iterrows():
        y_bin, x_bin = int(y_bin), int(x_bin)
        if support[y_bin, x_bin]:
            rate_maps[:, y_bin, x_bin] = rates_at_location.to_numpy()

    return {
        'rate_maps': rate_maps,
        'occupancy_s': occupancy_s,
        'sample_counts': sample_counts,
        'sample_bins': sample_bins,
        'support': support,
        'x_edges': x_edges,
        'y_edges': y_edges
    }

def example_spatial_map(df_pursuit):
    """Plot occupancy maps for the first supplied session with location data."""
    # Example session: skip YFC_0054, which has no location file.
    s = next(row.session_class for _, row in df_pursuit.iterrows() if row.session_class.locations is not None)
    s.trials_df = io.pursuit_trials_to_dataframe(s)
    spatial_samples, spatial_firing_rates = get_pursuit_spatial_samples(s)
    spatial_maps = {
        agent: make_spatial_rate_maps(spatial_samples, spatial_firing_rates, x_col=f'{agent}_x', y_col=f'{agent}_y', bin_size_ms=s.bin_size, n_spatial_bins=6)
        for agent in ['self', 'chosen', 'unchosen']
    }

    # Quick occupancy check before interpreting neuronal maps
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2), dpi=200, constrained_layout=True)
    for ax, agent in zip(axes, ['self', 'chosen', 'unchosen']):
        image = ax.imshow(
            spatial_maps[agent]['occupancy_s'], origin='lower', cmap='viridis',
            extent=PURSUIT_ARENA_BOUNDS, aspect='auto',
        )
        ax.set_title(f'{agent}: occupancy (s)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(image, ax=ax, shrink=0.75)
    plt.show()

    return spatial_maps, s

def example_spatial_tuning(spatial_maps, s):
    # Example maps for one unit
    unit_index = 0
    unit_label = s.chosen_chan.iloc[unit_index]['filename']
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2), dpi=200, constrained_layout=True)
    for ax, agent in zip(axes, ['self', 'chosen', 'unchosen']):
        image = ax.imshow(
            spatial_maps[agent]['rate_maps'][unit_index], origin='lower', cmap='magma',
            extent=PURSUIT_ARENA_BOUNDS, aspect='auto',
        )
        ax.set_title(agent)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(image, ax=ax, shrink=0.75, label='Hz')
    fig.suptitle(str(unit_label))
    plt.show()

def get_spatial_tuning_df(df_pursuit, rerun=False, output_dir='data/tuning/pursuit_tuning'):
    """Load or compute spatial selections, retaining predictor flags for summaries."""
    pursuit_output_dir = Path(output_dir)
    pursuit_output_dir.mkdir(parents=True, exist_ok=True)
    df = []
    for i in range(len(df_pursuit)):
        s = df_pursuit.iloc[i].session_class
        cache_path = pursuit_output_dir / f'{s.filename}_spatial_tuning.pkl'
        if cache_path.exists() and not rerun:
            glm_result = pd.read_pickle(cache_path)
            print(f'Loaded: {cache_path.name}')
        else:
            glm_data = make_chericoni_glm_data(s)
            if glm_data is None:  # No behavioral data available for this session.
                continue
            glm_result = fit_chericoni_glm(glm_data)
            glm_result['selection']['session_name'] = s.filename
            # Publish only a complete pickle, so an interrupted save is not reused.
            temporary_path = cache_path.with_suffix('.pkl.tmp')
            pd.to_pickle(glm_result, temporary_path)
            temporary_path.replace(cache_path)
            print(f'Saved: {cache_path.name}')
        df.append(glm_result['selection'])
    df = pd.concat(df, ignore_index=True)
    df = df[['neuron', 'session_name', 'selected_model', 'SelfPos', 'ChosenPreyPos', 'UnchosenPreyPos']].copy()
    df['spatial_tuning'] = ~df['selected_model'].isna()
    df = df.drop(columns=['selected_model'])
    return df

def summarize_spatial_tuning(df_spatial_tuning: pd.DataFrame, savefig=SAVEFIG):
    
    position_columns = ["SelfPos", "ChosenPreyPos", "UnchosenPreyPos"]
    counts = df_spatial_tuning[position_columns].sum()
    n_neurons = len(df_spatial_tuning)
    proportions = np.array((counts / n_neurons), dtype=float)

    summary = pd.DataFrame({
        "predictor": position_columns,
        "count": counts,
        "n_neurons": n_neurons,
        "percentage": 100 * proportions,
        "se_percentage": 100 * np.sqrt(proportions * (1 - proportions) / n_neurons),
    }).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(2.5,2.5), dpi=300)
    sns.barplot(data=summary, x="predictor", y="percentage", ax=ax)
    ax.errorbar(
        x=np.arange(len(summary)),
        y=summary["percentage"],
        yerr=summary["se_percentage"],
        fmt="none",
        color="black",
        capsize=4,
        linewidth=1,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Spatially tuned neurons (%)")
    ax.set_xticklabels(["Self", "Chosen prey", "Unchosen prey"])
    finish_plot(filename=f'spatial_tuning', save_dir='tuning', savefig=savefig)

def get_pursuit_event_modulation_df(df_pursuit, rerun=False, output_dir='data/tuning/pursuit_tuning'):
    """Load or compute Pursuit event selections and event-level statistics."""
    pursuit_output_dir = Path(output_dir)
    pursuit_output_dir.mkdir(parents=True, exist_ok=True)

    event_selections = []
    event_results = []
    for i in range(len(df_pursuit)):
        s = df_pursuit.iloc[i].session_class
        cache_path = pursuit_output_dir / f'{s.filename}_event_modulation.pkl'
        if cache_path.exists() and not rerun:
            event_result = pd.read_pickle(cache_path)
            print(f'Loaded: {cache_path.name}')
        else:
            event_data = make_pursuit_event_data(s)
            event_result = fit_event_modulation(event_data)
            event_result['selection']['session_name'] = s.filename
            event_result['event_results']['session_name'] = s.filename
            temporary_path = cache_path.with_suffix('.pkl.tmp')
            pd.to_pickle(event_result, temporary_path)
            temporary_path.replace(cache_path)
            print(f'Saved: {cache_path.name}')
        event_selections.append(event_result['selection'])
        event_results.append(event_result['event_results'])

    # event_selections is the one to use
    event_selections = pd.concat(event_selections, ignore_index=True)
    event_results = pd.concat(event_results, ignore_index=True)

    return event_selections, event_results



###########################
##### PODCAST RELATED #####
###########################


def get_podcast_event_modulation_df(df_podcast, rerun=False, output_dir='data/tuning/podcast_tuning'):
    """Load or compute Podcast event selections and event-level statistics.

    Use word onset and offset with the existing pre/post windows and paired
    Wilcoxon/Holm analysis from tuning_old.py. Return (selections, event_results),
    matching get_pursuit_event_modulation_df. Each table includes session_name.
    Existing session files are reused unless rerun=True.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selections, event_results = [], []
    for s in df_podcast.session_class:
        cache_path = output_dir / f'{s.filename}_event_modulation.pkl'
        if cache_path.exists() and not rerun:
            result = pd.read_pickle(cache_path)
            print(f'Loaded: {cache_path.name}')
        else:
            events = pd.DataFrame(s.event_times, columns=['word_onset', 'word_offset'])
            result = fit_event_modulation(make_event_modulation_data(s, events))
            result['selection']['session_name'] = s.filename
            result['event_results']['session_name'] = s.filename
            temporary = cache_path.with_suffix('.pkl.tmp')
            pd.to_pickle(result, temporary)
            temporary.replace(cache_path)
            print(f'Saved: {cache_path.name}')
        selections.append(result['selection'])
        event_results.append(result['event_results'])
    return pd.concat(selections, ignore_index=True), pd.concat(event_results, ignore_index=True)


# Supplied clusID assignments are authoritative. Do not modify or recode -100.
# The primary shuffle tests occurrence-level category association. Inspect the
# separate shift and lexical checks before interpreting this as semantic tuning.

WORD_CATEGORIES = {
    1: 'Body parts', 2: 'Places', 3: 'Emotional', 4: 'Mental',
    5: 'Social/people', 6: 'Objects', 7: 'Visual', 8: 'Numerical',
    9: 'Actions', 10: 'Identity (pronoun)', 11: 'Function words', 12: 'Proper nouns',
}


def make_category_occurrences(s):
    """One row per loaded word; raw count / 0.300 s in [onset+80, onset+380) ms.

    Keep the loader's YEX segment selection and original word/category labels.
    Invalid categories (including -100) and incomplete windows remain visible
    in the table, but are excluded from category tests. Word offset is metadata.
    """
    rows = []
    for i, event in enumerate(s.events):
        values = {key: _unwrap_matlab_scalar(event[key]) for key in ['text', 'onset', 'offset', 'clusID']}
        rows.append({'session': s.filename, 'event_index': i, 'word': values['text'],
                     'category_id': values['clusID'], 'onset_ms': values['onset'], 'offset_ms': values['offset']})
    df = pd.DataFrame(rows)
    # MATLAB-derived timing columns can retain object dtype in pandas.
    df[['onset_ms', 'offset_ms']] = df[['onset_ms', 'offset_ms']].apply(pd.to_numeric, errors='coerce').astype(float)
    df['category'] = df.category_id.map(WORD_CATEGORIES)
    df['response_start_ms'], df['response_stop_ms'] = df.onset_ms + 80, df.onset_ms + 380
    df['valid_window'] = (np.isfinite(df.onset_ms) & (df.response_start_ms >= 0)
                          & (df.response_stop_ms <= s.spikes.shape[1]))
    df['valid_category'] = df.category_id.isin(WORD_CATEGORIES) & df.word.notna()
    valid = df.valid_window.to_numpy()
    starts = np.ceil(df.loc[valid, 'response_start_ms']).astype(int).to_numpy()
    for channel_index, neuron in s.chosen_chan.filename.items():
        spikes = s.spikes.getrow(channel_index).tocsr(copy=True)
        spikes.sum_duplicates()
        spikes.sort_indices()
        cumulative = np.r_[0, np.cumsum(spikes.data)]
        counts = cumulative[np.searchsorted(spikes.indices, starts + 300)] - cumulative[np.searchsorted(spikes.indices, starts)]
        rates = np.full(len(df), np.nan)
        rates[valid] = counts / 0.300
        df[f'fr_hz_{neuron}'] = rates
    return df


def _category_contrasts(labels, moments, totals, categories):
    """Welch t and group means from exact integer spike-count sums/squares.

    moments contains [counts, counts**2]. Testing counts or Hz gives the same t.
    Both observed and null contrasts use this identical computation. Degenerate
    0/0 tests remain NaN; nonzero differences with zero variance give signed inf.
    """
    membership = csr_matrix((np.ones(len(labels), dtype=int), (labels, np.arange(len(labels)))), shape=(13, len(labels)))
    n = np.bincount(labels, minlength=13)[categories, None]
    m = len(labels) - n
    sums, squares = np.split((membership @ moments)[categories], 2, axis=1)
    total, total_squares = np.split(totals, 2)
    mean_target, mean_other = sums / n, (total - sums) / m
    with np.errstate(divide='ignore', invalid='ignore'):
        var_target = np.maximum(squares - sums**2 / n, 0) / (n - 1)
        var_other = np.maximum(total_squares - squares - (total - sums)**2 / m, 0) / (m - 1)
        t = (mean_target - mean_other) / np.sqrt(var_target / n + var_other / m)
    return t, mean_target / 0.300, mean_other / 0.300


def session_category_tuning(occurrences, min_occurrences=20, min_words=5, n_permutations=10000,
                            random_state=0, alpha=0.05, min_shift=100):
    """Category-versus-rest Welch permutations and within-neuron BH-FDR.

    The primary p_perm uses unrestricted occurrence-label permutations; it is
    an occurrence-association test whose validity requires exchangeability.
    p_shift separately enumerates circular label shifts of the valid occurrence
    sequence, excluding shifts < min_shift occurrences in either direction.
    Shifts preserve label order/counts and fixed responses, but assume suitable
    temporal stationarity; they are a sensitivity check, not an automatic fix.
    Enumerating unique shifts avoids pretending 10,000 repeats improve resolution.

    Reuse each null assignment across neurons sharing valid rows. Eligible
    targets need min_occurrences and min_words; other groups include ALL other
    classified valid occurrences. Return category results, neuron summaries,
    and exclusions. An untestable neuron has unknown, not negative, tuning.
    Leave-one-target-word-out effect ranges are descriptive, without new tests.
    They do not establish generalization to held-out word identities.
    """
    if occurrences.session.nunique() != 1 or occurrences.session.isna().any():
        raise ValueError('Provide occurrences from exactly one session.')
    if min_shift < 1 or n_permutations < 1:
        raise ValueError('Use positive min_shift and n_permutations.')
    session = occurrences.session.iloc[0]
    columns = [c for c in occurrences if c.startswith('fr_hz_')]
    rates = occurrences[columns].to_numpy(float)
    usable = (occurrences.valid_category & occurrences.valid_window).to_numpy()
    masks = np.isfinite(rates) & usable[:, None]
    groups = {}
    for j in range(len(columns)):
        groups.setdefault(masks[:, j].tobytes(), []).append(j)
    result_rows, neuron_rows, audit_rows = [], [], []
    for indices in groups.values():
        keep = masks[:, indices[0]]
        data = occurrences.loc[keep].reset_index(drop=True)
        labels, words = data.category_id.to_numpy(int), data.word.to_numpy()
        group_rates = rates[np.ix_(keep, indices)]
        # Responses come from raw integer counts in the same 300-ms window.
        counts = np.rint(group_rates * 0.300).astype(np.int64)
        moments = np.column_stack([counts, counts**2])
        totals = moments.sum(axis=0)
        sizes = data.groupby('category_id').word.agg(['size', 'nunique']).reindex(WORD_CATEGORIES, fill_value=0)
        eligible = (sizes['size'] >= min_occurrences) & (sizes['nunique'] >= min_words) & (len(data) - sizes['size'] >= 2)
        categories = sizes.index[eligible].to_numpy(int)
        # A roll by k and by N-k must each move at least min_shift occurrences.
        shifts = np.arange(min_shift, len(labels) - min_shift + 1)
        t_true, mean_target, mean_other = _category_contrasts(labels, moments, totals, categories)
        p_values = {}
        for null in ['perm', 'shift']:
            rng = np.random.default_rng(random_state)
            exceed = np.zeros_like(t_true, dtype=int)
            undefined = np.isnan(t_true)
            n_null = n_permutations if null == 'perm' else len(shifts)
            for k in range(n_null if len(categories) else 0):
                shuffled = rng.permutation(labels) if null == 'perm' else np.roll(labels, shifts[k])
                t_null, _, _ = _category_contrasts(shuffled, moments, totals, categories)
                exceed += np.abs(t_null) >= np.abs(t_true)
                undefined |= np.isnan(t_null)
            p = (1 + exceed) / (n_null + 1)
            p[undefined | (n_null == 0)] = np.nan
            p_values[null] = p

        for j, column_index in enumerate(indices):
            neuron = columns[column_index].removeprefix('fr_hz_')
            q_values = {}
            for null, p in p_values.items():
                q = np.full(len(categories), np.nan)
                finite = np.isfinite(p[:, j])
                if finite.any():
                    q[finite] = multipletests(p[finite, j], alpha=alpha, method='fdr_bh')[1]
                q_values[null] = q
            for k, category in enumerate(categories):
                target = labels == category
                # Remove every occurrence of each target word, one word at a time.
                lexical = pd.DataFrame({'word': words[target], 'rate': group_rates[target, j]}).groupby('word').rate.agg(['sum', 'size'])
                remaining = (group_rates[target, j].sum() - lexical['sum']) / (target.sum() - lexical['size'])
                loo_delta = remaining - mean_other[k, j]
                result_rows.append({
                    'session': session, 'neuron': neuron, 'category_id': category, 'category': WORD_CATEGORIES[category],
                    'n_target': int(target.sum()), 'n_other': int((~target).sum()), 'n_words': int(sizes.loc[category, 'nunique']),
                    'mean_target': mean_target[k, j], 'mean_other': mean_other[k, j], 'delta_hz': mean_target[k, j] - mean_other[k, j],
                    't_true': t_true[k, j], 'p_perm': p_values['perm'][k, j], 'q_value': q_values['perm'][k],
                    'significant_fdr': bool(q_values['perm'][k] < alpha),
                    'p_shift': p_values['shift'][k, j], 'q_shift': q_values['shift'][k], 'significant_shift': bool(q_values['shift'][k] < alpha),
                    'loo_min_delta_hz': loo_delta.min(), 'loo_max_delta_hz': loo_delta.max(),
                    'loo_same_direction': bool((np.sign(loo_delta) == np.sign(mean_target[k, j] - mean_other[k, j])).all()),
                    'n_permutations': n_permutations, 'n_shifts': len(shifts),
                })
            tested = int(np.isfinite(q_values['perm']).sum())
            tuned = bool((q_values['perm'] < alpha).any()) if tested else pd.NA
            shift_tested = int(np.isfinite(q_values['shift']).sum())
            neuron_rows.append({
                'session': session, 'neuron': neuron, 'n_categories_eligible': len(categories), 'n_categories_tested': tested,
                'n_valid_occurrences': len(data),
                'lag1_rate_correlation': pd.Series(group_rates[:, j]).autocorr() if len(data) > 2 else np.nan,
                'fraction_overlapping_windows': float((np.diff(data.onset_ms) < 300).mean()) if len(data) > 1 else np.nan,
                'n_fdr_significant': int((q_values['perm'] < alpha).sum()), 'category_tuned': tuned,
                'category_tuned_shift': bool((q_values['shift'] < alpha).any()) if shift_tested else pd.NA,
                'n_significant_both_nulls': int(((q_values['perm'] < alpha) & (q_values['shift'] < alpha)).sum()),
            })
            for category in WORD_CATEGORIES:
                audit_rows.append({
                    'session': session, 'neuron': neuron, 'category_id': category,
                    'n_valid_occurrences': int(sizes.loc[category, 'size']), 'n_words': int(sizes.loc[category, 'nunique']),
                    'eligible': bool(eligible.loc[category]), 'below_min_occurrences': bool(sizes.loc[category, 'size'] < min_occurrences),
                    'below_min_words': bool(sizes.loc[category, 'nunique'] < min_words),
                    'n_invalid_window': int((~occurrences.valid_window).sum()), 'n_unclassified': int((~occurrences.valid_category).sum()),
                    'n_nonfinite_response': int((usable & ~np.isfinite(rates[:, column_index])).sum()),
                })
    results = pd.DataFrame(result_rows, columns=[
        'session', 'neuron', 'category_id', 'category', 'n_target', 'n_other', 'n_words', 'mean_target', 'mean_other',
        'delta_hz', 't_true', 'p_perm', 'q_value', 'significant_fdr', 'p_shift', 'q_shift', 'significant_shift',
        'loo_min_delta_hz', 'loo_max_delta_hz', 'loo_same_direction', 'n_permutations', 'n_shifts',
    ])
    neurons = pd.DataFrame(neuron_rows).astype({'category_tuned': 'boolean', 'category_tuned_shift': 'boolean'})
    return results, neurons, pd.DataFrame(audit_rows)


def get_semantic_category_df(df_podcast, rerun=False, output_dir='data/tuning/podcast_category_tuning', **settings):
    """Load or compute Podcast category tuning, returning one row per unit/session.

    Return neuron, session_name, and nullable Boolean category_tuning. True means
    at least one category passed the primary permutation test with within-neuron
    BH-FDR; untestable units remain NA. Shift results stay in the detailed CSVs.

    Existing files are reused unless rerun=True. Pass session_category_tuning
    settings as keyword arguments; changing settings requires a manual rerun.
    Save the per-session results, exclusions, and combined summaries as before.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = dict(min_occurrences=20, min_words=5, n_permutations=10000, random_state=0, alpha=0.05, min_shift=100) | settings
    category_tables, neuron_tables, audit_tables = [], [], []
    for s in df_podcast.session_class:
        paths = [output_dir / f'{s.filename}_{name}.csv' for name in ['category_results', 'neuron_summary', 'exclusions']]
        if all(path.exists() for path in paths) and not rerun:
            results, neurons, audit = [pd.read_csv(path) for path in paths]
            print(f'Loaded category tuning: {s.filename}')
        else:
            print(f'Computing category tuning: {s.filename}', flush=True)
            occurrences = make_category_occurrences(s)
            results, neurons, audit = session_category_tuning(occurrences, **settings)
            for table, path in zip([results, neurons, audit], paths):
                temporary = path.with_suffix('.csv.tmp')
                table.to_csv(temporary, index=False)
                temporary.replace(path)
            parameters = {'version': 1, **settings, 'window_ms': [80, 380], 'categories': WORD_CATEGORIES}
            (output_dir / f'{s.filename}_parameters.json').write_text(json.dumps(parameters, sort_keys=True, indent=2))
        neurons = neurons.astype({'category_tuned': 'boolean', 'category_tuned_shift': 'boolean'})
        category_tables.append(results)
        neuron_tables.append(neurons)
        audit_tables.append(audit)

    results, neurons, audit = [pd.concat(tables, ignore_index=True) for tables in [category_tables, neuron_tables, audit_tables]]
    summary = neurons.groupby('session', as_index=False).agg(
        n_neurons=('neuron', 'size'), n_tested=('category_tuned', 'count'), n_category_tuned=('category_tuned', 'sum'),
        n_category_tuned_shift=('category_tuned_shift', 'sum'),
    )
    summary['pct_category_tuned'] = 100 * summary.n_category_tuned / summary.n_tested.replace(0, np.nan)
    for name, table in [('category_results', results), ('neuron_summary', neurons), ('exclusions', audit), ('session_summary', summary)]:
        table.to_csv(output_dir / f'podcast_{name}_all_sessions.csv', index=False)
    return neurons.rename(columns={'session': 'session_name', 'category_tuned': 'category_tuning'})[['neuron', 'session_name', 'category_tuning']]


#############################
##### ROTATIONS RELATED #####
#############################


def get_rotations_event_modulation_df(df_rotations, rerun=False, output_dir='data/tuning/rotations_tuning', min_occurrences=10):
    """Load or compute Rotations event selections and event-level statistics.

    Preserve tuning_old.py: use the loader's aligned trial starts/ends and the
    existing pre/post Wilcoxon/Holm analysis. Both events must have at least
    min_occurrences usable windows. Keep units from unalignable sessions with
    event_modulation=NA. Return (selections, event_results), like the other tasks.

    Cache each session, including alignment counts, and save the combined
    rotations_alignment.csv. Existing files are reused unless rerun=True.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selections, event_results, alignment = [], [], []
    for s in df_rotations.session_class:
        cache_path = output_dir / f'{s.filename}_event_modulation.pkl'
        if cache_path.exists() and not rerun:
            result = pd.read_pickle(cache_path)
            print(f'Loaded: {cache_path.name}')
        else:
            events = pd.DataFrame(s.event_times, columns=['trial_start', 'trial_end'])
            event_data = make_event_modulation_data(s, events)
            usable = event_data['valid_occurrences'].sum(axis=0)
            alignable = bool(len(events) and np.all(usable >= min_occurrences))
            if alignable:
                result = fit_event_modulation(event_data, min_occurrences=min_occurrences)
            else:
                names = event_data['channel_names']
                result = {
                    'selection': pd.DataFrame({'neuron_index': range(len(names)), 'neuron': names,
                                               'event_modulation': pd.array([pd.NA] * len(names), dtype='boolean')}),
                    'event_results': pd.DataFrame(columns=[
                        'neuron_index', 'neuron', 'event', 'n_occurrences', 'n_nonzero_differences',
                        'pre_rate_hz', 'post_rate_hz', 'mean_delta_hz', 'median_delta_hz',
                        'statistic', 'p_value', 'p_value_corrected', 'significant',
                    ]),
                }
            result['selection']['session_name'] = s.filename
            result['selection']['event_modulation'] = result['selection']['event_modulation'].astype('boolean')
            result['event_results']['session_name'] = s.filename
            result['alignment'] = {'session_name': s.filename, 'n_events': len(events),
                                   'n_usable_trial_starts': int(usable[0]), 'n_usable_trial_ends': int(usable[1]),
                                   'alignable': alignable, 'min_occurrences': min_occurrences}
            temporary = cache_path.with_suffix('.pkl.tmp')
            pd.to_pickle(result, temporary)
            temporary.replace(cache_path)
            print(f'Saved: {cache_path.name}')
        selections.append(result['selection'])
        event_results.append(result['event_results'])
        alignment.append(result['alignment'])
    pd.DataFrame(alignment).to_csv(output_dir / 'rotations_alignment.csv', index=False)
    event_results = [table for table in event_results if not table.empty] or event_results[:1]
    return pd.concat(selections, ignore_index=True), pd.concat(event_results, ignore_index=True)


def make_rotations_trial_dataframe(s):
    """One row per original behavioral trial, including missing/unusable rows.

    Align absolute or stimulus-relative responseTime to the neural clock using
    the existing offsets. response_ms is the actual answer; trial_end_ms retains
    the loader's additional 500 ms. Pair codes 1/2 are not relabeled as mirror.

    Raw spike counts / 0.500 s give responses in [stimulus+200, stimulus+700)
    and [response-700, response-200) ms. A full window must lie in the recording
    AND between stimulus and response. Invalid windows remain NaN, never clipped.
    Fractional starts use ceil(start), selecting exactly 500 millisecond samples.
    Also sample 500 ms centered halfway between
    stimulus and response. This is a midpoint sample, not a whole-trial average.
    """
    events = s.events
    fields = {'angle_deg': 'degreeForTrial', 'pair_type': 'pairTypeForTrial',
              'correct': 'correctAnswers', 'object_id': 'objectForTrial', 'response_key': 'inputForTrial'}
    trials = pd.DataFrame({name: [_unwrap_matlab_scalar(v) for v in events[key].ravel()] for name, key in fields.items()})
    for name in ['angle_deg', 'pair_type', 'correct', 'object_id']:
        trials[name] = pd.to_numeric(trials[name], errors='coerce')
    trials.insert(0, 'trial_index', np.arange(len(trials)))
    trials.insert(0, 'session_name', s.filename)
    trials['stimulus_ms'], trials['response_ms'] = np.nan, np.nan
    trials['rt_s'] = np.asarray(events.get('RTs', np.full(len(trials), np.nan))).ravel()
    timing_keys = ['stimulusStartTime', 'responseTime', 'nevTimeOffset', 'ns5TimeOffset']
    if all(key in events for key in timing_keys):
        onset = np.asarray(events['stimulusStartTime'], dtype=float).ravel()
        response = np.asarray(events['responseTime'], dtype=float).ravel()
        valid = np.isfinite(onset) & np.isfinite(response)
        if valid.any():
            if np.all(response[valid] < onset[valid]):
                response = onset + response
            elif not np.all(response[valid] > onset[valid]):
                raise ValueError(f'Inconsistent response-time convention: {s.filename}')
            offset = float(_unwrap_matlab_scalar(events['nevTimeOffset'])) + float(_unwrap_matlab_scalar(events['ns5TimeOffset']))
            trials['stimulus_ms'], trials['response_ms'] = (onset - offset) * 1000, (response - offset) * 1000
            trials['rt_s'] = response - onset
    trials['trial_end_ms'] = trials.response_ms + 500
    trials['valid_labels'] = trials.angle_deg.isin([0, 50, 100, 150]) & trials.pair_type.isin([1, 2])
    windows = [('stimulus', trials.stimulus_ms + 200), ('response', trials.response_ms - 700),
               ('decision', (trials.stimulus_ms + trials.response_ms) / 2 - 250)]
    for window, start in windows:
        stop = start + 500
        valid = (np.isfinite(start) & np.isfinite(trials.stimulus_ms) & np.isfinite(trials.response_ms)
                 & (start >= 0) & (stop <= s.spikes.shape[1]) & (start >= trials.stimulus_ms) & (stop <= trials.response_ms))
        trials[f'{window}_start_ms'], trials[f'{window}_stop_ms'], trials[f'valid_{window}'] = start, stop, valid
        starts = np.ceil(start[valid]).to_numpy(dtype=int)
        for channel_index, neuron in s.chosen_chan.filename.items():
            spikes = s.spikes.getrow(channel_index).tocsr(copy=True)
            spikes.sum_duplicates()
            spikes.sort_indices()
            cumulative = np.r_[0, np.cumsum(spikes.data)]
            counts = cumulative[np.searchsorted(spikes.indices, starts + 500)] - cumulative[np.searchsorted(spikes.indices, starts)]
            rates = np.full(len(trials), np.nan)
            rates[valid] = counts / 0.500
            trials[f'fr_hz_{window}_{neuron}'] = rates
    return trials


def session_rotations_task_tuning(trials, min_occurrences=10, n_permutations=10000, random_state=0, alpha=0.05,
                                  min_cell_occurrences=5):
    """Partial-F tuning tests using within-session Freedman-Lane permutations.

    Model: rate ~ C(angle) * pair type + linear trial order. Test angle, pair,
    and interaction in the stimulus, pre-response, and midpoint windows.
    Sum-centered coding makes main effects equal-weight marginal contrasts over
    the other factor. Separately add log(RT), response choice, or correctness to
    that stimulus model and test the added term. Behavioral effects adjust for
    stimulus and order, not for each other; associations do not imply causation.
    Choice codes o/d as 1 and w/s as 0 without assuming a hand/mirror mapping.

    Require min_occurrences per angle/pair level or behavioral binary level;
    interaction requires min_cell_occurrences in each of the eight cells.
    Missing covariates are removed only for the corresponding test. Always test
    six effects in three windows, giving 18 planned tests per neuron.

    Fit each reduced model, shuffle its residual trial rows within session, add
    back fitted responses, and recompute partial F. Reuse permutations across
    neurons sharing usable trials. p=(1+upper-tail exceedances)/(permutations+1).
    This approximate test assumes exchangeable errors; adjusting trial order
    alone does not ensure independence or constant variance (Winkler et al.,
    2014, doi:10.1016/j.neuroimage.2014.01.060). Constant responses give F=0, p=1.

    BH-FDR covers ALL planned effects x windows separately per neuron; untestable
    p-values count as 1 for correction but retain NaN results. Overall tuning is
    True if any test is significant, False if any test ran but none was significant,
    and NA if no tests ran. False means no tuning detected among eligible tests.
    Effect-specific flags retain NA for incomplete nonsignificant testing.
    No correction across neurons is implied.
    """
    if trials.session_name.nunique() != 1:
        raise ValueError('Provide trials from exactly one session.')
    if min_occurrences < 2 or n_permutations < 1:
        raise ValueError('Use min_occurrences >= 2 and n_permutations >= 1.')
    session = trials.session_name.iloc[0]
    neurons = [c.removeprefix('fr_hz_stimulus_') for c in trials if c.startswith('fr_hz_stimulus_')]
    windows = ['stimulus', 'response', 'decision']
    effects = ['angle', 'pair_type', 'interaction', 'log_rt', 'choice', 'correctness']
    covariates = {'log_rt': np.log(trials.rt_s.where(trials.rt_s > 0)),
                  'choice': trials.response_key.map({'w': 0., 's': 0., 'o': 1., 'd': 1.}),
                  'correctness': trials.correct.where(trials.correct.isin([0, 1]))}
    rows = []
    for window in windows:
        rates = trials[[f'fr_hz_{window}_{neuron}' for neuron in neurons]].to_numpy(float)
        for effect in effects:
            usable = trials.valid_labels & trials[f'valid_{window}']
            if effect in covariates:
                usable = usable & np.isfinite(covariates[effect])
            masks = usable.to_numpy()[:, None] & np.isfinite(rates)
            groups = {}
            for j in range(len(neurons)):
                groups.setdefault(masks[:, j].tobytes(), []).append(j)
            for indices in groups.values():
                keep = masks[:, indices[0]]
                data, y = trials.loc[keep], rates[np.ix_(keep, indices)]
                n = len(data)
                angle_counts = data.angle_deg.value_counts().reindex([0, 50, 100, 150], fill_value=0)
                pair_counts = data.pair_type.value_counts().reindex([1, 2], fill_value=0)
                cells = pd.MultiIndex.from_product([[0, 50, 100, 150], [1, 2]])
                cell_counts = data.groupby(['angle_deg', 'pair_type']).size().reindex(cells, fill_value=0)
                order = data.trial_index.to_numpy(float)
                order = (order - order.mean()) / max(np.ptp(order), 1) if n else order
                angle = (data.angle_deg.to_numpy()[:, None] == [50, 100, 150]).astype(float)
                pair = data.pair_type.eq(2).to_numpy(float)
                angle, pair = angle - .25, pair - .5
                x = np.column_stack([np.ones(n), angle, pair, order, angle * pair[:, None]])
                model = 'angle + pair_type + trial_order + angle:pair_type'
                columns = {'angle': [1, 2, 3], 'pair_type': [4], 'interaction': [6, 7, 8]}.get(effect, [9])
                counts = angle_counts if effect == 'angle' else pair_counts if effect == 'pair_type' else cell_counts
                threshold = min_cell_occurrences if effect == 'interaction' else min_occurrences
                if effect in covariates:
                    values = covariates[effect].loc[keep].to_numpy(float)
                    counts = pd.Series(values).value_counts().reindex([0, 1], fill_value=0) if effect != 'log_rt' else pd.Series([n])
                    x = np.column_stack([x, values])
                    model += f' + {effect}'
                residual_df = n - x.shape[1]
                rank = np.linalg.matrix_rank(x) if n else 0
                reason = ('no_valid_trials' if not n else 'insufficient_occurrences' if (counts < threshold).any()
                          else 'rank_deficient_model' if rank != x.shape[1] else 'insufficient_residual_df' if residual_df < 1 else '')
                f_true, p_perm = np.full(len(indices), np.nan), np.full(len(indices), np.nan)
                coefficients = np.full((x.shape[1], len(indices)), np.nan)
                y = y - y.mean(axis=0) if n else y
                if not reason:
                    reduced = np.delete(x, columns, axis=1)
                    qr = np.linalg.qr(reduced, mode='reduced')[0]
                    added = x[:, columns] - qr @ (qr.T @ x[:, columns])
                    qe = np.linalg.qr(added, mode='reduced')[0]

                    def f_stat(values):
                        """Partial F using orthogonal reduced/effect projections."""
                        total = np.sum(values**2, axis=0)
                        ss_effect = np.sum((qe.T @ values)**2, axis=0)
                        rss = np.maximum(total - np.sum((qr.T @ values)**2, axis=0) - ss_effect, 0)
                        tolerance = 1e-12 * np.maximum(total, 1)
                        with np.errstate(divide='ignore', invalid='ignore'):
                            f = (ss_effect / len(columns)) / (rss / residual_df)
                        f = np.where((rss <= tolerance) & (ss_effect > tolerance), np.inf, f)
                        return np.where((rss <= tolerance) & (ss_effect <= tolerance), 0, f)

                    f_true = f_stat(y)
                    fitted = qr @ (qr.T @ y)
                    residuals = y - fitted
                    exceed = np.zeros(len(indices), dtype=int)
                    rng = np.random.default_rng(random_state)
                    for _ in range(n_permutations):
                        f_null = f_stat(fitted + residuals[rng.permutation(n)])
                        exceed += (f_null >= f_true) | np.isclose(f_null, f_true, rtol=1e-10, atol=1e-12)
                    p_perm = (1 + exceed) / (n_permutations + 1)
                    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
                for j, column_index in enumerate(indices):
                    row = {'session_name': session, 'neuron': neurons[column_index], 'window': window, 'effect': effect,
                           'model': model, 'n_trials': n, 'n_excluded': len(trials) - n, 'exclusion_reason': reason,
                           'df_effect': len(columns), 'df_residual': residual_df if residual_df > 0 else np.nan,
                           'f_true': f_true[j], 'p_perm': p_perm[j], 'n_permutations': n_permutations,
                           'pair_type_2_minus_1_hz': coefficients[4, j], 'min_angle_pair_cell': int(cell_counts.min()),
                           'behavior_coefficient_hz': coefficients[-1, j] if effect in covariates else np.nan}
                    row.update({f'n_angle_{int(a)}': int(count) for a, count in angle_counts.items()})
                    row.update({f'n_pair_{int(p)}': int(count) for p, count in pair_counts.items()})
                    row.update({f'angle_{a}_minus_0_hz': coefficients[k, j] for k, a in enumerate([50, 100, 150], 1)})
                    if effect in ['choice', 'correctness']:
                        row.update(n_class_0=int(counts.loc[0]), n_class_1=int(counts.loc[1]))
                    rows.append(row)
    results = pd.DataFrame(rows)
    results['q_value'], results['significant_fdr'] = np.nan, False
    summaries = []
    flags = [f'{effect}_tuning' for effect in effects] + ['task_specific_tuning']
    for neuron, group in results.groupby('neuron', sort=False):
        valid = group.p_perm.notna()
        q = multipletests(group.p_perm.fillna(1), method='fdr_bh', alpha=alpha)[1]
        significant = (q < alpha) & valid.to_numpy()
        results.loc[group.index[valid], 'q_value'] = q[valid]
        results.loc[group.index, 'significant_fdr'] = significant
        summary = {'neuron': neuron, 'session_name': session, 'n_tests': int(valid.sum()),
                   'n_tests_planned': len(group), 'n_fdr_significant': int(significant.sum())}
        for flag, effect in zip(flags, effects + [None]):
            mask = group.effect.eq(effect) if effect else np.ones(len(group), dtype=bool)
            summary[flag] = True if (significant & mask).any() else False if valid[mask].all() else pd.NA
        summary['task_specific_tuning'] = bool(significant.any()) if valid.any() else pd.NA
        summaries.append(summary)
    return results, pd.DataFrame(summaries).astype({flag: 'boolean' for flag in flags})


def get_rotations_task_tuning_df(df_rotations, rerun=False, output_dir='data/tuning/rotations_tuning', **settings):
    """Load/compute Rotations tuning, returning merge-ready unit flags.

    Use file existence and rerun only, as for the event analysis. Cache trial
    responses, detailed tests, summary, and parameters per session. Save combined
    test/summary CSVs and trial metadata (responses remain in session pickles).
    Always use six effects and three 500-ms windows, including the trial midpoint.
    Refresh the overall detection flag from saved test counts when loading older
    summaries; this does not rerun tests or change their p-values or correction.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = dict(min_occurrences=10, n_permutations=10000, random_state=0, alpha=0.05,
                    min_cell_occurrences=5) | settings
    test_tables, summaries, metadata = [], [], []
    for s in df_rotations.session_class:
        path = output_dir / f'{s.filename}_task_tuning.pkl'
        if path.exists() and not rerun:
            result = pd.read_pickle(path)
            print(f'Loaded: {path.name}')
        else:
            print(f'Computing task-variable tuning: {s.filename}', flush=True)
            trials = make_rotations_trial_dataframe(s)
            tests, summary = session_rotations_task_tuning(trials, **settings)
            result = {'trials': trials, 'tests': tests, 'summary': summary,
                      'parameters': {**settings, 'stimulus_window_ms': [200, 700], 'response_window_ms': [-700, -200],
                                     'decision_window': 'midpoint +/- 250 ms',
                                     'models': tests.model.unique().tolist(),
                                     'behavioral_models': 'Separate additions; behavioral terms do not adjust for each other',
                                     'permutation': 'Freedman-Lane reduced-model residuals',
                                     'fdr_family': f'{summary.n_tests_planned.iloc[0]} planned tests per neuron'}}
            temporary = path.with_suffix('.pkl.tmp')
            pd.to_pickle(result, temporary)
            temporary.replace(path)
        summary = result['summary']
        detected = summary.n_fdr_significant.gt(0).astype('boolean').where(summary.n_tests.gt(0), pd.NA)
        if not summary.task_specific_tuning.equals(detected):
            summary['task_specific_tuning'] = detected
            temporary = path.with_suffix('.pkl.tmp')
            pd.to_pickle(result, temporary)
            temporary.replace(path)
        test_tables.append(result['tests'])
        summaries.append(result['summary'])
        metadata.append(result['trials'].loc[:, ~result['trials'].columns.str.startswith('fr_hz_')])
    tests, summary, trials = [pd.concat(tables, ignore_index=True) for tables in [test_tables, summaries, metadata]]
    for name, table in [('task_results', tests), ('task_summary', summary), ('trial_metadata', trials)]:
        table.to_csv(output_dir / f'rotations_{name}.csv', index=False)
    flags = [c for c in summary if c.endswith('_tuning')]
    return summary[['neuron', 'session_name'] + flags]


def get_rotations_tuning_df(df_rotations, rerun=False, output_dir='data/tuning/rotations_tuning', **settings):
    """Load/compute and save the complete Rotations tuning table.

    Always use six effects and three 500-ms windows with within-neuron BH-FDR.
    Merge event modulation. Overall tuning is False when eligible tests ran but
    none were significant; units with no eligible tests remain nullable Boolean NA.
    Existing caches are reused unless rerun=True. Return one row per unit/session.
    """
    result = get_rotations_task_tuning_df(df_rotations, rerun=rerun, output_dir=output_dir, **settings)
    events, _ = get_rotations_event_modulation_df(df_rotations, rerun=rerun, output_dir=output_dir)
    result = result.merge(events[['neuron', 'session_name', 'event_modulation']],
                          on=['neuron', 'session_name'], how='outer', validate='one_to_one')
    flags = [c for c in result if c.endswith('_tuning')] + ['event_modulation']
    result[flags] = result[flags].astype('boolean')
    result.to_csv(Path(output_dir) / 'rotations_tuning.csv', index=False)
    return result
