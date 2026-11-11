#%%
import sys
import glob
import os.path
import re
import json
import numpy as np
import pandas as pd
import h5py
import scipy.io
from collections import namedtuple
from collections import Counter
from scipy.sparse import csc_matrix
from datetime import datetime
from pathlib import Path
from decomposition import FAResults

REPO_ROOT = Path(__file__).resolve().parent

if sys.platform == 'linux':
    PREFIX = os.path.join(os.path.expanduser('~'), 'hungyun-elias', 'data')
elif sys.platform == 'darwin': # Data are stored on my MacBook locally. 
    PREFIX = str(REPO_ROOT / 'data')

DATADIR = os.path.join(PREFIX, 'sessions','perChannel')
SESSIONS_CSV = os.path.join(PREFIX, 'sessions.csv')
BRAIN_AREAS_JSON = os.path.join(PREFIX, 'brain_areas.json')
BEHAVIOR_DIR = os.path.join(PREFIX,'behavior')
WORDS_MAT = os.path.join(BEHAVIOR_DIR, 'podcast', 'HPC_YEXthurYFI_WordFRMatrix_300Post80msDelay.mat') # Containing all the words/event times for all patients

def load_sessions_csv(sessionfile=SESSIONS_CSV, datadir=DATADIR):
    """
    Example of sessionfile (csv):
        Patient,EMU,Task,Date,Time,Ranking
        YEX,96,Podcast,2024-02-08,3:55:00 PM,2
        YEX,99,Podcast,2024-02-08,4:36:00 PM,1

    The output df should have the following fields.
        1. Patient: Patient ID
        2. EMU: The EMU number for each task.
        3. Task: Task name, capitalized.
        4. Date
        5. Time
        6. Ranking: If nan, the task was done once. If a number, the xth time they did it.
        7. Datetime: Processed from Date and Time
        8. filepath: where session_name.mat files store.
        9. filename: abbreviation of the task info, should be like YEX_0099_podcast.
        10. session_key: more abbreviated version of task info, should be like X-C1.
    """

    def row_to_session_path(row, datadir=DATADIR):
        """
        converts a row from the sessions DataFrame to a session file path
        """
        session_dir = os.path.join(datadir, row['Task'].lower())
        filename = f'{row["Patient"]}_{str(row["EMU"]).zfill(4)}_{row["Task"].lower()}'
        return os.path.abspath(os.path.join(session_dir, filename + '.mat'))
    
    df = pd.read_csv(sessionfile)
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y-%m-%d %I:%M:%S %p')
    df['filepath'] = df.apply(lambda row: row_to_session_path(row, datadir), axis=1)

    # Helper functions for making session keys.
    task_key = {'Podcast': 'C', 'Pursuit': 'S', 'Rotations': 'R'}
    patient_f = lambda x: x[-1] if isinstance(x, str) else x
    rank_f = lambda x: str(int(x)) if pd.notna(x) else '1'

    df['filename'] = df.apply(lambda row: f'{row["Patient"]}_{str(row["EMU"]).zfill(4)}_{row["Task"].lower()}', axis=1)
    df['session_key'] = df.apply(lambda row: patient_f(row['Patient']) + '-' + task_key.get(row['Task']) + rank_f(row['Ranking']), axis=1)
    return df

def get_subj_channels(subj, datafile=BRAIN_AREAS_JSON):
    """
    subj is a str, the subject identifier (e.g., 'YEX')
    datafile is the path to the JSON file containing brain areas for each subject
    returns a dict mapping brain area names to lists of channel numbers for subj
    """
    with open(datafile) as f:
        brain_areas = json.load(f)
    csubj = subj[:3]
    if csubj.upper() not in brain_areas:
        raise Exception(f'Could not find subj={csubj} in {datafile}')
    subj_brain_areas = brain_areas[csubj.upper()]
    if 'ignore' in subj_brain_areas:
        subj_brain_areas.pop('ignore')
    return subj_brain_areas

def intersect_all(arrays, n=None):
    """
    arrays is a list of arrays
    n is the minimum number of arrays an element must appear in to be kept
    By default, n = len(arrays), which means the element must appear in all arrays
    """
    if not arrays:
        return np.array([])
    if n is None:
        n = len(arrays)
    
    # Flatten and count occurrences across arrays
    all_elements = []
    for arr in arrays:
        all_elements.extend(np.unique(arr))  # use unique to avoid duplicates within one array
    counts = Counter(all_elements)

    # Keep elements that appear in at least n arrays
    return np.array([elem for elem, count in counts.items() if count >= n])

def confirm_all_channels_match(sessions):
    """
    checks that all sessions have the same exact channels
    """
    all_chans = [session['channels'][session['ix']] for session in sessions]
    common_chans = intersect_all(all_chans)
    if not all([(len(common_chans) == len(chans)) and (common_chans == chans).all() for chans in all_chans]):
        raise Exception('Channels do not match across all results. Please check your data.')

def channels_labeled_by_brain_area(session, brain_areas):
    """
    labels channels by brain area, where each brain area is assigned a unique label
    session is a session dict
    brain_areas is a list of brain area names (e.g., ['HPC', 'ACC'])
    Returns an array of labels, with the same length as session['channels'], where each label corresponds to a brain area
    """
    labels = -1 * np.ones_like(session['channels'])
    for i, brain_area in enumerate(brain_areas):
        keys_to_use = [area for area in session['brain_areas'] if brain_area in area]
        if len(keys_to_use) == 0:
            continue
        chans_in_area = np.concatenate([session['brain_areas'][key] for key in keys_to_use])
        ixc = np.isin(session['channels'], chans_in_area)
        assert (labels[ixc] == -1).all()
        labels[ixc] = i
    return labels

def epoch_events(df, bin_times):
    """
    df is a DataFrame with epoch event times (per trial) in each row
    bin_times is a 1D array of bin times in ms
    returns an array of ints with the epoch index for each bin time
        where epoch index will be 0 for all bins not within any trial epoch
    note that the last epoch per trial will never be assigned, as it's assumed to indicate the end of the trial
    """
    epoch_labels = np.zeros_like(bin_times, dtype=int)
    col_names = df.columns
    for i, row in df.iterrows():
        epoch_times = row.values

        # first find bin times that are within the current trial
        ixc = (bin_times >= row.values.min()) & (bin_times <= row.values.max())
        cbin_times = bin_times[ixc]

        # for those bin times, find the index of the closest epoch time
        # by minimizing difference between bin time and epoch time
        # where distance between bin_time and epoch_time is infintite if epoch_time is larger than bin_time
        ds = (cbin_times[:, None] - epoch_times[None, :])
        ds[ds < 0] = np.inf  # set distance to inf if epoch time is larger than bin time
        closest_epoch_ix = np.argmin(ds, axis=1)+1
        epoch_labels[ixc] = closest_epoch_ix
    return epoch_labels

def load_events(filepath, events_dir='events'):
    """
    loads the events object corresponding to the filepath
    filepath is the path to the session file (e.g., 'data/sessions/YEW_0033_pursuit.mat')
    """
    basedir = os.path.dirname(filepath)
    eventsdir = os.path.join(basedir, events_dir)
    fnm = os.path.basename(filepath).replace('.mat', '_events.mat')
    if not os.path.exists(eventsdir):
        raise Exception(f'Events directory {eventsdir} does not exist. Please check the session file {filepath}.')
    events_file = os.path.join(eventsdir, fnm)
    f = scipy.io.loadmat(events_file)
    return f

def load_rotation_trials(filepath):
    f = load_events(filepath)
    if 'nevTimeOffset' not in f or 'ns5TimeOffset' not in f:
        print(f'WARNING: nevTimeOffset or ns5TimeOffset not found in {filepath}. Please check the session file.')
        return f, []
    
    t_offset = f['nevTimeOffset'][0,0] + f['ns5TimeOffset'][0,0]
    trial_start_times = f['stimulusStartTime'][:,0]
    trial_end_times = f['responseTime'][:,0] + 0.5 # pad response time by 0.5s
    ix = np.isnan(trial_start_times) | np.isnan(trial_end_times)
    if ix.sum() > 0: # Remove nan
        trial_start_times = trial_start_times[~ix]
        trial_end_times = trial_end_times[~ix]

    if np.all(trial_end_times > trial_start_times):
        # trial_end_times are in absolute times
        pass
    elif np.all(trial_end_times < trial_start_times):
        # trial_end_times are relative to trial_start_times
        trial_end_times += trial_start_times
    else:
        raise Exception(f'Inconsistent trial start and end times in {filepath}. Please check the session file.')

    # note: times are in seconds, so we convert to milliseconds
    ft = lambda x: (x - t_offset) * 1000 # convert to milliseconds
    event_times = list(zip(ft(trial_start_times), ft(trial_end_times)))
    return f, event_times

def load_pursuit_trials(filepath):
    """
    Load the events and locations files for each session.

    """
    trials = None
    locations = None
    event_times = []

    try:
        f = load_events(filepath)
        trials = f['events_info'][0,:]

    except FileNotFoundError:
        print(f'WARNING: events file not found for {filepath}. Please check the session file.')

    try:
        f = load_events(filepath, 'locations')
        locations = f['neuronData'][0,:]
    except FileNotFoundError:
        print(f'WARNING: locations file not found for {filepath}. Please check the session file.')

    # make dataframe of event times
    if trials is not None:
        event_order = ['wait_start', 'chase_start', 'chase_end', 'feedback_start', 'trial_end']
        event_data = { name: np.array([x[0, 0] for x in trials[name]]) for name in event_order }
        events = pd.DataFrame(event_data)
        event_times = list(zip(event_data["wait_start"],event_data["trial_end"],))

    # handle missing trials
    if trials is not None and locations is not None and len(locations) != len(trials):
        raise ValueError(f"Trial/location count mismatch: {len(trials)} trials versus {len(locations)} locations.")

    return trials, locations, events, event_times

def load_words(subj, words_file=WORDS_MAT):
    d = scipy.io.loadmat(words_file)
    key = f'{subj.lower()}_word'
    if key not in d:
        raise Exception(f'Could not find {subj=} in words_file.')
    words = d[key]
    words = [x for x in words if type(x['text'][0][0]) is np.str_]

    # handle cases (e.g., YEX) where word times got concatenated
    times = [x['onset'][0][0][0] for x in words]
    reversal_index = np.where(np.diff(times) < 0)[0].tolist()
    if len(reversal_index) > 0:
        # There could be multiple reversal times, but we only keep analyzing if there is just one.
        # If there are multiple, we raise an exception. 
        if len(reversal_index) > 1:
            raise Exception(f'Found multiple reversal times in {subj=} words. Please check the data.')
        nwords = len(words)
        words = words[reversal_index[0]+1:]  # keep only words after the first reversal
        print(f'WARNING: word onset times were not in order. Ignoring {nwords-len(words)} words before reversal time.')
    
    event_times = [(word['onset'][0][0][0], word['offset'][0][0][0]) for word in words]
    return words, event_times

def ix_during_events(event_times, bin_times, max_gap_dur_secs=1):
    """
    event_times is a list of (start_time, end_time) tuples of event times in ms
    bin_times is a 1D array of bin times in ms
    max_gap_dur_secs is the maximum duration of a gap to be considered significant

    returns a boolean array of the same length as bin_times, marking the bins containing event_times
    """

    def big_time_gaps(times, max_gap_dur_secs=1):
        """
        times is a list of (start_time, end_time) tuples of event times in ms
        max_gap_dur_secs is the maximum duration of a gap to be considered significant
        Returns a list of (start_time, end_time) tuples with too-large gaps between event times.
        """
        first_gap = (0, times[0][0])
        ts = [first_gap]
        for i in range(len(times)-1):
            t1 = times[i][1]
            t2 = times[i+1][0]
            if (t2-t1)/1000 > max_gap_dur_secs:
                ts.append((t1, t2))
        
        # add last gap
        t1 = times[-1][1]
        ts.append((t1, int(1e7)))
        return ts
    
    event_gaps = big_time_gaps(event_times, max_gap_dur_secs=max_gap_dur_secs)
    ix = np.ones_like(bin_times).astype(bool)
    for (t_start, t_end) in event_gaps:
        t_start_bin = np.argmin((bin_times - t_start)**2)
        t_end_bin = np.argmin((bin_times - t_end)**2)
        ix[t_start_bin:(t_end_bin+1)] = False
    return ix

def normalize(metric: np.ndarray):
    
    if metric.ndim != 2:
        print('May want to check the array dimension.')

    mean = np.mean(metric, axis=0) # (N,)
    std = np.std(metric, axis=0) # (N,)
    norm = (metric - mean[None,:]) / std[None,:] # (T x N)
    norm = np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0) # Replace NaN and inf with 0
    
    return norm

def permute_columns_independently(Y, rng=None):
    """
    Y: (time, neurons)
    Returns a copy where each neuron's time series is independently permuted.
    """
    if rng is None:
        rng = np.random.default_rng()

    Yperm = Y.copy()
    for j in range(Y.shape[1]):
        Yperm[:, j] = rng.permutation(Yperm[:, j])
    return Yperm

def pursuit_trials_to_dataframe(s):
    """Convert the pursuit event struct array into readable trial metadata."""
    trials = s.trials
    if trials is None:
        return pd.DataFrame()

    trials_df = pd.DataFrame.from_records(np.asarray(trials).reshape(-1))
    trials_df = trials_df.map(lambda x: x[0][0])
    trials_df.index.name = 'trial_index'
    trials_df['successful'] = trials_df['reward_val'] > 0
    return trials_df

class FactorAnalysisResult:
    """Legacy result container retained for reading pre-FAResults pickles."""

    def __init__(self, fit_result: dict):
        self.__dict__.update(fit_result)

    def __repr__(self):
        return f'{self.subspace.shape[0]} components from {self.subspace.shape[1]} channels'


class Session:
    """
    Turn a session dictionary into a class for easier access to attributes.
    Building from load_session() in _old/data_loader_old.py.
    """

    def __init__(self, session_info: pd.Series):

        # Known attributes
        self.session_info = session_info
        self.subj = session_info['Patient']
        self.task = session_info['Task'].lower()
        self.datetime = session_info['Datetime']
        self.filepath = session_info['filepath']
        self.filename = session_info['filename']
        self.session_key = session_info['session_key']
        self.fileexists = True
        self.brain_areas_in_session = get_subj_channels(subj=self.subj)
        self.available_areas = None
        self.df_chan = None
        self.chosen_chan = None
        self.matfile_type = None

        # Read the mat file
        if not os.path.exists(self.filepath):
            self.fileexists = False
            print(f'WARNING: session file {self.filepath} does not exist. Skip this row.')
            return
        else:
            with open(self.filepath, 'rb') as f:
                header = f.read(512) # Check the header to see if it's HDF5
            if b'HDF' in header: # This is a feature for MATLAB v7.3 files
                self.f = h5py.File(self.filepath, 'r')
                self.matfile_type = 'hdf5'
            else:
                self.f = scipy.io.loadmat(self.filepath)
                self.matfile_type = 'mat'

        # Attributes for analysis
        self.spikes = None
        self.unique_channels = None
        self.brain_areas_for_analysis = None
        self.Y = None
        self.bin_size = None
        self.bin_times = None
        self.spikes_mu = None
        self.spikes_sd = None
        self.Y_norm = None
        self.ix = None
        self.on_task = None
        self.events = None
        self.trials = None
        self.locations = None
        self.event_times = None
        self.mode: str = None
        self.max_gap_dur_secs = None
        self.verbose = True

        # For resampling
        self.Y_resample = None
        self.region_code = None
        self.resample_flag = False

    def __repr__(self):
        return f'Session {self.session_key}'

    def time_to_ix(self, t_ms):
        """Return the index of the bin containing ``t_ms``.

        ``bin_times`` stores the start time (ms) of every currently retained
        bin.  Consequently this method is also valid after ``get_events()``,
        but a time falling in a removed gap has no index and raises
        ``ValueError``.  Call ``get_spikes()`` before using this method.

        Parameters
        ----------
        t_ms : float
            Time in milliseconds, expressed in this session's neural-recording
            clock.

        Returns
        -------
        int
            Index into the current ``bin_times`` and ``spikes_per_bin`` arrays.
        """
        if self.bin_times is None or self.bin_size is None:
            raise RuntimeError('Bin times are unavailable. Call get_spikes() first.')
        if not isinstance(t_ms, (int, float, np.number)):
            raise TypeError('t_ms must be a single time in milliseconds.')
        if not np.isfinite(t_ms):
            raise ValueError('t_ms must be finite.')

        bin_ix = int(np.searchsorted(self.bin_times, t_ms, side='right') - 1)
        if bin_ix < 0 or t_ms >= self.bin_times[bin_ix] + self.bin_size:
            raise ValueError(f'{t_ms=} does not fall within a retained session bin.')
        return bin_ix

    def get_spikes(self, binsize: int = 50):
        """
        Load spikes from the session file and process them.

        This function yields a property spikes_per_bin (n_bins, n_channel).
        The shape[1] n_channel should be the same as the length of self.df_chan.

        The property self.spikes_per_bin should NOT be modified besides get_events().
        If want to sample a subset of channels to do FA, we should call self.find_Y().

        """
        self.bin_size = binsize

        if self.matfile_type == 'mat':
            self.spikes = self.f['spikes'].tocsr()
        elif self.matfile_type == 'hdf5':
            data = self.f['spikes']['data'][:]
            ir = self.f['spikes']['ir'][:]
            jc = self.f['spikes']['jc'][:]
            N = int(max(ir)) + 1  # Number of neurons (rows)
            T = len(jc) - 1  # Number of time bins (columns)
            self.spikes = csc_matrix((data, ir, jc), shape=(N, T))

        # count spikes in nonoverlapping bins
        t_to_drop = self.spikes.shape[1] % self.bin_size # trim off empty bins at end
        S = self.spikes[:,:self.spikes.shape[1]-t_to_drop].toarray().reshape(
            self.spikes.shape[0],                               # n_units
            (self.spikes.shape[1]-t_to_drop) // self.bin_size,  # n_timebins
            self.bin_size)                                      # bin_size
        spikes_per_bin = S.sum(axis=2)
        spikes_per_bin = spikes_per_bin * (1000 / self.bin_size) # convert to Hz
        self.bin_times = self.bin_size * np.arange(spikes_per_bin.shape[1])
        self.spikes_per_bin = spikes_per_bin.T # (n_channel, n_bins)

    def get_channels(self, 
                     brain_areas_for_analysis: list[str]|tuple[str] = None, 
                     strict_brain_areas = False):

        # Read the brain areas that we intended to analyze. 
        self.brain_areas_for_analysis = brain_areas_for_analysis
        if self.brain_areas_for_analysis is None:
            self.brain_areas_for_analysis = {'HPC','ACC'}
        
        # Check if brain areas for analysis are in this session.
        # Usually we choose HPC and ACC for analysis, but some patients do not have probes inserted in the ACC (e.g., YEY).
        # Therefore, we first find available areas in this patient, and then adjust our brain areas for analysis.
        # If stric_brain_areas is True, then it would raise an error for the YEY case.
        self.available_areas = set(i.split(' ')[0] for i in self.brain_areas_in_session.keys()) # A set of like (HPC,) or (HPC, ACC).
        if strict_brain_areas:
            if not set(self.brain_areas_for_analysis).issubset(self.available_areas):
                raise ValueError(f'Brain areas {self.brain_areas_for_analysis} not found in session {self.subj}. Available areas: {self.available_areas}')
        else:
            # If not strict, use the intersection of available areas and areas intended for analyses.
            self.brain_areas_for_analysis = list( self.available_areas & set(self.brain_areas_for_analysis))

        js = self.brain_areas_in_session # 'ignore' already popped. 
        js_map = []
        for key, val in js.items():
            js_map.extend([(v, key.split()) for v in val]) # [channel number, brain region, hemisphere]
        js_map = dict(js_map)

        def map_seq_num_to_reg(num):
            if num in js_map.keys():
                return js_map[num] # ['region','hemisphere']
            else:
                return ['ignored','ignored']

        if self.matfile_type == 'mat':
            # for the matfile that can be read by scipy.io.loadmat,
            # there are the following keys:
            # 'chan', 'cluster_ids', 'params', 'spikes', 'waveform', 'channel_file_names', 'n_spikes_excluded', 'clusters_excluded'
            filename = self.f['channel_file_names'].tolist()
            filename = [s.strip() for s in filename]
            df_chan = pd.DataFrame(filename, columns=['filename'])
            df_chan['raw_num'] = self.f['chan'].flatten()
            df_chan['seq_num'] = np.argsort(np.argsort(df_chan['raw_num']))+1
            df_chan[['region', 'hemi']] = df_chan['seq_num'].apply(lambda x: pd.Series(map_seq_num_to_reg(x)))
            df_chan['has_cluster'] = ~np.isnan(self.f['cluster_ids'].flatten())
            df_chan['fr'] = np.mean(self.spikes_per_bin, axis=0)
            self.df_chan = df_chan
        elif self.matfile_type == 'hdf5':
            # for the hdf5 matfile (MATLAB v7.3),
            # there are the following keys:
            # ['chan', 'qual', 'spikes', 'waveform']
            
            # This is the old format so the information is very limited.
            df_chan = pd.DataFrame(self.f['chan'][:].flatten(), columns=['raw_num'])
            df_chan['seq_num'] = np.argsort(np.argsort(df_chan['raw_num']))+1
            df_chan[['region', 'hemi']] = df_chan['seq_num'].apply(lambda x: pd.Series(map_seq_num_to_reg(x)))
            df_chan['has_cluster'] = True  # No cluster info in this format
            df_chan['fr'] = np.mean(self.spikes_per_bin, axis=0)
            self.df_chan = df_chan
    
    def choose_channels(self,
               brain_areas: list[str] = ['HPC','ACC'], 
               min_fr: float = 0.5, 
               must_have_cluster: bool = True):
        """
        Select channels based on some criteria, such as brain regions, 
        minimal firing rates, or if the channel must have clusters.
        
        Note that self.chosen_chan should NOT go through reset_index otherwise we will sample the wrong channels. 
        """
        
        if self.df_chan is None: 
            raise ValueError('Do get_channels first.')
        rule_area = (self.df_chan['region'].isin(brain_areas) if brain_areas is not None else self.df_chan['region'].notna())
        rule_fr = self.df_chan['fr'] > (min_fr if min_fr is not None else -np.inf)
        rule_cl = self.df_chan['has_cluster'] if must_have_cluster else self.df_chan['has_cluster'].notna()
        self.chosen_chan = self.df_chan[rule_area & rule_fr & rule_cl]
        self.chosen_chan['seq_index'] = np.arange(len(self.chosen_chan))
        return self

    def find_y(self, idx = None):
        """
        Generates self.Y
        """

        if self.matfile_type == 'mat':
            if idx is None:
                if self.chosen_chan is None:
                    raise ValueError('Must provide index or run choose_channels()')
                else:
                    idx = self.chosen_chan.index
            elif type(idx) == str:
                if idx == 'all':
                    idx = self.df_chan.index
                else:
                    raise ValueError('Unsupported input.')
            else:
                if len(idx) > self.spikes_per_bin.shape[1]:
                    raise ValueError('Index exceeds number of channels in spikes_per_bin.')
            self.Y = self.spikes_per_bin[:, idx]
        elif self.matfile_type == 'hdf5':
            # For hdf5 files, we do not have cluster info, so we just use all channels.
            self.Y = self.spikes_per_bin.copy()

        return self
    
    def resample_y(self, random_state=None):
        """
        Resample Y to have same number of channels in HPC and ACC.
        First half is HPC, second half is ACC.
        """

        rng = np.random.default_rng(random_state)
        self.choose_channels().find_y() # Reset self.Y
        hpc_ix = self.chosen_chan[self.chosen_chan['region'] == 'HPC'].seq_index
        acc_ix = self.chosen_chan[self.chosen_chan['region'] == 'ACC'].seq_index
        n_hpc, n_acc = len(hpc_ix), len(acc_ix)

        if len(acc_ix) == 0 or len(hpc_ix) == 0:
            print(f'WARNING: No ACC or HPC channels found for resampling in {self.session_key}. Skipping resampling.')
            return self

        # Choose smaller number of channels
        n_choose = min(n_hpc, n_acc)
        hpc_choose = np.sort(rng.choice(hpc_ix, size=n_choose, replace=False))
        acc_choose = np.sort(rng.choice(acc_ix, size=n_choose, replace=False))
        self.hpc_resample_ix = hpc_choose
        self.acc_resample_ix = acc_choose
        chosen_seq_ix = np.hstack((hpc_choose, acc_choose))

        # seq_index is relative to self.chosen_chan / self.Y, but find_y expects the
        # original channel indices from self.spikes_per_bin.
        chosen_ix = self.chosen_chan.index.to_numpy()[chosen_seq_ix]
        self.find_y(chosen_ix)

        return self

    def get_spikes_mu_sd(self, verbose: bool = True):
        """
        Calculate the mean and standard deviation of spikes across time bins.
        """
        if self.Y is None:
            raise ValueError('Spikes must be loaded before calculating mu and sd. Call get_spikes() first.')

        self.spikes_mu = np.mean(self.Y, axis=0)
        self.spikes_sd = np.std(self.Y, axis=0)
        self.Y_norm = normalize(self.Y)
        return self

    def get_events(self, mode: str = 'events', max_gap_dur_secs: int = 1):
        """
        Load evets from the session file and filter the spikes based on the events.

        Basically the same as load_events() but with some additional processing.
        The mode can be 'events', 'gaps', or 'all'.

        Sets ``self.on_task`` to a boolean array aligned with the retained
        ``bin_times`` and rows of ``spikes_per_bin``.
        """

        assert mode in ['events', 'gaps', 'all'], f'Invalid mode: {mode}. Choose from "events", "gaps", or "all".'
        self.mode = mode
        self.max_gap_dur_secs = max_gap_dur_secs  

        match self.task:
            case 'podcast':
                self.events, self.event_times = load_words(self.subj)
            case 'pursuit':
                filepath = os.path.join(BEHAVIOR_DIR, self.task.lower(), self.filename + '.mat')
                self.trials, self.locations, self.events, self.event_times = load_pursuit_trials(filepath)

            case 'rotations':
                filepath = os.path.join(BEHAVIOR_DIR, self.task.lower(), self.filename + '.mat')
                self.events, self.event_times = load_rotation_trials(filepath)
        
        if len(self.event_times) == 0:
            print(f'WARNING: no events found for {self.task}. Skipping event filtering.')
            ix_events = np.zeros(self.spikes_per_bin.shape[0], dtype=bool)
            ix_keep = np.ones(self.spikes_per_bin.shape[0], dtype=bool)  # keep all bins
            
        else:
            ix_events = ix_during_events(self.event_times, self.bin_times, max_gap_dur_secs=self.max_gap_dur_secs)

            match self.mode:
                case 'events':
                    ix_keep = ix_events
                case 'gaps':
                    ix_keep = np.logical_not(ix_events)
                case 'all':
                    # keep all bins, but mark which ones are during events
                    ix_keep = np.ones_like(ix_events, dtype=bool)
        self.spikes_per_bin = self.spikes_per_bin[ix_keep,:]
        self.bin_times = self.bin_times[ix_keep]
        self.on_task = ix_events[ix_keep]

        if self.verbose:
            print(f'{self}: {len(self.event_times)} events, "{self.mode}" mode. Keep {np.sum(ix_keep)} out of {len(ix_keep)} bins.')

    def find_subspace(self, metric = None, verbose: bool = True):

        # Do this as default
        self.get_spikes_mu_sd(verbose=False) # Call this every time because the dim of Y can change.
        self.FA = FAResults()
        self.FA.fa_fit(self.Y_norm)

        if metric == 'randomized':
            Y_rand = permute_columns_independently(self.Y_norm)
            self.FA_rand = FAResults()
            self.FA_rand.fa_fit(Y_rand, n_components=self.FA.d_shared, max_iter=int(1e5))
        elif type(metric) == np.ndarray: # Find subspace for other metric
            if verbose:
                print(f'{self}, array to FA: {metric.shape}.')
            try:
                # Return instead of setting to a fixed attribute
                # Can still do s.some_attribute = s.find_subspace(metric=some_array)
                result = FAResults()
                return result.fa_fit(normalize(metric))
            
            # Since the given metric is not guaranteed FA-able, raise error if needed.
            except Exception as e:
                print(f'Error finding subspace: {e}')
        return self

    def resample_ix_for_equal_channel_counts(self, seed: int = 123):

        if self.n_acc == self.n_hpc: # No need to resample
            return
        if (self.n_acc == 0) or (self.n_hpc == 0): # No need to resample if one of the regions has no channels
            return

        more_channel = 'hpc' if self.n_hpc > self.n_acc else 'acc' # Region that has more channel
        to_permute = getattr(self, more_channel + '_ix') # That region is the one to permute

        np.random.seed(seed)
        n_channels = np.min((self.n_hpc, self.n_acc)) # Resample based on the region of fewer channel counts
        ix = np.random.permutation(np.where(to_permute)[0])[:n_channels]

        setattr(self, more_channel + '_ix_perm', # Should be self.hpc_ix_perm or self.acc_ix_perm
                np.array([True if i in ix else False for i in range(len(to_permute))]))
        
        assert self.n_acc + self.n_hpc == self.n_total, ValueError(self)

        # Make sure the channel counts in resampled one is equal to the other one.
        # Make sure after resampling the sum is less than total channel counts.
        if hasattr(self, 'hpc_ix_perm'):
            assert np.sum(self.hpc_ix_perm) == self.n_acc, ValueError(self)
            assert np.sum(self.hpc_ix_perm) + self.n_acc < self.n_total, ValueError(self)
        elif hasattr(self, 'acc_ix_perm'):
            assert self.n_hpc == np.sum(self.acc_ix_perm), ValueError(self)
            assert self.n_hpc + np.sum(self.acc_ix_perm) < self.n_total, ValueError(self)

    def resample(self, seed: int = 123):

        self.resample_ix_for_equal_channel_counts(seed=seed)

        if hasattr(self, 'hpc_ix_perm'):
            self.Y_resample = np.concatenate((self.Y[:,self.hpc_ix_perm], self.Y[:,self.acc_ix]),axis=1)
            self.region_code = ['HPC'] * np.sum(self.hpc_ix_perm) + ['ACC'] * self.n_acc
        elif hasattr(self, 'acc_ix_perm'):
            self.Y_resample = np.concatenate((self.Y[:,self.hpc_ix], self.Y[:,self.acc_ix_perm]),axis=1)
            self.region_code = ['HPC'] * self.n_hpc + ['ACC'] * np.sum(self.acc_ix_perm)
        else:
            self.Y_resample = self.Y
            self.region_code = ['HPC' if i==True else 'ACC' for i in self.hpc_ix]

        assert len(self.region_code) == self.Y_resample.shape[1], ValueError(f'{self} resample error')
        self.region_code = np.array(self.region_code)
        self.resample_flag = True

# %%
TARGET_LABEL_TO_REGION = {'H': 'HPC','C': 'ACC','A': 'AMY','PH': 'PHG','I': 'INS','OF': 'OFC','OT': 'OTG'}
SENSA_CHANNEL_PATTERN = re.compile(
    r'^m(?P<hemi>[LR])'
    r'(?P<entry>[FPOT]\d+[a-z]?)'
    r'(?P<targets>(?:PH|OF|OT|[A-Z][a-z]?)+)'
    r'(?P<contact>\d+)-(?P<channel>\d+)$'
)

def parse_sensa_channel_name(channel_name):
    """
    Parse SENSA channel names into hemisphere and terminal target region.

    This follows the naming rules in SENSA_mapping_rules.txt:
    side is encoded by L/R, and the distal target label determines region.
    Non-neural analog inputs return NaN values.
    """
    if not isinstance(channel_name, str):
        return pd.Series({'hemi': np.nan, 'region': np.nan})

    match = SENSA_CHANNEL_PATTERN.match(channel_name)
    if match is None:
        return pd.Series({'hemi': np.nan, 'region': np.nan})

    hemisphere = 'left' if match.group('hemi') == 'L' else 'right'
    target_string = match.group('targets')

    # Targets can be chained (proximal -> distal), so keep the last target.
    target_labels = re.findall(r'PH|OF|OT|[A-Z]', target_string)
    target_label = target_labels[-1] if target_labels else None
    region = TARGET_LABEL_TO_REGION.get(target_label, np.nan)

    return pd.Series({'hemi': hemisphere, 'region': region})


#%% Raw recording and acoustic-energy helpers

NS5_DIR = REPO_ROOT / 'data' / 'ns5'
AUDIO_CHANNEL_NAMES = ('Audio', 'RoomMic1', 'RoomMic2', 'PS4_audio')

def load_blackrock_recording(path_to_file):
    """Load a Blackrock recording, importing SpikeInterface only when needed."""
    try:
        import spikeinterface.extractors as se
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'spikeinterface is required to load Blackrock recordings. '
            'Activate the spectral-subspace or subspace conda environment.'
        ) from exc
    return se.read_blackrock(path_to_file)


def get_channel_id(recording, channel_name: str) -> str:

    channels = pd.DataFrame({
        'channel_id': recording.channel_ids.astype(str),
        'channel_name': recording.get_property('channel_name').astype(str),
        'physical_unit': recording.get_property('physical_unit').astype(str),
        'gain_to_uV': recording.get_property('gain_to_uV'),
    })

    matches = channels.loc[channels.channel_name == channel_name, 'channel_id']
    if len(matches) != 1:
        available = ', '.join(channels.channel_name)
        raise ValueError(
            f'Expected exactly one channel named {channel_name!r}; found {len(matches)}. '
            f'Available channels: {available}'
        )
    return matches.iloc[0]


def get_audio_timeseries(recording,
                         channel_name: str = 'RoomMic1',
                         start_time_s: float = 0,
                         duration_s: float = 10,
                         demean: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Load a bounded audio waveform.

    Returns ``times_s, signal_uV, sampling_frequency``. Times are relative to
    the start of the NSx recording and therefore use the same origin as the
    matching millisecond spike matrix.
    """

    ch_id = get_channel_id(recording, channel_name)

    sampling_frequency = recording.get_sampling_frequency()
    total_duration_s = recording.get_total_duration()
    if duration_s is None: duration_s = total_duration_s - start_time_s
    stop_time_s = min(start_time_s + duration_s, total_duration_s)

    if start_time_s < 0 or duration_s <= 0 or start_time_s >= total_duration_s:
        raise ValueError(
            f'Requested interval ({start_time_s}, {stop_time_s}) is outside '
            f'the recording duration (0, {total_duration_s}).'
        )

    start_frame = round(start_time_s * sampling_frequency)
    end_frame = round(stop_time_s * sampling_frequency)
    signal_uV = recording.get_traces(
        start_frame=start_frame,
        end_frame=end_frame,
        channel_ids=[ch_id],
        return_in_uV=True,
    ).squeeze(axis=1)

    if demean:
        signal_uV = signal_uV - np.median(signal_uV)

    times_s = np.arange(start_frame, end_frame) / sampling_frequency
    return times_s, signal_uV, sampling_frequency


def get_rms_envelope(recording,
                     channel_name: str = 'RoomMic1',
                     window_s: float = 0.1,
                     chunk_duration_s: float = 60) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate a full-session RMS audio envelope without loading all samples.

    A local mean is removed within each window so that analog-channel DC
    offsets do not appear as acoustic energy. Returned times mark window
    centers relative to the start of the NSx recording.
    """

    ch_id = get_channel_id(recording, channel_name)

    sampling_frequency = recording.get_sampling_frequency()
    window_frames = round(window_s * sampling_frequency)
    windows_per_chunk = max(1, round(chunk_duration_s / window_s))
    n_windows = recording.get_num_samples() // window_frames
    rms_uV = np.empty(n_windows, dtype=float)

    for first_window in range(0, n_windows, windows_per_chunk):
        n_chunk_windows = min(windows_per_chunk, n_windows - first_window)
        start_frame = first_window * window_frames
        end_frame = start_frame + n_chunk_windows * window_frames
        signal_uV = recording.get_traces(
            start_frame=start_frame,
            end_frame=end_frame,
            channel_ids=[ch_id],
            return_in_uV=True,
        ).squeeze(axis=1)
        signal_uV = signal_uV.reshape(n_chunk_windows, window_frames)
        signal_uV = signal_uV - signal_uV.mean(axis=1, keepdims=True)
        rms_uV[first_window:first_window + n_chunk_windows] = np.sqrt(
            np.mean(signal_uV ** 2, axis=1)
        )

    times_s = (np.arange(n_windows) + 0.5) * window_s
    return times_s, rms_uV


def get_rms_at_bin_times(recording,
                         bin_times_ms: np.ndarray,
                         bin_size_ms: float,
                         channel_name: str = 'RoomMic1',
                         demean: bool = True,
                         chunk_duration_s: float = 60) -> np.ndarray:
    """
    Aggregate an audio channel directly into neural time bins using AC RMS.

    For every start time in ``bin_times_ms``, samples in the interval
    ``[bin_time, bin_time + bin_size_ms)`` are reduced to one RMS value. This
    works for both contiguous bins and event-filtered, non-contiguous bins and
    returns an array with exactly the same shape as ``bin_times_ms``.
    """
    bin_times_ms = np.asarray(bin_times_ms)
    if bin_times_ms.ndim != 1:
        raise ValueError('bin_times_ms must be one-dimensional.')
    if len(bin_times_ms) == 0:
        return np.empty(bin_times_ms.shape, dtype=float)
    if np.any(np.diff(bin_times_ms) <= 0):
        raise ValueError('bin_times_ms must be strictly increasing.')
    if bin_size_ms <= 0:
        raise ValueError('bin_size_ms must be positive.')

    sampling_frequency = recording.get_sampling_frequency()
    frames_per_bin_float = bin_size_ms * sampling_frequency / 1000
    frames_per_bin = round(frames_per_bin_float)
    if not np.isclose(frames_per_bin, frames_per_bin_float):
        raise ValueError(
            f'{bin_size_ms} ms does not contain an integer number of samples '
            f'at {sampling_frequency} Hz.'
        )

    start_frames_float = bin_times_ms * sampling_frequency / 1000
    start_frames = np.rint(start_frames_float).astype(np.int64)
    if not np.allclose(start_frames, start_frames_float):
        raise ValueError('Some bin start times do not align with audio samples.')
    if start_frames[0] < 0 or start_frames[-1] + frames_per_bin > recording.get_num_samples():
        raise ValueError(
            'At least one requested neural bin falls outside the audio recording.'
        )

    channel_id = get_channel_id(recording, channel_name)
    max_bins_per_chunk = max(1, int(chunk_duration_s * 1000 // bin_size_ms))
    rms_uV = np.empty(bin_times_ms.shape, dtype=float)

    first_bin = 0
    while first_bin < len(start_frames):
        stop_bin = first_bin + 1
        max_stop_bin = min(len(start_frames), first_bin + max_bins_per_chunk)
        while (
            stop_bin < max_stop_bin
            and start_frames[stop_bin] == start_frames[stop_bin - 1] + frames_per_bin
        ):
            stop_bin += 1

        traces_uV = recording.get_traces(
            start_frame=start_frames[first_bin],
            end_frame=start_frames[stop_bin - 1] + frames_per_bin,
            channel_ids=[channel_id],
            return_in_uV=True,
        ).squeeze(axis=1)
        traces_uV = traces_uV.reshape(stop_bin - first_bin, frames_per_bin)
        if demean:
            traces_uV = traces_uV - traces_uV.mean(axis=1, keepdims=True)
        rms_uV[first_bin:stop_bin] = np.sqrt(np.mean(traces_uV ** 2, axis=1))
        first_bin = stop_bin

    return rms_uV
