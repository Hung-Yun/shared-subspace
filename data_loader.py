import glob
import os.path
import json
import numpy as np
import pandas as pd
import h5py
import scipy.io
from collections import namedtuple
from collections import Counter
from scipy.sparse import csc_matrix
from datetime import datetime
import fa

DATADIR = os.path.join('data', 'sessions')
SESSIONS_FILE = os.path.join('data', 'sessions.csv')
BRAIN_AREAS_JSON = os.path.join('data', 'brain_areas.json')
PODCAST_WORDS_FILE = os.path.join('data', 'sessions', 'podcast', 'HPC_YEXthurYFI_WordFRMatrix_300Post80msDelay.mat') # Containing all the words/event times for all patients

def load_sessions(sessionfile=SESSIONS_FILE, datadir=DATADIR):
    """
    example of sessionfile (csv):
        Patient,EMU,Task,Date,Time,Ranking
        YEX,96,Podcast,2024-02-08,3:55:00 PM,2
        YEX,99,Podcast,2024-02-08,4:36:00 PM,1
        YEY,72,Podcast,2024-03-31,12:21:00 PM,
        YEZ,40,Podcast,2024-04-10,3:55:00 PM,
        ...
        YEX,81,Pursuit,2024-02-07,4:41:59 PM
    """
    df = pd.read_csv(sessionfile)
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y-%m-%d %I:%M:%S %p')
    df['session_file'] = df.apply(lambda row: row_to_session_path(row, datadir), axis=1)

    for i, row in df.iterrows():
        if not os.path.exists(row['session_file']):
            print(f'WARNING: session file {row["session_file"]} does not exist. Removing row {i}.')
            df.drop(i, inplace=True)

    task_key = {'Podcast': 'C', 'Pursuit': 'S', 'Rotations': 'R'}
    patient_f = lambda x: x[-1] if isinstance(x, str) else x
    rank_f = lambda x: str(int(x)) if pd.notna(x) else '1'

    df['session_key'] = df.apply(lambda row: patient_f(row['Patient']) + '-' + task_key.get(row['Task']) + rank_f(row['Ranking']), axis=1)
    return df

def row_to_session_path(row, datadir=DATADIR):
    """
    converts a row from the sessions DataFrame to a session file path
    """
    session_dir = os.path.join(datadir, row['Task'].lower())
    filename = f'{row["Patient"]}_{str(row["EMU"]).zfill(4)}_{row["Task"].lower()}'
    return os.path.abspath(os.path.join(session_dir, filename + '.mat'))

def get_session_by_subj_task(subj, task, sessionfile=SESSIONS_FILE):
    """
    returns the session for a given subject and task
    """
    df = load_sessions(sessionfile)
    dfc = df[(df['Patient'] == subj.upper()) & (df['Task'] == task.capitalize())]
    if len(dfc) == 0:
        return None
    if len(dfc) > 1:
        # here we pick the session with smallest value of dfc['Ranking']
        row = dfc.loc[dfc['Ranking'].idxmin()] 
    else:
        # when there is only one session, choose that one
        row = dfc.iloc[0]
    return row.to_dict()

def get_session_names(task_name, brain_areas=None):
    """
    returns all sessions for a given task_name (e.g., 'Podcast') (optional: with channels in brain_areas)
    """
    df = load_sessions()
    dfc = df[df['Task'] == task_name.capitalize()]
    sessions = []
    for i, row in dfc.iterrows():
        if brain_areas is not None:
            # check if session has channels in brain_areas
            channels = get_subj_channels(row['Patient'])
            chans_in_area = get_subj_channels_in_area(channels, brain_areas)
            if len(chans_in_area) == 0:
                continue
        # if we got here, we have a valid session (return as dict)
        sessions.append(row.to_dict())
    print(f'Found {len(sessions)} sessions for task {task_name} with brain areas {brain_areas}.')
    return sessions

def get_session_pairs(task_names, brain_areas=None, max_hours_apart=30, ignore_same_task=True):
    """
    finds all pairs of sessions from same patient for tasks in task_names (optional: with channels in brain_areas)
        - each item in pair is [subj, emu_number, task_name]
    """
    df = load_sessions()
    # iterate through rows to find pairs of sessions
    pairs = []
    for i, row1 in df.iterrows():
        if row1['Task'].lower() not in task_names:
            continue
        for j, row2 in df.iterrows():
            if i <= j or row2['Task'].lower() not in task_names:
                continue
            if ignore_same_task and row1['Task'] == row2['Task']:
                continue
            if row1['Patient'] != row2['Patient']:
                continue
            if np.abs((row1['Datetime'] - row2['Datetime']).total_seconds() / 3600) > max_hours_apart:
                continue
            if brain_areas is not None:
                # check if both sessions have channels in brain_areas
                channels1 = get_subj_channels(row1['Patient'])
                channels2 = get_subj_channels(row2['Patient'])
                chans_in_area1 = get_subj_channels_in_area(channels1, brain_areas)
                chans_in_area2 = get_subj_channels_in_area(channels2, brain_areas)
                if len(chans_in_area1) == 0 or len(chans_in_area2) == 0:
                    continue
            
            # if we got here, we have a valid pair (return as dicts)
            if row1['Task'] == 'Rotations':
                pairs.append((row2.to_dict(), row1.to_dict()))
            else:
                pairs.append((row1.to_dict(), row2.to_dict()))
    print(f'Found {len(pairs)} pairs for tasks {task_names} with brain areas {brain_areas} within {max_hours_apart} hours.')
    return pairs

def load_spikes(infile):
    """
    loads spikes from mat file containing the spikes as a sparse matrix
    """
    f = h5py.File(infile)
    data = f['spikes']['data'][:]
    ir = f['spikes']['ir'][:]
    jc = f['spikes']['jc'][:]
    N = int(max(ir)) + 1  # Number of neurons (rows)
    T = len(jc) - 1  # Number of time bins (columns)
    spikes = csc_matrix((data, ir, jc), shape=(N, T))
    return spikes, f

def binned_spike_counts(spikes, bin_size=50, scale_to_hz=True):
    """
    inputs:
    - spikes: (nneurons x time) sparse matrix
    - bin_size: int, in ms
    """
    t_to_drop = spikes.shape[1] % bin_size # trim off empty bins at end

    # reshape spikes to be n_units x n_timebins x bin_size
    S = spikes[:,:spikes.shape[1]-t_to_drop].toarray().reshape(spikes.shape[0], (spikes.shape[1]-t_to_drop) // bin_size, bin_size)    
    spikes_per_bin = S.sum(axis=2)
    if scale_to_hz:
        spikes_per_bin = spikes_per_bin * (1000 / bin_size) # convert to Hz
    bin_times = bin_size * np.arange(spikes_per_bin.shape[1])
    return spikes_per_bin, bin_times

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

def get_subj_channels_in_area(channels_by_area, brain_areas):
    """
    channels_by_area is a dict mapping brain area names to lists of channel numbers
    brain_areas is a list of brain area names (e.g., ['HPC', 'ACC'])
    returns a list of channel numbers that are in any of the brain_areas
    """
    channels_to_keep = []
    for key, chans in channels_by_area.items():
        if np.any([brain_area in key for brain_area in brain_areas]):
            channels_to_keep.extend(chans)
    return channels_to_keep

def filter_brain_areas(subj, spikes, channels, brain_areas, silent=True):
    """
    subj is a str, the subject identifier (e.g., 'YEX')
    spikes is a sparse matrix of shape (nneurons x time)
    channels is a 1D array of channel numbers corresponding to the neurons in spikes
    brain_areas is a list of brain area names (e.g., ['HPC', 'ACC'])

    returns spikes and channels, filtered to only include channels in brain_areas
    """
    channels_by_area = get_subj_channels(subj)
    channels_to_keep = get_subj_channels_in_area(channels_by_area, brain_areas)
    if len(channels_to_keep) == 0:
        if not silent:
            raise Exception(f'No channels for {subj} matching area {brain_areas}')
        else:
            print(f'No channels for {subj} matching area {brain_areas}')
    
    ix = np.isin(channels, np.array(channels_to_keep))

    # In some cases, channels_to_keep is equal to channels[ix], but not guaranteed.
    # To make sure that we are indexing the channels we are sure in the channels (f['chan'][:][:,0].astype(int))
    # We should use channels[ix] and spikes[ix].

    return spikes[ix], channels[ix]

def load_session(session_info, bin_size=50, ignore_sorting=True, verbose=True, brain_areas=None, make_channels_consecutive=True, mode='events', max_gap_dur_secs=1):
    """
    session_info is a dict with keys:
        - 'Patient': subject identifier (e.g., 'YEX')
        - 'Task': task name (e.g., 'Podcast')
        - 'EMU': EMU number (e.g., 96)
        - 'session_file': path to the session file
    """
    if session_info is None:
        return None
    subj = session_info['Patient']
    task = session_info['Task'].lower()

    # load spikes file
    spikes, f = load_spikes(session_info['session_file'])
    channels = f['chan'][:][:,0].astype(int)
    if make_channels_consecutive:
        # in spreadsheet, channel numbers act like probes are not missing
        # but wave_clus code returns channel numbers relative to lowest channel
        # so here we need to make them consecutive
        # e.g., if channels are [5,6,7,1,2,3] we want [4,5,6,1,2,3]
        # note that if channels is already a permutation, nothing will change
        channels = np.argsort(np.argsort(channels))+1
    if brain_areas is not None:
        nchannels_prev = len(np.unique(channels))
        spikes, channels = filter_brain_areas(subj, spikes, channels, brain_areas)
        nchannels = len(np.unique(channels))
        if verbose:
            print(f'{subj=}: Keeping {nchannels} of {nchannels_prev} channels matching brain areas {brain_areas}')

    # count spikes in nonoverlapping bins
    Y, bin_times = binned_spike_counts(spikes, bin_size=bin_size)
    Y = Y.T # (T x N)

    if mode in ['events', 'gaps', 'all']:
        if task == 'podcast':
            events, event_times = load_words(subj)
        elif task == 'pursuit':
            events, event_times = load_pursuit_trials(session_info['session_file'])
        elif task == 'rotations':
            events, event_times = load_rotation_trials(session_info['session_file'])
        if len(event_times) == 0:
            print(f'WARNING: no events found for {task}. Skipping event filtering.')
            ix_events = []
        else:
            # if event_times[-1][1]-1000 > bin_times[-1]:
            #     raise Exception(f'Last event time {event_times[-1][1]} is after last bin time {bin_times[-1]}. Please check the session file {session_info["session_file"]}.')
            ix_events = ix_during_events(event_times, bin_times, max_gap_dur_secs=max_gap_dur_secs)
            if mode == 'events':
                ix_keep = ix_events
            elif mode == 'gaps':
                ix_keep = np.logical_not(ix_events)
            elif mode == 'all':
                # keep all bins, but mark which ones are during events
                ix_keep = np.ones_like(ix_events, dtype=bool)
            if verbose:
                print(f'{subj=}, {task=}: {len(events)} events, {ix_events.sum()} bins during events, {len(bin_times)-ix_events.sum()} bins during gaps. Proportion of time bins during events: {ix_events.mean():0.3f}. Keeping {ix_keep.sum()} bins.')
            Y = Y[ix_keep,:]
            bin_times = bin_times[ix_keep]
    else:
        events = None
        event_times = None
        ix_events = None

    if ignore_sorting:
        # sum up all spikes on same channel
        unique_channels = np.unique(channels)
        Y_combined = np.zeros((Y.shape[0], len(unique_channels)))
        for i, channel in enumerate(unique_channels):
            Y_combined[:,i] = Y[:,channels == channel].sum(axis=1)
        Y = Y_combined
        channels = unique_channels

    # get brain areas
    probe_inds = (channels-1) // 8
    brain_areas = get_subj_channels(subj)
    
    # get mean and sd of spikes per unit
    spikes_mu = np.mean(Y, axis=0) # (N,)
    spikes_sd = np.std(Y, axis=0) # (N,)
    Y_norm = (Y - spikes_mu[None,:]) / spikes_sd[None,:] # (T x N)
    ix = spikes_sd > 0 

    if verbose:
        print(f'Loaded {subj=}, {task=}: spikes.shape={Y.shape}')
    session = {'subj': subj, 'task': task, 'session_info': session_info, 'f': f, 'spikes': spikes, 
               'channels': channels, 'probe_inds': probe_inds, 'brain_areas': brain_areas, 
               'Y': Y, 'bin_times': bin_times, 'spikes_mu': spikes_mu, 'spikes_sd': spikes_sd, 
               'Y_norm': Y_norm, 'ix': ix, 'events': events, 'event_times': event_times, 
               'ix_events': ix_events, 'mode': mode}
    return session

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

def filter_channels(results, chans_to_keep):
    """
    removes data from any channel that isn't in chans_to_keep
    """
    for result in results:
        ix = np.isin(result['channels'], chans_to_keep)
        if ix.sum() < len(ix):
            print(f'Filtering {result["subj"]} {result["task"]}: keeping {ix.sum()} of {len(ix)} channels')
        result['ix_total'] = ix
        result['Y'] = result['Y'][:,ix]
        result['Y_norm'] = result['Y_norm'][:,ix]
        result['channels'] = result['channels'][ix]
        result['probe_inds'] = result['probe_inds'][ix]
        result['spikes_mu'] = result['spikes_mu'][ix]
        result['spikes_sd'] = result['spikes_sd'][ix]
    return results

def filter_matching_channels(sessions):
    """
    removes data from any channel that isn't present in all results
    """    
    all_chans = [session['channels'][session['ix']] for session in sessions]
    common_chans = intersect_all(all_chans)
    print(f'found {len(common_chans)} common channels from {[len(x) for x in all_chans]} channels')
    return filter_channels(sessions, common_chans)

def subsample_channels(sessions, n_units, seed=123):
    """
    randomly subsample the number of units in each session to match the number of units in unit_counts
    """
    if n_units == 0:
        return []
    cur_chans = sessions[0]['channels']
    if len(cur_chans) < n_units:
        # keep session as-is if we have fewer channels than n_units
        return sessions

    rng = np.random.default_rng(seed)
    chans_to_keep = rng.permuted(cur_chans)[:n_units]
    print(f'subsampling {n_units} of {len(cur_chans)} channels')
    return filter_channels(sessions, chans_to_keep)

# def remove_sessions_too_far_apart(results, max_hours=30):
#     dts = [session['datetime'] for session in results]
#     inds = np.argsort(dts)
#     dt_min = dts[inds[0]]
#     scale_to_hours = 1 / (60*60.)
#     filtered_results = [session for session in results if scale_to_hours*(session['datetime'] - dt_min).total_seconds() <= max_hours]
#     if len(filtered_results) != len(results):
#         print(f'Removed {len(results) - len(filtered_results)} sessions that started more than {max_hours} apart.')
#     return filtered_results

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

#%% event helper

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
    t1 = times[i+1][1]
    ts.append((t1, int(1e7)))
    return ts

def ix_during_events(event_times, bin_times, max_gap_dur_secs=1):
    """
    event_times is a list of (start_time, end_time) tuples of event times in ms
    bin_times is a 1D array of bin times in ms
    max_gap_dur_secs is the maximum duration of a gap to be considered significant

    returns a boolean array of the same length as bin_times, marking the bins containing event_times
    """
    event_gaps = big_time_gaps(event_times, max_gap_dur_secs=max_gap_dur_secs)
    ix = np.ones_like(bin_times).astype(bool)
    for (t_start, t_end) in event_gaps:
        t_start_bin = np.argmin((bin_times - t_start)**2)
        t_end_bin = np.argmin((bin_times - t_end)**2)
        ix[t_start_bin:(t_end_bin+1)] = False
    return ix

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

#%% Load events/trials

def load_events(session_file):
    """
    loads the events object corresponding to the session_file
    session_file is the path to the session file (e.g., 'data/sessions/YEX_96_Podcast.mat')
    """
    basedir = os.path.dirname(session_file)
    eventsdir = os.path.join(basedir, 'events')
    fnm = os.path.basename(session_file).replace('.mat', '_events.mat')
    if not os.path.exists(eventsdir):
        raise Exception(f'Events directory {eventsdir} does not exist. Please check the session file {session_file}.')
    events_file = os.path.join(eventsdir, fnm)
    f = scipy.io.loadmat(events_file)
    return f

# load Rotation trials
def load_rotation_trials(session_file):
    f = load_events(session_file)
    if 'nevTimeOffset' not in f or 'ns5TimeOffset' not in f:
        print(f'WARNING: nevTimeOffset or ns5TimeOffset not found in {session_file}. Please check the session file.')
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
        raise Exception(f'Inconsistent trial start and end times in {session_file}. Please check the session file.')

    # note: times are in seconds, so we convert to milliseconds
    ft = lambda x: (x - t_offset) * 1000 # convert to milliseconds
    event_times = list(zip(ft(trial_start_times), ft(trial_end_times)))
    return f, event_times

# load Pursuit trials
def load_pursuit_trials(session_file):
    """
    example:
       trial_start: 73441
         iti_start: 73460
           iti_end: 74100
        wait_start: 74112
       chase_start: 75077
         chase_end: 77697
    feedback_start: 78050
         trial_end: 79567
    """
    try:
        f = load_events(session_file)
    except FileNotFoundError:
        print(f'WARNING: events file not found for {session_file}. Please check the session file.')
        return None, []
    trials = f['events_info'][0,:]
    # n.b. trial start includes ITI, and trial_end includes feedback period

    # make dataframe of event times
    event_order = ['wait_start', 'chase_start', 'chase_end', 'feedback_start', 'trial_end']
    event_times = {}
    for event_name in event_order:
        event_times[event_name] = np.array([x[0,0] for x in trials[event_name]])
    df = pd.DataFrame(event_times)
    event_times = list(zip(df['wait_start'].values, df['trial_end'].values))
    return df, event_times

# load Podcast words
def load_words(subj, words_file=PODCAST_WORDS_FILE):
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

def report_all_sessions(sessions):

    print(f'Total sessions loaded: {sum(len(s) for s in sessions.values())}')
    for task_name, task_sessions in sessions.items():
        print(f'Task: {task_name}')
        keys = set()
        for session_key, (S1, S2) in task_sessions.items():
            print(f'  {session_key}. Shapes - S1: {S1["Y"].shape}, S2: {S2["Y"].shape}')
            keys.add(session_key[0]) # Patient ID
        print(f'Unique patients for task {task_name}: {len(keys)}') 

#%% classes

class FactorAnalysisResult:
    def __init__(self, fit_result: dict):
        for key, value in fit_result.items():
            setattr(self, key, value)

    def __repr__(self):
        return f'{self.d_shared} components from {self.n_components} channels'

class Session:
    """
    Turn a session dictionary into a class for easier access to attributes.
    Building from load_session().
    """

    def __init__(self, session_info: dict):

        # Known attributes
        self.session_info = session_info
        self.subj = session_info['Patient']
        self.task = session_info['Task'].lower()
        self.datetime = session_info['Datetime']
        self.session_file = session_info['session_file']
        self.session_key = session_info['session_key']
        self.brain_areas_in_session = get_subj_channels(subj=self.subj)
        self.available_areas = None

        # Attributes for analysis
        self.f = None
        self.spikes = None
        self.channels = None
        self.unique_channels = None
        self.brain_areas_for_analysis = None
        self.Y = None
        self.bin_size = None
        self.bin_times = None
        self.spikes_mu = None
        self.spikes_sd = None
        self.Y_norm = None
        self.ix = None
        self.events = None
        self.event_times = None
        self.mode: str = None
        self.max_gap_dur_secs = None
        self.make_channels_consecutive = True
        self.ignore_sorting = True
        self.verbose = True
        self.probe_inds = None

        # For resampling
        self.Y_resample = None
        self.region_code = None
        self.resample_flag = False

    def __repr__(self):
        return f'Session {self.session_key}'
    
    def filter_brain_areas(self, brain_areas: list[str], silent: bool =True):
        """
        brain_areas is a list of brain area names (e.g., ['HPC', 'ACC'])
        """
        channels_by_area = get_subj_channels(self.subj)
        channels_to_keep = get_subj_channels_in_area(channels_by_area, brain_areas)
        if len(channels_to_keep) == 0:
            if not silent:
                raise Exception(f'No channels for {self.subj} matching area {brain_areas}')
        
        # Here we are using the latest self.channels >> This also applies when we filter out low FR channels.
        ix = np.isin(self.channels, np.array(channels_to_keep))
        return ix
    
    @property
    def hpc_ix(self):
        return self.filter_brain_areas(['HPC'])
    
    @property
    def n_hpc(self):
        return np.sum(self.hpc_ix)
    
    @property
    def acc_ix(self):
        return self.filter_brain_areas(['ACC'])
    
    @property
    def n_acc(self):
        return np.sum(self.acc_ix)
    
    @property
    def n_total(self):
        return len(self.channels)

    def get_spikes(self, binsize: int = 50, brain_areas_for_analysis: list[str]|tuple[str] = None, strict_brain_areas = False):
        """
        Load spikes from the session file and process them.

        Basically the same as load_spikes() but with some additional processing.
        And making this more compatible with the Session class.

        strict_brain_areas: whether reading brain areas that are strictly defined in brain_areas_for_analysis
        """
        
        self.bin_size = binsize
        self.brain_areas_for_analysis = brain_areas_for_analysis
        # check if self.brain_areas_for_analysis is in self.brain_areas_in_session
        self.available_areas = set(i.split(' ')[0] for i in self.brain_areas_in_session.keys())

        if self.brain_areas_for_analysis is None:
            self.brain_areas_for_analysis = {'HPC','ACC'}

        if strict_brain_areas:
            if not set(self.brain_areas_for_analysis).issubset(self.available_areas):
                raise ValueError(f'Brain areas {self.brain_areas_for_analysis} not found in session {self.subj}. Available areas: {self.available_areas}')
        else:
            # If not strict, use the intersection of available areas and areas intended for analyses.
            self.brain_areas_for_analysis = list( self.available_areas & set(self.brain_areas_for_analysis))

        self.spikes, self.f = load_spikes(self.session_info['session_file'])
        channels = self.f['chan'][:][:,0].astype(int)
        if self.make_channels_consecutive:
            # in spreadsheet, channel numbers act like probes are not missing
            # but wave_clus code returns channel numbers relative to lowest channel
            # so here we need to make them consecutive
            # e.g., if channels are [5,6,7,1,2,3] we want [4,5,6,1,2,3]
            # note that if channels is already a permutation, nothing will change
            self.channels = np.argsort(np.argsort(channels))+1
        if self.brain_areas_for_analysis is not None:
            nchannels_prev = len(np.unique(channels))
            spikes, self.channels = filter_brain_areas(self.subj, self.spikes, self.channels, self.brain_areas_for_analysis)
            nchannels = len(np.unique(self.channels))
            if self.verbose:
                print(f'{self}: Keeping {nchannels} of {nchannels_prev} channels matching brain areas {self.brain_areas_for_analysis}')

        # count spikes in nonoverlapping bins
        Y, self.bin_times = binned_spike_counts(spikes, bin_size=self.bin_size)
        self.Y = Y.T # (T x N)

    def get_events(self, mode: str = 'events', max_gap_dur_secs: int = 1):
        """
        Load evets from the session file and filter the spikes based on the events.

        Basically the same as load_events() but with some additional processing.
        The mode can be 'events', 'gaps', or 'all'.
        """

        assert mode in ['events', 'gaps', 'all'], f'Invalid mode: {mode}. Choose from "events", "gaps", or "all".'
        self.mode = mode
        self.max_gap_dur_secs = max_gap_dur_secs
        if self.Y is None:
            raise ValueError('Spikes must be loaded before events. Call get_spikes() first.')  

        match self.task:
            case 'podcast':
                self.events, self.event_times = load_words(self.subj)
            case 'pursuit':
                self.events, self.event_times = load_pursuit_trials(self.session_file)
            case 'rotations':
                self.events, self.event_times = load_rotation_trials(self.session_file)
        
        if len(self.event_times) == 0:
            print(f'WARNING: no events found for {self.task}. Skipping event filtering.')
            ix_events = []
            ix_keep = self.Y.shape[0] * [True]  # keep all bins
            
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
        self.Y = self.Y[ix_keep,:]
        self.bin_times = self.bin_times[ix_keep]

        if self.verbose:
            print(f'{self}: {len(self.event_times)} events, "{self.mode}" mode. Keep {np.sum(ix_keep)} out of {len(ix_keep)} bins.')
        
        if self.ignore_sorting:
            # sum up all spikes on same channel
            self.unique_channels = np.unique(self.channels)
            Y_combined = np.zeros((self.Y.shape[0], len(self.unique_channels)))
            for i, channel in enumerate(self.unique_channels):
                Y_combined[:,i] = self.Y[:,self.channels == channel].sum(axis=1)
            self.Y = Y_combined
            self.channels = self.unique_channels

        self.probe_inds = (self.channels-1) // 8 # ?

    def normalize(self, metric: np.ndarray):
        
        if metric.ndim != 2:
            print('May want to check the array dimension.')

        mean = np.mean(metric, axis=0) # (N,)
        std = np.std(metric, axis=0) # (N,)
        norm = (metric - mean[None,:]) / std[None,:] # (T x N)
        norm = np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0) # Replace NaN and inf with 0
        return norm

    def get_spikes_mu_sd(self, verbose: bool = True):
        """
        Calculate the mean and standard deviation of spikes across time bins.
        """
        if self.Y is None:
            raise ValueError('Spikes must be loaded before calculating mu and sd. Call get_spikes() first.')

        self.spikes_mu = np.mean(self.Y, axis=0)
        self.spikes_sd = np.std(self.Y, axis=0)
        self.Y_norm = self.normalize(self.Y)

    def find_subspace(self, metric: np.ndarray = None, verbose: bool = True):

        if metric is None: # Normal case
            if self.Y_norm is None:
                self.get_spikes_mu_sd(verbose=False)
            self.FA = FactorAnalysisResult(fa.fa_fit(self.Y_norm))

        else: # Find subspace for other metric
            if verbose:
                print(f'{self}, array to FA: {metric.shape}.')
            try:
                # Return instead of setting to a fixed attribute
                # Can still do s.some_attribute = s.find_subspace(metric=some_array)
                return FactorAnalysisResult(fa.fa_fit(self.normalize(metric))) 
            
            # Since the given metric is not guaranteed FA-able, raise error if needed.
            except Exception as e:
                print(f'Error finding subspace: {e}')


    def filter_low_firing_rate(self, verbose: bool = False):
        """
        To filter out low FR channels, it has to be done in the 'all' mode. 
        Otherwise, the excluded channels may be different across 'events' and 'gaps' modes.
        """

        self.get_events(mode='all')
        self.get_spikes_mu_sd(verbose=False)

        good_fr = self.spikes_mu > 0.5 # Greater than 0.5 Hz
        self.Y = self.Y[:, good_fr]
        excluded = self.channels[~good_fr]
        self.channels = self.channels[good_fr]

        if verbose:
            print(f'{self}: excluding channel {excluded}.')

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

