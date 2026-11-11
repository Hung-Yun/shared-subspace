"""Extract RoomMic1 acoustic RMS in the neural bins used for Fig. S11–S12.

Inputs: sessions_ns5path.csv, its NSP-2 recordings, and bin_times/*.npy (ms).
Outputs: energy/<patient>_<EMU>_<task>_RoomMic1_acoustic_energy.npy (microvolts).
Run this script for the batch; the functions also support waveform inspection.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Batch settings. The supplied bin-time arrays start at zero with 100 ms spacing.
SCRIPT_DIR = Path(__file__).resolve().parent
NS5_PATHS_CSV = Path(os.environ.get('NS5_PATHS_CSV', SCRIPT_DIR / 'sessions_ns5path.csv'))
BIN_TIMES_DIR = Path(os.environ.get('BIN_TIMES_DIR', SCRIPT_DIR / 'bin_times'))
ENERGY_DIR = Path(os.environ.get('ENERGY_DIR', SCRIPT_DIR / 'energy'))
AUDIO_CHANNEL = 'RoomMic1'
BIN_SIZE_MS = 100
AUDIO_CHANNEL_NAMES = ('Audio', 'RoomMic1', 'RoomMic2', 'PS4_audio')


def load_blackrock_recording(path_to_file):
    """Return a lazy SpikeInterface reader for an NS5 path; no traces are loaded."""
    import spikeinterface.extractors as se

    return se.read_blackrock(path_to_file)


def get_channel_id(recording, channel_name: str) -> str:
    """Return the unique channel ID for a name in a recording, or raise ValueError."""
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


def main():
    """Read the configured session table, save RMS arrays, and return an exit code.

    Existing arrays are retained. Missing recording paths are skipped; missing or
    ambiguous bin files and processing errors are reported as failures. A failed
    session never supplies its bin times to the next session.
    """
    sessions = pd.read_csv(NS5_PATHS_CSV)
    ENERGY_DIR.mkdir(parents=True, exist_ok=True)
    saved = existing = skipped = failed = 0

    for row in sessions.itertuples(index=False):
        session = f'{row.Patient}_{int(row.EMU):04d}_{row.Task.lower()}'
        ns5 = row.ns5_path
        if not isinstance(ns5, str) or not ns5.strip():
            print(f'SKIP {session}: no NS5 path.')
            skipped += 1
            continue

        output_path = ENERGY_DIR / f'{session}_{AUDIO_CHANNEL}_acoustic_energy.npy'
        if output_path.exists():
            print(f'EXISTS {output_path}')
            existing += 1
            continue

        # Patient and EMU jointly identify the recording; require one bin file.
        bin_files = list(BIN_TIMES_DIR.glob(f'{row.Patient}_{int(row.EMU):04d}_*_bin_times.npy'))
        if len(bin_files) != 1:
            print(f'FAILED {session}: expected one bin-time file; found {len(bin_files)}.')
            failed += 1
            continue

        try:
            bin_times = np.load(bin_files[0])
            if bin_times.ndim != 1 or len(bin_times) == 0 or not np.all(np.diff(bin_times) == BIN_SIZE_MS):
                raise ValueError(f'Expected nonempty full-session bin times with {BIN_SIZE_MS} ms spacing.')

            print(f'PROCESSING {session}: {ns5}')
            recording = load_blackrock_recording(ns5)
            room_mic_rms = get_rms_at_bin_times(
                recording,
                bin_times_ms=bin_times,
                bin_size_ms=BIN_SIZE_MS,
                channel_name=AUDIO_CHANNEL,
            )
            np.save(output_path, room_mic_rms)
        except Exception as error:
            print(f'FAILED {session}: {error}')
            failed += 1
            continue

        print(f'SAVED {output_path}')
        saved += 1

    print(f'Acoustic energy: {saved} saved; {existing} existing; {skipped} without recording paths; {failed} failed.')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
