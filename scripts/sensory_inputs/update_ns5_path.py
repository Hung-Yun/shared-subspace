"""Locate the NSP-2 recordings used for acoustic energy in Fig. S11–S12.

Input: sessions.csv (Patient, EMU, Task) and the stitched recording directory.
Output: sessions_ns5path.csv, retaining every session and adding ns5_path.
Excluded sessions and ambiguous or missing matches have an empty path.
"""

import os
from pathlib import Path

import pandas as pd

# Settings: defaults describe the original server layout. Paths can be overridden
# without editing the script; see README.md for the corresponding variables.
SCRIPT_DIR = Path(__file__).resolve().parent
SESSIONS_CSV = Path(os.environ.get('SESSIONS_CSV', SCRIPT_DIR / 'sessions.csv'))
NS5_PATHS_CSV = Path(os.environ.get('NS5_PATHS_CSV', SCRIPT_DIR / 'sessions_ns5path.csv'))
EMU_SOURCE_ROOT = Path(os.environ.get('EMU_SOURCE_ROOT', '/home/hungyun/wrangell/stitched/EMU-18112'))
EXCLUDED_PATIENTS = {'YEX', 'YEW'}


def matched(paths, description):
    """Return the sole matching Path, or None with a diagnostic for this input."""
    paths = list(paths)
    if len(paths) != 1:
        print(f'SKIP {description}: expected one match; found {len(paths)}.')
        return None
    return paths[0]


def find_path(row):
    """Map one session-table row to its unique NSP-2 file, or return None."""
    patient, emu = row['Patient'], int(row['EMU'])
    if patient in EXCLUDED_PATIENTS:
        return None

    root = EMU_SOURCE_ROOT / patient
    folders = (path for path in root.glob(f'EMU-{emu:04d}*') if path.is_dir())
    folder = matched(folders, f'{patient}_{emu:04d} recording folder')
    if folder is None:
        return None
    recording = matched(folder.glob('*NSP-2.ns5'), f'{patient}_{emu:04d} NSP-2 file')
    return str(recording) if recording is not None else None


def main():
    """Read the session list and write the path table; return no in-memory result."""
    if not EMU_SOURCE_ROOT.is_dir():
        raise FileNotFoundError(f'Recording root does not exist: {EMU_SOURCE_ROOT}')

    sessions = pd.read_csv(SESSIONS_CSV)
    sessions['ns5_path'] = sessions.apply(find_path, axis=1)
    print(sessions[['Patient', 'EMU', 'ns5_path']])
    NS5_PATHS_CSV.parent.mkdir(parents=True, exist_ok=True)
    sessions.to_csv(NS5_PATHS_CSV, index=False)
    print(f'Saved {sessions.ns5_path.notna().sum()} matched paths to {NS5_PATHS_CSV}')


if __name__ == "__main__":
    main()
