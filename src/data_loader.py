# src/data_loader.py
"""
CWRU Bearing Dataset loader.

Loads .mat files from the Case Western Reserve University Bearing Dataset
and returns structured Pandas DataFrames ready for feature extraction.

Dataset source: https://engineering.case.edu/bearingdatacenter
Sampling rate: 12,000 Hz (12k Drive End Bearing Fault Data)
Bearing model: SKF 6205-2RS JEM
"""

import numpy as np
import pandas as pd
import scipy.io
from pathlib import Path


class CWRUDataLoader:
    """
    Loads CWRU .mat files and segments them into fixed-size windows.

    Each window becomes one row in the output DataFrame.
    Rows are labelled with fault type, severity, and load condition.
    """

    SAMPLING_RATE = 12000   # Hz — fixed by CWRU acquisition hardware
    WINDOW_SIZE   = 4096    # samples per window (~0.34 seconds)
    OVERLAP       = 0.5     # 50% overlap between consecutive windows

    # Integer labels for ML classification
    FAULT_LABELS = {
        'normal':      0,
        'inner_race':  1,
        'outer_race':  2,
        'ball':        3
    }

    def _find_de_key(self, mat_dict: dict) -> str:
        """
        Find the drive-end accelerometer data key in a .mat file.

        CWRU files use inconsistent key naming across files:
        e.g. 'X097_DE_time', 'X105_DE_time', 'DE_time'
        This method finds the correct key regardless of filename.

        Returns the key string, or None if not found.
        """
        # Strategy 1: look for key containing both 'DE' and 'time'
        for key in mat_dict.keys():
            if 'DE' in key and 'time' in key.lower():
                return key

        # Strategy 2: take the first non-metadata key
        candidates = [k for k in mat_dict.keys()
                      if not k.startswith('_')]
        if candidates:
            return candidates[0]

        return None

    def load_mat_file(self, filepath: str, fault_type: str,
                      severity: float = 0.0,
                      load_hp: int = 0) -> dict | None:
        """
        Load one .mat file and return a dictionary with signal + metadata.

        Args:
            filepath  : Full path to the .mat file
            fault_type: 'normal' | 'inner_race' | 'outer_race' | 'ball'
            severity  : Fault diameter in inches (0.0 for normal motors)
            load_hp   : Motor load in horsepower (0, 1, 2, or 3)

        Returns:
            dict with keys: signal, label, fault_type, severity, load_hp
            Returns None if file cannot be loaded.
        """
        try:
            mat = scipy.io.loadmat(str(filepath))
        except Exception as e:
            print(f"  Could not load {filepath}: {e}")
            return None

        de_key = self._find_de_key(mat)

        if de_key is None:
            print(f"  No drive-end key found in {filepath}")
            print(f"  Available keys: {[k for k in mat.keys() if not k.startswith('_')]}")
            return None

        signal = mat[de_key].flatten().astype(np.float64)

        return {
            'signal':      signal,
            'label':       self.FAULT_LABELS[fault_type],
            'fault_type':  fault_type,
            'severity':    severity,
            'load_hp':     load_hp
        }

    def segment_signal(self, signal: np.ndarray) -> list:
        """
        Cut a long signal into overlapping fixed-size windows.

        Each window becomes one training/inference sample.
        50% overlap doubles the number of samples from each recording.

        Returns: list of numpy arrays, each of length WINDOW_SIZE
        """
        step    = int(self.WINDOW_SIZE * (1 - self.OVERLAP))
        windows = []
        start   = 0

        while start + self.WINDOW_SIZE <= len(signal):
            windows.append(signal[start : start + self.WINDOW_SIZE])
            start += step

        return windows

    def load_dataset(self, data_dir: str,
                     verbose: bool = True) -> pd.DataFrame:
        """
        Load all .mat files from the structured data directory.

        Expected directory structure:
            data_dir/
                normal/       ← healthy motor .mat files
                inner_race/   ← inner race fault .mat files
                outer_race/   ← outer race fault .mat files
                ball/         ← ball element fault .mat files

        Each .mat file is segmented into windows.
        Each window becomes one row in the returned DataFrame.

        Columns returned:
            window_id  : unique integer ID
            signal     : numpy array of WINDOW_SIZE samples
            label      : integer class label (0–3)
            fault_type : string fault class name
            severity   : fault diameter in inches (0.0 for normal)
            load_hp    : motor load in horsepower

        Returns: pd.DataFrame
        """
        data_path = Path(data_dir)

        # Define which files map to which fault/severity/load
        # Format: (subfolder, fault_type, severity_inches, load_hp)
        # Adjust this list to match exactly which files you downloaded
        file_configs = [
            # ── Normal baseline ──────────────────────────────────────
            ('normal', 'normal', 0.000, 0),   # 97.mat
            ('normal', 'normal', 0.000, 1),   # 98.mat
            ('normal', 'normal', 0.000, 2),   # 99.mat
            ('normal', 'normal', 0.000, 3),   # 100.mat

            # ── Inner race fault — 0.007 inch ────────────────────────
            ('inner_race', 'inner_race', 0.007, 0),  # 105.mat
            ('inner_race', 'inner_race', 0.007, 1),  # 106.mat
            ('inner_race', 'inner_race', 0.007, 2),  # 107.mat
            ('inner_race', 'inner_race', 0.007, 3),  # 108.mat

            # ── Inner race fault — 0.014 inch ────────────────────────
            ('inner_race', 'inner_race', 0.014, 0),  # 169.mat
            ('inner_race', 'inner_race', 0.014, 1),  # 170.mat
            ('inner_race', 'inner_race', 0.014, 2),  # 171.mat
            ('inner_race', 'inner_race', 0.014, 3),  # 172.mat

            # ── Inner race fault — 0.021 inch ────────────────────────
            ('inner_race', 'inner_race', 0.021, 0),  # 209.mat
            ('inner_race', 'inner_race', 0.021, 1),  # 210.mat
            ('inner_race', 'inner_race', 0.021, 2),  # 211.mat
            ('inner_race', 'inner_race', 0.021, 3),  # 212.mat

            # ── Outer race fault — 0.007 inch ────────────────────────
            ('outer_race', 'outer_race', 0.007, 0),  # 130.mat
            ('outer_race', 'outer_race', 0.007, 1),  # 131.mat
            ('outer_race', 'outer_race', 0.007, 2),  # 132.mat

            # ── Outer race fault — 0.014 inch ────────────────────────
            ('outer_race', 'outer_race', 0.014, 0),  # 197.mat
            ('outer_race', 'outer_race', 0.014, 1),  # 198.mat
            ('outer_race', 'outer_race', 0.014, 2),  # 199.mat

            # ── Ball element fault — 0.007 inch ──────────────────────
            ('ball', 'ball', 0.007, 0),  # 118.mat
            ('ball', 'ball', 0.007, 1),  # 119.mat
            ('ball', 'ball', 0.007, 2),  # 120.mat
            ('ball', 'ball', 0.007, 3),  # 121.mat

            # ── Ball element fault — 0.014 inch ──────────────────────
            ('ball', 'ball', 0.014, 0),  # 185.mat
            ('ball', 'ball', 0.014, 1),  # 186.mat
            ('ball', 'ball', 0.014, 2),  # 187.mat
            ('ball', 'ball', 0.014, 3),  # 188.mat
        ]

        records    = []
        window_id  = 0
        files_loaded = 0
        files_missing = 0

        for subfolder, fault_type, severity, load_hp in file_configs:
            folder = data_path / subfolder

            if not folder.exists():
                if verbose:
                    print(f"  Folder not found: {folder}")
                continue

            mat_files = sorted(folder.glob('*.mat'))

            if not mat_files:
                if verbose:
                    print(f"  No .mat files in: {folder}")
                files_missing += 1
                continue

            # Take the next available file for this config
            # Files are sorted alphabetically — 97, 98, 99, 100 etc.
            # We iterate through files in order, one per config entry
            # Find which config index this is for this subfolder
            subfolder_configs = [c for c in file_configs
                                  if c[0] == subfolder]
            config_index = subfolder_configs.index(
                (subfolder, fault_type, severity, load_hp)
            )

            if config_index >= len(mat_files):
                if verbose:
                    print(f"  Not enough files in {subfolder} "
                          f"for config index {config_index}")
                files_missing += 1
                continue

            mat_file = mat_files[config_index]

            if verbose:
                print(f"  Loading: {mat_file.name} "
                      f"[{fault_type}, {severity}\", {load_hp}HP]",
                      end=' ')

            result = self.load_mat_file(
                str(mat_file), fault_type, severity, load_hp
            )

            if result is None:
                files_missing += 1
                continue

            windows = self.segment_signal(result['signal'])

            for window in windows:
                records.append({
                    'window_id':  window_id,
                    'signal':     window,
                    'label':      result['label'],
                    'fault_type': result['fault_type'],
                    'severity':   result['severity'],
                    'load_hp':    result['load_hp'],
                    'source_file': mat_file.name
                })
                window_id += 1

            if verbose:
                print(f"→ {len(windows)} windows")

            files_loaded += 1

        df = pd.DataFrame(records)

        if verbose:
            print(f"\n{'─'*50}")
            print(f"Files loaded:  {files_loaded}")
            print(f"Files missing: {files_missing}")
            print(f"Total windows: {len(df)}")
            print(f"\nClass distribution:")
            for ft, count in df['fault_type'].value_counts().items():
                label = self.FAULT_LABELS[ft]
                print(f"  [{label}] {ft:15s}: {count:5d} windows")

        return df


# ─── Quick self-test ──────────────────────────────────────────────
if __name__ == '__main__':
    loader = CWRUDataLoader()
    df = loader.load_dataset('data/raw/', verbose=True)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample row (no signal data):")
    print(df[['window_id', 'label', 'fault_type',
              'severity', 'load_hp', 'source_file']].head(3))