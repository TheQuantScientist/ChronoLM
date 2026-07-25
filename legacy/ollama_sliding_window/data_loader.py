"""
data_loader.py - Load processed clinical CSV data and build forecasting windows.
"""
import pandas as pd
import numpy as np
from pathlib import Path

def load_clinical_data(csv_path):
    """
    Load a processed clinical CSV, for example NIMAP.csv or RespRate.csv.
    Rows are sorted by patient and time to preserve sequence order.
    """
    df = pd.read_csv(csv_path)
    df = df.sort_values(by=['RecordID', 'Time']).reset_index(drop=True)
    return df

def split_patients(df, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Split by patient, not by row, to avoid leakage.
    Returns: dict {'train': [ids], 'val': [ids], 'test': [ids]}
    """
    rng = np.random.RandomState(seed)
    patients = df['RecordID'].unique()
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        'train': patients[:n_train].tolist(),
        'val':   patients[n_train:n_train + n_val].tolist(),
        'test':  patients[n_train + n_val:].tolist(),
    }

def build_windows(df, patient_ids, lookback=14, horizon=1, stride=1, max_windows=None):
    """
    Build a sliding-window dataset.
    
    Args:
        df: Input DataFrame.
        patient_ids: Patient IDs to include.
        lookback: Number of historical observations.
        horizon: Number of future observations to forecast.
        stride: Sliding-window stride.
        max_windows: Optional cap for quick tests.

    Returns:
        A list of dictionaries containing history and target values.
    """
    windows = []
    
    for pid in patient_ids:
        pdata = df[df['RecordID'] == pid].sort_values(by='Time')
        n_points = len(pdata)
        
        if n_points < lookback + horizon:
            continue

        for start in range(0, n_points - lookback - horizon + 1, stride):
            history = pdata.iloc[start : start + lookback]
            actual = pdata.iloc[start + lookback : start + lookback + horizon]

            windows.append({
                'patient_id': pid,
                'history_times':  history['Time'].tolist(),
                'history_values': history['Value'].tolist(),
                'actual_times':   actual['Time'].tolist(),
                'actual_values':  actual['Value'].tolist(),
            })

            if max_windows is not None and len(windows) >= max_windows:
                return windows

    return windows

# ==========================================
# Quick smoke test for the old sliding-window prototype.
# ==========================================
if __name__ == "__main__":
    CSV = "processed_data/extracted_features/RespRate.csv" 
    
    print("[*] Loading clinical data...")
    df = load_clinical_data(CSV)
    
    print("\n[*] Splitting patients (70/15/15)...")
    splits = split_patients(df)
    
    for lb in [1, 14, 21, 28]:
        test_windows = build_windows(
            df, splits['test'],
            lookback=lb, 
            horizon=1,
            max_windows=100
        )
        print(f"    Lookback {lb:2d} -> Created {len(test_windows):3d} windows")

    if test_windows:
        example = test_windows[0]
        print(f"\n[*] Example Window (LB={len(example['history_values'])}, H={len(example['actual_values'])}):")
        print(f"    History: {example['history_values']}")
        print(f"    Target:  {example['actual_values']}")
