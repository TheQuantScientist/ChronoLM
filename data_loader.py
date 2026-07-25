"""
data_loader.py - Load data P12 và sinh windows cho medical forecasting.
Cải tiến: Sử dụng Sliding Window và hỗ trợ linh hoạt các chỉ số y tế (NIMAP, RespRate...).
"""
import pandas as pd
import numpy as np
from pathlib import Path

def load_clinical_data(csv_path):
    """
    Load dữ liệu y tế (ví dụ: NIMAP.csv hoặc RespRate.csv).
    Sắp xếp theo RecordID và Time để đảm bảo tính tuần tự.
    """
    df = pd.read_csv(csv_path)
    # Đảm bảo dữ liệu được sắp xếp đúng thứ tự thời gian của từng bệnh nhân
    df = df.sort_values(by=['RecordID', 'Time']).reset_index(drop=True)
    return df

def split_patients(df, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Chia bệnh nhân (KHÔNG chia theo dòng) để tránh data leakage.
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
    Sử dụng kỹ thuật Sliding Window để tạo tập dữ liệu test.
    
    Args:
        df: DataFrame chứa dữ liệu.
        patient_ids: Danh sách ID bệnh nhân thuộc tập Test.
        lookback: Số điểm dữ liệu quá khứ (1, 14, 21, 28).
        horizon: Số điểm cần dự báo (mặc định là 1 theo yêu cầu).
        stride: Bước nhảy của cửa sổ (mặc định là 1 để lấy tối đa dữ liệu).
        max_windows: Giới hạn tổng số window nếu muốn test nhanh.

    Returns: list of dict chứa history và actual values.
    """
    windows = []
    
    for pid in patient_ids:
        pdata = df[df['RecordID'] == pid].sort_values(by='Time')
        n_points = len(pdata)
        
        # Kiểm tra nếu dữ liệu bệnh nhân không đủ độ dài tối thiểu
        if n_points < lookback + horizon:
            continue

        # Trượt cửa sổ qua chuỗi thời gian của bệnh nhân
        # Window: [0...lookback-1] -> Predict [lookback...lookback+horizon-1]
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

            # Dừng nếu đã đủ số lượng windows yêu cầu (để tiết kiệm tài nguyên khi chạy LLM)
            if max_windows is not None and len(windows) >= max_windows:
                return windows

    return windows

# ==========================================
# TEST NHANH (Ablation Study Preparation)
# ==========================================
if __name__ == "__main__":
    # Lưu ý: Bạn có thể đổi sang NIMAP.csv nếu file đã sẵn sàng
    CSV = "processed_data/extracted_features/RespRate.csv" 
    
    print("[*] Loading clinical data...")
    df = load_clinical_data(CSV)
    
    print("\n[*] Splitting patients (70/15/15)...")
    splits = split_patients(df)
    
    # Giả lập chạy qua các Lookback yêu cầu của Mentor
    for lb in [1, 14, 21, 28]:
        test_windows = build_windows(
            df, splits['test'],
            lookback=lb, 
            horizon=1, # Theo yêu cầu Prediction horizon: 1
            max_windows=100
        )
        print(f"    Lookback {lb:2d} -> Created {len(test_windows):3d} windows")

    if test_windows:
        example = test_windows[0]
        print(f"\n[*] Example Window (LB={len(example['history_values'])}, H={len(example['actual_values'])}):")
        print(f"    History: {example['history_values']}")
        print(f"    Target:  {example['actual_values']}")