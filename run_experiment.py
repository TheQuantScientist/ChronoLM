import sys
import time
import pandas as pd
from pathlib import Path

# Cấu hình đường dẫn hệ thống
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_clinical_data, split_patients, build_windows
from src.metrics import compute_all_metrics
from src.models.llm_ollama import Gemma2_2B, Gemma3_4B # 2 model tốt nhất

# CẤU HÌNH THỰC NGHIỆM
CSV_PATH = "processed_data/extracted_features/NIMAP.csv" 
LOOKBACK = 14  
HORIZON = 1
SEED = 42

def main():
    # 1. Chuẩn bị dữ liệu
    print(f"[*] Đang tải dữ liệu từ {CSV_PATH}...")
    df = load_clinical_data(CSV_PATH)
    
    # LẤY ĐÚNG TRỰC TIẾP 10 BỆNH NHÂN TỪ BIỂU ĐỒ GRID
    test_patient_ids = [132539, 132540, 132541, 132543, 132545, 132547, 132548, 132551, 132554, 132555]
    
    # Kiểm tra kiểu dữ liệu trong DataFrame để tránh lệch kiểu dữ liệu (string vs int)
    sample_id = df['RecordID'].iloc[0]
    if isinstance(sample_id, str):
        test_patient_ids = [str(pid) for pid in test_patient_ids]
    elif isinstance(sample_id, (int, float)):
        test_patient_ids = [int(pid) for pid in test_patient_ids]

    print(f"[*] Chạy Định Danh cho 10 bệnh nhân biểu đồ: {test_patient_ids}")

    # 2. Xây dựng cửa sổ trượt (Chạy full data tới obs cuối)
    windows = build_windows(
        df, 
        test_patient_ids, 
        lookback=LOOKBACK, 
        horizon=HORIZON, 
        max_windows=None 
    )
    print(f"[*] Tổng số cửa sổ dự báo: {len(windows)}\n")
    
    # 3. Danh sách Model tham chiến
    models = [Gemma2_2B(), Gemma3_4B()]
    results_table = []

    for model in models:
        print(f"==> Đang thực thi model: {model.name}")
        y_true_all, y_pred_all, y_last_obs_all = [], [], []
        start_time = time.time()
        
        for w in windows:
            # Dự báo điểm tiếp theo
            pred, _ = model.predict(w['history_times'], w['history_values'], n_steps=HORIZON)
            
            y_true_all.extend(w['actual_values'])
            y_pred_all.extend(pred)
            # Lưu giá trị cuối của history (Điểm A) để so sánh trend
            y_last_obs_all.append(w['history_values'][-1])
        
        total_time = time.time() - start_time
        
        # Tính toán metric bao gồm cả DA
        metrics = compute_all_metrics(y_true_all, y_pred_all, y_last_obs_all)
        
        results_table.append({
            'Model': model.name,
            'MAE': round(metrics['MAE'], 4),
            'RMSE': round(metrics['RMSE'], 4),
            'DA (%)': f"{round(metrics['DA'], 2)}%",
            'MaxAE': round(metrics['MaxAE'], 4),
            'Avg Time/Obs (s)': round(total_time / len(windows), 3)
        })

    # 4. Xuất kết quả
    summary_df = pd.DataFrame(results_table)
    print("\n" + "="*90)
    print(f" KẾT QUẢ THỰC NGHIỆM ĐỊNH DANH: DIRECTIONAL ACCURACY FOCUS (N=10) ")
    print("="*90)
    print(summary_df.to_string(index=False))
    print("="*90)
    
    # Lưu kết quả
    output_dir = Path("result")
    output_dir.mkdir(exist_ok=True)
    summary_df.to_csv(output_dir / "directional_test14_results.csv", index=False)
    print(f"[Xong] Kết quả chuẩn đã được lưu tại: {output_dir}/directional_testLB14_results.csv")

if __name__ == "__main__":
    main()