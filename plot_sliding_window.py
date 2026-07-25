import pandas as pd
import matplotlib.pyplot as plt
import ollama
import re
from pathlib import Path

# CẤU HÌNH LINH HOẠT - CHỈ SỬA Ở ĐÂY
LOOKBACK = 1           
NUM_PATIENTS = 10      
CSV_PATH = "processed_data/extracted_features/NIMAP.csv"

# Chỉ chạy 2 model như ông bảo
MODEL_LIST = [
    ('gemma3:4b',      'Gemma3-4B',     'red',      '^'),
    ('llama3.2:3b',    'Llama3.2-3B',   'orange',   'D'),
]


# 1. LOAD DATA
df = pd.read_csv(CSV_PATH)
df = df.sort_values(by=['RecordID', 'Time']).reset_index(drop=True)
patient_ids = df['RecordID'].unique()[:NUM_PATIENTS]
Path("result").mkdir(parents=True, exist_ok=True)


# 2. KHỞI TẠO GRID 10 BIỂU ĐỒ 
# Tui tăng figsize lên rất to (25x30) để khi gom lại chữ không bị lí nhí
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(25, 30))
axes_flat = axes.flatten() 

print(f"[*] Đang vẽ lưới 10 bệnh nhân. Lookback: {LOOKBACK}")


# 3. VÒNG LẶP CHẠY VÀ VẼ
for i, pid in enumerate(patient_ids):
    ax = axes_flat[i]
    pdata = df[df['RecordID'] == pid].reset_index(drop=True)
    
    start_idx = LOOKBACK
    all_times = pdata['Time'].tolist()
    all_values = pdata['Value'].tolist()
    
    forecast_times = all_times[start_idx:]
    actual_values = all_values[start_idx:]
    history_times = all_times[:start_idx]
    history_values = all_values[:start_idx]

    # --- CHẠY MODEL  ---
    all_forecasts = {}
    for model_name, display_name, color, marker in MODEL_LIST:
        forecasts = []
        for j in range(len(forecast_times)):
            current_pos = start_idx + j
            input_values = all_values[current_pos - LOOKBACK : current_pos]
            vals = [round(v, 1) for v in input_values]

            prompt = f"Given these NIMAP values: {vals}\nWhat is the next value? Reply ONLY with a number."
            try:
                response = ollama.chat(model=model_name, 
                                       messages=[{'role': 'user', 'content': prompt}],
                                       options={'temperature': 0.0})
                raw = response['message']['content'].strip()
                numbers = re.findall(r'\b(\d+\.?\d*)\b', raw)
                pred = float(numbers[0]) if numbers else input_values[-1]
            except:
                pred = input_values[-1]
            forecasts.append(pred)
        all_forecasts[display_name] = (forecasts, color, marker)

    # --- VẼ LÊN SUBPLOT TƯƠNG ỨNG ---
    # History & Actual
    ax.plot(range(len(history_times)), history_values, 'b-o', label='History', alpha=0.6)
    forecast_x = range(len(history_times), len(history_times) + len(forecast_times))
    ax.plot(forecast_x, actual_values, 'g-s', label='Actual', linewidth=2, zorder=5)

    # Các Model
    for name, (preds, color, marker_style) in all_forecasts.items():
        ax.plot(forecast_x, preds, f'--{marker_style}', color=color, label=name, markersize=4)

    # Decorate cho từng ô nhỏ
    ax.set_title(f'Patient {pid}', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Chỉ hiện Legend ở ô đầu tiên cho đỡ rối
    if i == 0:
        ax.legend(loc='best', fontsize=10)

    # Ẩn nhãn trục X nếu quá dày, hoặc xoay nghiêng
    if len(all_times) > 15:
        ax.set_xticks(range(0, len(all_times), 2)) # Hiện cách quãng cho đỡ chật
    else:
        ax.set_xticks(range(len(all_times)))
    ax.set_xticklabels([all_times[t] for t in ax.get_xticks()], rotation=45, fontsize=8)

# 4. HOÀN THIỆN VÀ LƯU
plt.suptitle(f'NIMAP Multi-Patient Comparison (Lookback={LOOKBACK})', fontsize=24, y=1.02)
plt.tight_layout()
output_name = f'result/comparison_grid_LB{LOOKBACK}.png'
plt.savefig(output_name, dpi=200, bbox_inches='tight')
print(f"\n[DONE] Đã gom 10 người vào 1 tấm! Check file: {output_name}")
plt.show()