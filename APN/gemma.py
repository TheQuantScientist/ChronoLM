# -*- coding: utf-8 -*-
"""
gemma.py - Chay Gemma tren CUNG data pipeline cua APN
Setup GIONG APN: lookback 36h, forecast 3 steps, split 80/10/10, scaled metric
Prompt theo mentor: JSON format + system prompt
API: LiteLLM proxy (OpenAI-compatible)
"""
import sys
sys.path.insert(0, '.')

import os
import json
import numpy as np
import pandas as pd
import re
import warnings
import requests
import time
import logging

sys.path.append(".")
from data.dependencies.tsdm.tasks.P12 import Physionet2012

warnings.filterwarnings("ignore")

# Log ra file + terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler('gemma_temp06_debug.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()

# ==========================================
# 1. CAU HINH
# ==========================================
API_URL = "https://they-intranet-medicine-stuff.trycloudflare.com/v1/chat/completions"
HF_MODEL_ID = "google/gemma-3-4b-it"
API_KEY = "litellm-sml-2026"

SEQ_LEN = 36
PRED_LEN = 3
MAX_TEST_SAMPLES = None

OUTPUT_CSV = "Gemma_temp06_P12_Results.csv"
CHECKPOINT_CSV = "Gemma_temp06_P12_Checkpoint.csv"
LOG_CSV = "Gemma_temp06_P12_DetailLog.csv"

MAX_RETRIES = 3

# ==========================================
# 2. LOAD DATA TU APN PIPELINE
# ==========================================
log.info("Loading PhysioNet P12 data tu APN pipeline...")
t_load = time.time()
task = Physionet2012(seq_len=SEQ_LEN, pred_len=PRED_LEN)
encoder = task.encoder.column_encoders
columns = list(task.dataset.columns)

test_dataset = task.get_dataset((0, "test"))
n_total_test = len(test_dataset)
n_samples = MAX_TEST_SAMPLES if MAX_TEST_SAMPLES else n_total_test

log.info(f"  Data loaded in {time.time()-t_load:.1f}s")
log.info(f"  Test samples: {n_total_test}, Using: {n_samples}")
log.info(f"  Variables: {len(columns)}")

MY_VARS = columns
MY_VAR_INDICES = list(range(len(columns)))

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def hours_to_timestamp(h):
    hh = int(h)
    mm = int((h - hh) * 60)
    ss = int(((h - hh) * 60 - mm) * 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}"

def scaled_to_raw(scaled_val, var_idx):
    return scaled_val * encoder.stdv[var_idx] + encoder.mean[var_idx]

def raw_to_scaled(raw_val, var_idx):
    return (raw_val - encoder.mean[var_idx]) / encoder.stdv[var_idx]

def parse_numbers(text, count):
    if not text:
        return None
    nums = re.findall(r'[-+]?\d+\.?\d*', text)
    parsed = []
    for n in nums:
        try:
            val = float(n)
            if -10000 < val < 10000:  # Chap nhan rong, clamp sau
                parsed.append(val)
        except:
            continue
    if len(parsed) >= count:
        return parsed[:count]
    return None

def call_gemma(user_prompt, system_prompt=""):
    try:
        full_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
        messages = [{"role": "user", "content": full_prompt}]

        resp = requests.post(API_URL, json={
            "model": HF_MODEL_ID,
            "messages": messages,
            "temperature": 0.6,
            "max_tokens": 100
        }, headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }, timeout=120)
        
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            if not content:
                log.info(f"    [EMPTY] finish_reason={data['choices'][0].get('finish_reason')}, full={str(data)[:300]}")
            return content
        else:
            log.info(f"    [HTTP_{resp.status_code}] {resp.text[:200]}")
    except Exception as e:
        log.info(f"    [EXCEPTION] {str(e)[:200]}")
    return ""

# ==========================================
# 4. LOAD CHECKPOINT
# ==========================================
all_results = []
processed_vars = []

if os.path.exists(CHECKPOINT_CSV):
    try:
        df_cp = pd.read_csv(CHECKPOINT_CSV)
        all_results = df_cp.to_dict('records')
        processed_vars = df_cp['Variable'].tolist()
        log.info(f"  CHECKPOINT: Da hoan thanh {len(processed_vars)} bien:")
        log.info(f"  {processed_vars}")
    except:
        log.info("  Checkpoint loi, chay lai tu dau.")

# ==========================================
# 5. MAIN LOOP - 36 BIEN
# ==========================================
log.info(f"\n{'='*70}")
log.info(f"  GEMMA-3-4B vs APN - 36 Variables")
log.info(f"  Lookback={SEQ_LEN}h, Forecast={PRED_LEN} steps, One-shot")
log.info(f"  Test samples: {n_samples}")
log.info(f"{'='*70}")

total_start = time.time()

for var_name, var_idx in zip(MY_VARS, MY_VAR_INDICES):
    if var_name in processed_vars:
        log.info(f"\n  SKIP: {var_name} (da co trong checkpoint)")
        continue

    log.info(f"\n{'─'*70}")
    log.info(f"  Variable: {var_name} (index={var_idx})")
    log.info(f"{'─'*70}")

    var_start = time.time()
    mae_scaled_list = []
    mse_scaled_list = []
    mae_raw_list = []
    total_predictions = 0
    fallback_count = 0
    detail_logs = []

    for sample_i in range(n_samples):
        sample = test_dataset[sample_i]
        t_input, x_input, t_target = sample.inputs
        y_target = sample.targets

        history_scaled = x_input[:, var_idx]
        target_scaled = y_target[:, var_idx]

        valid_history_mask = ~history_scaled.isnan()
        valid_history = history_scaled[valid_history_mask]
        valid_history_t = t_input[valid_history_mask] * 48
        valid_target_mask = ~target_scaled.isnan()

        if len(valid_history) < 3 or valid_target_mask.sum() == 0:
            continue

        history_raw = [scaled_to_raw(v.item(), var_idx) for v in valid_history]
        history_times = [round(v.item(), 2) for v in valid_history_t]

        # JSON format theo mentor
        series_data = []
        for t, v in zip(history_times, history_raw):
            series_data.append({
                "timestamp": hours_to_timestamp(t),
                "value": round(v, 1)
            })

        history_json = json.dumps({
            "symbol": var_name,
            "interval": "irregular",
            "series": series_data
        }, indent=2)

        # Dem target
        n_forecast = 0
        actual_scaled_list = []
        actual_raw_list = []
        for step_i in range(min(PRED_LEN, len(target_scaled))):
            if valid_target_mask[step_i]:
                actual_scaled_list.append(target_scaled[step_i].item())
                actual_raw_list.append(scaled_to_raw(target_scaled[step_i].item(), var_idx))
                n_forecast += 1

        if n_forecast == 0:
            continue

# System prompt - ngan gon hon, giu JSON
        system_prompt = (
            f"Predict next {n_forecast} values for {var_name}.\n"
            f"Output ONLY {n_forecast} numbers, comma separated. No text.\n\n"
            f"{history_json}"
        )

        user_prompt = "Predict me next values"
        # System prompt (theo mentor) - chua data JSON
        # system_prompt = (
        #     f"You are a healthcare forecasting expert.\n"
        #     f"Your job is to base on the previous historical data of patients for {var_name} "
        #     f"and predict its {n_forecast} upcoming values.\n"
        #     f"Values must be precise and close to real-world healthcare environments.\n"
        #     f"You must analyze the underlying trends, statistics, and nature carefully "
        #     f"before making a decision because healthcare data values can move irregularly "
        #     f"across diverse time intervals.\n"
        #     f"You must output only {n_forecast} numerical observations.\n"
        #     f"Numbers must be separated by commas.\n\n"
        #     f"Here is the previous data of {var_name}:\n"
        #     f"{history_json}"
        # )

        # # User prompt (theo mentor)
        # user_prompt = f"Output ONLY {n_forecast} numbers separated by commas. No text, no explanation. Example: 7.3, 7.4, 7.2"

        # Retry - one-shot forecast (co logging)
        pred_raws = None
        for attempt in range(MAX_RETRIES):
            raw_output = call_gemma(user_prompt, system_prompt)
            if not raw_output:  # Server tra trong -> doi roi thu lai
                time.sleep(2)
                continue

            # LOG INPUT/OUTPUT
            if sample_i < 3 or (sample_i % 200 == 0):
                log.info(f"    [LOG] Patient={sample.key}, Var={var_name}, Attempt={attempt+1}")
                log.info(f"    [INPUT] {len(history_raw)} obs, {len(system_prompt)} chars")
                log.info(f"    [PROMPT] ...{system_prompt[-200:]}")
                log.info(f"    [OUTPUT] {raw_output[:200]}")

            if raw_output:
                pred_raws = parse_numbers(raw_output, n_forecast)
                if pred_raws is not None:
                    if sample_i < 3 or (sample_i % 200 == 0):
                        log.info(f"    [PARSED] {pred_raws}")
                    break

        if pred_raws is None:
            pred_raws = [history_raw[-1]] * n_forecast
            fallback_count += 1
            if fallback_count <= 5:
                log.info(f"    [FALLBACK] Patient={sample.key}, output='{raw_output[:100]}'")

        # Clamp: gioi han prediction trong 3 sigma cua history
        if len(history_raw) >= 3:
            h_mean = np.mean(history_raw[-10:])
            h_std = max(np.std(history_raw[-10:]), 0.1)
            clamped = []
            for p in pred_raws:
                lower = h_mean - 3 * h_std
                upper = h_mean + 3 * h_std
                clamped.append(max(lower, min(upper, p)))
            pred_raws = clamped


        # Metric tung step
        for step_i in range(n_forecast):
            pred_scaled = raw_to_scaled(pred_raws[step_i], var_idx)

            mae_scaled_list.append(abs(actual_scaled_list[step_i] - pred_scaled))
            mse_scaled_list.append((actual_scaled_list[step_i] - pred_scaled) ** 2)
            mae_raw_list.append(abs(actual_raw_list[step_i] - pred_raws[step_i]))

            detail_logs.append({
                'Variable': var_name,
                'Patient': sample.key,
                'Step': step_i,
                'Actual_scaled': round(actual_scaled_list[step_i], 6),
                'Predicted_scaled': round(pred_scaled, 6),
                'Actual_raw': round(actual_raw_list[step_i], 4),
                'Predicted_raw': round(pred_raws[step_i], 4),
                'AE_scaled': round(abs(actual_scaled_list[step_i] - pred_scaled), 6),
            })

            total_predictions += 1

        # Progress
        if (sample_i + 1) % 50 == 0 or (sample_i + 1) == n_samples:
            elapsed = time.time() - var_start
            speed = (sample_i + 1) / elapsed if elapsed > 0 else 0
            remaining = (n_samples - sample_i - 1) / speed if speed > 0 else 0
            log.info(f"  [{var_name}] {sample_i+1}/{n_samples} "
                     f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    var_elapsed = time.time() - var_start

    if mae_scaled_list:
        row = {
            'Variable': var_name,
            'Model': 'Gemma-3-4b',
            'MAE_scaled': round(np.mean(mae_scaled_list), 6),
            'MSE_scaled': round(np.mean(mse_scaled_list), 6),
            'MAE_raw': round(np.mean(mae_raw_list), 4),
            'Predictions': total_predictions,
            'Fallback': fallback_count,
            'Time_s': round(var_elapsed, 1),
        }
        all_results.append(row)

        log.info(f"\n  {var_name}: MAE_scaled={row['MAE_scaled']}, "
                 f"MSE_scaled={row['MSE_scaled']}, MAE_raw={row['MAE_raw']}, "
                 f"Fallback={fallback_count}, Time={row['Time_s']}s")

        pd.DataFrame(all_results).to_csv(CHECKPOINT_CSV, index=False)
        log.info(f"  Checkpoint saved ({len(all_results)}/36 bien)")

        if detail_logs:
            header = not os.path.exists(LOG_CSV) or var_name == MY_VARS[0]
            pd.DataFrame(detail_logs).to_csv(LOG_CSV, mode='a',
                                              header=header, index=False)
    else:
        log.info(f"  {var_name}: KHONG CO DATA (skip)")

# ==========================================
# 6. XUAT KET QUA
# ==========================================
if all_results:
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)

    total_elapsed = time.time() - total_start

    log.info(f"\n{'='*70}")
    log.info(f"  KET QUA GEMMA-3-4B vs APN - PhysioNet P12")
    log.info(f"{'='*70}")
    log.info("\n" + df[['Variable', 'MAE_scaled', 'MSE_scaled', 'MAE_raw',
              'Predictions', 'Fallback']].to_string(index=False))

    avg_mae = df['MAE_scaled'].mean()
    avg_mse = df['MSE_scaled'].mean()
    total_preds = df['Predictions'].sum()
    total_fallback = df['Fallback'].sum()

    log.info(f"\n{'─'*70}")
    log.info(f"  TRUNG BINH 36 BIEN:")
    log.info(f"    MAE_scaled = {avg_mae:.6f}")
    log.info(f"    MSE_scaled = {avg_mse:.6f}")
    log.info(f"    Total predictions: {total_preds:,}")
    log.info(f"    Total fallback: {total_fallback}")
    log.info(f"    Total time: {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)")
    log.info(f"\n  APN (paper): MAE=0.3762, MSE=0.2936")
    log.info(f"  Gemma:       MAE={avg_mae:.4f}, MSE={avg_mse:.4f}")

    if avg_mae < 0.3762:
        log.info(f"\n  >>> GEMMA THANG APN! (MAE thap hon {0.3762-avg_mae:.4f})")
    else:
        log.info(f"\n  >>> APN van tot hon (MAE cao hon {avg_mae-0.3762:.4f})")

    log.info(f"{'='*70}")
    log.info(f"\n[+] Da luu: {OUTPUT_CSV}")
    log.info(f"[+] Detail log: {LOG_CSV}")
    log.info(f"[+] Debug log: gemma_temp06_debug.log")
    log.info(f"[+] Checkpoint: {CHECKPOINT_CSV}")

    if len(all_results) >= len(MY_VARS):
        log.info(f"\n  Tat ca 36 bien da chay xong!")