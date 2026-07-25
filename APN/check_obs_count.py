# -*- coding: utf-8 -*-
"""
check_obs_count.py - Kiem tra so obs moi patient trong 36h lookback
Chay trong folder APN
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from data.dependencies.tsdm.tasks.P12 import Physionet2012

print("Loading PhysioNet P12 data...")
task = Physionet2012(seq_len=36, pred_len=3)
columns = list(task.dataset.columns)

test_dataset = task.get_dataset((0, "test"))
n_samples = len(test_dataset)
print(f"Test samples: {n_samples}")
print(f"Variables: {len(columns)}\n")

# Dem obs cho tung bien, tung patient
var_obs = {var: [] for var in columns}
total_obs_per_patient = []

for i in range(n_samples):
    sample = test_dataset[i]
    t_input, x_input, t_target = sample.inputs
    y_target = sample.targets

    patient_total = 0
    for var_idx, var_name in enumerate(columns):
        history = x_input[:, var_idx]
        n_valid = (~history.isnan()).sum().item()
        var_obs[var_name].append(n_valid)
        patient_total += n_valid

    total_obs_per_patient.append(patient_total)

# In ket qua tung bien
print(f"{'='*70}")
print(f"{'Bien':<15} {'Min':>6} {'Mean':>8} {'Median':>8} {'Max':>6} {'Co data':>10}")
print(f"{'='*70}")

results = []
for var_name in columns:
    obs = var_obs[var_name]
    has_data = sum(1 for x in obs if x > 0)
    results.append({
        'Variable': var_name,
        'Min': min(obs),
        'Mean': round(np.mean(obs), 1),
        'Median': int(np.median(obs)),
        'Max': max(obs),
        'Patients_with_data': f"{has_data}/{n_samples}",
    })
    print(f"{var_name:<15} {min(obs):>6} {np.mean(obs):>8.1f} {int(np.median(obs)):>8} {max(obs):>6} {has_data:>5}/{n_samples}")

# Tong obs moi patient (tat ca 36 bien)
print(f"\n{'='*70}")
print(f"TONG OBS MOI PATIENT (36 bien gop lai):")
print(f"  Min:    {min(total_obs_per_patient)}")
print(f"  Mean:   {np.mean(total_obs_per_patient):.1f}")
print(f"  Median: {int(np.median(total_obs_per_patient))}")
print(f"  Max:    {max(total_obs_per_patient)}")
print(f"{'='*70}")

# Luu CSV
df = pd.DataFrame(results)
df.to_csv("P12_obs_count_36h.csv", index=False)
print(f"\n[+] Da luu: P12_obs_count_36h.csv")