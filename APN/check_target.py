import sys
sys.path.insert(0, '.')
import numpy as np
from data.dependencies.tsdm.tasks.P12 import Physionet2012

print("Loading data...")
task = Physionet2012(seq_len=36, pred_len=3)
columns = list(task.dataset.columns)
test = task.get_dataset((0, 'test'))
n = len(test)
print(f"Test samples: {n}\n")

print(f"{'Bien':<15} {'History_mean':>12} {'Target_mean':>12} {'Has_target':>12}")
print("=" * 55)

for vi, vn in enumerate(columns):
    h_counts = []
    t_counts = []
    for i in range(n):
        s = test[i]
        t_in, x_in, t_tgt = s.inputs
        y_tgt = s.targets
        h_n = (~x_in[:, vi].isnan()).sum().item()
        t_n = (~y_tgt[:, vi].isnan()).sum().item()
        h_counts.append(h_n)
        t_counts.append(t_n)
    has = sum(1 for c in t_counts if c > 0)
    print(f"{vn:<15} {np.mean(h_counts):>12.1f} {np.mean(t_counts):>12.2f} {has:>8}/{n}")