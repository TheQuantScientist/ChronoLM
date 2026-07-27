# -*- coding: utf-8 -*-
"""
download_data.py - Tai truoc dataset PhysioNet 2012 (P12) qua thu vien tsdm.

Chay 1 lan duy nhat truoc khi chay gemma.py.
tsdm se tu dong tai va cache data vao ~/.tsdm/datasets/Physionet2012/

Cach dung:
    cd APN
    python download_data.py
"""
import sys
sys.path.insert(0, '.')

import time
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("  TAI DATASET PHYSIONET 2012 (P12)")
print("=" * 60)
print("  Lan dau chay se tai ~18MB data tu nguon goc.")
print("  Data se duoc cache vao ~/.tsdm/datasets/Physionet2012/")
print("  Qua trinh nay mat khoang 5-10 phut tuy toc do mang.")
print("=" * 60)
print()

t0 = time.time()

try:
    print("[1/3] Dang import thu vien tsdm...")
    from data.dependencies.tsdm.tasks.P12 import Physionet2012

    print("[2/3] Dang khoi tao task (se tu dong tai data neu chua co)...")
    task = Physionet2012(seq_len=36, pred_len=3)

    print("[3/3] Dang kiem tra data da san sang chua...")
    columns = list(task.dataset.columns)
    test_dataset = task.get_dataset((0, "test"))
    n_test = len(test_dataset)

    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print("  TAI DATA THANH CONG!")
    print("=" * 60)
    print(f"  So bien (variables) : {len(columns)}")
    print(f"  So test samples     : {n_test}")
    print(f"  Thoi gian           : {elapsed:.1f}s")
    print(f"  Data cache tai       : ~/.tsdm/datasets/Physionet2012/")
    print("=" * 60)
    print()
    print("  Gio ban co the chay: python gemma.py")
    print()

except Exception as e:
    print()
    print("=" * 60)
    print("  LOI KHI TAI DATA!")
    print("=" * 60)
    print(f"  Chi tiet: {e}")
    print()
    print("  Kiem tra:")
    print("  1. Da cai day du thu vien chua? (pip install -r requirements.txt)")
    print("  2. Co ket noi internet khong?")
    print("  3. Dang chay trong folder APN chua?")
    print("  4. Neu thieu 'pyarrow': pip install pyarrow")
    print("=" * 60)
    import traceback
    traceback.print_exc()
    sys.exit(1)
