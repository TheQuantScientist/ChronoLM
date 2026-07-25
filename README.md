# ChronoLM — Small Language Models as Irregular Time Series Forecasters

Dự án so sánh khả năng forecast chuỗi thời gian y tế bất quy tắc (irregular clinical time series) giữa **Small Language Models (SLM)** như Gemma / Llama / Hermes và mô hình deep learning **APN** trên bộ dữ liệu **PhysioNet 2012 (P12)**.

Ý tưởng cốt lõi: dùng chung toàn bộ data pipeline của APN (cùng cách load, scale, split, cùng metric) nhưng thay phần model bằng SLM chạy zero-shot qua API, để so sánh công bằng.

---

## 1. Yêu cầu hệ thống

- **Python 3.11** (khuyến nghị, khớp với môi trường APN gốc)
- Kết nối internet (lần đầu chạy sẽ tự tải data PhysioNet)
- Một API endpoint chạy SLM theo chuẩn OpenAI (LiteLLM proxy hoặc tương tự)

---

## 2. Cài đặt

### 2.1. Clone repo

```bash
git clone -b chrono https://github.com/TheQuantScientist/ChronoLM.git
cd ChronoLM
```

### 2.2. Tạo môi trường ảo

**Windows (PowerShell):**
```bash
python -m venv apn_env
apn_env\Scripts\activate
```

**Linux / Mac:**
```bash
python -m venv apn_env
source apn_env/bin/activate
```

### 2.3. Cài thư viện

```bash
pip install -r requirements.txt
```

Nếu chưa có `requirements.txt`, cài các gói chính:

```bash
pip install torch pandas numpy scikit-learn requests pyarrow
cd APN
pip install -r requirements.txt
```

---

## 3. Dữ liệu — TỰ ĐỘNG TẢI

**Không cần copy data thủ công.** Folder `data/` không được đẩy lên git vì dung lượng lớn.

Khi chạy lần đầu, thư viện `tsdm` sẽ tự động:
1. Tải PhysioNet 2012 (set A, B, C) từ nguồn gốc
2. Giải nén và cache vào `~/.tsdm/datasets/Physionet2012/`
3. Các lần chạy sau đọc từ cache, không tải lại

Quá trình tải + xử lý lần đầu mất khoảng 5–10 phút tùy mạng.

Class load data nằm ở:
```
APN/data/dependencies/tsdm/tasks/P12.py  ->  class Physionet2012
```

Cấu hình mặc định (giống paper APN):
- `seq_len = 36` — lookback 36 giờ đầu
- `pred_len = 3` — forecast 3 quan sát kế tiếp
- Split 80% train / 10% val / 10% test
- Scale bằng `Standardizer` (z-score), fit trên toàn bộ data
- 36 biến lâm sàng

---

## 4. Cấu hình API (SLM)

Mở file `APN/gemma.py`, chỉnh 3 dòng đầu cho khớp server của bạn:

```python
API_URL   = "https://<your-server>.trycloudflare.com/v1/chat/completions"
HF_MODEL_ID = "google/gemma-3-4b-it"   # hoặc model khác
API_KEY   = "your-api-key"
```

Server phải hỗ trợ chuẩn OpenAI `/v1/chat/completions`. Kiểm tra nhanh trước khi chạy:

```bash
python -c "import requests; r=requests.post('https://<your-server>/v1/chat/completions', json={'model':'google/gemma-3-4b-it','messages':[{'role':'user','content':'1+1=?'}],'max_tokens':10}, headers={'Authorization':'Bearer your-api-key','Content-Type':'application/json'}, timeout=30); print(r.status_code, r.text[:200])"
```

Thấy `200` và có nội dung trả về là OK.

---

## 5. Chạy thực nghiệm

Từ trong folder `APN`:

```bash
cd APN
python gemma.py
```

Hoặc chạy nền trên server (Linux), treo cho tới khi xong:

```bash
nohup python gemma.py > gemma_run.log 2>&1 &
```

Theo dõi tiến trình:

```bash
tail -f gemma_run.log
```

---

## 6. Cơ chế của `gemma.py`

Với mỗi biến (trong 36 biến) và mỗi bệnh nhân trong tập test:

1. Lấy toàn bộ observation trong 36h đầu (bỏ giá trị NaN)
2. Chuyển scaled → raw để đưa vào prompt (SLM hiểu giá trị thật tốt hơn)
3. Đóng gói thành JSON có `timestamp` + `value`
4. Gọi SLM dự đoán 3 giá trị kế tiếp cùng lúc (one-shot, không dùng giá trị thật giữa các bước)
5. Parse số từ output, clamp trong khoảng 3-sigma của history để loại outlier
6. Nếu parse thất bại → fallback dùng giá trị quan sát cuối
7. Chuyển prediction về scaled, tính MAE / MSE trên scaled (khớp cách APN báo cáo)

**Checkpoint:** kết quả từng biến được lưu ngay sau khi chạy xong. Nếu bị ngắt giữa chừng, chạy lại sẽ tự bỏ qua biến đã xong và tiếp tục phần còn lại.

---

## 7. File kết quả

Sau khi chạy xong, sinh ra:

| File | Nội dung |
|------|----------|
| `*_Results.csv` | Bảng tổng hợp MAE / MSE / fallback cho 36 biến |
| `*_Checkpoint.csv` | Checkpoint chống mất tiến trình |
| `*_DetailLog.csv` | Log từng prediction (actual vs predicted) |
| `gemma_debug.log` | Log input/output để debug |

Cuối log sẽ in bảng so sánh trung bình 36 biến giữa SLM và APN (paper: MAE=0.3762, MSE=0.2936).

---

## 8. So sánh nhiều model

Đổi `HF_MODEL_ID` và tên file output trong `gemma.py`, rồi chạy lại để benchmark model khác:

```python
HF_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_CSV  = "Llama_vs_APN_P12_Results.csv"
```

---

## 9. Chạy APN gốc (baseline)

Để lấy số liệu APN so sánh, chạy script gốc trong folder `APN`:

```bash
cd APN
# Windows: mở scripts/APN/P12.sh xem lệnh python bên trong rồi chạy trực tiếp
# Linux:
chmod +x ./scripts/APN/P12.sh
./scripts/APN/P12.sh
```

APN cần GPU để train (200 epochs). Máy không GPU chỉ chạy được để kiểm tra setup.

---

## Ghi chú

- Điểm khác biệt chính: APN là neural network được train (nhận tensor số), còn SLM chạy zero-shot (nhận text). Setup thí nghiệm giống hệt nhau; chỉ khác paradigm xử lý.
- Data không đẩy lên git — luôn để `tsdm` tự tải để đảm bảo đúng bản gốc.