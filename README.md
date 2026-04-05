# 🪖 Phát hiện Mũ Bảo hộ trong Công trường bằng YOLOv8

Giúp phát hiện vi phạm an toàn lao động tự động từ ảnh và video, sử dụng mô hình **YOLOv8n**.

🔗 **Web App:** [mubaoho.streamlit.app](https://mubaoho-rrxjabkbpgrtp5t8addomr.streamlit.app/)

---

## Giới thiệu

Bài toán giúp phát hiện theo thời gian thực các trường hợp người lao động **không đội mũ bảo hộ** tại công trường xây dựng. Thay vì chỉ phát hiện class, bài toán kiểm tra quan hệ không gian (spatial overlap) giữa bounding box `head` và `helmet` để xác định vi phạm chính xác hơn.

**3 lớp phát hiện:**
- `helmet` — mũ bảo hộ (tuân thủ)
- `head` — đầu không đội mũ (vi phạm)
- `person` — người

---

## Kết quả mô hình

| Chỉ số | Giá trị |
|---|---|
| mAP@50 | 0.644 |
| mAP@50-95 | 0.418 |
| Precision | 0.623 |
| Recall | 0.594 |
| Recall class `head` | **92%** |
| Best epoch | 44 / 50 |

---

## Cấu trúc project

```
Mu_Bao_Ho/
├── app.py              # Web App Streamlit (3 trang)
├── requirements.txt    # Thư viện cần thiết
├── Packages.txt        # System packages cho Streamlit Cloud
├── models/
│   └── best.pt         # YOLOv8n trained weights (tự động download nếu thiếu)
├──Train.ipynb          # file train model
└── README.md

```

---

## Chạy local

**Yêu cầu:** Python 3.9+

```bash
# 1. Clone repo
git clone https://github.com/sinh2210/Mu_Bao_Ho.git
cd Mu_Bao_Ho

# 2. Cài thư viện
pip install -r requirements.txt

# 3. Chạy app
streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501`

> **Lưu ý:** Nếu `models/best.pt` chưa có, app sẽ tự động download từ Google Drive khi khởi động.

---

## Tính năng Web App

### Trang 1 — Giới thiệu & EDA
- Thông tin đề tài, pipeline hoạt động
- Biểu đồ phân phối nhãn theo tập train/val/test
- Phân tích mất cân bằng dữ liệu

### Trang 2 — Demo Phát hiện
- **Upload ảnh** (jpg/png): inference trực tiếp, hiển thị bounding box, cảnh báo vi phạm
- **Upload video** (mp4/avi/mov): lấy mẫu đều toàn video, detection từng frame, preview grid, download video annotated
- **Webcam**: chụp ảnh từ camera và inference

### Trang 3 — Đánh giá & Hiệu năng
- Chỉ số thực từ quá trình training (results.csv)
- Biểu đồ Loss / mAP theo epoch
- Confusion matrix và phân tích từng class

---

## Dataset

**Hard Hat Detection** — Kaggle ([link](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection))

| Tập | helmet | head | person | Tổng |
|---|---|---|---|---|
| Train (80%) | 17.840 | 4.621 | 11.273 | 33.734 |
| Val (15%) | 3.342 | 867 | 2.113 | 6.322 |
| Test (5%) | 1.654 | 430 | 1.046 | 3.130 |

Tổng: **43.186 bounding box** từ ~5.000 ảnh (trung bình ~8-9 box/ảnh).

---
