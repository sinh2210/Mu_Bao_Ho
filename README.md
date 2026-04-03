# 🪖 Hard Hat Detection — Streamlit App

Ứng dụng phát hiện mũ bảo hộ tại công trường sử dụng **YOLOv8**.

## 🚀 Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

## 📁 Cấu trúc

```
hardhat_app/
├── app.py              ← File chạy chính
├── requirements.txt    ← Thư viện
├── models/
│   └── best.pt         ← YOLOv8 weight (sau khi train)
└── README.md
```

## ⚙️ Chạy local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Deploy lên Streamlit Cloud

1. Push repo lên GitHub (public)
2. Truy cập [share.streamlit.io](https://share.streamlit.io)
3. Kết nối GitHub → chọn repo → Main file: `app.py`
4. Deploy!

## 📌 Lưu ý về `best.pt`

- File `best.pt` **không nên push lên GitHub** (quá nặng)
- Dùng **Streamlit Secrets** hoặc tải từ Google Drive khi khởi động
- Hoặc dùng [Git LFS](https://git-lfs.github.com/) nếu < 100MB

## 🏷️ Classes

| ID | Tên     | Ý nghĩa                      |
|----|---------|------------------------------|
| 0  | helmet  | Mũ bảo hộ (đang đội)         |
| 1  | head    | Đầu không đội mũ → **Vi phạm** |
| 2  | person  | Người                         |
