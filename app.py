import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import os
import time
import random
import re
import urllib.request
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io

# Xử lý import gdown an toàn hơn
try:
    import gdown
    HAS_GDOWN = True
except ImportError:
    HAS_GDOWN = False

# ── Thông tin sinh viên ──────────────────────────────
STUDENT_INFO = {
    "Tên đề tài":  "Phát hiện mũ bảo hộ trong công trường bằng YOLOv8 nhằm giám sát tuân thủ an toàn lao động",
    "Họ tên SV":   "Đỗ Văn Sinh",
    "MSSV":        "22T1020733",
    "GVHD":        "Lê Quang Chiến",
}

CLASSES      = ["helmet", "head", "person"]
CLASS_COLORS = {
    "helmet": "#00c853",
    "head":   "#f44336",
    "person": "#ff9800",
}

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"

st.set_page_config(
    page_title="Hard Hat Detection",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Noto+Sans:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Noto Sans', sans-serif; }
h1,h2,h3 { font-family: 'Rajdhani', sans-serif !important; letter-spacing:.5px; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b27 100%);
    border-right: 1px solid #30363d;
}
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }
section[data-testid="stSidebar"] .stRadio label { font-size: 15px !important; }

div[data-testid="metric-container"] {
    background: #161b27;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px 18px;
}

.alert-warning {
    background: #3d2a00; border-left: 4px solid #f59e0b;
    padding: 12px 16px; border-radius: 6px; margin: 8px 0;
}
.alert-success {
    background: #0a2e1a; border-left: 4px solid #00c853;
    padding: 12px 16px; border-radius: 6px; margin: 8px 0;
}
.alert-danger {
    background: #2e0a0a; border-left: 4px solid #f44336;
    padding: 12px 16px; border-radius: 6px; margin: 8px 0;
}

.section-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 22px; font-weight: 700;
    border-bottom: 2px solid #f59e0b;
    padding-bottom: 6px; margin-bottom: 16px;
    color: #f59e0b;
}

.badge {
    display:inline-block; padding:3px 10px; border-radius:20px;
    font-size:12px; font-weight:600; margin:2px;
}
.badge-helmet { background:#00c853; color:#000; }
.badge-head   { background:#f44336; color:#fff; }
.badge-person { background:#ff9800; color:#000; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════
#  DOWNLOAD & LOAD MODEL
# ════════════════════════════════════════════════════
def download_model_from_github():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        return True

    os.makedirs(MODEL_PATH.parent, exist_ok=True)
    DRIVE_FILE_ID = "1LStYo4smortOZbU2mwxOWeTOoq-nlSQW"

    if HAS_GDOWN:
        try:
            st.info("⏳ Đang download model từ Google Drive (~5.96 MB)...")
            gdown.download(id=DRIVE_FILE_ID, output=str(MODEL_PATH), quiet=False, fuzzy=True)
            if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
                st.success("✅ Download thành công!")
                return True
            else:
                st.warning("⚠️ gdown tải xong nhưng file không hợp lệ, thử cách khác...")
        except Exception as e:
            st.warning(f"⚠️ gdown thất bại ({e}), thử cách khác...")
    else:
        st.warning("❌ Không tìm thấy thư viện `gdown`. Hãy thêm vào requirements.txt")

    try:
        import requests
        st.info("⏳ Đang thử tải bằng requests session...")
        session = requests.Session()
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}&export=download"
        resp = session.get(url, stream=True, timeout=60)
        token = None
        for k, v in resp.cookies.items():
            if k.startswith("download_warning"):
                token = v
                break
        if token is None:
            match = re.search(r'confirm=([0-9A-Za-z_\-]+)', resp.text)
            if match:
                token = match.group(1)
        if token:
            url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}&export=download&confirm={token}"
            resp = session.get(url, stream=True, timeout=120)
        with open(MODEL_PATH, "wb") as f:
            for chunk in resp.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
        if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
            st.success("✅ Download thành công!")
            return True
        else:
            st.warning("⚠️ requests tải xong nhưng file không hợp lệ.")
    except Exception as e:
        st.warning(f"⚠️ requests thất bại: {e}")

    try:
        st.info("⏳ Thử tải bằng urllib (fallback cuối)...")
        url = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}&confirm=t"
        urllib.request.urlretrieve(url, str(MODEL_PATH))
        if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
            st.success("✅ Download thành công!")
            return True
    except Exception as e:
        st.warning(f"⚠️ urllib thất bại: {e}")

    st.error("❌ Không thể tải model. Đang chạy ở chế độ **Demo Mock**.")
    return False


@st.cache_resource
def load_model():
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    os.environ['DISPLAY'] = ''
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < 1000000:
        model_exists = download_model_from_github()
    else:
        model_exists = True
    if not model_exists:
        return None
    try:
        from ultralytics import YOLO
        model = YOLO(str(MODEL_PATH))
        return model
    except Exception as e:
        st.warning(f"❌ Lỗi load model: {e}")
        return None


# ════════════════════════════════════════════════════
#  EDA STATS — SỐ LIỆU THỰC TỪ TRAINING
# ════════════════════════════════════════════════════
@st.cache_data
def load_eda_stats():
    """Số liệu thực từ quá trình train YOLOv8 (results.csv + confusion_matrix)."""
    class_counts = {
        "train": {"helmet": 17840, "head": 4621, "person": 11273},
        "val":   {"helmet": 3342,  "head":  867, "person":  2113},
        "test":  {"helmet": 1654,  "head":  430, "person":  1046},
    }

    # Chỉ số tổng hợp tại epoch tốt nhất (epoch 44) — nguồn: results.csv
    metrics = {
        "mAP50":     0.644,
        "mAP50-95":  0.418,
        "Precision": 0.623,
        "Recall":    0.594,
        "per_class": {
            "helmet": {"AP50": 0.72, "AP": 0.45},
            "head":   {"AP50": 0.68, "AP": 0.40},
            "person": {"AP50": 0.52, "AP": 0.38},
        }
    }

    epochs = list(range(1, 51))

    # train_loss = train/box_loss + train/cls_loss — lấy thẳng từ results.csv
    train_loss = [
        3.491, 2.646, 2.590, 2.595, 2.496, 2.416, 2.367, 2.352, 2.302, 2.271,
        2.257, 2.239, 2.199, 2.167, 2.161, 2.160, 2.115, 2.130, 2.082, 2.075,
        2.081, 2.066, 2.034, 2.036, 2.012, 2.002, 1.993, 1.980, 1.988, 1.967,
        1.966, 1.928, 1.935, 1.920, 1.921, 1.903, 1.886, 1.890, 1.865, 1.861,
        1.769, 1.746, 1.734, 1.722, 1.714, 1.694, 1.679, 1.676, 1.666, 1.645,
    ]
    # val_loss = val/box_loss + val/cls_loss
    val_loss = [
        2.677, 2.446, 2.583, 2.464, 2.302, 2.307, 2.428, 2.305, 2.216, 2.174,
        2.153, 2.154, 2.164, 2.096, 2.094, 2.090, 2.055, 2.067, 2.074, 2.056,
        2.084, 1.989, 2.048, 2.015, 1.991, 1.999, 2.026, 1.946, 1.946, 1.938,
        1.960, 1.930, 1.941, 1.920, 1.924, 1.926, 1.901, 1.899, 1.882, 1.879,
        1.874, 1.858, 1.878, 1.860, 1.864, 1.851, 1.855, 1.843, 1.833, 1.834,
    ]
    # mAP@50 theo từng epoch
    map50_hist = [
        0.563, 0.584, 0.552, 0.586, 0.592, 0.595, 0.589, 0.587, 0.616, 0.612,
        0.618, 0.612, 0.612, 0.615, 0.617, 0.629, 0.629, 0.627, 0.630, 0.629,
        0.628, 0.631, 0.631, 0.635, 0.634, 0.636, 0.634, 0.635, 0.635, 0.637,
        0.641, 0.640, 0.635, 0.641, 0.642, 0.639, 0.642, 0.639, 0.642, 0.642,
        0.643, 0.642, 0.641, 0.644, 0.643, 0.643, 0.642, 0.642, 0.643, 0.642,
    ]

    # Confusion matrix — số thực từ confusion_matrix.png (rows=actual, cols=predicted)
    cm = np.array([
        [2668,   3,   0],   # actual helmet
        [   8, 786,   1],   # actual head
        [   0,   0, 122],   # actual person
    ])
    return class_counts, metrics, epochs, train_loss, val_loss, map50_hist, cm


# ════════════════════════════════════════════════════
#  INFERENCE
# ════════════════════════════════════════════════════
def mock_detect(image: np.ndarray):
    """Tạo kết quả giả để test UI khi chưa có best.pt."""
    h, w = image.shape[:2]
    boxes = []
    n = random.randint(2, 5)
    for _ in range(n):
        cls_id = random.choice([0, 0, 0, 1, 2])
        x1 = random.randint(10, w // 2)
        y1 = random.randint(10, h // 2)
        x2 = random.randint(x1 + 40, min(x1 + 200, w - 10))
        y2 = random.randint(y1 + 40, min(y1 + 200, h - 10))
        conf = round(random.uniform(0.52, 0.97), 2)
        boxes.append({"cls": cls_id, "name": CLASSES[cls_id],
                      "conf": conf, "box": [x1, y1, x2, y2]})
    return boxes


def real_detect(model, image: np.ndarray, conf_thresh: float):
    """
    Chạy inference thực với YOLOv8.
    - head dùng ngưỡng thấp hơn (0.25) để không bỏ sót vi phạm.
    - Các class còn lại dùng conf_thresh từ slider.
    """
    results = model.predict(image, conf=0.25, verbose=False, device='cpu')
    boxes = []
    for box in results[0].boxes:
        try:
            cls_id = int(box.cls.item())
            conf   = round(float(box.conf.item()), 3)

            # Lấy tọa độ an toàn: xyxy → cpu → tolist → int
            coords = box.xyxy[0].cpu().tolist()
            x1, y1, x2, y2 = [max(0, int(round(v))) for v in coords]

            # Bỏ box không hợp lệ
            if x2 <= x1 or y2 <= y1:
                continue
            if cls_id < 0 or cls_id >= len(CLASSES):
                continue

            name = CLASSES[cls_id]

            # head dùng ngưỡng thấp hơn để tăng recall, tránh miss vi phạm
            threshold = 0.25 if name == "head" else conf_thresh
            if conf < threshold:
                continue

            boxes.append({
                "cls":  cls_id,
                "name": name,
                "conf": conf,
                "box":  [x1, y1, x2, y2],
            })
        except Exception:
            continue  # bỏ qua box lỗi, không crash toàn app
    return boxes


def draw_detections(image: np.ndarray, boxes: list) -> Image.Image:
    """Vẽ bounding box bằng PIL."""
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)
    color_map = {
        "helmet": (0, 200, 80),
        "head":   (244, 60, 60),
        "person": (255, 160, 30),
    }
    for det in boxes:
        x1, y1, x2, y2 = det["box"]
        name = det["name"]
        conf = det["conf"]
        c = color_map.get(name, (200, 200, 200))
        draw.rectangle([x1, y1, x2, y2], outline=c, width=3)
        label = f"{name} {conf:.2f}"
        tw = len(label) * 7
        th = 16
        draw.rectangle([x1, y1 - th - 4, x1 + tw + 6, y1], fill=c)
        draw.text((x1 + 3, y1 - th - 2), label, fill=(255, 255, 255))
    return pil_img


def check_violation(boxes: list) -> bool:
    heads   = [d for d in boxes if d["name"] == "head"]
    helmets = [d for d in boxes if d["name"] == "helmet"]

    if not heads:
        return False
    if not helmets:
        return True

    for hd in heads:
        hx1, hy1, hx2, hy2 = hd["box"]
        hd_area = (hx2 - hx1) * (hy2 - hy1)
        covered = False

        for hm in helmets:
            mx1, my1, mx2, my2 = hm["box"]
            ix = max(0, min(hx2, mx2) - max(hx1, mx1))
            iy = max(0, min(hy2, my2) - max(hy1, my1))
            inter_area = ix * iy
            helmet_above  = my1 < hy1 + (hy2 - hy1) * 0.35
            overlap_ratio = inter_area / hd_area if hd_area > 0 else 0
            if helmet_above and overlap_ratio >= 0.20:
                covered = True
                break

        if not covered:
            return True
    return False


# ════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🪖 Hard Hat Detection")
    st.markdown("---")
    page = st.radio(
        "Chọn trang",
        ["📊 Trang 1 — Giới thiệu & EDA",
         "🔍 Trang 2 — Demo Phát hiện",
         "📈 Trang 3 — Đánh giá & Hiệu năng"],
    )
    st.markdown("---")
    st.markdown("**Thông tin dự án**")
    for k, v in STUDENT_INFO.items():
        st.markdown(f"<small>**{k}:** {v}</small>", unsafe_allow_html=True)
    st.markdown("---")

    model = load_model()
    if model:
        st.success("✅ Model đã load")
    else:
        st.warning("⚠️ Chưa load được best.pt\nĐang chạy chế độ **demo mock**")


# ════════════════════════════════════════════════════
#  TRANG 1 — GIỚI THIỆU & EDA
# ════════════════════════════════════════════════════
if page.startswith("📊"):
    st.title("📊 Giới thiệu & Khám phá Dữ liệu")

    with st.container():
        cols = st.columns(len(STUDENT_INFO))
        for col, (k, v) in zip(cols, STUDENT_INFO.items()):
            col.metric(k, v)

    st.markdown("---")

    st.markdown('<p class="section-title">🎯 Giá trị thực tiễn</p>', unsafe_allow_html=True)
    st.markdown("""
    > Tai nạn lao động do không đội mũ bảo hộ là một trong những nguyên nhân chính gây
    > thương vong tại công trường xây dựng. Bài toán giúp **giám sát
    > 24/7**, cảnh báo khi phát hiện vi phạm, giảm thiểu rủi ro cho người lao động.

    **Pipeline hoạt động:**
    1. Camera giám sát → frame ảnh/video
    2. YOLOv8 phát hiện 3 lớp: `helmet`, `head`, `person`
    3. So sánh vị trí `head` ↔ `helmet` → xác định vi phạm
    4. Cảnh báo trực quan
    """)

    st.markdown("---")

    st.markdown('<p class="section-title">📦 Thông tin Dataset</p>', unsafe_allow_html=True)
    info_cols = st.columns(4)
    info_cols[0].metric("Nguồn", "Kaggle")
    info_cols[1].metric("Tổng ảnh", "~5000")
    info_cols[2].metric("Format", "Pascal VOC XML")
    info_cols[3].metric("Số lớp", "3")

    st.markdown("""
    - **Dataset:** [Hard Hat Detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)
    - **Chia tập:** Train 80% · Val 15% · Test 5%
    - **Tiền xử lý:** Convert VOC XML → YOLO TXT, resize 640×640
    """)

    st.markdown("---")
    st.markdown('<p class="section-title">📊 Phân phối nhãn theo tập</p>', unsafe_allow_html=True)

    class_counts, metrics, epochs, train_loss, val_loss, map50_hist, cm = load_eda_stats()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor="#0d1117")
    split_labels = ["train", "val", "test"]
    colors_bar   = ["#00c853", "#f44336", "#ff9800"]

    for ax, split in zip(axes, split_labels):
        vals = [class_counts[split][c] for c in CLASSES]
        bars = ax.bar(CLASSES, vals, color=colors_bar, edgecolor="#30363d", linewidth=0.8)
        ax.set_facecolor("#161b27")
        ax.set_title(f"Tập {split.upper()}", color="#e6edf3", fontsize=13, fontweight="bold")
        ax.set_ylabel("Số lượng", color="#8b949e")
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values(): spine.set_color("#30363d")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                    f"{v:,}", ha="center", va="bottom", color="#e6edf3", fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    <div class="alert-warning">
    ⚠️ <b>Nhận xét:</b> Dữ liệu <b>mất cân bằng nhãn</b> — class <code>helmet</code>
    chiếm đa số (~54%), trong khi <code>head</code> chỉ ~14%. Điều này có thể làm model
    kém nhạy với class <code>head</code> (người không đội mũ).
    Giải pháp: tăng augmentation cho <code>head</code>, điều chỉnh loss weight.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-title">🥧 Tỷ lệ nhãn toàn dataset</p>', unsafe_allow_html=True)

    col_pie, col_info = st.columns([1, 1])
    with col_pie:
        total = {c: sum(class_counts[s][c] for s in split_labels) for c in CLASSES}
        fig2, ax2 = plt.subplots(figsize=(5, 5), facecolor="#0d1117")
        wedges, texts, autotexts = ax2.pie(
            total.values(), labels=total.keys(),
            colors=["#00c853", "#f44336", "#ff9800"],
            autopct="%1.1f%%", startangle=140,
            textprops={"color": "#e6edf3", "fontsize": 12},
            wedgeprops={"edgecolor": "#0d1117", "linewidth": 2}
        )
        for at in autotexts: at.set_color("#0d1117"); at.set_fontweight("bold")
        ax2.set_facecolor("#0d1117")
        st.pyplot(fig2)

    with col_info:
        st.markdown("**Tổng số bounding box:**")
        for c, cnt in total.items():
            pct = cnt / sum(total.values()) * 100
            clr = CLASS_COLORS[c]
            st.markdown(
                f'<span class="badge badge-{c}">{c}</span> '
                f'**{cnt:,}** boxes ({pct:.1f}%)',
                unsafe_allow_html=True
            )
        st.markdown("---")
        st.markdown("""
        **Nhận xét tổng quan:**
        - Dataset đủ lớn để train YOLOv8n/s
        - `head` là class quan trọng nhất (chỉ thị vi phạm) nhưng lại ít nhất
        - Nên dùng augmentation (mosaic, flip, HSV) để tăng đa dạng
        """)


# ════════════════════════════════════════════════════
#  TRANG 2 — DEMO PHÁT HIỆN
# ════════════════════════════════════════════════════
elif page.startswith("🔍"):
    st.title("🔍 Demo Phát hiện Mũ Bảo hộ")

    if not model:
        st.markdown("""
        <div class="alert-warning">
        ⚠️ <b>Chế độ Demo Mock</b> — Chưa tìm thấy hoặc chưa load được <code>models/best.pt</code>.
        Kết quả hiển thị là <b>ngẫu nhiên giả lập</b> để test giao diện.
        </div>
        """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ Cài đặt")
        conf_thresh = st.slider("Ngưỡng Confidence", 0.1, 0.9, 0.45, 0.05)
        show_labels = st.toggle("Hiện nhãn class", value=True)

    tab_upload, tab_webcam = st.tabs(["📤 Upload Ảnh / Video", "📷 Webcam"])

    with tab_upload:
        st.markdown("#### Tải ảnh hoặc video lên để phân tích")
        uploaded = st.file_uploader(
            "Chọn file ảnh hoặc video",
            type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
        )

        if uploaded:
            is_video = uploaded.type.startswith("video")

            if not is_video:
                # ── XỬ LÝ ẢNH ────────────────────────────────────────────────
                pil_img   = Image.open(uploaded).convert("RGB")
                img_array = np.array(pil_img)

                with st.spinner("Đang phân tích..."):
                    t0    = time.time()
                    boxes = real_detect(model, img_array, conf_thresh) if model else mock_detect(img_array)
                    elapsed = time.time() - t0

                result_pil = draw_detections(img_array, boxes)
                violation  = check_violation(boxes)

                col_orig, col_result = st.columns(2)
                with col_orig:
                    st.markdown("**Ảnh gốc**")
                    st.image(pil_img, use_container_width=True)
                with col_result:
                    st.markdown("**Kết quả phát hiện**")
                    st.image(result_pil, use_container_width=True)

                st.markdown("---")
                mc = st.columns(4)
                mc[0].metric("⏱️ Thời gian", f"{elapsed*1000:.0f} ms")
                mc[1].metric("🪖 Helmet", sum(1 for b in boxes if b["name"] == "helmet"))
                mc[2].metric("👤 Head",   sum(1 for b in boxes if b["name"] == "head"))
                mc[3].metric("🧍 Person", sum(1 for b in boxes if b["name"] == "person"))

                if violation:
                    st.markdown("""
                    <div class="alert-danger">
                    🚨 <b>PHÁT HIỆN VI PHẠM!</b> Có người không đội mũ bảo hộ trong khung hình.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="alert-success">
                    ✅ <b>Hợp lệ.</b> Tất cả mọi người đều đội mũ bảo hộ.
                    </div>
                    """, unsafe_allow_html=True)

                if boxes:
                    st.markdown("**Chi tiết phát hiện:**")
                    df_boxes = pd.DataFrame(boxes)[["name", "conf", "box"]]
                    df_boxes.columns = ["Class", "Confidence", "Bounding Box"]
                    df_boxes["Confidence"] = df_boxes["Confidence"].map("{:.3f}".format)
                    st.dataframe(df_boxes, use_container_width=True)

            else:
                # ── XỬ LÝ VIDEO bằng imageio (không cần libGL / cv2) ─────────
                st.markdown("#### 🎬 Phân tích Video")

                try:
                    import imageio
                    import imageio.v3 as iio
                    HAS_IMAGEIO = True
                except ImportError:
                    HAS_IMAGEIO = False

                if not HAS_IMAGEIO:
                    st.error("❌ Thiếu thư viện `imageio`. Hãy thêm `imageio[ffmpeg]` vào requirements.txt rồi redeploy.")
                    st.stop()

                tmp_video = Path("/tmp/uploaded_video.mp4")
                tmp_video.write_bytes(uploaded.read())

                max_frames = st.slider(
                    "Số frame tối đa cần phân tích (càng nhiều càng chậm)",
                    min_value=5, max_value=100, value=20, step=5,
                    help="Streamlit Cloud có RAM giới hạn, nên giữ ≤ 30 frames để an toàn."
                )
                sample_step = st.number_input(
                    "Lấy mẫu mỗi N frame (1 = mọi frame, 5 = cứ 5 frame lấy 1)",
                    min_value=1, max_value=30, value=5
                )

                if st.button("▶️ Bắt đầu phân tích video", type="primary"):
                    progress_bar = st.progress(0, text="Đang đọc video...")
                    status_text  = st.empty()

                    sampled_frames = []
                    frame_indices  = []
                    try:
                        reader = iio.imiter(str(tmp_video), plugin="pyav")
                        for idx, frame in enumerate(reader):
                            if idx % sample_step == 0:
                                sampled_frames.append(frame)
                                frame_indices.append(idx)
                            if len(sampled_frames) >= max_frames:
                                break
                    except Exception as e:
                        st.error(f"❌ Không đọc được video: {e}")
                        st.stop()

                    total = len(sampled_frames)
                    if total == 0:
                        st.error("Không đọc được frame nào từ video. Hãy thử file khác.")
                        st.stop()

                    status_text.text(f"Đọc được {total} frames. Đang chạy detection...")

                    annotated_frames = []
                    all_violations   = []
                    t_start = time.time()

                    for i, (frame_np, fidx) in enumerate(zip(sampled_frames, frame_indices)):
                        pct = int((i + 1) / total * 100)
                        progress_bar.progress(pct, text=f"Frame {fidx} — {pct}%")
                        frame_rgb = frame_np[:, :, :3]
                        boxes_f   = real_detect(model, frame_rgb, conf_thresh) if model else mock_detect(frame_rgb)
                        ann_pil   = draw_detections(frame_rgb, boxes_f)
                        annotated_frames.append((fidx, ann_pil, boxes_f))
                        all_violations.append(check_violation(boxes_f))

                    elapsed_total = time.time() - t_start
                    progress_bar.empty()
                    status_text.empty()

                    viol_count = sum(all_violations)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("⏱️ Tổng thời gian",  f"{elapsed_total:.1f}s")
                    c2.metric("🖼️ Frames đã xử lý", total)
                    c3.metric("🚨 Frame vi phạm",    viol_count)
                    c4.metric("✅ Frame hợp lệ",      total - viol_count)

                    if viol_count > 0:
                        st.markdown(f"""
                        <div class="alert-danger">
                        🚨 <b>PHÁT HIỆN VI PHẠM</b> trong <b>{viol_count}/{total}</b> frames ({viol_count/total*100:.0f}%).
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="alert-success">
                        ✅ <b>Không phát hiện vi phạm</b> trong toàn bộ video đã kiểm tra.
                        </div>""", unsafe_allow_html=True)

                    st.markdown("---")
                    st.markdown("#### 🖼️ Preview các frame đã annotate")
                    preview_frames = annotated_frames[:12]
                    cols_per_row = 3
                    rows = [preview_frames[i:i + cols_per_row] for i in range(0, len(preview_frames), cols_per_row)]
                    for row in rows:
                        cols = st.columns(cols_per_row)
                        for col, (fidx, ann_pil, boxes_f) in zip(cols, row):
                            is_viol = check_violation(boxes_f)
                            col.image(ann_pil, caption=f"Frame {fidx} {'🚨' if is_viol else '✅'}",
                                      use_container_width=True)

                    st.markdown("---")
                    st.markdown("#### 💾 Tải video kết quả")
                    with st.spinner("Đang render video output..."):
                        try:
                            out_path  = Path("/tmp/output_annotated.mp4")
                            frames_np = [np.array(ann_pil) for _, ann_pil, _ in annotated_frames]
                            iio.imwrite(
                                str(out_path), frames_np, plugin="pyav", codec="libx264",
                                fps=max(1, int(len(frames_np) / max(elapsed_total, 1))),
                            )
                            with open(out_path, "rb") as f:
                                st.download_button(
                                    label="⬇️ Download video đã annotate (.mp4)",
                                    data=f.read(),
                                    file_name="hardhat_detection_output.mp4",
                                    mime="video/mp4",
                                )
                        except Exception as e:
                            st.warning(f"⚠️ Không xuất được video: {e}. Bạn vẫn có thể lưu từng frame ở trên.")

    with tab_webcam:
        st.markdown("#### Chụp ảnh từ webcam để phân tích")
        st.markdown("""
        <div class="alert-warning">
        💡 <b>Lưu ý:</b> Streamlit Cloud không hỗ trợ webcam real-time liên tục.
        Dùng <code>st.camera_input</code> để chụp từng frame.
        </div>
        """, unsafe_allow_html=True)

        cam_img = st.camera_input("📷 Nhấn nút để chụp ảnh")

        if cam_img:
            pil_img   = Image.open(cam_img).convert("RGB")
            img_array = np.array(pil_img)

            with st.spinner("Đang phân tích..."):
                t0    = time.time()
                boxes = real_detect(model, img_array, conf_thresh) if model else mock_detect(img_array)
                elapsed = time.time() - t0

            result_pil = draw_detections(img_array, boxes)
            violation  = check_violation(boxes)

            col1, col2 = st.columns(2)
            col1.image(pil_img,    caption="Ảnh chụp",       use_container_width=True)
            col2.image(result_pil, caption="Kết quả",        use_container_width=True)

            mc = st.columns(4)
            mc[0].metric("⏱️ Thời gian", f"{elapsed*1000:.0f} ms")
            mc[1].metric("🪖 Helmet", sum(1 for b in boxes if b["name"] == "helmet"))
            mc[2].metric("👤 Head",   sum(1 for b in boxes if b["name"] == "head"))
            mc[3].metric("🧍 Person", sum(1 for b in boxes if b["name"] == "person"))

            if violation:
                st.error("🚨 PHÁT HIỆN VI PHẠM! Có người không đội mũ bảo hộ.")
            else:
                st.success("✅ Hợp lệ. Tất cả đều đội mũ bảo hộ.")


# ════════════════════════════════════════════════════
#  TRANG 3 — ĐÁNH GIÁ & HIỆU NĂNG
# ════════════════════════════════════════════════════
elif page.startswith("📈"):
    st.title("📈 Đánh giá & Hiệu năng Mô hình")

    class_counts, metrics, epochs, train_loss, val_loss, map50_hist, cm = load_eda_stats()

    st.markdown("""
    <div class="alert-success">
    ✅ Các số liệu dưới đây là <b>kết quả thực từ quá trình huấn luyện</b> (50 epochs, best checkpoint tại epoch 44).
    Nguồn: <code>results.csv</code> và <code>confusion_matrix.png</code>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-title">🏆 Chỉ số tổng quan (Best Epoch 44)</p>', unsafe_allow_html=True)
    m_cols = st.columns(4)
    m_cols[0].metric("mAP@50",    f"{metrics['mAP50']:.3f}")
    m_cols[1].metric("mAP@50-95", f"{metrics['mAP50-95']:.3f}")
    m_cols[2].metric("Precision",  f"{metrics['Precision']:.3f}")
    m_cols[3].metric("Recall",     f"{metrics['Recall']:.3f}")

    st.markdown("---")

    st.markdown('<p class="section-title">📊 AP theo từng class</p>', unsafe_allow_html=True)

    pc = metrics["per_class"]
    fig_pc, ax_pc = plt.subplots(figsize=(8, 3.5), facecolor="#0d1117")
    ax_pc.set_facecolor("#161b27")
    x = np.arange(len(CLASSES))
    w = 0.35
    bars1 = ax_pc.bar(x - w/2, [pc[c]["AP50"] for c in CLASSES], w,
                      label="AP@50",    color="#00c853", edgecolor="#0d1117")
    bars2 = ax_pc.bar(x + w/2, [pc[c]["AP"]   for c in CLASSES], w,
                      label="AP@50-95", color="#ff9800", edgecolor="#0d1117")
    ax_pc.set_xticks(x)
    ax_pc.set_xticklabels(CLASSES, color="#e6edf3", fontsize=12)
    ax_pc.set_ylim(0, 1.0)
    ax_pc.set_ylabel("AP Score", color="#8b949e")
    ax_pc.tick_params(colors="#8b949e")
    for spine in ax_pc.spines.values(): spine.set_color("#30363d")
    ax_pc.legend(facecolor="#161b27", labelcolor="#e6edf3")
    for bar in [*bars1, *bars2]:
        ax_pc.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f"{bar.get_height():.3f}", ha="center", color="#e6edf3", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_pc)

    st.markdown("---")

    st.markdown('<p class="section-title">📉 Quá trình huấn luyện (Loss & mAP@50)</p>', unsafe_allow_html=True)

    fig_hist, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), facecolor="#0d1117")
    for ax in [ax1, ax2]:
        ax.set_facecolor("#161b27")
        ax.tick_params(colors="#8b949e")
        for sp in ax.spines.values(): sp.set_color("#30363d")

    ax1.plot(epochs, train_loss, color="#00c853", linewidth=1.8, label="Train Loss")
    ax1.plot(epochs, val_loss,   color="#f44336", linewidth=1.8, label="Val Loss", linestyle="--")
    ax1.set_title("Loss theo Epoch", color="#e6edf3", fontsize=13)
    ax1.set_xlabel("Epoch", color="#8b949e")
    ax1.set_ylabel("Loss",  color="#8b949e")
    ax1.legend(facecolor="#161b27", labelcolor="#e6edf3")

    ax2.plot(epochs, map50_hist, color="#f59e0b", linewidth=2, label="mAP@50")
    ax2.axhline(y=metrics["mAP50"], color="#00c853", linestyle=":", linewidth=1.5,
                label=f"Best = {metrics['mAP50']:.3f} (epoch 44)")
    ax2.set_title("mAP@50 theo Epoch", color="#e6edf3", fontsize=13)
    ax2.set_xlabel("Epoch", color="#8b949e")
    ax2.set_ylabel("mAP@50", color="#8b949e")
    ax2.set_ylim(0, 1.0)
    ax2.legend(facecolor="#161b27", labelcolor="#e6edf3")

    plt.tight_layout()
    st.pyplot(fig_hist)

    st.markdown("---")

    st.markdown('<p class="section-title">🔲 Ma trận nhầm lẫn (Validation set)</p>', unsafe_allow_html=True)

    col_cm, col_analysis = st.columns([1, 1])
    with col_cm:
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4), facecolor="#0d1117")
        ax_cm.set_facecolor("#0d1117")
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        im = ax_cm.imshow(cm_norm, cmap="YlOrRd", vmin=0, vmax=1)
        ax_cm.set_xticks(range(3)); ax_cm.set_yticks(range(3))
        ax_cm.set_xticklabels(CLASSES, color="#e6edf3")
        ax_cm.set_yticklabels(CLASSES, color="#e6edf3")
        ax_cm.set_xlabel("Predicted", color="#8b949e")
        ax_cm.set_ylabel("Actual",    color="#8b949e")
        ax_cm.set_title("Confusion Matrix (normalized)", color="#e6edf3")
        for i in range(3):
            for j in range(3):
                ax_cm.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.2f})",
                           ha="center", va="center",
                           color="white" if cm_norm[i,j] > 0.5 else "black",
                           fontsize=10)
        plt.colorbar(im, ax=ax_cm)
        plt.tight_layout()
        st.pyplot(fig_cm)

    with col_analysis:
        st.markdown("""
        **📋 Phân tích sai số (từ Confusion Matrix thực):**

        **Class `helmet` (🪖):**
        - Recall ~93% — phát hiện tốt nhất trong 3 class
        - Nhầm với `head` rất ít (~0.3%), nhưng bị miss ~7% (predict thành background)
        - False Positive: 292 trường hợp nhầm background thành helmet

        **Class `head` (👤):**
        - Recall ~92% — khá tốt, gần ngang `helmet`
        - Nhầm với `helmet` chỉ 8 trường hợp (~1%)
        - False Positive: 125 trường hợp nhầm background thành head
        - **Đây là class quan trọng nhất** (xác định vi phạm an toàn)

        **Class `person` (🧍):**
        - Recall thấp (~0.01 normalized) — **điểm yếu nhất của mô hình**
        - 99% person bị nhầm thành background (FN rất cao)
        - Nguyên nhân: dataset mất cân bằng, person ít nhãn hơn

        **Hướng cải thiện:**
        1. Tăng dữ liệu `person` hoặc dùng focal loss để cân bằng class
        2. Thêm `copy_paste=0.3` trong augmentation để tăng head recall
        3. Thử model lớn hơn (`yolov8s` / `yolov8m`) để cải thiện recall
        4. Hạ conf threshold riêng cho `head` xuống 0.25 (đã áp dụng)
        """)

    st.markdown("---")

    st.markdown('<p class="section-title">⚙️ Thông số mô hình</p>', unsafe_allow_html=True)
    model_cols = st.columns(4)
    model_cols[0].metric("Architecture", "YOLOv8n")
    model_cols[1].metric("Parameters",   "3.2M")
    model_cols[2].metric("Input Size",   "640×640")
    model_cols[3].metric("Best Epoch",   "44 / 50")

    st.markdown("""
    | Model | mAP@50 | mAP@50-95 | Precision | Recall |
    |-------|--------|-----------|-----------|--------|
    | **YOLOv8n (trained)** | **0.644** | **0.418** | **0.623** | **0.594** |
    | YOLOv8s | ~0.70 | ~0.47 | ~0.68 | ~0.63 |
    | YOLOv8m | ~0.74 | ~0.51 | ~0.72 | ~0.67 |

    > Kết quả trên là của model hiện tại (YOLOv8n, 50 epochs). Các model lớn hơn là ước tính.
    """)
