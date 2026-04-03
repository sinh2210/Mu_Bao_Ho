"""
╔══════════════════════════════════════════════════════╗
║   Hard Hat Detection — Streamlit App                 ║
║   YOLOv8 | 3 Trang | Đủ yêu cầu đồ án              ║
╚══════════════════════════════════════════════════════╝
Cấu trúc:
  Trang 1 — Giới thiệu & EDA
  Trang 2 — Demo phát hiện (Upload ảnh / Webcam)
  Trang 3 — Đánh giá & Hiệu năng
"""

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
import urllib.request
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io

# ── Thông tin sinh viên ──────────────────────────────
STUDENT_INFO = {
    "Tên đề tài":  "Phát hiện mũ bảo hộ trong công trường bằng YOLOv8 nhằm giám sát tuân thủ an toàn lao động",
    "Họ tên SV":   "Đỗ Văn Sinh",        # ← đổi tên của bạn
    "MSSV":        "22T1020733",             # ← đổi MSSV
    "GVHD":        "Lê Quang Chiến",                  # ← đổi tên GV
}

CLASSES      = ["helmet", "head", "person"]
CLASS_COLORS = {
    "helmet": "#00c853",   # xanh lá
    "head":   "#f44336",   # đỏ (vi phạm)
    "person": "#ff9800",   # cam
}
MODEL_PATH = "models/best.pt"

# ════════════════════════════════════════════════════
#  PAGE CONFIG & GLOBAL CSS
# ════════════════════════════════════════════════════
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

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b27 100%);
    border-right: 1px solid #30363d;
}
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }
section[data-testid="stSidebar"] .stRadio label { font-size: 15px !important; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: #161b27;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px 18px;
}

/* Alert boxes */
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

/* Section title */
.section-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 22px; font-weight: 700;
    border-bottom: 2px solid #f59e0b;
    padding-bottom: 6px; margin-bottom: 16px;
    color: #f59e0b;
}

/* Badge */
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
#  CACHE: LOAD MODEL FROM GITHUB
# ════════════════════════════════════════════════════
def download_model_from_github():
    """Download best.pt từ GitHub nếu chưa có."""
    if os.path.exists(MODEL_PATH):
        return True
    
    os.makedirs("models", exist_ok=True)
    
    try:
        st.info("⏳ Đang download model từ GitHub (~100MB)...")
        
        # Raw GitHub URL
        url = "https://raw.githubusercontent.com/sinh2210/Mu_Bao_Ho/main/models/best.pt"
        urllib.request.urlretrieve(url, MODEL_PATH)
        
        if os.path.exists(MODEL_PATH):
            st.success("✅ Download thành công!")
            return True
        else:
            st.warning("❌ Download thất bại: file không tạo được")
            return False
            
    except Exception as e:
        st.warning(f"❌ Lỗi download: {str(e)}")
        return False


@st.cache_resource
def load_model():
    """Load YOLOv8 model từ GitHub. Headless mode cho Streamlit Cloud."""
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    os.environ['DISPLAY'] = ''
    
    if not os.path.exists(MODEL_PATH):
        st.warning("⏳ Model chưa có, đang tải từ GitHub...")
        model_exists = download_model_from_github()
    else:
        model_exists = True
    
    if not model_exists:
        st.warning("⚠️ Không thể tải model. Đang chạy chế độ **Demo Mock**.")
        return None
    
    try:
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        error_msg = str(e)
        if "libGL" in error_msg or "OpenGL" in error_msg:
            st.warning("⚠️ Lỗi OpenGL (headless). Chạy **Demo Mock**.")
        else:
            st.warning(f"❌ Lỗi load model: {error_msg}")
        return None


@st.cache_data
def load_eda_stats():
    """Trả về thống kê giả lập (thay bằng số liệu thực sau khi train)."""
    # ── Số lượng mỗi class trong dataset ──────────────
    class_counts = {
        "train": {"helmet": 17840, "head": 4621, "person": 11273},
        "val":   {"helmet": 3342,  "head":  867, "person":  2113},
        "test":  {"helmet": 1654,  "head":  430, "person":  1046},
    }
    # ── Metrics sau khi train ──────────────────────────
    metrics = {
        "mAP50":      0.887,
        "mAP50-95":   0.612,
        "Precision":  0.901,
        "Recall":     0.864,
        "per_class": {
            "helmet": {"AP50": 0.934, "AP": 0.671},
            "head":   {"AP50": 0.812, "AP": 0.548},
            "person": {"AP50": 0.915, "AP": 0.618},
        }
    }
    # ── Loss theo epoch (giả lập) ──────────────────────
    epochs = list(range(1, 51))
    np.random.seed(42)
    train_loss = [1.8 * np.exp(-0.07*e) + 0.18 + np.random.normal(0, 0.02) for e in epochs]
    val_loss   = [1.9 * np.exp(-0.065*e) + 0.22 + np.random.normal(0, 0.025) for e in epochs]
    map50_hist = [min(0.887, 0.3 + 0.012*e + np.random.normal(0, 0.01)) for e in epochs]

    # ── Confusion matrix ───────────────────────────────
    cm = np.array([
        [938,  41,  21],   # helmet
        [58,  342,  30],   # head
        [18,   22, 406],   # person
    ])
    return class_counts, metrics, epochs, train_loss, val_loss, map50_hist, cm


# ════════════════════════════════════════════════════
#  MOCK INFERENCE (khi chưa có model)
# ════════════════════════════════════════════════════
def mock_detect(image: np.ndarray):
    """Tạo kết quả giả để test UI khi chưa có best.pt."""
    h, w = image.shape[:2]
    boxes = []
    n = random.randint(2, 5)
    for _ in range(n):
        cls_id = random.choice([0, 0, 0, 1, 2])
        x1 = random.randint(10, w//2)
        y1 = random.randint(10, h//2)
        x2 = random.randint(x1+40, min(x1+200, w-10))
        y2 = random.randint(y1+40, min(y1+200, h-10))
        conf = round(random.uniform(0.52, 0.97), 2)
        boxes.append({"cls": cls_id, "name": CLASSES[cls_id],
                      "conf": conf, "box": [x1, y1, x2, y2]})
    return boxes


def real_detect(model, image: np.ndarray, conf_thresh: float):
    """Chạy inference thực với YOLOv8."""
    results = model.predict(image, conf=conf_thresh, verbose=False)
    boxes = []
    for box in results[0].boxes:
        cls_id = int(box.cls)
        boxes.append({
            "cls":  cls_id,
            "name": CLASSES[cls_id],
            "conf": round(float(box.conf), 3),
            "box":  list(map(int, box.xyxy[0].tolist())),
        })
    return boxes


def draw_detections(image: np.ndarray, boxes: list) -> np.ndarray:
    """Vẽ bounding box lên ảnh."""
    img = image.copy()
    color_map = {
        "helmet": (0, 200, 80),
        "head":   (60, 60, 244),
        "person": (30, 160, 255),
    }
    for det in boxes:
        x1, y1, x2, y2 = det["box"]
        name = det["name"]
        conf = det["conf"]
        c = color_map.get(name, (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1-th-8), (x1+tw+6, y1), c, -1)
        cv2.putText(img, label, (x1+3, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
    return img


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
            
            # Tính vùng giao nhau
            ix = max(0, min(hx2, mx2) - max(hx1, mx1))
            iy = max(0, min(hy2, my2) - max(hy1, my1))
            inter_area = ix * iy
            
            # Helmet phải nằm PHÍA TRÊN đầu (my1 < hy1 + 1/3 chiều cao đầu)
            helmet_above = my1 < hy1 + (hy2 - hy1) * 0.35
            
            # Overlap phải đủ lớn (>= 20% diện tích đầu)
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

    # Trạng thái model
    model = load_model()
    if model:
        st.success("✅ Model đã load")
    else:
        st.warning("⚠️ Chưa có best.pt\nĐang chạy chế độ **demo mock**")


# ════════════════════════════════════════════════════
#  TRANG 1 — GIỚI THIỆU & EDA
# ════════════════════════════════════════════════════
if page.startswith("📊"):
    st.title("📊 Giới thiệu & Khám phá Dữ liệu")

    # ── Thông tin sinh viên ──────────────────────────
    with st.container():
        cols = st.columns(len(STUDENT_INFO))
        for col, (k, v) in zip(cols, STUDENT_INFO.items()):
            col.metric(k, v)

    st.markdown("---")

    # ── Mô tả bài toán ──────────────────────────────
    st.markdown('<p class="section-title">🎯 Giá trị thực tiễn</p>', unsafe_allow_html=True)
    st.markdown("""
    > Tai nạn lao động do không đội mũ bảo hộ là một trong những nguyên nhân chính gây
    > thương vong tại công trường xây dựng. Hệ thống phát hiện tự động giúp **giám sát
    > 24/7**, cảnh báo ngay lập tức khi phát hiện vi phạm, giảm thiểu rủi ro cho người lao động.

    **Pipeline hoạt động:**
    1. Camera giám sát → frame ảnh/video
    2. YOLOv8 phát hiện 3 lớp: `helmet`, `head`, `person`
    3. So sánh vị trí `head` ↔ `helmet` → xác định vi phạm
    4. Cảnh báo trực quan + âm thanh
    """)

    st.markdown("---")

    # ── Dataset info ─────────────────────────────────
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

    # ── Biểu đồ 1: phân phối class ───────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">📊 Phân phối nhãn theo tập</p>', unsafe_allow_html=True)

    class_counts, metrics, epochs, train_loss, val_loss, map50_hist, cm = load_eda_stats()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor="#0d1117")
    split_labels = ["train", "val", "test"]
    colors_bar = ["#00c853", "#f44336", "#ff9800"]

    for ax, split in zip(axes, split_labels):
        vals = [class_counts[split][c] for c in CLASSES]
        bars = ax.bar(CLASSES, vals, color=colors_bar, edgecolor="#30363d", linewidth=0.8)
        ax.set_facecolor("#161b27")
        ax.set_title(f"Tập {split.upper()}", color="#e6edf3", fontsize=13, fontweight="bold")
        ax.set_ylabel("Số lượng", color="#8b949e")
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values(): spine.set_color("#30363d")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
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

    # ── Biểu đồ 2: phân phối tổng (pie) ─────────────
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
        ⚠️ <b>Chế độ Demo Mock</b> — Chưa tìm thấy <code>models/best.pt</code>.
        Kết quả hiển thị là <b>ngẫu nhiên giả lập</b> để test giao diện.
        Sau khi train xong, đặt file <code>best.pt</code> vào thư mục <code>models/</code> và restart app.
        </div>
        """, unsafe_allow_html=True)

    # ── Cài đặt ──────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Cài đặt")
        conf_thresh = st.slider("Ngưỡng Confidence", 0.1, 0.9, 0.45, 0.05)
        show_labels = st.toggle("Hiện nhãn class", value=True)

    tab_upload, tab_webcam = st.tabs(["📤 Upload Ảnh / Video", "📷 Webcam"])

    # ── Tab Upload ────────────────────────────────────
    with tab_upload:
        st.markdown("#### Tải ảnh hoặc video lên để phân tích")
        uploaded = st.file_uploader(
            "Chọn file ảnh (jpg/png) hoặc video (mp4/avi)",
            type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
        )

        if uploaded:
            is_video = uploaded.type.startswith("video")

            if not is_video:
                # ── Xử lý ảnh ─────────────────────────
                file_bytes = np.frombuffer(uploaded.read(), np.uint8)
                image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                image_rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                with st.spinner("Đang phân tích..."):
                    t0 = time.time()
                    boxes = real_detect(model, image_bgr, conf_thresh) if model else mock_detect(image_bgr)
                    elapsed = time.time() - t0

                result_img = draw_detections(image_rgb, boxes)
                violation  = check_violation(boxes)

                col_orig, col_result = st.columns(2)
                with col_orig:
                    st.markdown("**Ảnh gốc**")
                    st.image(image_rgb, use_container_width=True)
                with col_result:
                    st.markdown("**Kết quả phát hiện**")
                    st.image(result_img, use_container_width=True)

                # ── Thống kê ──────────────────────────
                st.markdown("---")
                mc = st.columns(4)
                mc[0].metric("⏱️ Thời gian", f"{elapsed*1000:.0f} ms")
                mc[1].metric("🪖 Helmet",  sum(1 for b in boxes if b["name"]=="helmet"))
                mc[2].metric("👤 Head",    sum(1 for b in boxes if b["name"]=="head"))
                mc[3].metric("🧍 Person",  sum(1 for b in boxes if b["name"]=="person"))

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

                # ── Chi tiết từng box ─────────────────
                if boxes:
                    st.markdown("**Chi tiết phát hiện:**")
                    df_boxes = pd.DataFrame(boxes)[["name","conf","box"]]
                    df_boxes.columns = ["Class", "Confidence", "Bounding Box"]
                    df_boxes["Confidence"] = df_boxes["Confidence"].map("{:.3f}".format)
                    st.dataframe(df_boxes, use_container_width=True)

            else:
                # ── Xử lý video (frame-by-frame) ──────
                tfile = f"/tmp/upload_{uploaded.name}"
                with open(tfile, "wb") as f:
                    f.write(uploaded.read())

                st.info("Đang xử lý video... (chỉ hiển thị 30 frame đầu để demo)")
                cap = cv2.VideoCapture(tfile)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25

                frame_placeholder = st.empty()
                stat_placeholder  = st.empty()
                progress = st.progress(0)

                max_frames = min(total_frames, 90)
                for i in range(max_frames):
                    ok, frame = cap.read()
                    if not ok: break
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    boxes = real_detect(model, frame, conf_thresh) if model else mock_detect(frame)
                    result = draw_detections(frame_rgb, boxes)
                    violation = check_violation(boxes)
                    if violation:
                        cv2.putText(result, "⚠ VI PHAM!", (10, result.shape[0]-20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,60,60), 3)
                    frame_placeholder.image(result, use_container_width=True,
                                            caption=f"Frame {i+1}/{max_frames}")
                    stat_placeholder.markdown(
                        f"🪖 {sum(1 for b in boxes if b['name']=='helmet')}  "
                        f"👤 {sum(1 for b in boxes if b['name']=='head')}  "
                        f"🧍 {sum(1 for b in boxes if b['name']=='person')}  "
                        + ("🚨 **VI PHẠM**" if violation else "✅ OK")
                    )
                    progress.progress((i+1)/max_frames)
                    time.sleep(1/fps)

                cap.release()
                st.success("✅ Xử lý video hoàn tất!")

    # ── Tab Webcam ────────────────────────────────────
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
            file_bytes = np.frombuffer(cam_img.getvalue(), np.uint8)
            image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            with st.spinner("Đang phân tích..."):
                t0 = time.time()
                boxes = real_detect(model, image_bgr, conf_thresh) if model else mock_detect(image_bgr)
                elapsed = time.time() - t0

            result_img = draw_detections(image_rgb, boxes)
            violation  = check_violation(boxes)

            col1, col2 = st.columns(2)
            col1.image(image_rgb, caption="Ảnh chụp", use_container_width=True)
            col2.image(result_img, caption="Kết quả", use_container_width=True)

            mc = st.columns(4)
            mc[0].metric("⏱️ Thời gian", f"{elapsed*1000:.0f} ms")
            mc[1].metric("🪖 Helmet",  sum(1 for b in boxes if b["name"]=="helmet"))
            mc[2].metric("👤 Head",    sum(1 for b in boxes if b["name"]=="head"))
            mc[3].metric("🧍 Person",  sum(1 for b in boxes if b["name"]=="person"))

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
    <div class="alert-warning">
    📌 Các số liệu dưới đây là <b>kết quả tham khảo / giả lập</b>.
    Sau khi train xong, cập nhật hàm <code>load_eda_stats()</code> bằng số liệu thực từ
    <code>runs/hardhat_yolov8/results.csv</code>.
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics tổng quan ─────────────────────────────
    st.markdown('<p class="section-title">🏆 Chỉ số tổng quan (Test set)</p>', unsafe_allow_html=True)
    m_cols = st.columns(4)
    m_cols[0].metric("mAP@50",    f"{metrics['mAP50']:.3f}",    "↑ tốt")
    m_cols[1].metric("mAP@50-95", f"{metrics['mAP50-95']:.3f}", "")
    m_cols[2].metric("Precision",  f"{metrics['Precision']:.3f}","")
    m_cols[3].metric("Recall",     f"{metrics['Recall']:.3f}",  "")

    st.markdown("---")

    # ── Per-class AP ──────────────────────────────────
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
    ax_pc.set_xticks(x); ax_pc.set_xticklabels(CLASSES, color="#e6edf3", fontsize=12)
    ax_pc.set_ylim(0, 1.0); ax_pc.set_ylabel("AP Score", color="#8b949e")
    ax_pc.tick_params(colors="#8b949e")
    for spine in ax_pc.spines.values(): spine.set_color("#30363d")
    ax_pc.legend(facecolor="#161b27", labelcolor="#e6edf3")
    for bar in [*bars1, *bars2]:
        ax_pc.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                   f"{bar.get_height():.3f}", ha="center", color="#e6edf3", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_pc)

    st.markdown("---")

    # ── Loss & mAP50 history ──────────────────────────
    st.markdown('<p class="section-title">📉 Quá trình huấn luyện (Loss & mAP@50)</p>', unsafe_allow_html=True)

    fig_hist, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), facecolor="#0d1117")
    for ax in [ax1, ax2]:
        ax.set_facecolor("#161b27")
        ax.tick_params(colors="#8b949e")
        for sp in ax.spines.values(): sp.set_color("#30363d")

    ax1.plot(epochs, train_loss, color="#00c853", linewidth=1.8, label="Train Loss")
    ax1.plot(epochs, val_loss,   color="#f44336", linewidth=1.8, label="Val Loss",   linestyle="--")
    ax1.set_title("Loss theo Epoch", color="#e6edf3", fontsize=13)
    ax1.set_xlabel("Epoch", color="#8b949e"); ax1.set_ylabel("Loss", color="#8b949e")
    ax1.legend(facecolor="#161b27", labelcolor="#e6edf3")

    ax2.plot(epochs, map50_hist, color="#f59e0b", linewidth=2, label="mAP@50")
    ax2.axhline(y=metrics["mAP50"], color="#00c853", linestyle=":", linewidth=1.5, label=f"Best={metrics['mAP50']}")
    ax2.set_title("mAP@50 theo Epoch", color="#e6edf3", fontsize=13)
    ax2.set_xlabel("Epoch", color="#8b949e"); ax2.set_ylabel("mAP@50", color="#8b949e")
    ax2.set_ylim(0, 1.0); ax2.legend(facecolor="#161b27", labelcolor="#e6edf3")

    plt.tight_layout()
    st.pyplot(fig_hist)

    st.markdown("---")

    # ── Confusion Matrix ──────────────────────────────
    st.markdown('<p class="section-title">🔲 Ma trận nhầm lẫn (Test set)</p>', unsafe_allow_html=True)

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
        **📋 Phân tích sai số:**

        **Class `helmet` (🪖):**
        - Accuracy ~93% — phát hiện tốt nhất
        - Nhầm với `head` (~4%) khi mũ bị che khuất 1 phần

        **Class `head` (👤):**
        - Accuracy ~80% — kém nhất do ít dữ liệu
        - Nhầm với `helmet` (~13%) khi đầu tóc sẫm màu
        - **Đây là class quan trọng nhất** (xác định vi phạm)

        **Class `person` (🧍):**
        - Accuracy ~91% — ổn định
        - Nhầm nhỏ với `head` khi người đứng xa camera

        **Hướng cải thiện:**
        1. Thu thập thêm ảnh `head` trong các điều kiện ánh sáng khác nhau
        2. Tăng weight loss cho class `head`
        3. Thử model lớn hơn (`yolov8s` / `yolov8m`)
        4. Áp dụng Test-Time Augmentation (TTA)
        """)

    st.markdown("---")

    # ── Model info ────────────────────────────────────
    st.markdown('<p class="section-title">⚙️ Thông số mô hình</p>', unsafe_allow_html=True)
    model_cols = st.columns(4)
    model_cols[0].metric("Architecture", "YOLOv8n")
    model_cols[1].metric("Parameters",   "3.2M")
    model_cols[2].metric("Input Size",   "640×640")
    model_cols[3].metric("Inference",    "~8ms/frame (T4)")

    st.markdown("""
    | Model     | mAP@50 | mAP@50-95 | Speed (T4) |
    |-----------|--------|-----------|------------|
    | YOLOv8n   | 0.887  | 0.612     | ~8 ms      |
    | YOLOv8s   | ~0.91  | ~0.64     | ~12 ms     |
    | YOLOv8m   | ~0.93  | ~0.67     | ~22 ms     |
    """)
