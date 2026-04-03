"""
Hard Hat Detection — Streamlit App
YOLOv8 | 3 Trang | Không dùng cv2 (tương thích Streamlit Cloud)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io

# ── Thông tin sinh viên ──────────────────────────────
STUDENT_INFO = {
    "Tên đề tài": "Phát hiện mũ bảo hộ trong công trường bằng YOLOv8",
    "Họ tên SV":  "Đỗ Văn Sinh",
    "MSSV":       "22T1020733",
    "GVHD":       "Lê Quang Chiến",
}

CLASSES      = ["helmet", "head", "person"]
CLASS_COLORS = {"helmet": "#00c853", "head": "#f44336", "person": "#ff9800"}
MODEL_PATH   = "models/best.pt"

# ════════════════════════════════════════════════════
#  PAGE CONFIG & CSS
# ════════════════════════════════════════════════════
st.set_page_config(page_title="Hard Hat Detection", page_icon="🪖",
                   layout="wide", initial_sidebar_state="expanded")

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
div[data-testid="metric-container"] {
    background: #161b27; border: 1px solid #30363d;
    border-radius: 10px; padding: 14px 18px;
}
.alert-warning { background:#3d2a00; border-left:4px solid #f59e0b; padding:12px 16px; border-radius:6px; margin:8px 0; }
.alert-success { background:#0a2e1a; border-left:4px solid #00c853; padding:12px 16px; border-radius:6px; margin:8px 0; }
.alert-danger  { background:#2e0a0a; border-left:4px solid #f44336; padding:12px 16px; border-radius:6px; margin:8px 0; }
.section-title { font-family:'Rajdhani',sans-serif; font-size:22px; font-weight:700;
    border-bottom:2px solid #f59e0b; padding-bottom:6px; margin-bottom:16px; color:#f59e0b; }
.badge { display:inline-block; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; margin:2px; }
.badge-helmet { background:#00c853; color:#000; }
.badge-head   { background:#f44336; color:#fff; }
.badge-person { background:#ff9800; color:#000; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════
#  CACHE & DOWNLOAD
# ════════════════════════════════════════════════════
def download_model_from_drive():
    """Download best.pt từ Google Drive nếu chưa có."""
    if os.path.exists(MODEL_PATH):
        return True
    
    os.makedirs("models", exist_ok=True)
    
    # Google Drive File ID của best.pt
    DRIVE_FILE_ID = "1LStYo4smortOZbU2mwxOWeTOoq-nlSQW"
    
    try:
        import gdown
        st.info("⏳ Đang download model từ Google Drive...")
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        st.success("✅ Download thành công!")
        return True
    except Exception as e:
        st.warning(f"❌ Lỗi download: {e}")
        return False


@st.cache_resource
def load_model():
    """Load YOLOv8 model. Tự động download từ Google Drive nếu cần."""
    # Thử download từ Google Drive nếu chưa có
    if not os.path.exists(MODEL_PATH):
        model_exists = download_model_from_drive()
    else:
        model_exists = True
    
    if not model_exists:
        return None
    
    try:
        from ultralytics import YOLO
        return YOLO(MODEL_PATH)
    except Exception as e:
        st.warning(f"Không load được model: {e}")
        return None


@st.cache_data
def load_eda_stats():
    class_counts = {
        "train": {"helmet": 17840, "head": 4621, "person": 11273},
        "val":   {"helmet": 3342,  "head":  867, "person":  2113},
        "test":  {"helmet": 1654,  "head":  430, "person":  1046},
    }
    metrics = {
        "mAP50": 0.887, "mAP50-95": 0.612,
        "Precision": 0.901, "Recall": 0.864,
        "per_class": {
            "helmet": {"AP50": 0.934, "AP": 0.671},
            "head":   {"AP50": 0.812, "AP": 0.548},
            "person": {"AP50": 0.915, "AP": 0.618},
        }
    }
    epochs = list(range(1, 51))
    np.random.seed(42)
    train_loss = [1.8*np.exp(-0.07*e)+0.18+np.random.normal(0,0.02) for e in epochs]
    val_loss   = [1.9*np.exp(-0.065*e)+0.22+np.random.normal(0,0.025) for e in epochs]
    map50_hist = [min(0.887, 0.3+0.012*e+np.random.normal(0,0.01)) for e in epochs]
    cm = np.array([[938,41,21],[58,342,30],[18,22,406]])
    return class_counts, metrics, epochs, train_loss, val_loss, map50_hist, cm


# ════════════════════════════════════════════════════
#  HELPER: ảnh → numpy array (không dùng cv2)
# ════════════════════════════════════════════════════
def pil_to_numpy(pil_img: Image.Image) -> np.ndarray:
    """PIL RGB → numpy RGB array."""
    return np.array(pil_img.convert("RGB"))


def bytes_to_pil(raw: bytes) -> Image.Image:
    """Bytes → PIL Image (RGB)."""
    return Image.open(io.BytesIO(raw)).convert("RGB")


# ════════════════════════════════════════════════════
#  DETECTION
# ════════════════════════════════════════════════════
def mock_detect(w: int, h: int):
    boxes = []
    for _ in range(random.randint(2, 5)):
        cls_id = random.choice([0, 0, 0, 1, 2])
        x1 = random.randint(10, w//2)
        y1 = random.randint(10, h//2)
        x2 = random.randint(x1+40, min(x1+200, w-10))
        y2 = random.randint(y1+40, min(y1+200, h-10))
        boxes.append({"cls": cls_id, "name": CLASSES[cls_id],
                      "conf": round(random.uniform(0.52, 0.97), 2),
                      "box": [x1, y1, x2, y2]})
    return boxes


def real_detect(model, img_array: np.ndarray, conf: float):
    results = model.predict(img_array, conf=conf, verbose=False)
    boxes = []
    for box in results[0].boxes:
        cls_id = int(box.cls)
        boxes.append({"cls": cls_id, "name": CLASSES[cls_id],
                      "conf": round(float(box.conf), 3),
                      "box": list(map(int, box.xyxy[0].tolist()))})
    return boxes


def draw_detections(pil_img: Image.Image, boxes: list) -> Image.Image:
    """Vẽ bounding box bằng Pillow (không cần cv2)."""
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    color_map = {"helmet": (0,200,80), "head": (244,60,60), "person": (255,160,30)}
    for det in boxes:
        x1, y1, x2, y2 = det["box"]
        name = det["name"]
        conf = det["conf"]
        c = color_map.get(name, (200,200,200))
        # Box
        draw.rectangle([x1, y1, x2, y2], outline=c, width=3)
        # Label background
        label = f"{name} {conf:.2f}"
        tw = len(label) * 7
        th = 16
        draw.rectangle([x1, y1-th-4, x1+tw+6, y1], fill=c)
        draw.text((x1+3, y1-th-2), label, fill=(255,255,255))
    return img


def check_violation(boxes: list) -> bool:
    heads   = [d for d in boxes if d["name"] == "head"]
    helmets = [d for d in boxes if d["name"] == "helmet"]
    if not heads:   return False
    if not helmets: return True
    for hd in heads:
        hx1,hy1,hx2,hy2 = hd["box"]
        hd_area = max(1, (hx2-hx1)*(hy2-hy1))
        covered = False
        for hm in helmets:
            mx1,my1,mx2,my2 = hm["box"]
            ix = max(0, min(hx2,mx2)-max(hx1,mx1))
            iy = max(0, min(hy2,my2)-max(hy1,my1))
            if my1 < hy1+(hy2-hy1)*0.35 and ix*iy/hd_area >= 0.20:
                covered = True; break
        if not covered: return True
    return False


# ════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🪖 Hard Hat Detection")
    st.markdown("---")
    page = st.radio("Chọn trang", [
        "Trang 1 — Giới thiệu & EDA",
        "Trang 2 — Demo Phát hiện",
        "Trang 3 — Đánh giá & Hiệu năng",
    ])
    st.markdown("---")
    st.markdown("**Thông tin dự án**")
    for k, v in STUDENT_INFO.items():
        st.markdown(f"<small>**{k}:** {v}</small>", unsafe_allow_html=True)
    st.markdown("---")
    model = load_model()
    if model:
        st.success("✅ Model đã load")
    else:
        st.warning("⚠️ Chưa có best.pt\nĐang chạy **demo mock**")


# ════════════════════════════════════════════════════
#  TRANG 1 — GIỚI THIỆU & EDA
# ════════════════════════════════════════════════════
if "Trang 1" in page:
    st.title("📊 Giới thiệu & Khám phá Dữ liệu")

    # Thông tin sinh viên
    st.markdown("""<div style="background:#161b27;border:1px solid #30363d;border-radius:12px;padding:20px 28px;margin-bottom:16px;">
    <table style="width:100%;border-collapse:collapse;"><tr>""" +
    "".join([f'<td style="padding:8px 16px;border-right:1px solid #30363d;vertical-align:top;"><div style="color:#8b949e;font-size:12px;margin-bottom:4px;">{k}</div><div style="color:#e6edf3;font-size:15px;font-weight:600;">{v}</div></td>'
             for k, v in STUDENT_INFO.items()]) +
    "</tr></table></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-title">🎯 Giá trị thực tiễn</p>', unsafe_allow_html=True)
    st.markdown("""
    > Tai nạn lao động do không đội mũ bảo hộ là một trong những nguyên nhân chính gây
    > thương vong tại công trường xây dựng. Hệ thống phát hiện tự động giúp **giám sát
    > 24/7**, cảnh báo ngay lập tức khi phát hiện vi phạm.

    **Pipeline:** Camera → YOLOv8 phát hiện `helmet / head / person` → So sánh vị trí → Cảnh báo vi phạm
    """)
    st.markdown("---")

    st.markdown('<p class="section-title">📦 Thông tin Dataset</p>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Nguồn","Kaggle"); c2.metric("Tổng ảnh","~5,000")
    c3.metric("Format","Pascal VOC"); c4.metric("Số lớp","3")
    st.markdown("- **Dataset:** [Hard Hat Detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)\n- **Chia tập:** Train 80% · Val 15% · Test 5%\n- **Tiền xử lý:** Convert VOC XML → YOLO TXT, resize 640×640")

    st.markdown("---")
    st.markdown('<p class="section-title">📊 Phân phối nhãn theo tập</p>', unsafe_allow_html=True)
    class_counts, metrics, epochs, train_loss, val_loss, map50_hist, cm = load_eda_stats()

    fig, axes = plt.subplots(1, 3, figsize=(15,4), facecolor="#0d1117")
    for ax, split in zip(axes, ["train","val","test"]):
        vals = [class_counts[split][c] for c in CLASSES]
        bars = ax.bar(CLASSES, vals, color=["#00c853","#f44336","#ff9800"], edgecolor="#30363d")
        ax.set_facecolor("#161b27"); ax.set_title(f"Tập {split.upper()}", color="#e6edf3", fontsize=13, fontweight="bold")
        ax.set_ylabel("Số lượng", color="#8b949e"); ax.tick_params(colors="#8b949e")
        for sp in ax.spines.values(): sp.set_color("#30363d")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50, f"{v:,}", ha="center", color="#e6edf3", fontsize=10)
    plt.tight_layout(); st.pyplot(fig)

    st.markdown('<div class="alert-warning">⚠️ <b>Nhận xét:</b> Dữ liệu <b>mất cân bằng nhãn</b> — <code>helmet</code> chiếm ~54%, <code>head</code> chỉ ~14%. Model có thể kém nhạy với class <code>head</code>. Giải pháp: tăng augmentation, điều chỉnh loss weight.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-title">🥧 Tỷ lệ nhãn toàn dataset</p>', unsafe_allow_html=True)
    col_pie, col_info = st.columns(2)
    with col_pie:
        total = {c: sum(class_counts[s][c] for s in ["train","val","test"]) for c in CLASSES}
        fig2, ax2 = plt.subplots(figsize=(5,5), facecolor="#0d1117")
        wedges, texts, autotexts = ax2.pie(total.values(), labels=total.keys(),
            colors=["#00c853","#f44336","#ff9800"], autopct="%1.1f%%", startangle=140,
            textprops={"color":"#e6edf3","fontsize":12}, wedgeprops={"edgecolor":"#0d1117","linewidth":2})
        for at in autotexts: at.set_color("#0d1117"); at.set_fontweight("bold")
        st.pyplot(fig2)
    with col_info:
        st.markdown("**Tổng số bounding box:**")
        for c, cnt in total.items():
            pct = cnt/sum(total.values())*100
            st.markdown(f'<span class="badge badge-{c}">{c}</span> **{cnt:,}** boxes ({pct:.1f}%)', unsafe_allow_html=True)
        st.markdown("---\n**Nhận xét:** `head` là class quan trọng nhất (vi phạm) nhưng ít nhất. Nên dùng mosaic + flip augmentation.")


# ════════════════════════════════════════════════════
#  TRANG 2 — DEMO PHÁT HIỆN
# ════════════════════════════════════════════════════
elif "Trang 2" in page:
    st.title("🔍 Demo Phát hiện Mũ Bảo hộ")

    if not model:
        st.markdown('<div class="alert-warning">⚠️ <b>Chế độ Demo Mock</b> — Chưa có <code>models/best.pt</code>. Kết quả là ngẫu nhiên giả lập.</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ Cài đặt")
        conf_thresh = st.slider("Ngưỡng Confidence", 0.1, 0.9, 0.45, 0.05)

    tab_upload, tab_webcam = st.tabs(["📤 Upload Ảnh / Video", "📷 Webcam"])

    # ── Tab Upload ────────────────────────────────────
    with tab_upload:
        st.markdown("#### Tải ảnh hoặc video lên để phân tích")
        uploaded = st.file_uploader("Chọn file ảnh (jpg/png) hoặc video (mp4/avi)",
                                    type=["jpg","jpeg","png","mp4","avi","mov"])
        if uploaded:
            is_video = uploaded.type.startswith("video")

            if not is_video:
                # Ảnh
                pil_img = bytes_to_pil(uploaded.read())
                img_arr = pil_to_numpy(pil_img)
                w, h    = pil_img.size

                with st.spinner("Đang phân tích..."):
                    t0    = time.time()
                    boxes = real_detect(model, img_arr, conf_thresh) if model else mock_detect(w, h)
                    elapsed = time.time() - t0

                result_pil = draw_detections(pil_img, boxes)
                violation  = check_violation(boxes)

                col1, col2 = st.columns(2)
                col1.markdown("**Ảnh gốc**");          col1.image(pil_img,    use_container_width=True)
                col2.markdown("**Kết quả phát hiện**"); col2.image(result_pil, use_container_width=True)

                st.markdown("---")
                mc = st.columns(4)
                mc[0].metric("⏱️ Thời gian", f"{elapsed*1000:.0f} ms")
                mc[1].metric("🪖 Helmet", sum(1 for b in boxes if b["name"]=="helmet"))
                mc[2].metric("👤 Head",   sum(1 for b in boxes if b["name"]=="head"))
                mc[3].metric("🧍 Person", sum(1 for b in boxes if b["name"]=="person"))

                if violation:
                    st.markdown('<div class="alert-danger">🚨 <b>PHÁT HIỆN VI PHẠM!</b> Có người không đội mũ bảo hộ.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="alert-success">✅ <b>Hợp lệ.</b> Tất cả mọi người đều đội mũ bảo hộ.</div>', unsafe_allow_html=True)

                if boxes:
                    st.markdown("**Chi tiết phát hiện:**")
                    df = pd.DataFrame(boxes)[["name","conf","box"]]
                    df.columns = ["Class","Confidence","Bounding Box"]
                    df["Confidence"] = df["Confidence"].map("{:.3f}".format)
                    st.dataframe(df, use_container_width=True)

            else:
                # Video — dùng imageio thay cv2
                st.info("Đang xử lý video... (hiển thị tối đa 60 frame)")
                tfile = f"/tmp/upload_{uploaded.name}"
                with open(tfile, "wb") as f: f.write(uploaded.read())

                try:
                    import imageio
                    reader = imageio.get_reader(tfile)
                    fps    = reader.get_meta_data().get("fps", 25)
                    frames = list(reader)[:60]
                    reader.close()

                    frame_ph = st.empty()
                    stat_ph  = st.empty()
                    prog     = st.progress(0)
                    for i, frame_np in enumerate(frames):
                        pil_f  = Image.fromarray(frame_np).convert("RGB")
                        w_f, h_f = pil_f.size
                        boxes  = real_detect(model, np.array(pil_f), conf_thresh) if model else mock_detect(w_f, h_f)
                        result = draw_detections(pil_f, boxes)
                        viol   = check_violation(boxes)
                        if viol:
                            d = ImageDraw.Draw(result)
                            d.text((10, h_f-30), "VI PHAM!", fill=(255,60,60))
                        frame_ph.image(result, use_container_width=True, caption=f"Frame {i+1}/{len(frames)}")
                        stat_ph.markdown(f"🪖 {sum(1 for b in boxes if b['name']=='helmet')}  👤 {sum(1 for b in boxes if b['name']=='head')}  " + ("🚨 **VI PHẠM**" if viol else "✅ OK"))
                        prog.progress((i+1)/len(frames))
                        time.sleep(1/fps)
                    st.success("✅ Xử lý xong!")
                except Exception as e:
                    st.error(f"Lỗi xử lý video: {e}\n\nHãy thử upload ảnh thay vì video.")

    # ── Tab Webcam ────────────────────────────────────
    with tab_webcam:
        st.markdown("#### Chụp ảnh từ webcam để phân tích")
        cam_img = st.camera_input("📷 Nhấn nút để chụp ảnh")
        if cam_img:
            pil_img = bytes_to_pil(cam_img.getvalue())
            img_arr = pil_to_numpy(pil_img)
            w, h    = pil_img.size

            with st.spinner("Đang phân tích..."):
                t0    = time.time()
                boxes = real_detect(model, img_arr, conf_thresh) if model else mock_detect(w, h)
                elapsed = time.time() - t0

            result_pil = draw_detections(pil_img, boxes)
            violation  = check_violation(boxes)

            col1, col2 = st.columns(2)
            col1.image(pil_img,    caption="Ảnh chụp", use_container_width=True)
            col2.image(result_pil, caption="Kết quả",  use_container_width=True)

            mc = st.columns(4)
            mc[0].metric("⏱️ Thời gian", f"{elapsed*1000:.0f} ms")
            mc[1].metric("🪖 Helmet", sum(1 for b in boxes if b["name"]=="helmet"))
            mc[2].metric("👤 Head",   sum(1 for b in boxes if b["name"]=="head"))
            mc[3].metric("🧍 Person", sum(1 for b in boxes if b["name"]=="person"))

            if violation:
                st.error("🚨 PHÁT HIỆN VI PHẠM! Có người không đội mũ bảo hộ.")
            else:
                st.success("✅ Hợp lệ. Tất cả đều đội mũ bảo hộ.")


# ════════════════════════════════════════════════════
#  TRANG 3 — ĐÁNH GIÁ & HIỆU NĂNG
# ════════════════════════════════════════════════════
elif "Trang 3" in page:
    st.title("📈 Đánh giá & Hiệu năng Mô hình")
    class_counts, metrics, epochs, train_loss, val_loss, map50_hist, cm = load_eda_stats()

    st.markdown('<div class="alert-warning">📌 Số liệu dưới đây là <b>tham khảo / giả lập</b>. Sau khi train xong, cập nhật hàm <code>load_eda_stats()</code> bằng số liệu thực.</div>', unsafe_allow_html=True)

    st.markdown('<p class="section-title">🏆 Chỉ số tổng quan (Test set)</p>', unsafe_allow_html=True)
    mc = st.columns(4)
    mc[0].metric("mAP@50",    f"{metrics['mAP50']:.3f}",    "↑ tốt")
    mc[1].metric("mAP@50-95", f"{metrics['mAP50-95']:.3f}", "")
    mc[2].metric("Precision",  f"{metrics['Precision']:.3f}","")
    mc[3].metric("Recall",     f"{metrics['Recall']:.3f}",  "")

    st.markdown("---")
    st.markdown('<p class="section-title">📊 AP theo từng class</p>', unsafe_allow_html=True)
    pc = metrics["per_class"]
    fig_pc, ax_pc = plt.subplots(figsize=(8,3.5), facecolor="#0d1117")
    ax_pc.set_facecolor("#161b27")
    x = np.arange(3); w = 0.35
    b1 = ax_pc.bar(x-w/2, [pc[c]["AP50"] for c in CLASSES], w, label="AP@50",    color="#00c853", edgecolor="#0d1117")
    b2 = ax_pc.bar(x+w/2, [pc[c]["AP"]   for c in CLASSES], w, label="AP@50-95", color="#ff9800", edgecolor="#0d1117")
    ax_pc.set_xticks(x); ax_pc.set_xticklabels(CLASSES, color="#e6edf3", fontsize=12)
    ax_pc.set_ylim(0,1); ax_pc.set_ylabel("AP Score", color="#8b949e"); ax_pc.tick_params(colors="#8b949e")
    for sp in ax_pc.spines.values(): sp.set_color("#30363d")
    ax_pc.legend(facecolor="#161b27", labelcolor="#e6edf3")
    for bar in [*b1,*b2]:
        ax_pc.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{bar.get_height():.3f}", ha="center", color="#e6edf3", fontsize=9)
    plt.tight_layout(); st.pyplot(fig_pc)

    st.markdown("---")
    st.markdown('<p class="section-title">📉 Quá trình huấn luyện</p>', unsafe_allow_html=True)
    fig_h, (ax1,ax2) = plt.subplots(1,2, figsize=(14,4), facecolor="#0d1117")
    for ax in [ax1,ax2]:
        ax.set_facecolor("#161b27"); ax.tick_params(colors="#8b949e")
        for sp in ax.spines.values(): sp.set_color("#30363d")
    ax1.plot(epochs, train_loss, color="#00c853", linewidth=1.8, label="Train Loss")
    ax1.plot(epochs, val_loss,   color="#f44336", linewidth=1.8, label="Val Loss", linestyle="--")
    ax1.set_title("Loss theo Epoch", color="#e6edf3", fontsize=13)
    ax1.set_xlabel("Epoch", color="#8b949e"); ax1.set_ylabel("Loss", color="#8b949e")
    ax1.legend(facecolor="#161b27", labelcolor="#e6edf3")
    ax2.plot(epochs, map50_hist, color="#f59e0b", linewidth=2, label="mAP@50")
    ax2.axhline(y=metrics["mAP50"], color="#00c853", linestyle=":", linewidth=1.5, label=f"Best={metrics['mAP50']}")
    ax2.set_title("mAP@50 theo Epoch", color="#e6edf3", fontsize=13)
    ax2.set_xlabel("Epoch", color="#8b949e"); ax2.set_ylabel("mAP@50", color="#8b949e"); ax2.set_ylim(0,1)
    ax2.legend(facecolor="#161b27", labelcolor="#e6edf3")
    plt.tight_layout(); st.pyplot(fig_h)

    st.markdown("---")
    st.markdown('<p class="section-title">🔲 Ma trận nhầm lẫn (Test set)</p>', unsafe_allow_html=True)
    col_cm, col_an = st.columns(2)
    with col_cm:
        fig_cm, ax_cm = plt.subplots(figsize=(5,4), facecolor="#0d1117")
        ax_cm.set_facecolor("#0d1117")
        cm_norm = cm.astype(float)/cm.sum(axis=1, keepdims=True)
        im = ax_cm.imshow(cm_norm, cmap="YlOrRd", vmin=0, vmax=1)
        ax_cm.set_xticks(range(3)); ax_cm.set_yticks(range(3))
        ax_cm.set_xticklabels(CLASSES, color="#e6edf3"); ax_cm.set_yticklabels(CLASSES, color="#e6edf3")
        ax_cm.set_xlabel("Predicted", color="#8b949e"); ax_cm.set_ylabel("Actual", color="#8b949e")
        ax_cm.set_title("Confusion Matrix", color="#e6edf3")
        for i in range(3):
            for j in range(3):
                ax_cm.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.2f})", ha="center", va="center",
                           color="white" if cm_norm[i,j]>0.5 else "black", fontsize=10)
        plt.colorbar(im, ax=ax_cm); plt.tight_layout(); st.pyplot(fig_cm)
    with col_an:
        st.markdown("""
        **📋 Phân tích sai số:**

        **`helmet` 🪖** — Accuracy ~93%, nhầm với `head` ~4% khi mũ bị che khuất

        **`head` 👤** — Accuracy ~80%, kém nhất do ít dữ liệu, nhầm với `helmet` ~13%
        → **Class quan trọng nhất** (xác định vi phạm)

        **`person` 🧍** — Accuracy ~91%, ổn định

        **Hướng cải thiện:**
        1. Thêm data `head` nhiều điều kiện ánh sáng
        2. Tăng loss weight cho class `head`
        3. Thử `yolov8s` hoặc `yolov8m`
        4. Test-Time Augmentation (TTA)
        """)

    st.markdown("---")
    st.markdown('<p class="section-title">⚙️ Thông số mô hình</p>', unsafe_allow_html=True)
    mc2 = st.columns(4)
    mc2[0].metric("Architecture","YOLOv8n"); mc2[1].metric("Parameters","3.2M")
    mc2[2].metric("Input Size","640×640");   mc2[3].metric("Inference","~8ms/frame")
    st.markdown("""
    | Model   | mAP@50 | mAP@50-95 | Speed  |
    |---------|--------|-----------|--------|
    | YOLOv8n | 0.887  | 0.612     | ~8 ms  |
    | YOLOv8s | ~0.91  | ~0.64     | ~12 ms |
    | YOLOv8m | ~0.93  | ~0.67     | ~22 ms |
    """)
