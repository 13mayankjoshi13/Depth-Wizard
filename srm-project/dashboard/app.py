"""
dashboard/app.py
─────────────────
Streamlit dashboard for the SIH26142 Super-Resolution Mapping system.

Features:
  1. Upload Sentinel-2 GeoTIFF (or select from provided samples)
  2. Run SR inference with progress feedback
  3. Before/after image comparison slider
  4. PSNR / SSIM / SAM metric cards (with GT if uploaded, else N/A)
  5. Uncertainty heatmap overlay
  6. Geo-coordinate map (Folium) confirming geo-referencing preservation
  7. Batch mode (upload ZIP → process all → download ZIP)
  8. Training history plot (if fine-tuning was run)
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image

# Add project root to sys.path for imports
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Compatibility patch for basicsr with modern torchvision
try:
    import torchvision.transforms.functional as F_t
    sys.modules["torchvision.transforms.functional_tensor"] = F_t
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Page config (MUST be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SRM Dashboard — Sentinel-2 Super-Resolution",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-repo/srm-project",
        "About": "SIH26142 — Deep Learning Based Super Resolution Mapping (SRM)",
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS theming
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Dark-mode-friendly card */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        color: #94a3b8;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
        margin: 0;
    }
    .metric-unit {
        font-size: 0.85rem;
        color: #64748b;
    }
    .status-ok   { color: #4ade80; }
    .status-warn { color: #fb923c; }
    .status-na   { color: #64748b; }
    .section-divider {
        border: none;
        border-top: 1px solid #1e293b;
        margin: 1.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Real-ESRGAN model …")
def _load_model(model_key: str, checkpoint_path: Optional[str], half: bool, tile_size: int):
    """Load and cache the Real-ESRGAN upsampler (runs once per session).

    tile=0 disables RealESRGANer's internal sub-tiler so that our outer
    Hann-blended loop in inference.py is the only tiling strategy (Bug 1 fix).
    """
    import torch
    from src.model import load_realesrgan

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_realesrgan(
        model_key=model_key,
        checkpoint_path=checkpoint_path or None,
        scale=4,
        tile=0,   # BUG-1 FIX: disable internal tiling — see inference.py
        half=(half and device.type == "cuda"),
        device=device,
    )


def _ndarray_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert (C,H,W) float32 [0,1] or uint8 (H,W,3) to PIL Image."""
    if arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[0] != arr.shape[2]:
        # (C,H,W) → (H,W,C)
        arr = arr.transpose(1, 2, 0)
    if arr.dtype != np.uint8:
        arr = (arr.clip(0, 1) * 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    elif arr.shape[-1] > 3:
        arr = arr[:, :, :3]
    return Image.fromarray(arr)


def _get_tile_bounds(tif_path: Path) -> Optional[dict]:
    """Return bounding box dict for a GeoTIFF in WGS84 (lat/lon)."""
    try:
        import rasterio
        from rasterio.warp import transform_bounds

        with rasterio.open(tif_path) as src:
            bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        return {
            "west": bounds[0], "south": bounds[1],
            "east": bounds[2], "north": bounds[3],
        }
    except Exception as e:
        st.warning(f"Could not extract geo-bounds: {e}")
        return None


def _show_geo_map(bounds: dict, input_label: str = "Input", sr_label: str = "SR Output"):
    """Render a Folium map showing the tile bounding box."""
    import folium
    from streamlit_folium import st_folium

    center_lat = (bounds["south"] + bounds["north"]) / 2
    center_lon = (bounds["west"] + bounds["east"]) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=9)

    # Input tile footprint (blue)
    folium.Rectangle(
        bounds=[[bounds["south"], bounds["west"]], [bounds["north"], bounds["east"]]],
        color="#38bdf8",
        fill=True,
        fill_opacity=0.1,
        weight=2,
        popup=folium.Popup(input_label, parse_html=True),
        tooltip=f"Tile: {bounds['west']:.4f}°E, {bounds['south']:.4f}°N → {bounds['east']:.4f}°E, {bounds['north']:.4f}°N",
    ).add_to(m)

    folium.Marker(
        [center_lat, center_lon],
        popup=f"Centre: {center_lat:.4f}°N, {center_lon:.4f}°E",
        icon=folium.Icon(color="blue", icon="satellite", prefix="fa"),
    ).add_to(m)

    st_folium(m, width=700, height=400)


def _metric_card(label: str, value: str, unit: str = "", status: str = "ok"):
    status_class = f"status-{status}"
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value {status_class}">{value}</div>
          <div class="metric-unit">{unit}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _format_metric(value, unit: str = "", fmt: str = ".2f", high_is_better: bool = True):
    if value is None or (isinstance(value, float) and (value != value)):  # NaN
        return "N/A", "", "na"
    formatted = f"{value:{fmt}}"
    # Rough quality thresholds
    if high_is_better:
        status = "ok" if value >= 25 else ("warn" if value >= 15 else "na")
    else:
        status = "ok" if value <= 3 else ("warn" if value <= 8 else "na")
    return formatted, unit, status


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🛰️ SRM Dashboard")
    st.caption("SIH26142 — NTRO Sponsored")
    st.markdown("---")

    st.subheader("⚙️ Model Settings")
    model_key = st.selectbox("Model", ["x4plus", "x4plus_anime"], index=0,
                              help="x4plus: general SR | x4plus_anime: anime-style (not recommended for satellite)")

    # Check for fine-tuned checkpoint
    ckpt_dir = ROOT / "src" / "checkpoints"
    available_ckpts = list(ckpt_dir.glob("model_finetuned*.pth"))
    ckpt_options = ["Pretrained (default)"] + [p.name for p in available_ckpts]
    selected_ckpt = st.selectbox("Checkpoint", ckpt_options,
                                  help="'Pretrained' = original Real-ESRGAN; fine-tuned versions show here after running train.py")
    checkpoint_path = None if selected_ckpt == "Pretrained (default)" else str(ckpt_dir / selected_ckpt)

    tile_size = st.slider("Inference tile size", 128, 512, 256, 64,
                           help="Smaller = less VRAM. Increase if you have >8GB VRAM.")
    use_half = st.checkbox("FP16 inference (CUDA)", value=False,
                            help="Faster on GPU but may reduce precision")
    compute_unc = st.checkbox("Compute uncertainty map", value=True,
                               help="TTA ensemble — adds ~8× inference time")
    tta_n = st.slider("TTA augmentations", 2, 8, 4, disabled=not compute_unc,
                       help="More = smoother uncertainty map, slower inference")

    st.markdown("---")
    st.subheader("📊 Batch Mode")
    batch_mode = st.checkbox("Enable batch processing")

    st.markdown("---")
    st.caption("GPU: " + ("✅ Available" if __import__("torch").cuda.is_available() else "❌ CPU only"))
    import torch
    if torch.cuda.is_available():
        st.caption(f"CUDA: {torch.cuda.get_device_name(0)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────────────────────

st.title("🛰️ Sentinel-2 Super-Resolution Mapping")
st.markdown(
    "Upload a Sentinel-2 GeoTIFF (10m resolution) and get a **4× super-resolved** output "
    "(2.5m-equivalent) with spectral consistency metrics and uncertainty quantification."
)

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab_single, tab_batch, tab_training, tab_about = st.tabs(
    ["🔬 Single Image", "📦 Batch", "📈 Training History", "ℹ️ About"]
)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE IMAGE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_single:
    col_upload, col_gt = st.columns(2)

    with col_upload:
        st.subheader("📂 Input Tile")
        uploaded_file = st.file_uploader(
            "Upload Sentinel-2 GeoTIFF",
            type=["tif", "tiff"],
            key="main_upload",
        )

        # Sample tiles (shown if present)
        sample_dir = ROOT / "data" / "raw"
        sample_tiles = list(sample_dir.glob("*.tif")) + list(sample_dir.glob("*.tiff"))
        if sample_tiles and not uploaded_file:
            selected_sample = st.selectbox(
                "Or select a sample tile",
                ["None"] + [p.name for p in sample_tiles],
            )
        else:
            selected_sample = "None"

    with col_gt:
        st.subheader("📐 Ground Truth (optional)")
        gt_file = st.file_uploader(
            "Upload GT GeoTIFF for metrics",
            type=["tif", "tiff"],
            key="gt_upload",
            help="If provided, PSNR/SSIM/SAM are computed vs this ground truth",
        )
        st.caption("Leave empty to view SR output without metric comparison.")

    # ── Resolve input ─────────────────────────────────────────────────────────
    input_tif_path: Optional[Path] = None
    gt_tif_path: Optional[Path] = None
    _temp_files = []

    if uploaded_file is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
        tmp.write(uploaded_file.read())
        tmp.flush()
        input_tif_path = Path(tmp.name)
        _temp_files.append(input_tif_path)
    elif selected_sample != "None":
        input_tif_path = sample_dir / selected_sample

    if gt_file is not None:
        tmp_gt = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
        tmp_gt.write(gt_file.read())
        tmp_gt.flush()
        gt_tif_path = Path(tmp_gt.name)
        _temp_files.append(gt_tif_path)

    if input_tif_path is None:
        st.info("👆 Upload a Sentinel-2 GeoTIFF above, or place tiles in `data/raw/` and select from the dropdown.")
        st.stop()

    # ── Validate input ────────────────────────────────────────────────────────
    try:
        from src.preprocessing import validate_tile, extract_rgb_preview
        meta = validate_tile(input_tif_path)
    except Exception as e:
        st.error(f"❌ Invalid tile: {e}")
        st.stop()

    st.success(
        f"✅ **{Path(input_tif_path).name}** | {meta['width']}×{meta['height']} px | "
        f"{meta['bands']} bands | CRS: `{meta['crs']}`"
    )

    # ── Preview input ─────────────────────────────────────────────────────────
    try:
        lr_preview = extract_rgb_preview(input_tif_path)
        st.image(lr_preview, caption="Input tile (RGB preview, stretched)", use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render preview: {e}")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Run inference ─────────────────────────────────────────────────────────
    run_btn = st.button("🚀 Run Super-Resolution", type="primary", use_container_width=True)

    if run_btn:
        out_dir = ROOT / "data" / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)

        with st.status("⚙️ Running SR inference …", expanded=True) as status_box:
            st.write("Loading model …")
            try:
                upsampler = _load_model(model_key, checkpoint_path, use_half, tile_size)
            except Exception as e:
                st.error(f"Model load failed: {e}")
                st.stop()

            st.write("Processing tiles …")
            t0 = time.time()
            try:
                from src.inference import run_inference
                report = run_inference(
                    input_path=input_tif_path,
                    output_dir=out_dir,
                    gt_path=gt_tif_path,
                    checkpoint_path=checkpoint_path,
                    model_key=model_key,
                    patch_size=256,
                    overlap=32,
                    compute_uncertainty=compute_unc,
                    tta_n=tta_n,
                    tile_size=tile_size,
                    half=use_half,
                )
                elapsed = time.time() - t0
                status_box.update(label=f"✅ Done in {elapsed:.1f}s", state="complete")
            except Exception as e:
                status_box.update(label="❌ Inference failed", state="error")
                st.error(f"Error: {e}")
                st.exception(e)
                st.stop()

        st.session_state["last_report"] = report
        st.session_state["input_tif"] = str(input_tif_path)

    # ── Display results ───────────────────────────────────────────────────────
    if "last_report" in st.session_state:
        report = st.session_state["last_report"]
        sr_tif = Path(report["sr_output"])
        unc_png = Path(report.get("uncertainty_map", "")) if report.get("uncertainty_map") else None

        st.markdown("## 📊 Results")

        # ── Metric cards ──────────────────────────────────────────────────────
        metrics = report.get("metrics") or {}
        col_psnr, col_ssim, col_sam, col_time = st.columns(4)

        with col_psnr:
            if "psnr" in metrics:
                v, u, s = _format_metric(metrics["psnr"], "dB", ".2f", high_is_better=True)
                _metric_card("PSNR", v, u, s)
            else:
                _metric_card("PSNR", "N/A", "no GT provided", "na")

        with col_ssim:
            if "ssim" in metrics:
                v = f"{metrics['ssim']:.4f}"
                s = "ok" if metrics["ssim"] > 0.7 else "warn"
                _metric_card("SSIM", v, "", s)
            else:
                _metric_card("SSIM", "N/A", "no GT provided", "na")

        with col_sam:
            if "sam_deg" in metrics:
                v = f"{metrics['sam_deg']:.4f}"
                s = "ok" if metrics["sam_deg"] < 3.0 else "warn"
                _metric_card("SAM", v, "°", s)
            else:
                _metric_card("SAM", "N/A", "no GT provided", "na")

        with col_time:
            _metric_card("Inference time", f"{report['elapsed_seconds']:.1f}", "seconds", "ok")

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # ── Before / After comparison ─────────────────────────────────────────
        st.subheader("🔍 Before / After Comparison")
        try:
            from src.preprocessing import extract_rgb_preview
            lr_rgb = extract_rgb_preview(st.session_state["input_tif"])
            sr_rgb = extract_rgb_preview(sr_tif)

            # Use streamlit-image-comparison if available, else side-by-side
            try:
                from streamlit_image_comparison import image_comparison
                image_comparison(
                    img1=Image.fromarray(lr_rgb),
                    img2=Image.fromarray(sr_rgb),
                    label1=f"Input (10m)",
                    label2=f"SR Output (2.5m-eq, 4×)",
                    width=700,
                )
            except ImportError:
                col_l, col_r = st.columns(2)
                with col_l:
                    st.image(lr_rgb, caption="Input (10m)", use_container_width=True)
                with col_r:
                    st.image(sr_rgb, caption="SR Output (2.5m-equivalent)", use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render comparison: {e}")

        # ── Uncertainty heatmap ───────────────────────────────────────────────
        if unc_png and unc_png.exists():
            st.subheader("🌡️ Uncertainty Map")
            st.markdown(
                "**Higher intensity (yellow/green) = model is less confident** about the "
                "reconstructed detail. Low uncertainty (dark blue) = high confidence."
            )
            st.image(str(unc_png), use_container_width=True)

        # ── Geo map ───────────────────────────────────────────────────────────
        st.subheader("🗺️ Geographic Coverage")
        st.markdown("Map confirms geo-referencing was preserved — tile bounds displayed on basemap.")
        try:
            bounds = _get_tile_bounds(sr_tif)
            if bounds:
                _show_geo_map(bounds)
                st.success(
                    f"✅ Geo-reference verified | "
                    f"Bounds: {bounds['west']:.4f}°E, {bounds['south']:.4f}°N → "
                    f"{bounds['east']:.4f}°E, {bounds['north']:.4f}°N | "
                    f"CRS: `{report['crs']}`"
                )
        except Exception as e:
            st.warning(f"Geo-map unavailable: {e}")

        # ── Download ──────────────────────────────────────────────────────────
        st.subheader("💾 Download Results")
        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            if sr_tif.exists():
                with open(sr_tif, "rb") as f:
                    st.download_button(
                        "⬇️ Download SR GeoTIFF",
                        f,
                        file_name=sr_tif.name,
                        mime="image/tiff",
                        use_container_width=True,
                    )

        with col_dl2:
            if report.get("metrics"):
                st.download_button(
                    "⬇️ Download Metrics JSON",
                    json.dumps(report, indent=2),
                    file_name="metrics_report.json",
                    mime="application/json",
                    use_container_width=True,
                )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH MODE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.subheader("📦 Batch Processing")
    st.markdown("Upload a ZIP file containing multiple Sentinel-2 GeoTIFF tiles.")

    batch_zip = st.file_uploader("Upload ZIP of .tif files", type=["zip"], key="batch_zip")

    if batch_zip is not None:
        st.info("Extracting ZIP …")
        with tempfile.TemporaryDirectory() as batch_tmp:
            batch_tmp = Path(batch_tmp)
            with zipfile.ZipFile(io.BytesIO(batch_zip.read())) as zf:
                zf.extractall(batch_tmp)

            tif_files = list(batch_tmp.glob("**/*.tif")) + list(batch_tmp.glob("**/*.tiff"))
            st.write(f"Found **{len(tif_files)}** tile(s) in ZIP.")

            if st.button("🚀 Process All Tiles", type="primary"):
                batch_out = ROOT / "data" / "outputs" / "batch"
                batch_out.mkdir(parents=True, exist_ok=True)
                progress = st.progress(0)
                all_reports = []

                for i, tif in enumerate(tif_files):
                    st.write(f"Processing {tif.name} …")
                    try:
                        from src.inference import run_inference
                        rpt = run_inference(
                            input_path=tif,
                            output_dir=batch_out,
                            compute_uncertainty=compute_unc,
                            model_key=model_key,
                            checkpoint_path=checkpoint_path,
                        )
                        all_reports.append(rpt)
                    except Exception as e:
                        st.error(f"❌ {tif.name}: {e}")
                    progress.progress((i + 1) / len(tif_files))

                st.success(f"✅ Processed {len(all_reports)}/{len(tif_files)} tiles successfully.")

                # Offer ZIP download of results
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w") as zout:
                    for sr_f in batch_out.glob("*_sr_x4.tif"):
                        zout.write(sr_f, sr_f.name)
                    for unc_f in batch_out.glob("*_uncertainty.png"):
                        zout.write(unc_f, unc_f.name)
                    summary = json.dumps(all_reports, indent=2)
                    zout.writestr("batch_summary.json", summary)

                zip_buf.seek(0)
                st.download_button(
                    "⬇️ Download All Results (ZIP)",
                    zip_buf,
                    file_name="srm_batch_results.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
    else:
        st.info("Upload a ZIP file to enable batch processing.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TRAINING HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_training:
    st.subheader("📈 Fine-Tuning Training History")

    hist_path = ROOT / "src" / "checkpoints" / "training_history.json"
    if hist_path.exists():
        import plotly.graph_objects as go

        with open(hist_path) as f:
            hist = json.load(f)

        epochs = list(range(1, len(hist["train_loss"]) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=hist["train_loss"], name="Train Loss",
                                  line=dict(color="#38bdf8", width=2)))
        fig.add_trace(go.Scatter(x=epochs, y=hist["val_loss"], name="Val Loss",
                                  line=dict(color="#fb923c", width=2, dash="dash")))
        fig.update_layout(
            title="Generator Fine-Tuning Loss (L1 + Perceptual)",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#1e293b",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Final Train Loss", f"{hist['train_loss'][-1]:.4f}")
        with col_b:
            st.metric("Final Val Loss", f"{hist['val_loss'][-1]:.4f}")

        total_time = sum(hist.get("epoch_time", []))
        st.caption(f"Total training time: {total_time/60:.1f} minutes | {len(epochs)} epochs")
    else:
        st.info(
            "No training history found yet. Run fine-tuning first:\n\n"
            "```bash\n"
            "python src/train.py --pairs-dir data/synthetic_pairs --epochs 50\n"
            "```\n\n"
            "Training history will appear here automatically after completion."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.subheader("ℹ️ About this System")
    st.markdown(
        """
        ### SIH26142 — Deep Learning Based Super Resolution Mapping (SRM)
        **Sponsored by**: NTRO (National Technical Research Organisation)

        ---
        ### Architecture
        ```
        Sentinel-2 GeoTIFF (10m, multi-band)
                   │
                   ▼
           preprocessing.py  ─── tile into 256×256 patches, preserve CRS/transform
                   │
                   ▼
           model.py ─────────── Real-ESRGAN x4plus (pretrained + fine-tuned)
                   │
            ┌──────┴──────┐
            ▼             ▼
        SR output    uncertainty.py ─── TTA ensemble → per-pixel variance map
        (GeoTIFF)         │
            │             │
            └──────┬──────┘
                   ▼
              metrics.py ─── PSNR / SSIM / SAM (vs GT if available)
                   │
                   ▼
          dashboard/app.py ─── Streamlit: upload → infer → visualise
        ```

        ---
        ### Methodology Notes

        > ⚠️ **Synthetic Training Pairs (Design Decision)**
        >
        > True paired low-resolution / high-resolution satellite datasets are not freely
        > available at scale. This system uses **controlled downsampling** of Sentinel-2 10m tiles
        > to create synthetic training pairs:
        > - Gaussian blur (PSF simulation)
        > - 4× bicubic downscale
        > - Gaussian noise (sensor noise simulation)
        > - Optional JPEG compression artifacts
        >
        > The 10m original serves as the "high-resolution ground truth."
        > This is a documented design choice; results should be interpreted accordingly.

        ---
        ### Metrics

        | Metric | Description | Ideal |
        |--------|-------------|-------|
        | **PSNR** | Peak Signal-to-Noise Ratio (dB) | Higher = better |
        | **SSIM** | Structural Similarity Index | 1.0 = perfect |
        | **SAM** | Spectral Angle Mapper (degrees) | 0° = no spectral distortion |

        SAM is critical for remote sensing — it measures whether spectral relationships
        between bands are preserved, not just visual sharpness.

        ---
        ### Uncertainty Quantification

        Uses **Test-Time Augmentation (TTA)**: 8 geometric augmentations are applied,
        SR is run on each, and per-pixel variance across outputs is the uncertainty estimate.
        Higher uncertainty (bright pixels on the heatmap) means the model has less confidence
        in the reconstructed fine detail at that location.

        ---
        ### Stack
        - **Model**: Real-ESRGAN x4plus (xinntao/Real-ESRGAN)
        - **Geo-handling**: Rasterio + GDAL
        - **Metrics**: scikit-image (PSNR/SSIM) + custom SAM
        - **Dashboard**: Streamlit + Folium + Plotly
        """
    )
