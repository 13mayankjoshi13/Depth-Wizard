# Streamlit Community Cloud Deployment Guide
## SIH26142 — Sentinel-2 Super-Resolution Mapping Dashboard

---

## Prerequisites

- A **GitHub account** (free)
- A **Streamlit Community Cloud account** (free) — sign up at https://share.streamlit.io
- The `srm-project/` folder pushed to a GitHub repository

---

## Step 1 — Push to GitHub

Open a terminal in `srm-project/` and run:

```bash
git init
git add .
git commit -m "Initial commit: SRM project SIH26142"

# Create a new GitHub repo named 'srm-project' via github.com,
# then link it:
git remote add origin https://github.com/YOUR_USERNAME/srm-project.git
git branch -M main
git push -u origin main
```

> **Important**: The `.gitignore` already excludes:
> - `.venv/` (Python environment — rebuilt by Streamlit Cloud)
> - `data/raw/` (your private satellite tiles)
> - `src/checkpoints/*.pth` (model weights — auto-downloaded on first run)
>
> These are NOT committed; Streamlit Cloud downloads model weights automatically.

---

## Step 2 — Rename requirements file for Cloud

Streamlit Cloud uses `requirements.txt` at the repo root. The cloud version uses
CPU-only PyTorch and basicsr from GitHub to avoid Python 3.12+ compat issues:

```bash
# In the repo root:
cp requirements_cloud.txt requirements.txt
git add requirements.txt
git commit -m "Switch to cloud requirements (CPU-only PyTorch, basicsr from GitHub)"
git push
```

---

## Step 3 — Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io  
2. Click **"New app"**
3. Connect your GitHub account if not done already
4. Fill in:

   | Field | Value |
   |-------|-------|
   | **Repository** | `YOUR_USERNAME/srm-project` |
   | **Branch** | `main` |
   | **Main file path** | `dashboard/app.py` |

5. Click **"Deploy!"**

Streamlit Cloud will:
- Install system packages from `packages.txt` (GDAL/rasterio deps)
- Install Python packages from `requirements.txt`
- Start the app

First startup takes **5–10 minutes** (downloading ML packages + model weights from GitHub).

---

## Step 4 — Verify the deployment

Once deployed, you'll get a URL like:
```
https://YOUR_USERNAME-srm-project-dashboard-app-XXXX.streamlit.app
```

Test it by:
1. Going to the **About** tab — should render with no errors
2. Uploading a small Sentinel-2 GeoTIFF (a cropped 256×256 patch works)
3. Clicking **"Run Super-Resolution"** — model auto-downloads on first run (~65MB)

> ⚠️ **Cloud inference is CPU-only** — expect ~30–60 seconds per tile.
> The dashboard automatically detects CPU and disables FP16 inference.

---

## Cloud Limitations vs Local

| Feature | Local | Streamlit Cloud |
|---------|-------|-----------------|
| GPU inference | ✅ Fast (CUDA) | ❌ CPU only |
| Training | ✅ `train.py` | ❌ Not available |
| Tile size limit | 4GB+ | ~200MB (RAM limit) |
| Batch processing | ✅ | ✅ (smaller batches) |
| Uncertainty maps | ✅ (8 TTA) | ✅ (reduce to 2–4 TTA) |

---

## Troubleshooting

### "Module not found: basicsr"
→ Make sure `requirements_cloud.txt` was copied as `requirements.txt` and pushed.

### "GDAL not found" / rasterio import error
→ Check `packages.txt` is in the repo root and contains the GDAL deps.

### App is slow / times out
→ In the sidebar, reduce:
- **Inference tile size** → 128
- **TTA augmentations** → 2
- Upload smaller tiles (256×256 crops)

### Model weights not downloading
→ The model downloads from GitHub Releases on first run. If blocked:
1. Download `RealESRGAN_x4plus.pth` manually from:
   https://github.com/xinntao/Real-ESRGAN/releases
2. Add it to `src/checkpoints/` and re-push

---

## Environment Variables (Optional)

Set these in the Streamlit Cloud app **Secrets** (⚙️ Settings → Secrets):

```toml
# No secrets required for the base system.
# Add if you later integrate paid APIs:
# COPERNICUS_USER = "your@email.com"
# COPERNICUS_PASSWORD = "yourpassword"
```

---

## Updating the deployed app

Any `git push` to `main` automatically triggers a redeployment on Streamlit Cloud.

```bash
# Make changes, then:
git add -A
git commit -m "Your update message"
git push
```

---

*Generated for SIH26142 — Deep Learning Based Super Resolution Mapping*  
*Sponsored by: NTRO | Python 3.11 | Real-ESRGAN x4plus | Streamlit 1.31+*
