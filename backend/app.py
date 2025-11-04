# app.py
# Simple CPU-only cartoonizer backend compatible with the HTML UI
# Endpoints:
#   POST /v1/jobs  (multipart: file, style, strength)
#   GET  /v1/jobs/{job_id}
# Serves output images at /files/<filename>

import os
import uuid
from typing import Dict, Optional

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import numpy as np
from PIL import Image
import cv2

APP_PORT = int(os.environ.get("PORT", 8000))
DATA_DIR = os.environ.get("DATA_DIR", "./data")
IN_DIR = os.path.join(DATA_DIR, "inputs")
OUT_DIR = os.path.join(DATA_DIR, "outputs")
os.makedirs(IN_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

app = FastAPI(title="Cartoonizer Backend")

# Allow local dev origins (tighten in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve output files
app.mount("/files", StaticFiles(directory=OUT_DIR), name="files")

# In-memory job store (simple for personal use)
class Job(BaseModel):
    job_id: str
    status: str  # queued, running, done, error
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    error: Optional[str] = None
    progress_percent: Optional[int] = 0

JOBS: Dict[str, Job] = {}

@app.post("/v1/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    style: str = Form("anime_v1"),
    strength: float = Form(0.6),
):
    job_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename or "image.jpg")[1].lower() or ".jpg"
    in_path = os.path.join(IN_DIR, f"{job_id}{ext}")
    with open(in_path, "wb") as f:
        f.write(await file.read())

    job = Job(job_id=job_id, status="queued", input_path=in_path, progress_percent=0)
    JOBS[job_id] = job

    background_tasks.add_task(process_job, job_id, style, strength)
    return {"job_id": job_id, "status": job.status}

@app.get("/v1/jobs/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return {"status": "error", "error": "job not found"}

    out_url = None
    if job.output_path:
        fname = os.path.basename(job.output_path)
        out_url = f"/files/{fname}"

    return {
        "job_id": job.job_id,
        "status": job.status,
        "error": job.error,
        "progress_percent": job.progress_percent,
        "output_url": out_url,
    }

# ---- Image processing ----
def process_job(job_id: str, style: str, strength: float):
    job = JOBS[job_id]
    try:
        job.status = "running"
        job.progress_percent = 5

        img = Image.open(job.input_path).convert("RGB")
        img_np = np.array(img)
        bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        job.progress_percent = 20

        h, w = bgr.shape[:2]
        max_side = max(h, w)
        if max_side > 2048:
            scale = 2048 / max_side
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        job.progress_percent = 35

        out = cartoonize(bgr, style=style, strength=float(strength))
        job.progress_percent = 80

        fname = f"{job_id}.webp"
        out_path = os.path.join(OUT_DIR, fname)
        cv2.imwrite(out_path, out, [int(cv2.IMWRITE_WEBP_QUALITY), 95])

        job.output_path = out_path
        job.status = "done"
        job.progress_percent = 100
    except Exception as e:
        job.status = "error"
        job.error = str(e)
        job.progress_percent = None

def cartoonize(bgr: np.ndarray, style: str = "anime_v1", strength: float = 0.6) -> np.ndarray:
    """Fast cartoonization using OpenCV only (CPU)."""
    strength = float(np.clip(strength, 0.0, 1.0))

    if style == "sketch_v1":
        gray, _ = cv2.pencilSketch(
            bgr,
            sigma_s=int(60 + 40*strength),
            sigma_r=0.07 + 0.1*strength,
            shade_factor=0.03 + 0.05*strength
        )
        return gray

    if style == "flat_v1":
        smooth = cv2.bilateralFilter(
            bgr, d=9,
            sigmaColor=75 + 125*strength,
            sigmaSpace=75 + 125*strength
        )
        Z = smooth.reshape((-1,3)).astype(np.float32)
        K = 8 - int(5*strength)  # 8..3 clusters
        K = max(3, min(8, K))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(Z, K, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
        centers = np.uint8(centers)
        quant = centers[labels.flatten()].reshape(bgr.shape)
        edges = edge_mask(bgr, thresh=30)
        out = cv2.bitwise_and(quant, quant, mask=edges)
        return out

    if style == "comic_v1":
        styl = cv2.stylization(
            bgr,
            sigma_s=int(80 + 40*strength),
            sigma_r=0.2 + 0.2*strength
        )
        edges = edge_mask(bgr, thresh=50)
        out = cv2.bitwise_and(styl, styl, mask=edges)
        return out

    # default anime_v1
    styl = cv2.stylization(
        bgr,
        sigma_s=int(60 + 60*strength),
        sigma_r=0.05 + 0.25*strength
    )
    edges = edge_mask(bgr, thresh=40)
    out = cv2.bitwise_and(styl, styl, mask=edges)
    return out

def edge_mask(bgr: np.ndarray, thresh: int = 40) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.Canny(gray, thresh, thresh*3)
    edges = cv2.bitwise_not(edges)
    return edges

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=APP_PORT)
