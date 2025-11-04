import os, base64, requests
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Kartunin Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://kartunin.netlify.app", "http://localhost:8080", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}

@app.post("/v1/gemini")
async def gemini_cartoonize(file: UploadFile = File(...)):
    if not GEMINI_API_KEY:
        return {"error": "Missing GEMINI_API_KEY"}
    img = await file.read()
    b64 = base64.b64encode(img).decode()

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image-preview:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{
            "parts": [
                {"text": "Ubah foto ini jadi kartun 2D clean, outline tegas, warna flat; wajah tetap mirip; rapikan noise."},
                {"inline_data": {"mime_type": file.content_type or "image/jpeg", "data": b64}}
            ]
        }]
    }

    r = requests.post(url, json=payload, timeout=60)
    try:
        return r.json()
    except Exception:
        return {"error": "Gemini raw response", "text": r.text, "status": r.status_code}
