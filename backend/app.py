# backend/app.py (hanya endpoint /v1/gemini yang diubah)
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
    img_bytes = await file.read()
    b64 = base64.b64encode(img_bytes).decode()

    prompt = (
        "Ubah foto ini menjadi ilustrasi kartun 2D bergaya digital: "
        "outline tebal/rapi, warna flat/bersih, proporsi wajah tetap mirip, "
        "minim noise dan artefak."
    )

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash-image-preview:generateContent?key={GEMINI_API_KEY}"
    )

    payload = {
        # 👉 WAJIB untuk minta keluaran berupa gambar
        "generationConfig": { "response_mime_type": "image/png" },

        # (opsional) boleh ditambah supaya lebih longgar
        # "safetySettings": [{"category":"HARM_CATEGORY_HATE_SPEECH","threshold":"BLOCK_NONE"}],

        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": file.content_type or "image/jpeg", "data": b64}}
            ]
        }]
    }

    r = requests.post(url, json=payload, timeout=120)
    # Return mentah supaya frontend bisa baca parts/inline_data
    try:
        return r.json()
    except Exception:
        return {"error": "Gemini raw response", "status": r.status_code, "text": r.text}
