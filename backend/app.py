# --- Tambahan di atas: ---
import os
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")  # ✅ aman, tidak hardcodefrom io import BytesIO
from fastapi import UploadFile, File
# (Sudah ada di file kamu, pastikan tidak dobel import)

HF_TOKEN = os.getenv("wHEIkqdunzkGLiNuzPfEFyaxPhYSnHyONm")
HF_MODEL = os.getenv("HUGGINGFACE_MODEL", "akhaliq/AnimeGANv2")  # default model gratis

@app.post("/v1/cartoonify")
async def cartoonify_via_hf(file: UploadFile = File(...)):
    if not HF_TOKEN:
        return {"error": "Missing HUGGINGFACE_API_TOKEN"}

    # Baca gambar dari user
    img_bytes = await file.read()
    mime = file.content_type or "image/jpeg"

    # Panggil Hosted Inference API (image-to-image: kirim data biner langsung)
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        # Biarkan API memilih Content-Type dari data (jangan paksa application/json)
        "Accept": "image/png"
    }

    try:
        r = requests.post(url, headers=headers, data=img_bytes, timeout=120)
    except requests.RequestException as e:
        return {"error": "request_failed", "message": str(e)}

    # Kalau model belum warm / rate limit, API mengembalikan JSON; kalau sukses -> image bytes
    content_type = r.headers.get("content-type", "")
    if r.status_code != 200:
        # kirimkan pesan error apa adanya untuk diagnosa di UI
        err = None
        try:
            err = r.json()
        except Exception:
            err = {"text": r.text}
        return {"error": "api_error", "status": r.status_code, "detail": err}

    if content_type.startswith("image/"):
        # Bungkus ke format yang frontend-mu sudah pahami (inline_data base64)
        b64 = base64.b64encode(r.content).decode()
        return {
            "candidates": [{
                "content": {
                    "parts": [{
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": b64
                        }
                    }]
                }
            }]
        }
    else:
        # Mungkin balasan JSON (queued / loading / rate limited)
        try:
            data = r.json()
        except Exception:
            data = {"text": r.text}
        return {"error": "no_image", "message": "Model belum mengembalikan gambar.", "raw": data}
