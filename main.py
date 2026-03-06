import json
import math
import os
import re
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ─── Config ───────────────────────────────────────────────────────────────────
API_KEY        = os.environ.get("API_KEY", "sk-70303af38b561de6712b6f2f91f6a755e5bc388a7d8ab262")
BASE_URL       = os.environ.get("BASE_URL", "https://ai.dinoiki.com/v1")
FONNTE_TOKEN   = os.environ.get("FONNTE_TOKEN", "AkRPLwk1PmsrDvjXYf37")
SIMILARITY_THR = float(os.environ.get("SIMILARITY_THR", "0.50"))

ADMIN_NUMBER     = os.environ.get("ADMIN_NUMBER", "6285261781320")
NO_ANSWER_SIGNAL = "TIDAK_ADA_JAWABAN"

# ─── Load embeddings sekali saat startup ──────────────────────────────────────
EMBEDDINGS: list[dict] = json.loads(Path("ebvl_embeddings.json").read_text())
print(f"✅ Loaded {len(EMBEDDINGS)} embeddings")

app = FastAPI()

# ─── Greeting detection ───────────────────────────────────────────────────────
GREETING_RE = re.compile(
    r"^(halo|hai|hi|hei|hey|selamat\s(pagi|siang|sore|malam)|assalamu'?alaikum|permisi|hola|hello|pagi)[!.,?]?\s*$",
    re.IGNORECASE
)

def is_greeting(text: str) -> bool:
    return bool(GREETING_RE.match(text.strip()))

# ─── Cosine similarity ────────────────────────────────────────────────────────
def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)

# ─── Get embedding ────────────────────────────────────────────────────────────
async def get_embedding(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{BASE_URL}/embeddings",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={"model": "text-embedding-3-small", "input": text},
        )
        return resp.json()["data"][0]["embedding"]

# ─── Retrieve top-K chunks ────────────────────────────────────────────────────
def retrieve_top_k(question_embedding: list[float], k: int = 3) -> list[dict]:
    scored = [
        {**item, "score": cosine_similarity(question_embedding, item["embedding"])}
        for item in EMBEDDINGS
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]

# ─── Call LLM ─────────────────────────────────────────────────────────────────
async def ask_llm(question: str, context: str) -> str | None:
    system_prompt = f"""Anda adalah admin resmi sistem EBVL Pertamina Patra Niaga.

ATURAN PALING PENTING — TIDAK BOLEH DILANGGAR:
Anda hanya boleh menjawab menggunakan informasi yang ada di bagian REFERENSI di bawah ini.
Jika informasi tidak ada di REFERENSI, Anda WAJIB balas hanya dengan: {NO_ANSWER_SIGNAL}
Tidak ada pengecualian. Tidak peduli seberapa relevan pertanyaannya.

YANG DILARANG KERAS:
- DILARANG mengarang jawaban meskipun Anda merasa tahu jawabannya.
- DILARANG menyarankan pengguna untuk menghubungi pihak lain, tim support, atau siapapun.
- DILARANG memberikan saran, alternatif, atau opini jika tidak ada di referensi.
- DILARANG menyimpulkan, berasumsi, atau mengisi kekosongan informasi.
- DILARANG bertanya balik kepada pengguna.
- DILARANG menggunakan simbol * atau ** atau markdown apapun.
- DILARANG menggunakan kalimat pembuka seperti "Halo!", "Baik,", "Tentu,", "Saya bantu ya".
- DILARANG menggunakan kalimat penutup seperti "Semoga membantu", "Silakan tanya lagi".
- DILARANG memperkenalkan diri kecuali saat menjawab sapaan.
- DILARANG balas jika pengguna mengajak bertemu atau diskusi langsung — balas: {NO_ANSWER_SIGNAL}

CARA MENJAWAB SAPAAN:
Jika pengguna menyapa (halo, hai, selamat pagi, dll), balas dengan ramah, perkenalkan diri sebagai admin EBVL, dan tawarkan bantuan.

JIKA PENGGUNA MEMINTA PANDUAN ATAU MANUAL EBVL:
Jawab: "Berikut saya lampirkan manual book EBVL: https://drive.google.com/file/d/1XM5vFMZFh4PtmXFgif-BGfOb0Af3KfTm/view?usp=sharing"

GAYA BAHASA:
- Bahasa Indonesia yang sopan, ramah, dan lugas.
- Singkat dan langsung ke inti jawaban.
- Format teks biasa untuk WhatsApp, tanpa markdown.

{f"REFERENSI:{chr(10)}{context}" if context else "REFERENSI: (kosong — tidak ada referensi yang tersedia)"}"""

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "claude-haiku-4-5",
                "temperature": 0.0,
                "max_tokens": 512,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": question},
                ],
            },
        )
        answer = resp.json()["choices"][0]["message"]["content"].strip()

        if NO_ANSWER_SIGNAL in answer:
            return None

        return answer

# ─── Send WhatsApp via Fonnte ─────────────────────────────────────────────────
async def send_whatsapp(target: str, message: str) -> None:
    async with httpx.AsyncClient(timeout=30) as client:
        await client.post(
            "https://api.fonnte.com/send",
            headers={"Authorization": FONNTE_TOKEN},
            data={"target": target, "message": message},
        )

# ─── Notifikasi ke admin jika bot skip ────────────────────────────────────────
async def notify_admin(sender: str, message: str) -> None:
    notif = (
        f"⚠️ Pesan belum terjawab oleh bot\n"
        f"Dari: {sender}\n"
        f"Pesan: {message}\n\n"
        f"Silakan balas manual."
    )
    await send_whatsapp(ADMIN_NUMBER, notif)

# ─── RAG core ─────────────────────────────────────────────────────────────────
async def process_rag(question: str) -> str | None:
    if is_greeting(question):
        return await ask_llm(question, "")

    embedding  = await get_embedding(question)
    top_chunks = retrieve_top_k(embedding, k=3)
    best_score = top_chunks[0]["score"] if top_chunks else 0

    if best_score < SIMILARITY_THR:
        return None

    context = "\n\n".join(
        c["text"] for c in top_chunks if c["score"] >= SIMILARITY_THR
    )

    if not context:
        return None

    return await ask_llm(question, context)

# ─── Webhook endpoint (Fonnte kirim ke sini) ──────────────────────────────────
@app.post("/webhook")
async def webhook(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = dict(await request.form())

    print("BODY LENGKAP:", body)

    sender  = body.get("pengirim") or body.get("sender") or body.get("from", "")
    message = body.get("pesan") or body.get("message") or body.get("text", "")

    if not sender or not message:
        return JSONResponse({"status": "ignored"})

    # Abaikan pesan dari diri sendiri
    if body.get("device") and sender == body.get("device"):
        return JSONResponse({"status": "self"})

    print(f"📩 [{sender}]: {message}")

    answer = await process_rag(message)

    if answer is None:
        print(f"⏭️ No answer for [{sender}], skipping...")
        return JSONResponse({"status": "skipped"})

    await send_whatsapp(sender, answer)
    print(f"✅ Replied to {sender}")

    return JSONResponse({"status": "ok"})

# ─── Health check ─────────────────────────────────────────────────────────────
@app.get("/")
async def health():
    return {"status": "ok", "embeddings": len(EMBEDDINGS)}