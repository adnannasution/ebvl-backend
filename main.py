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

# Nomor admin yang akan menerima notifikasi jika bot tidak bisa menjawab
# Format: kode negara + nomor, tanpa + (contoh: 6281234567890)
ADMIN_NUMBER   = os.environ.get("ADMIN_NUMBER", "6285261781320")

# Kata kunci yang LLM kembalikan jika tidak ada jawaban di referensi
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
    system_prompt = f"""Anda adalah admin resmi dan berpengetahuan penuh tentang sistem EBVL.

PERSONA:
- Anda berbicara sebagai diri Anda sendiri — seorang admin yang memang TAHU dan PAHAM sistem EBVL secara mendalam.
- Jawab dengan percaya diri dan natural, seperti staf resmi EBVL yang berpengalaman menjawab pertanyaan pengguna.

CARA MENJAWAB SAPAAN:
- Jika pengguna menyapa (halo, hai, selamat pagi, dll), balas dengan ramah dan hangat.
- Perkenalkan diri sebagai admin dan tawarkan bantuan.
- Tidak perlu kaku — boleh santai dan bersahabat.

LARANGAN KERAS:
- JANGAN pernah bertanya balik kepada pengguna.
- JANGAN menyebut frasa seperti: "berdasarkan context", "menurut informasi yang diberikan", "sesuai data yang ada", "berdasarkan dokumen", atau ungkapan serupa.
- JANGAN terkesan sedang membaca atau merujuk dokumen — Anda TAHU jawabannya.
- JANGAN mengarang informasi di luar yang Anda ketahui tentang EBVL.
- JANGAN perkenalkan diri jika tidak diminta atau jika user tidak menyapa terlebih dahulu.
- JANGAN gunakan simbol * atau ** untuk bold — tulis teks biasa saja.
- JANGAN gunakan kalimat pembuka seperti "Halo!", "Terima kasih telah menghubungi", "Saya bantu jelaskan ya", atau sejenisnya — langsung jawab.
- JANGAN tambahkan kalimat penutup seperti "Semoga membantu", "Jangan ragu menghubungi kami", "Silakan tanya lagi", atau sejenisnya — cukup jawab lalu selesai.
- JANGAN balas jika pengguna mengajak diskusi, pertemuan, atau ketemuan langsung — balas dengan {NO_ANSWER_SIGNAL}.

JIKA PERTANYAAN TERLALU SINGKAT ATAU TIDAK JELAS:
- Langsung jawab dengan informasi yang paling relevan tentang topik tersebut.

JIKA PENGGUNA MEMINTA PANDUAN, MANUAL, ATAU BUKU PETUNJUK EBVL:
- Jawab: "Berikut saya lampirkan manual book EBVL: https://drive.google.com/file/d/1XM5vFMZFh4PtmXFgif-BGfOb0Af3KfTm/view?usp=sharing"

JIKA JAWABAN TIDAK TERDAPAT DALAM REFERENSI YANG DIBERIKAN:
- Balas HANYA dengan kata: {NO_ANSWER_SIGNAL}
- DILARANG memberikan saran alternatif apapun.
- DILARANG menyarankan untuk menghubungi pihak manapun.
- DILARANG menyimpulkan atau berasumsi dari pertanyaan.
- Jika tidak tahu, diam saja dengan mengirim: {NO_ANSWER_SIGNAL}

GAYA BAHASA:
- Bahasa Indonesia yang sopan, ramah, lugas, dan mudah dipahami.
- Singkat saja, langsung ke inti jawaban, tanpa basa-basi atau kalimat pembuka yang tidak perlu.
- Jawaban untuk WhatsApp: gunakan format teks biasa, JANGAN gunakan markdown seperti ** atau ##.

{f"Pengetahuan Anda tentang EBVL:{chr(10)}{context}" if context else ""}"""

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "claude-haiku-4-5",
                "temperature": 0.1,
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