# Cara Deploy ke Google Cloud Run

## Struktur folder
```
ebvl-backend/
├── main.py
├── requirements.txt
├── Dockerfile
└── ebvl_embeddings.json   ← copy dari assets app kamu
```

## Langkah deploy

### 1. Install Google Cloud CLI
https://cloud.google.com/sdk/docs/install

### 2. Login & set project
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### 3. Build & deploy ke Cloud Run
```bash
cd ebvl-backend
gcloud run deploy ebvl-backend \
  --source . \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --set-env-vars API_KEY=sk-xxx,FONNTE_TOKEN=xxx,BASE_URL=https://ai.dinoiki.com/v1
```

Setelah deploy, kamu akan dapat URL seperti:
https://ebvl-backend-xxxx-as.a.run.app

### 4. Set webhook di Fonnte
- Login ke https://fonnte.com
- Pilih device WA kamu
- Set Webhook URL: https://ebvl-backend-xxxx-as.a.run.app/webhook
- Method: POST

## Environment variables
| Key              | Value                          |
|------------------|-------------------------------|
| API_KEY          | API key OpenAI/dinoiki kamu   |
| BASE_URL         | https://ai.dinoiki.com/v1     |
| FONNTE_TOKEN     | Token dari fonnte.com         |
| SIMILARITY_THR   | 0.30 (opsional)               |

## Test manual
```bash
curl -X POST https://ebvl-backend-xxxx-as.a.run.app/webhook \
  -H "Content-Type: application/json" \
  -d '{"sender": "6281234567890", "message": "halo"}'
```

## Catatan
- File ebvl_embeddings.json harus ada di folder yang sama
- Cloud Run gratis sampai 2 juta request/bulan
- Auto-scale: mati sendiri kalau tidak ada request, hidup sendiri kalau ada
