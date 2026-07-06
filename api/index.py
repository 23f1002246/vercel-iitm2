# api/index.py — COMBINED: both /answer-image and /extract
import os, json, httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "")
AIPIPE_URL   = "https://aipipe.org/openai/v1/chat/completions"

# ─── Models ───────────────────────────────────────────────────────────────────

class ImageQA(BaseModel):
    image_base64: str
    question: str

class InvoiceRequest(BaseModel):
    invoice_text: str

# ─── /answer-image ────────────────────────────────────────────────────────────

@app.post("/answer-image")
async def answer_image(body: ImageQA):
    prompt = (
        f"{body.question}\n\n"
        "Look carefully at every number visible in the image. "
        "If asked for a total or sum, add up ALL the values shown. "
        "Return ONLY the raw answer value — no units, no currency symbols, "
        "no extra text. For numeric answers return just the number (e.g. 4089.35)."
    )
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{body.image_base64}"}},
                {"type": "text", "text": prompt}
            ]
        }],
        "max_tokens": 200
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            AIPIPE_URL,
            json=payload,
            headers={"Authorization": f"Bearer {AIPIPE_TOKEN}"}
        )
        r.raise_for_status()
        answer = r.json()["choices"][0]["message"]["content"].strip()
    return {"answer": answer}

# ─── /extract ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an invoice data extraction assistant.
Extract exactly these 6 fields from the invoice text and return ONLY valid JSON:

{
  "invoice_no": "string or null",
  "date": "YYYY-MM-DD string or null",
  "vendor": "string or null",
  "amount": number (subtotal BEFORE tax) or null,
  "tax": number (tax amount only) or null,
  "currency": "3-letter currency code e.g. INR, USD, EUR or null"
}

Rules:
- date MUST be ISO format YYYY-MM-DD. Convert any format (e.g. "April 3, 2026" -> "2026-04-03").
- amount is the SUBTOTAL before tax (NOT the grand total).
- tax is the tax amount only (NOT the rate/percentage).
- currency: infer from symbols (Rs./Rs -> INR, $ -> USD, EUR/euro -> EUR, pound/GBP -> GBP).
- Indian number format: Rs. 1,40,000 = 140000 (not 140 or 1400).
- Return null for any field not found.
- Return ONLY the JSON object, no extra text, no markdown."""

@app.post("/extract")
async def extract_invoice(body: InvoiceRequest):
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": body.invoice_text}
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 300
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            AIPIPE_URL,
            json=payload,
            headers={"Authorization": f"Bearer {AIPIPE_TOKEN}"}
        )
        r.raise_for_status()
        result = json.loads(r.json()["choices"][0]["message"]["content"])

    keys = ["invoice_no", "date", "vendor", "amount", "tax", "currency"]
    return {k: result.get(k, None) for k in keys}

# ─── Health check ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "endpoints": ["POST /answer-image", "POST /extract"]}
