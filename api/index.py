# api/index.py — COMBINED: /answer-image + /extract
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

IMAGE_SYSTEM = """You are a precise data extraction assistant for images.
Your job is to answer questions about charts, receipts, invoices, tables, and diagrams.

Rules:
- Read EVERY number visible in the image carefully before answering.
- For bar charts: read the exact value of EACH bar (check axis scale and labels). If asked for a total/sum, add ALL bars.
- For receipts/invoices: find the specific field asked (subtotal, tax, grand total, item price, etc.).
- For tables: read row/column values exactly as printed.
- Return ONLY the raw numeric or text answer — no units, no currency symbols, no commas, no explanation.
- For numbers: return as a plain decimal (e.g. 4089.35 not $4,089.35 or Rs. 4,089).
- Do NOT round unless the answer is already a whole number.
- If the question asks for a sum/total, compute it yourself from the individual values shown."""

@app.post("/answer-image")
async def answer_image(body: ImageQA):
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": IMAGE_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{body.image_base64}",
                                   "detail": "high"}},
                    {"type": "text",
                     "text": f"{body.question}\n\nRemember: return ONLY the raw answer value, nothing else."}
                ]
            }
        ],
        "max_tokens": 200,
        "temperature": 0
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            AIPIPE_URL,
            json=payload,
            headers={"Authorization": f"Bearer {AIPIPE_TOKEN}"}
        )
        r.raise_for_status()
        answer = r.json()["choices"][0]["message"]["content"].strip()
        # Strip any accidental currency symbols or commas
        answer = answer.replace(",", "").replace("$", "").replace("£", "")
        answer = answer.replace("₹", "").replace("€", "").replace("Rs.", "").strip()
    return {"answer": answer}

# ─── /extract ─────────────────────────────────────────────────────────────────

EXTRACT_SYSTEM = """You are an invoice data extraction assistant.
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
- currency: infer from symbols (Rs./Rs/INR -> INR, $ -> USD, EUR/euro -> EUR, pound/GBP -> GBP).
- Indian number format: Rs. 1,40,000 = 140000 (not 140 or 1400).
- Return null for any field not found.
- Return ONLY the JSON object, no extra text, no markdown."""

@app.post("/extract")
async def extract_invoice(body: InvoiceRequest):
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user",   "content": body.invoice_text}
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 300,
        "temperature": 0
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
