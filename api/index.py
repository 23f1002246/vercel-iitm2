# api/index.py — add this to your existing Vercel project (or deploy separately)
import os, json, httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "")
AIPIPE_URL   = "https://aipipe.org/openai/v1/chat/completions"

class InvoiceRequest(BaseModel):
    invoice_text: str

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
- date MUST be ISO format YYYY-MM-DD. Convert any format (e.g. "April 3, 2026" → "2026-04-03").
- amount is the SUBTOTAL before tax (NOT the grand total).
- tax is the tax amount only (NOT the rate/percentage).
- currency: infer from symbols (Rs./₹ → INR, $ → USD, € → EUR, £ → GBP) or explicit text.
- Return null for any field not found.
- Return ONLY the JSON object, no extra text."""

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

    # Ensure all 6 keys are always present
    keys = ["invoice_no", "date", "vendor", "amount", "tax", "currency"]
    return {k: result.get(k, None) for k in keys}

# Keep existing /answer-image endpoint if merging into same project
@app.get("/")
def root():
    return {"status": "ok", "endpoints": ["POST /extract", "POST /answer-image"]}
