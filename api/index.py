# For Vercel deployment: save as api/index.py
# requirements.txt: fastapi httpx

import os, httpx
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

class ImageQA(BaseModel):
    image_base64: str
    question: str

@app.post("/answer-image")
async def answer_image(body: ImageQA):
    prompt = (
        f"{body.question}\n\n"
        "Return ONLY the raw answer — no units, no currency symbols, no extra text. "
        "For numeric answers return just the number (e.g. 4089.35)."
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

@app.get("/")
def root():
    return {"status": "ok"}
