from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional

try:
    load_dotenv()
except Exception:
    pass

app = FastAPI()
client = OpenAI()


class ChatRequest(BaseModel):
    message: str
    image_url: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    content: list = [{"type": "text", "text": request.message}]

    url = (request.image_url or "").strip()
    if url:
        if not url.startswith(("http://", "https://", "data:")):
            url = f"data:image/jpeg;base64,{url}"
        content.append({"type": "image_url", "image_url": {"url": url}})

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        max_tokens=1024,
        messages=[{"role": "user", "content": content}],
    )
    return ChatResponse(reply=response.choices[0].message.content)


OCR_SYSTEM_PROMPT = (
    "You are an OCR specialist for reading measurement instruments and digital displays. "
    "Look at this image and extract ALL visible text and numbers. Rules:\n"
    '- Prioritize numeric readings shown on LCD/LED digital displays (e.g. "081", "25.6°C")\n'
    "- Also read printed labels, units, and text on the device body\n"
    "- Return ONLY the raw extracted text, one item per line\n"
    "- No explanations, no markdown, no formatting\n"
    "- Format: put the main digital reading on the first line, then other text below"
)


class OCRRequest(BaseModel):
    image_base64: str


class OCRResponse(BaseModel):
    text: str


@app.post("/ocr", response_model=OCRResponse)
async def ocr(request: OCRRequest) -> OCRResponse:
    if not request.image_base64.strip():
        raise HTTPException(status_code=400, detail="image_base64 is empty")

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": OCR_SYSTEM_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{request.image_base64}"
                            },
                        },
                    ],
                }
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return OCRResponse(text=response.choices[0].message.content)
