from fastapi import FastAPI
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
        content.append({"type": "image_url", "image_url": {"url": url}})

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        max_tokens=1024,
        messages=[{"role": "user", "content": content}],
    )
    return ChatResponse(reply=response.choices[0].message.content)
