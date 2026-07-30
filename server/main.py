from __future__ import annotations

from fastapi import FastAPI

from routes.conversations import router as conversations_router
from routes.responses import router as responses_router

app = FastAPI(title="OpenAI Responses API Compatible Server", version="0.1.0")

app.include_router(responses_router, prefix="/v1")
app.include_router(conversations_router, prefix="/v1")


@app.get("/health")
def health():
    return {"status": "ok"}
