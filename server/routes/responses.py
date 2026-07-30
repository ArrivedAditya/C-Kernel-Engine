from __future__ import annotations

import json
import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from schemas.common import ResponseStatus
from schemas.response import CreateResponseRequest, Response

router = APIRouter()

_response_store: dict[str, Response] = {}


def _make_mock_response(model: str) -> Response:
    resp_id = f"resp_{uuid.uuid4().hex[:24]}"
    now = int(time.time())
    resp = Response(
        id=resp_id,
        created_at=now,
        status=ResponseStatus.completed,
        model=model,
        output_items=[],
    )
    _response_store[resp_id] = resp
    return resp


@router.post("/responses", response_model=Response)
def create_response(body: CreateResponseRequest):
    resp = _make_mock_response(body.model)
    return resp


@router.get("/responses/{response_id}", response_model=Response)
def get_response(response_id: str):
    resp = _response_store.get(response_id)
    if not resp:
        raise HTTPException(status_code=404, detail="Response not found")
    return resp


@router.post("/responses/{response_id}/cancel", response_model=Response)
def cancel_response(response_id: str):
    resp = _response_store.get(response_id)
    if not resp:
        raise HTTPException(status_code=404, detail="Response not found")
    resp.status = ResponseStatus.cancelled
    return resp


@router.post("/responses/{response_id}/stream")
def stream_response(response_id: str):
    resp = _response_store.get(response_id)
    if not resp:
        raise HTTPException(status_code=404, detail="Response not found")

    events = [
        {"type": "response.created", "response": resp.model_dump(mode="json")},
        {"type": "response.in_progress", "response": resp.model_dump(mode="json")},
        {"type": "response.completed", "response": resp.model_dump(mode="json")},
    ]

    async def event_generator():
        for event in events:
            yield f"event: {event['type']}\n"
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
