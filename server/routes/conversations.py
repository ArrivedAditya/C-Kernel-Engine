from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, HTTPException

from schemas.conversation import (
    AddConversationItemsRequest,
    Conversation,
    CreateConversationRequest,
)

router = APIRouter()

_conversation_store: dict[str, Conversation] = {}


def _make_conversation(items: list | None = None) -> Conversation:
    conv_id = f"conv_{uuid.uuid4().hex[:24]}"
    conv = Conversation(
        id=conv_id,
        created_at=int(time.time()),
        items=items or [],
    )
    _conversation_store[conv_id] = conv
    return conv


@router.post("/conversations", response_model=Conversation)
def create_conversation(body: CreateConversationRequest):
    return _make_conversation(body.items)


@router.get("/conversations/{conversation_id}", response_model=Conversation)
def get_conversation(conversation_id: str):
    conv = _conversation_store.get(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@router.post("/conversations/{conversation_id}/items", response_model=Conversation)
def add_conversation_items(conversation_id: str, body: AddConversationItemsRequest):
    conv = _conversation_store.get(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    conv.items.extend(body.items)
    return conv
