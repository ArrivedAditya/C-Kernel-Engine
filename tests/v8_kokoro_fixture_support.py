"""Pinned Kokoro fixture loading without compilation or model execution."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

PREFIX = "phoneme_encoder.encoder.albert_layer_groups.0.albert_layers.0.attention"
BASE_WEIGHTS = {
    "word": "phoneme_encoder.embeddings.word_embeddings.weight",
    "position": "phoneme_encoder.embeddings.position_embeddings.weight",
    "token_type": "phoneme_encoder.embeddings.token_type_embeddings.weight",
    "gamma": "phoneme_encoder.embeddings.LayerNorm.weight",
    "beta": "phoneme_encoder.embeddings.LayerNorm.bias",
    "projection_weight": "phoneme_encoder.encoder.embedding_hidden_mapping_in.weight",
    "projection_bias": "phoneme_encoder.encoder.embedding_hidden_mapping_in.bias",
}


def load_first_layer_fixtures(cls):
    fixtures = ROOT / "tests/fixtures/tts"
    cls.embed = dict(np.load(fixtures / "bert_embedding_pinned.npz"))
    cls.projection = dict(np.load(fixtures / "kokoro_projection_pinned.npz"))
    fixture_path = fixtures / "kokoro_qkv_pinned.npz"
    cls.qkv = dict(np.load(fixture_path))
    cls.meta = json.loads(fixture_path.with_suffix(".json").read_text())
    context_path = fixtures / "kokoro_attention_context_pinned.npz"
    cls.context = dict(np.load(context_path))
    cls.context_meta = json.loads(context_path.with_suffix(".json").read_text())
    if hashlib.sha256(context_path.read_bytes()).hexdigest() != cls.context_meta["fixture_sha256"]:
        raise RuntimeError("attention-context fixture hash mismatch")
    if hashlib.sha256(fixture_path.read_bytes()).hexdigest() != cls.meta["fixture_sha256"]:
        raise RuntimeError("Q/K/V fixture hash mismatch")
    layer_path = fixtures / "kokoro_albert_layer_pinned.npz"
    cls.layer = dict(np.load(layer_path))
    cls.layer_meta = json.loads(layer_path.with_suffix(".json").read_text())
    if hashlib.sha256(layer_path.read_bytes()).hexdigest() != cls.layer_meta["fixture_sha256"]:
        raise RuntimeError("first-layer oracle fixture hash mismatch")
    tensors = {BASE_WEIGHTS[key]: cls.embed[key] for key in
               ("word", "position", "token_type", "gamma", "beta")}
    tensors[BASE_WEIGHTS["projection_weight"]] = cls.projection["weight"]
    tensors[BASE_WEIGHTS["projection_bias"]] = cls.projection["bias"]
    for name in ("query", "key", "value"):
        for suffix in ("weight", "bias"):
            tensors[f"{PREFIX}.{name}.{suffix}"] = cls.qkv[f"{name}_{suffix}"]
    for name, tensor in cls.layer.items():
        if name.startswith("weight__"):
            canonical = PREFIX.removesuffix(".attention") + "." + name.removeprefix("weight__").replace("__", ".")
            tensors[canonical] = tensor
    return tensors
