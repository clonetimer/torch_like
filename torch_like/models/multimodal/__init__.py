"""Multimodal domain base interfaces and models."""

from torch_like.models.multimodal.clip import CLIP, CLIPConfig
from torch_like.models.multimodal.common import MultimodalBatch, MultimodalConfig, MultimodalModel
from torch_like.models.registry import register_model_catalog

model_zoo = {
    "CLIP": {
        "cls": CLIP,
        "family": "image_text_dual_encoder",
        "layer": "Multimodal / image-text representation",
        "integration_level": "framework_oriented",
        "supports": ["image_text_retrieval", "zero_shot_transfer", "dual_encoder", "similarity_logits"],
        "paper": "Learning Transferable Visual Models From Natural Language Supervision (ICML 2021)",
        "description": "CLIP-style image-text dual encoder returning normalized embeddings and image-text similarity logits",
        "complexity": "O(B * image_patches^2 * D + B * text_len^2 * D + B^2 * embed_dim)",
    },
}

register_model_catalog("multimodal", model_zoo)

__all__ = [
    "MultimodalConfig",
    "MultimodalBatch",
    "MultimodalModel",
    "CLIPConfig",
    "CLIP",
    "model_zoo",
]
