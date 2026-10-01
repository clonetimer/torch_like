"""Multimodal domain base interfaces and models."""

from torch_like.models.multimodal.blip import BLIP, BLIPConfig
from torch_like.models.multimodal.blip2 import BLIP2, BLIP2Config
from torch_like.models.multimodal.clip import CLIP, CLIPConfig
from torch_like.models.multimodal.common import MultimodalBatch, MultimodalConfig, MultimodalModel
from torch_like.models.multimodal.llava import LLaVA, LLaVAConfig
from torch_like.models.multimodal.siglip import SigLIP, SigLIPConfig
from torch_like.models.registry import register_model_catalog

model_zoo = {
    "SigLIP": {
        "cls": SigLIP,
        "family": "image_text_dual_encoder",
        "layer": "Multimodal / image-text representation",
        "integration_level": "paper_aligned",
        "supports": ["image_text_retrieval", "zero_shot_transfer", "dual_encoder", "sigmoid_loss"],
        "paper": "Sigmoid Loss for Language Image Pre-Training (ICCV 2023)",
        "description": "CLIP-compatible dual encoder trained with independent pairwise sigmoid loss",
        "complexity": "O(B * image_patches^2 * D + B * text_len^2 * D + B^2 * embed_dim)",
    },
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
    "BLIP": {
        "cls": BLIP,
        "family": "vision_language_understanding_generation",
        "layer": "Multimodal / vision-language generation",
        "integration_level": "paper_aligned_core",
        "supports": [
            "image_text_retrieval",
            "image_text_matching",
            "captioning",
            "cross_attention",
            "language_modeling",
        ],
        "paper": "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation (ICML 2022)",
        "description": "Vision-language model exposing dual-encoder retrieval, image-text fusion, ITM logits, and caption logits",
        "complexity": "O(B * image_patches^2 * D + B * text_len^2 * D + B * text_len * image_patches * D)",
    },
    "BLIP2": {
        "cls": BLIP2,
        "family": "vision_language_qformer",
        "layer": "Multimodal / efficient vision-language bridge",
        "integration_level": "paper_aligned_core",
        "supports": [
            "image_text_retrieval",
            "q_former",
            "image_to_language_bridge",
            "image_text_matching",
            "captioning",
        ],
        "paper": "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models (ICML 2023)",
        "description": "BLIP-2-style Q-Former bridge from image tokens to language-prefix features with retrieval, ITM, and caption logits",
        "complexity": "O(B * image_patches^2 * D + B * query_tokens * image_patches * D + B * text_len * lm_dim)",
    },
    "LLaVA": {
        "cls": LLaVA,
        "family": "visual_instruction_language_model",
        "layer": "Multimodal / visual instruction following",
        "integration_level": "framework_oriented",
        "supports": ["visual_instruction", "image_prefix", "projector", "causal_language_modeling", "generation"],
        "paper": "Visual Instruction Tuning / LLaVA (NeurIPS 2023)",
        "description": "LLaVA-style vision encoder plus projector feeding visual tokens into a causal language decoder",
        "complexity": "O(B * image_patches^2 * D + B * (image_patches + text_len)^2 * D)",
    },
}

register_model_catalog("multimodal", model_zoo)

__all__ = [
    "MultimodalConfig",
    "MultimodalBatch",
    "MultimodalModel",
    "CLIPConfig",
    "CLIP",
    "SigLIPConfig",
    "SigLIP",
    "BLIPConfig",
    "BLIP",
    "BLIP2Config",
    "BLIP2",
    "LLaVAConfig",
    "LLaVA",
    "model_zoo",
]
