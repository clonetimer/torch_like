"""Sequence domain base interfaces."""

from torch_like.models.sequence.bert import BERT, BERTConfig
from torch_like.models.sequence.caracal import Caracal
from torch_like.models.sequence.common import Seq2SeqBatch, Seq2SeqModel, SequenceBatch, SequenceConfig, SequenceModel
from torch_like.models.sequence.gpt import GPT, GPTConfig
from torch_like.models.sequence.lru import LRU
from torch_like.models.sequence.mamba import Mamba
from torch_like.models.sequence.mamba_like import MambaLike
from torch_like.models.sequence.t5 import T5, T5Config
from torch_like.models.sequence.transformer_decoder import TransformerDecoder
from torch_like.models.registry import register_model_catalog

model_zoo = {
    "TransformerDecoder": {
        "cls": TransformerDecoder,
        "family": "causal_decoder",
        "layer": "General sequence modeling",
        "integration_level": "framework_oriented",
        "supports": ["causal_mask", "next_token", "generation"],
        "paper": "Transformer decoder baseline for torch_like sequence domain (framework-oriented)",
        "description": "Causal decoder-only Transformer baseline for sequence-domain expansion",
        "complexity": "O(B * L^2 * D) causal self-attention over sequence length L and hidden width D",
    },
    "GPT": {
        "cls": GPT,
        "family": "causal_decoder",
        "layer": "General sequence modeling",
        "integration_level": "framework_oriented",
        "supports": ["causal_mask", "next_token", "generation", "pre_norm_transformer", "language_modeling"],
        "paper": "Improving Language Understanding by Generative Pre-Training (2018) / GPT-2 Language Models are Unsupervised Multitask Learners (2019)",
        "description": "GPT-style decoder-only Transformer language model with pre-norm blocks and tied token embeddings",
        "complexity": "O(B * L^2 * D) causal self-attention over sequence length L and hidden width D",
    },
    "LRU": {
        "cls": LRU,
        "family": "state_space",
        "layer": "General sequence modeling",
        "integration_level": "framework_oriented",
        "supports": ["next_token", "generation", "linear_recurrence", "long_context"],
        "paper": "Resurrecting Recurrent Neural Networks for Long Sequences (ICML 2023), PyTorch-native framework integration",
        "description": "Linear recurrent unit backbone with diagonal complex recurrence and residual feed-forward refinement",
        "complexity": "O(B * L * S) recurrent scan over sequence length L and recurrent state size S",
    },
    "Mamba": {
        "cls": Mamba,
        "family": "state_space",
        "layer": "General sequence modeling",
        "integration_level": "paper_aligned",
        "supports": ["next_token", "generation", "state_space", "long_context"],
        "paper": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces (COLM 2024), PyTorch-native framework integration",
        "description": "Selective state-space causal sequence backbone with input-dependent delta, B, and C projections",
        "complexity": "O(B * L * D * S) selective scan over sequence length L, hidden width D, and state size S",
    },
    "MambaLike": {
        "cls": MambaLike,
        "family": "state_space",
        "layer": "General sequence modeling",
        "integration_level": "framework_oriented",
        "supports": ["next_token", "generation", "state_space_like", "long_context"],
        "paper": "Mamba-like state-space baseline for torch_like sequence domain (framework-oriented)",
        "description": "Selective state-space inspired causal sequence baseline with local convolutional mixing",
        "complexity": "O(B * L * D) local selective mixing over sequence length L and hidden width D",
    },
    "Caracal": {
        "cls": Caracal,
        "family": "spectral_mixing",
        "layer": "General sequence modeling",
        "integration_level": "framework_oriented",
        "supports": ["causal_mask", "next_token", "generation", "fft_mixing", "long_context"],
        "paper": "Caracal: Causal Architecture via Spectral Mixing (ICML 2026), framework-oriented integration",
        "description": "Causal spectral mixing backbone with Multi-Head Fourier blocks and periodic sliding-window attention",
        "complexity": "O(B * L log L * D) spectral token mixing plus sparse local attention every few blocks",
    },
    "BERT": {
        "cls": BERT,
        "family": "bidirectional_encoder",
        "layer": "General sequence modeling",
        "integration_level": "framework_oriented",
        "supports": ["masked_language_modeling", "bidirectional_attention", "pooled_output", "encoder_only"],
        "paper": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (NAACL 2019)",
        "description": "Bidirectional Transformer encoder with token/position/type embeddings and masked-token prediction head",
        "complexity": "O(B * L^2 * D) bidirectional self-attention over sequence length L and hidden width D",
    },
    "T5": {
        "cls": T5,
        "family": "text_to_text_encoder_decoder",
        "layer": "General sequence modeling",
        "integration_level": "framework_oriented",
        "supports": ["seq2seq", "encoder_decoder", "text_to_text", "generation"],
        "paper": "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (JMLR 2020)",
        "description": "T5-style encoder-decoder Transformer with text-to-text logits and autoregressive decoder generation",
        "complexity": "O(B * (src_len^2 + tgt_len^2 + src_len * tgt_len) * D) encoder-decoder attention",
    },
}

register_model_catalog("sequence", model_zoo)

__all__ = [
    "SequenceConfig",
    "SequenceBatch",
    "SequenceModel",
    "Seq2SeqBatch",
    "Seq2SeqModel",
    "TransformerDecoder",
    "GPTConfig",
    "GPT",
    "LRU",
    "Mamba",
    "MambaLike",
    "Caracal",
    "BERTConfig",
    "BERT",
    "T5Config",
    "T5",
    "model_zoo",
]
