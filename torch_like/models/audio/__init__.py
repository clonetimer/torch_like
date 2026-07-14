"""Audio domain interfaces and models."""

from torch_like.models.audio.audio_spectrogram_transformer import (
    AudioSpectrogramTransformer,
    AudioSpectrogramTransformerConfig,
)
from torch_like.models.audio.common import AudioBatch, AudioConfig, AudioModel
from torch_like.models.audio.conformer import Conformer, ConformerConfig
from torch_like.models.audio.encodec import EnCodec, EnCodecConfig, ResidualVectorQuantizer
from torch_like.models.audio.hubert import HuBERT, HuBERTConfig
from torch_like.models.audio.wav2vec2 import Wav2Vec2, Wav2Vec2Config
from torch_like.models.audio.whisper import Whisper, WhisperConfig
from torch_like.models.registry import register_model_catalog

model_zoo = {
    "EnCodec": {
        "cls": EnCodec,
        "family": "neural_audio_codec",
        "layer": "Audio / neural codec",
        "integration_level": "paper_aligned_portable_subset",
        "supports": [
            "raw_waveform",
            "audio_encoding",
            "audio_decoding",
            "residual_vector_quantization",
            "bandwidth_control",
            "reconstruction_loss",
        ],
        "paper": "High Fidelity Neural Audio Compression (2022)",
        "description": "Portable neural waveform codec subset with bandwidth-scalable residual vector quantization",
        "complexity": "O(B * T * conv + B * frames * codebooks * codebook_size * latent_dim)",
    },
    "Wav2Vec2": {
        "cls": Wav2Vec2,
        "family": "self_supervised_speech_representation",
        "layer": "Audio / speech representation",
        "integration_level": "framework_oriented",
        "supports": ["raw_waveform", "conv_feature_encoder", "transformer_encoder", "ctc_logits"],
        "paper": "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (NeurIPS 2020)",
        "description": "Raw-waveform convolutional feature encoder followed by Transformer contextualization",
        "complexity": "O(B * T * conv + B * frames^2 * d_model) speech representation encoder",
    },
    "Conformer": {
        "cls": Conformer,
        "family": "speech_transformer",
        "layer": "Audio / speech recognition",
        "integration_level": "paper_aligned_core",
        "supports": ["acoustic_features", "attention", "depthwise_convolution", "ctc_logits"],
        "paper": "Conformer: Convolution-augmented Transformer for Speech Recognition (Interspeech 2020)",
        "description": "Macaron feed-forward, self-attention, and depthwise convolution blocks for acoustic sequences",
        "complexity": "O(B * T^2 * d_model + B * T * conv_kernel * d_model) acoustic sequence encoder",
    },
    "HuBERT": {
        "cls": HuBERT,
        "family": "self_supervised_speech_representation",
        "layer": "Audio / speech representation",
        "integration_level": "framework_oriented",
        "supports": [
            "raw_waveform",
            "conv_feature_encoder",
            "transformer_encoder",
            "masked_prediction",
            "hidden_unit_prediction",
        ],
        "paper": "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units (IEEE/ACM TASLP 2021)",
        "description": "HuBERT-style raw-waveform encoder with feature masking and hidden-unit prediction logits",
        "complexity": "O(B * T * conv + B * frames^2 * d_model) masked speech representation encoder",
    },
    "Whisper": {
        "cls": Whisper,
        "family": "encoder_decoder_speech_recognition",
        "layer": "Audio / speech recognition",
        "integration_level": "framework_oriented",
        "supports": ["log_mel", "encoder_decoder", "cross_attention", "autoregressive_generation", "asr_logits"],
        "paper": "Robust Speech Recognition via Large-Scale Weak Supervision (arXiv 2022)",
        "description": "Whisper-style log-mel audio encoder and autoregressive text decoder for speech recognition",
        "complexity": "O(B * T^2 * d_model + B * L^2 * d_model + B * L * T * d_model) encoder-decoder ASR",
    },
    "AudioSpectrogramTransformer": {
        "cls": AudioSpectrogramTransformer,
        "family": "audio_classification_transformer",
        "layer": "Audio / classification",
        "integration_level": "framework_oriented",
        "supports": ["spectrogram", "patch_embedding", "transformer_encoder", "classification"],
        "paper": "AST: Audio Spectrogram Transformer (Interspeech 2021)",
        "description": "ViT-style patch Transformer for log-mel spectrogram classification",
        "complexity": "O(B * patches^2 * d_model) Transformer attention over spectrogram patches",
    },
}

register_model_catalog("audio", model_zoo)

__all__ = [
    "AudioConfig",
    "AudioBatch",
    "AudioModel",
    "EnCodecConfig",
    "ResidualVectorQuantizer",
    "EnCodec",
    "Wav2Vec2Config",
    "Wav2Vec2",
    "ConformerConfig",
    "Conformer",
    "HuBERTConfig",
    "HuBERT",
    "WhisperConfig",
    "Whisper",
    "AudioSpectrogramTransformerConfig",
    "AudioSpectrogramTransformer",
    "model_zoo",
]
