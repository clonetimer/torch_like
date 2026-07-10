# Sequence Domain Draft / 序列域草案

中文：本文解释为什么 `torch_like` 需要独立的 `sequence` 域，以及它和 `forecasting` 的边界。

English: This document explains why `torch_like` needs a separate `sequence` domain and how it differs from `forecasting`.

## Goal / 目标

中文：`sequence` 域用于通用序列建模，例如 decoder-only Transformer、LRU、Mamba、MambaLike、Caracal 等。

English: The `sequence` domain is for general sequence modeling, including decoder-only Transformers, LRU, Mamba, MambaLike, and Caracal.

中文：它不要求输入输出必须是时间序列预测中的 `(B, history, channels) -> (B, horizon, channels)`。

English: It does not require the forecasting-style shape `(B, history, channels) -> (B, horizon, channels)`.

## Why A New Domain Is Needed / 为什么需要新域

中文：如果把所有序列模型都塞进 forecasting，会导致 `ForecastingBatch` 被迫承载 token、causal mask、next-token prediction 和 generation 等语义。

English: If all sequence models are forced into forecasting, `ForecastingBatch` would have to carry token, causal-mask, next-token-prediction, and generation semantics.

Problems / 问题:

- 中文：forecasting 的目标是多步数值预测。
- English: Forecasting targets multi-step numeric prediction.
- 中文：sequence 的目标可能是 next-token、分类、生成或长上下文编码。
- English: Sequence tasks may target next-token prediction, classification, generation, or long-context encoding.
- 中文：混在一起会让 API 变模糊。
- English: Mixing them would blur the API.

## Domain Name / 域名

中文：域名使用 `sequence`，不是 `language` 或 `ssm`。

English: The domain name is `sequence`, not `language` or `ssm`.

Reason / 原因:

- 中文：它覆盖 token 序列、连续序列和长上下文序列。
- English: It covers token sequences, continuous sequences, and long-context sequences.
- 中文：它不绑定某一种架构。
- English: It is not tied to a single architecture family.

## Base Types / 基础类型

Current files / 当前文件:

- `torch_like/models/sequence/common.py`
- `SequenceConfig`
- `SequenceBatch`
- `SequenceModel`

Expected contract / 预期合同:

- 中文：`predict(batch)` 返回模型输出。
- English: `predict(batch)` returns model outputs.
- 中文：`make_sequence_batch(...)` 构造统一 batch。
- English: `make_sequence_batch(...)` builds a unified batch.
- 中文：generation 类模型可以额外暴露 `generate(...)`。
- English: Generative models may additionally expose `generate(...)`.

## Input / Output Semantics / 输入输出语义

Input / 输入:

- 中文：token ids、连续 embedding、或通用序列张量。
- English: Token IDs, continuous embeddings, or general sequence tensors.
- 中文：可选 causal mask、attention mask、position ids。
- English: Optional causal masks, attention masks, and position IDs.

Output / 输出:

- 中文：logits、hidden states、next-step predictions 或任务头输出。
- English: Logits, hidden states, next-step predictions, or task-head outputs.

## Registry Positioning / Registry 定位

中文：`sequence` 是独立 domain。

English: `sequence` is an independent domain.

```python
from torch_like.models import create_model

model = create_model("LRU", domain="sequence", ...)
```

## Why Caracal Fits / 为什么 Caracal 适合 sequence

中文：Caracal 是谱/频域序列建模代表，更接近长上下文 sequence backbone，而不是标准 forecasting 模型。

English: Caracal is a spectral/frequency-domain sequence representative. It is closer to a long-context sequence backbone than a standard forecasting model.

## Why Not Forecasting / 为什么不作为 forecasting 一等模型

中文：forecasting 模型要遵守 `ForecastingModel` 和 `ForecastingBatch`。Caracal/Mamba/LRU 这类模型的自然接口不是历史窗口到未来窗口。

English: Forecasting models should follow `ForecastingModel` and `ForecastingBatch`. Models like Caracal, Mamba, and LRU do not naturally map history windows to future windows.

## Implementation Plan / 实现计划

Done / 已完成:

- 中文：建立 `sequence` 域基础接口。
- English: Base interfaces for the `sequence` domain exist.
- 中文：接入 `TransformerDecoder`、`LRU`、`Mamba`、`MambaLike`、`Caracal`。
- English: `TransformerDecoder`, `LRU`, `Mamba`, `MambaLike`, and `Caracal` are integrated.

Future / 后续:

- 中文：如需要，可增加 sequence-side catalog 输出。
- English: Add sequence-side catalog output if needed.
- 中文：如需要，可增加 generation runner，但不作为核心要求。
- English: Add a generation runner if needed, but it is not required for the core framework.

## Non-goals / 非目标

- 中文：不把 `sequence` 做成完整 NLP 框架。
- English: Do not turn `sequence` into a full NLP framework.
- 中文：不把所有 SSM 变体都加入核心库。
- English: Do not add every SSM variant to the core library.
- 中文：不复刻 Hugging Face Transformers。
- English: Do not clone Hugging Face Transformers.

## Decision Summary / 决策总结

中文：`sequence` 域是必要的，因为它保护 forecasting API 不被 token/generation 语义污染，同时给长上下文和状态空间模型一个合理归宿。

English: The `sequence` domain is necessary because it keeps token/generation semantics out of the forecasting API while providing a natural home for long-context and state-space models.
