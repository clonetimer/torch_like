"""Attention mechanisms: Full, ProbSparse, AutoCorrelation, and LogSparse.

- FullAttention: standard scaled dot-product multi-head attention.
- ProbSparseAttention: Informer (AAAI 2021) — selects top-u queries only.
- AutoCorrelation: Autoformer (NeurIPS 2021) — FFT-based period-based attention.
- LogSparseSelfAttention: LogTrans (NeurIPS 2019) — exponential step mask.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FullAttention(nn.Module):
    """Standard scaled dot-product multi-head attention.

    Args:
        mask_flag: if True, expects an attention mask.
        scale: sqrt(d_k) or None for auto-computation.
        attention_dropout: dropout on attention weights.
        output_attention: if True, returns attention weights alongside output.
    """

    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, _ = keys.shape
        scale = self.scale or 1.0 / math.sqrt(E)

        if not self.output_attention:
            mask = attn_mask if self.mask_flag else None
            out = F.scaled_dot_product_attention(
                queries.transpose(1, 2),
                keys.transpose(1, 2),
                values.transpose(1, 2),
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                scale=self.scale,
            )
            return out.transpose(1, 2).contiguous(), None

        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * scale

        if self.mask_flag and attn_mask is not None:
            scores.masked_fill_(~attn_mask, -float("inf"))

        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.einsum("bhls,bshd->blhd", attn, values)

        if self.output_attention:
            return out.contiguous(), attn
        return out.contiguous(), None


class ProbSparseAttention(nn.Module):
    """ProbSparse self-attention — computes only top-u query-key pairs.

    Args:
        mask_flag: if True, applies triangular mask (for decoder self-attn).
        factor: sampling factor c; u = c * ln(L_Q).
        scale: sqrt(d_k); auto-computed if None.
        attention_dropout: dropout rate on attention weights.
        output_attention: if True, returns attention weights.
    """

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        """Measure query sparsity via max-mean divergence.

        Args:
            Q: (B, H, L_Q, E)
            K: (B, H, L_K, E)
            sample_k: number of keys to sample for M computation.
            n_top: top-u (number of queries to select).

        Returns:
            Q: (B, H, u, E) — selected top-u queries.
            M_top: (B, H, u) — sparsity scores.
        """
        B, H, L_Q, E = Q.shape
        _, _, L_K, _ = K.shape

        # Sample a subset of keys for efficiency
        sample_k = min(sample_k, L_K)
        K_sample = K[:, :, torch.randint(L_K, (sample_k,), device=K.device), :]

        # QK^T for the sampled keys: (B, H, L_Q, sample_k)
        Q_K_sample = torch.matmul(Q, K_sample.transpose(-2, -1))

        # M(q_i, K) = max_j(q_i k_j^T) - mean_j(q_i k_j^T)   [Eq. 4 in paper]
        M = Q_K_sample.max(dim=-1)[0] - Q_K_sample.mean(dim=-1)

        # Select top-u queries
        M_top = M.topk(n_top, dim=-1, sorted=False)[1]  # (B, H, u) indices

        # Gather selected queries
        gather_index = M_top.unsqueeze(-1).expand(-1, -1, -1, E)
        Q_reduce = torch.gather(Q, dim=2, index=gather_index)

        return Q_reduce, M_top

    def _get_initial_context(self, V, L_Q):
        """Mean of V as context for all queries, broadcast to L_Q length."""
        # V: (B, H, L_V, D) — return (B, H, L_Q, D)
        return torch.mean(V, dim=2, keepdim=True).expand(-1, -1, L_Q, -1)

    def _update_context(self, context_in, V, scores, index):
        """Scatter dot-product results back to full context tensor."""
        B, H, _, E = V.shape

        attn = self.dropout(torch.softmax(scores, dim=-1))
        context_out = torch.matmul(attn, V)
        scatter_index = index.unsqueeze(-1).expand(-1, -1, -1, E)
        context_in.scatter_(2, scatter_index, context_out)
        return context_in

    def forward(self, queries, keys, values, attn_mask=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        # Transpose to (B, H, L, D) for matmul convenience
        Q = queries.transpose(1, 2)
        K = keys.transpose(1, 2)
        V = values.transpose(1, 2)

        scale = self.scale or 1.0 / math.sqrt(D)

        # Determine number of selected queries: u = c * ln(L_Q)
        u = int(self.factor * math.ceil(math.log(L_Q)))
        u = min(u, L_Q)

        # Compute sparsity and select top-u queries
        Q_reduce, index = self._prob_QK(Q, K, sample_k=int(math.log(L_K)), n_top=u)

        # Dot-product for selected queries only
        QK = torch.matmul(Q_reduce, K.transpose(-2, -1)) * scale

        if self.mask_flag and attn_mask is not None:
            mask = attn_mask.unsqueeze(1).expand(-1, H, -1, -1)
            gathered_index = index.unsqueeze(-1).expand(-1, -1, -1, L_K)
            mask = torch.gather(mask, dim=2, index=gathered_index)
            QK.masked_fill_(~mask, -float("inf"))

        # Context: mean of V for all queries, then scatter updates for selected
        context = self._get_initial_context(V, L_Q).contiguous()
        context = self._update_context(context, V, QK, index)

        # Transpose back to (B, L_Q, H, D)
        out = context.transpose(1, 2)

        if self.output_attention:
            attn_map = torch.zeros(B, H, L_Q, L_K, device=queries.device)
            attn_probs = torch.softmax(QK, dim=-1)
            scatter_index = index.unsqueeze(-1).expand(-1, -1, -1, L_K)
            attn_map.scatter_(2, scatter_index, attn_probs)
            return out.contiguous(), attn_map
        return out.contiguous(), None


# ---------------------------------------------------------------------------
#  Autoformer — AutoCorrelation attention (NeurIPS 2021)
# ---------------------------------------------------------------------------


class AutoCorrelation(nn.Module):
    """Auto-Correlation mechanism: replace dot-product with period-based dependencies.

    Uses FFT to compute autocorrelation efficiently, then aggregates via
    Time Delay Aggregation: roll values by top-k dominant lags.

    Args:
        mask_flag: apply attention mask (triangular for decoder).
        factor: controls top-k = factor * log(L).
        scale: not used (correlation is normalized differently).
        attention_dropout: dropout on attention weights.
        output_attention: if True, returns attention weights.
    """

    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.factor = factor
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg(self, queries, keys, values, attn_mask=None):
        """FFT-based autocorrelation + top-k time delay aggregation.

        Args:
            queries: (B, L, H, E)
            keys:   (B, S, H, E)
            values: (B, S, H, E)

        Returns:
            out: (B, L, H, E) aggregated values.
            corr: (B, H, L, S) attention map (only if output_attention).
        """
        B, L, H, E = queries.shape
        S = keys.shape[1]
        device = queries.device

        # Reshape to (B, H, E, T) for batched FFT along time dim
        Q = queries.permute(0, 2, 3, 1).float()  # (B, H, E, L)
        K = keys.permute(0, 2, 3, 1).float()  # (B, H, E, S)
        V = values.permute(0, 2, 3, 1)  # (B, H, E, S)

        # Handle unequal sequence lengths: pad the shorter to the longer
        T = max(L, S)
        if L < T:
            Q = F.pad(Q, (0, T - L))
        if S < T:
            K = F.pad(K, (0, T - S))
            V = F.pad(V, (0, T - S))

        # FFT-based cross-correlation
        Q_fft = torch.fft.rfft(Q, dim=-1)  # (B, H, E, T//2+1)
        K_fft = torch.fft.rfft(K, dim=-1)  # (B, H, E, T//2+1)

        res = Q_fft * torch.conj(K_fft)  # (B, H, E, T//2+1)
        corr = torch.fft.irfft(res, dim=-1, n=T)  # (B, H, E, T)

        # Take only the first L lags (output length for queries)
        corr = corr[..., :L]  # (B, H, E, L)

        # Sum over feature channels: (B, H, L)
        corr_sum = corr.sum(dim=2)

        # Top-k selection
        top_k = int(self.factor * math.log(L))
        top_k = max(1, min(top_k, L))

        corr_weights = torch.softmax(corr_sum.abs(), dim=-1)  # (B, H, L)
        top_weights, top_lags = torch.topk(corr_weights, top_k, dim=-1)  # (B, H, top_k)
        top_weights = F.softmax(top_weights, dim=-1)  # renormalize top-k

        # Time Delay Aggregation — vectorized via gather
        # V: (B, H, E, S), we want to roll V by each lag
        V_exp = V.unsqueeze(2).expand(-1, -1, top_k, -1, -1)  # (B, H, top_k, E, S)

        # Compute rolled indices: rolled_idx[b, h, k, s] = (s - lag) % S
        time_idx = torch.arange(S, device=device).view(1, 1, 1, S)  # (1, 1, 1, S)
        lags = top_lags.unsqueeze(-1)  # (B, H, top_k, 1)
        raw_idx = time_idx - lags  # (B, H, top_k, S)
        rolled_idx = raw_idx % S  # (B, H, top_k, S)
        valid = None
        if self.mask_flag:
            valid = (raw_idx >= 0) & (raw_idx < S)
            rolled_idx = raw_idx.clamp(0, S - 1)
            if attn_mask is not None:
                if L != S:
                    raise ValueError("Masked AutoCorrelation currently requires equal query and key lengths")
                allowed = attn_mask.to(device=device, dtype=torch.bool)
                if allowed.dim() == 2:
                    allowed = allowed.unsqueeze(0).unsqueeze(0)
                elif allowed.dim() == 3:
                    allowed = allowed.unsqueeze(1)
                if allowed.dim() != 4 or allowed.shape[-2:] != (L, S):
                    raise ValueError(f"Expected attention mask ending in ({L}, {S}), got {tuple(allowed.shape)}")
                allowed = allowed.expand(B, H, L, S)
                selected = torch.gather(allowed, dim=-1, index=rolled_idx.permute(0, 1, 3, 2))
                valid = valid & selected.permute(0, 1, 3, 2)
        rolled_idx = rolled_idx.unsqueeze(-2).expand(-1, -1, -1, E, -1)  # (B, H, top_k, E, S)

        V_rolled = torch.gather(V_exp, dim=-1, index=rolled_idx)  # (B, H, top_k, E, S)

        # Weighted sum over top-k lags
        if valid is None:
            weights = top_weights.view(B, H, top_k, 1, 1)
        else:
            masked_weights = top_weights.unsqueeze(-1) * valid.to(top_weights.dtype)
            masked_weights = masked_weights / masked_weights.sum(dim=2, keepdim=True).clamp_min(1e-8)
            weights = masked_weights.unsqueeze(-2)
        out = (weights * V_rolled).sum(dim=2)  # (B, H, E, S)

        out = out.permute(0, 3, 1, 2)  # (B, S, H, E)

        # Pad/truncate to desired output length L
        if L != S:
            if L < S:
                out = out[:, :L]
            else:
                out = F.pad(out, (0, 0, 0, 0, 0, L - S))

        # Attention map for output_attention (only when requested)
        attn_map = None
        if self.output_attention:
            attn_map = corr_sum  # (B, H, L)

        return out, attn_map

    def forward(self, queries, keys, values, attn_mask=None):
        # Apply mask if needed — mask future lags for decoder self-attn
        return self.time_delay_agg(queries, keys, values, attn_mask=attn_mask)


class AutoCorrelationLayer(nn.Module):
    """Full attention layer wrapping AutoCorrelation with QKV projections.

    This mirrors the standard multi-head attention interface but uses
    AutoCorrelation instead of dot-product attention internally.
    """

    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or d_model // n_heads
        d_values = d_values or d_model // n_heads
        self.n_heads = n_heads
        self.inner_attention = attention

        self.w_q = nn.Linear(d_model, d_keys * n_heads, bias=False)
        self.w_k = nn.Linear(d_model, d_keys * n_heads, bias=False)
        self.w_v = nn.Linear(d_model, d_values * n_heads, bias=False)
        self.out_proj = nn.Linear(d_values * n_heads, d_model, bias=False)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L = queries.shape[:2]
        S = keys.shape[1]
        H = self.n_heads

        q = self.w_q(queries).view(B, L, H, -1)
        k = self.w_k(keys).view(B, S, H, -1)
        v = self.w_v(values).view(B, S, H, -1)

        out, attn = self.inner_attention(q, k, v, attn_mask=attn_mask)
        out = out.reshape(B, L, -1)
        out = self.out_proj(out)

        return out, attn


# ---------------------------------------------------------------------------
#  LogTrans — LogSparse self-attention (NeurIPS 2019)
# ---------------------------------------------------------------------------


def log_sparse_mask(seq_len, device=None):
    """Generate a causal log-sparse attention mask.

    Each query at position t attends to keys at positions:
        t (itself), t-1, t-2, t-4, t-8, ..., t-2^floor(log2(t))

    This yields O(L log L) complexity — each row has ~log_2(L) non-zero entries.

    Args:
        seq_len: sequence length.
        device: torch device.

    Returns:
        mask: (seq_len, seq_len) — True where attention is allowed.
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    for t in range(seq_len):
        # Always attend to self
        mask[t, t] = True
        # Exponential steps backward
        step = 1
        while step <= t:
            mask[t, t - step] = True
            step *= 2

    return mask


class LogSparseSelfAttention(FullAttention):
    """Log-sparse self-attention: each query attends to O(log L) keys.

    Uses a pre-computed binary mask that selects exponential-step positions.
    Inherits from FullAttention and overrides the masking logic.

    Reference: Li et al., "Enhancing the Locality and Breaking the Memory
    Bottleneck of Transformer on Time Series Forecasting" (NeurIPS 2019).
    """

    def __init__(self, mask_flag=False, scale=None, attention_dropout=0.1, output_attention=False):
        # mask_flag=False because we use our own sparse mask, not triangular
        super().__init__(
            mask_flag=False, scale=scale, attention_dropout=attention_dropout, output_attention=output_attention
        )
        self.register_buffer("sparse_mask", None, persistent=False)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, _ = keys.shape

        # Build log-sparse mask if not cached or shape changed
        if self.sparse_mask is None or self.sparse_mask.shape[0] != L:
            self.sparse_mask = log_sparse_mask(L, queries.device)

        scale = self.scale or 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * scale

        # Apply log-sparse mask: mask out positions not in the sparse pattern
        # sparse_mask: (L, L) → broadcast to (1, 1, L, L) → (1, 1, L, S)
        mask_4d = ~self.sparse_mask[:L, :S].unsqueeze(0).unsqueeze(0)
        scores.masked_fill_(mask_4d, -float("inf"))

        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.einsum("bhls,bshd->blhd", attn, values)

        if self.output_attention:
            return out.contiguous(), attn
        return out.contiguous(), None
