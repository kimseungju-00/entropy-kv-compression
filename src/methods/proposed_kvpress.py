"""
Entropy-based KV cache compression (single-pass).

Token selection uses the mean attention over a recent observation window
(SnapKV-style). The per-layer keep ratio is set adaptively from the attention
entropy of that layer, so the budget is chosen per input without manual tuning.

Classes:
  ProposedOnePassPress  - role-weighted budget (with head-role classification)
  EntropyBudgetPress    - final method: mean-entropy budget (no head classification)
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import get_prerope_query_states


def _window_attention(module, hidden_states, keys, window_size, position_embeddings):
    """Attention from the last `window_size` queries to all keys.

    Returns a tensor of shape (B, num_heads, window_size, k_len). This is the
    flash-compatible observation-window attention used for both entropy
    estimation and token scoring.
    """
    bsz, _, k_len, _ = keys.shape
    num_heads = module.config.num_attention_heads
    head_dim = module.head_dim
    n_group = num_heads // module.config.num_key_value_heads

    q = get_prerope_query_states(module, hidden_states[:, -window_size:])
    cos, sin = position_embeddings
    cos, sin = cos[:, -window_size:], sin[:, -window_size:]
    q = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))

    k = repeat_kv(keys, n_group)
    attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
    mask = torch.ones_like(attn) * float("-inf")
    mask = torch.triu(mask, diagonal=k_len - window_size + 1)
    attn = attn + mask
    attn = F.softmax(attn, dim=-1, dtype=torch.float32)
    return attn


@dataclass
class ProposedOnePassPress(ScorerPress):
    """Single-pass entropy-based compression with a role-weighted budget.

    For each layer, during the prefill forward hook:
      1. compute head entropies from the observation-window attention,
      2. set the layer keep ratio from those entropies (local phi),
      3. score tokens by their window-averaged attention and keep the top-k.

    Compression happens in a single prefill pass, so the reduced budget
    translates directly into reduced KV cache memory.
    """
    compression_ratio: float = 0.0  # kept for parent compatibility; budget is dynamic
    window_size: int = 32
    kernel_size: int = 5
    epsilon: float = 0.7
    alpha: float = 2.0
    beta: float = 1.0
    tau_percentile: float = 30.0
    min_ratio: float = 0.05   # lower bound on keep ratio
    max_ratio: float = 0.50   # upper bound on keep ratio

    def _layer_keep_ratio(self, entropy):
        """Keep ratio for this layer from its head entropies (num_heads,).

        Heads with low entropy are treated as retrieval heads and up-weighted
        (alpha vs beta) when forming the layer budget phi.
        """
        tau = torch.quantile(entropy, self.tau_percentile / 100.0)
        roles = entropy <= tau  # retrieval heads
        weights = torch.where(roles, torch.full_like(entropy, self.alpha),
                              torch.full_like(entropy, self.beta))
        H_norm = entropy / (entropy.max() + 1e-9)
        phi = (weights * H_norm).sum().item() / (weights.sum().item() + 1e-9)
        g = phi ** (1.0 / max(self.epsilon, 0.01))
        keep = self.min_ratio + (self.max_ratio - self.min_ratio) * g
        return float(np.clip(keep, self.min_ratio, self.max_ratio))

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        bsz, n_kv, k_len, _ = keys.shape
        n_group = module.config.num_attention_heads // n_kv
        w = self.window_size
        if hidden_states.shape[1] <= w:
            return keys, values  # skip compression for short inputs

        # Observation-window attention
        if attentions is not None:
            attn = attentions[..., -w:, :]
        else:
            attn = _window_attention(module, hidden_states, keys, w, kwargs["position_embeddings"])

        # Head entropies (used to set the layer keep ratio)
        head_attn = attn.mean(dim=-2)[0]  # (num_heads, k_len)
        p = head_attn / (head_attn.sum(dim=-1, keepdim=True) + 1e-9)
        logp = torch.where(p > 0, p.log(), torch.zeros_like(p))
        entropy = -(p * logp).sum(dim=-1)  # (num_heads,)
        keep_ratio = self._layer_keep_ratio(entropy)
        if not hasattr(self, "_budget_log"):
            self._budget_log = []
        self._budget_log.append(keep_ratio)
        n_keep = max(1, int(k_len * keep_ratio))
        if n_keep >= k_len:
            return keys, values

        # Token scores = window-averaged attention per key
        scores = attn.mean(dim=-2)  # (B, num_heads, k_len)
        scores = F.avg_pool1d(scores, kernel_size=self.kernel_size,
                              padding=self.kernel_size // 2, stride=1)
        scores = scores.view(bsz, n_kv, n_group, k_len).mean(2)  # (B, n_kv, k_len)
        scores[..., -w:] = scores.max().item() + 1  # always keep the recent window

        # Top-k selection (order preserved)
        idx = scores.topk(n_keep, dim=-1).indices  # (B, n_kv, n_keep)
        idx = idx.sort(dim=-1).values
        idx_k = idx.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
        keys_c = keys.gather(2, idx_k)
        values_c = values.gather(2, idx_k)
        return keys_c, values_c


@dataclass
class EntropyBudgetPress(ProposedOnePassPress):
    """Final proposed method: mean-entropy budget (no head-role classification).

    Same token selection and compression as the parent; only the budget rule
    differs. The keep ratio is derived from the mean normalized head entropy of
    each layer, which gives a better accuracy-memory trade-off than the
    role-weighted variant.
    """
    def _layer_keep_ratio(self, entropy):
        h_norm = entropy / (entropy.max() + 1e-9)
        phi = h_norm.mean().item()
        g = phi ** (1.0 / max(self.epsilon, 0.01))
        keep = self.min_ratio + (self.max_ratio - self.min_ratio) * g
        return float(np.clip(keep, self.min_ratio, self.max_ratio))