"""tiny causal transformer —— 离线端到端训练单测用的可导 policy。

几十万参数，CPU 秒级前向/反向。真机换成 Qwen3-4B(HF AutoModelForCausalLM)，本模块
提供的 logprobs_of / value head 接口保持一致，训练循环代码零改动。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, hidden: int = 32, n_layers: int = 2,
                 n_heads: int = 2, max_len: int = 2048, with_value: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, hidden)
        self.pos_emb = nn.Embedding(max_len, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 4,
            batch_first=True, dropout=0.0,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab_size)
        self.value_head = nn.Linear(hidden, 1) if with_value else None
        self.max_len = max_len

    def forward(self, input_ids: torch.Tensor):
        """input_ids: [B, T] → logits [B, T, V], values [B, T] (若有 value head)。"""
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        h = self.tok_emb(input_ids) + self.pos_emb(pos)
        # causal mask
        mask = torch.triu(torch.ones(T, T, device=input_ids.device), diagonal=1).bool()
        h = self.blocks(h, mask=mask)
        h = self.ln(h)
        logits = self.lm_head(h)
        values = self.value_head(h).squeeze(-1) if self.value_head is not None else None
        return logits, values


def token_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """计算每个位置对 next-token 的 log π。

    logits: [B, T, V]；input_ids: [B, T]。
    返回 [B, T-1]：位置 t 的 logprob = log P(input_ids[t+1] | input_ids[:t+1])。
    """
    logp = torch.log_softmax(logits[:, :-1, :], dim=-1)     # [B, T-1, V]
    targets = input_ids[:, 1:].unsqueeze(-1)                # [B, T-1, 1]
    return logp.gather(-1, targets).squeeze(-1)             # [B, T-1]
