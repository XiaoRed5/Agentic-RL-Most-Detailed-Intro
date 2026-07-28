"""tiny 字节级 tokenizer —— 离线单测用，无需下载真实 tokenizer。

真机换成 Qwen3 的 AutoTokenizer；本 tokenizer 只需保证：
- 可逆(encode/decode 往返一致)，
- 词表大小固定(config.model.vocab_size)，
- 特殊 token 稳定，
以便 masking / rollout / 训练在 CPU 上跑通。
"""
from __future__ import annotations


class ByteTokenizer:
    """把字符串按 UTF-8 字节编码。词表 = 256 字节 + 少量特殊 token。"""

    # 特殊 token 占用 256 之后的 id
    PAD = 256
    BOS = 257
    EOS = 258

    def __init__(self, vocab_size: int = 512):
        assert vocab_size >= 259
        self.vocab_size = vocab_size

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        if add_special:
            ids = [self.BOS] + ids + [self.EOS]
        return ids

    def decode(self, ids: list[int]) -> str:
        b = bytes([i for i in ids if i < 256])
        return b.decode("utf-8", errors="replace")

    def __len__(self) -> int:
        return self.vocab_size
