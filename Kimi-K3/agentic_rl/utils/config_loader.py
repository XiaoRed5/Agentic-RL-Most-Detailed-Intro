"""YAML 配置加载：把 configs/*.yaml 解析成 Config dataclass。

离线单测默认用 default_offline_config()；本加载器让真机能从 real_qwen3_4b.yaml 一键构建
同一个 Config 结构，切换只改 YAML。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_rl.utils.config import (
    AlgoConfig,
    Config,
    CreditConfig,
    ModelConfig,
    RolloutConfig,
    ShapingConfig,
)


def _apply(dc, data: dict[str, Any]):
    """把 dict 里已知字段覆盖到 dataclass 实例(忽略未知字段，如真机的 runtime 段)。"""
    for k, v in data.items():
        if hasattr(dc, k):
            setattr(dc, k, v)
    return dc


def load_config(path: str | Path) -> Config:
    """从 YAML 加载 Config。不依赖 pyyaml 时回退到极简解析(仅支持本项目的扁平结构)。"""
    text = Path(path).read_text()
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text)
    except Exception:
        data = _minimal_yaml_parse(text)

    cfg = Config()
    if "model" in data:
        _apply(cfg.model, data["model"])
    if "credit" in data:
        _apply(cfg.credit, data["credit"])
    if "shaping" in data:
        _apply(cfg.shaping, data["shaping"])
    if "algo" in data:
        _apply(cfg.algo, data["algo"])
    if "rollout" in data:
        _apply(cfg.rollout, data["rollout"])
    if "seed" in data:
        cfg.seed = data["seed"]
    return cfg


def _minimal_yaml_parse(text: str) -> dict[str, Any]:
    """极简 YAML 解析：仅支持两级缩进 + 标量(pyyaml 不可用时的兜底)。"""
    root: dict[str, Any] = {}
    cur: dict[str, Any] | None = None
    for line in text.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        if not line.startswith(" "):
            key = line.split(":", 1)[0].strip()
            rest = line.split(":", 1)[1].strip()
            if rest:
                root[key] = _coerce(rest)
                cur = None
            else:
                cur = {}
                root[key] = cur
        else:
            if cur is None:
                continue
            k, _, v = line.strip().partition(":")
            v = v.strip()
            if v and not v.startswith("#"):
                cur[k.strip()] = _coerce(v.split("#")[0].strip())
    return root


def _coerce(s: str):
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        return [x.strip() for x in inner.split(",")] if inner else []
    for caster in (int, float):
        try:
            return caster(s)
        except ValueError:
            pass
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    return s
