"""配置加载测试：离线/真机 YAML 都能解析成 Config，字段正确。"""
from pathlib import Path

import pytest

from agentic_rl.utils.config_loader import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_load_offline_config():
    cfg = load_config(CONFIGS / "offline.yaml")
    assert cfg.model.name == "tiny"
    assert cfg.credit.method == "r2g"
    assert cfg.credit.gamma == 0.8
    assert cfg.algo.estimator == "grpo"
    assert cfg.rollout.max_steps == 16


def test_load_real_config():
    cfg = load_config(CONFIGS / "real_qwen3_4b.yaml")
    assert cfg.model.name == "Jarrodbarnes/Qwen3-4B-tau2-sft1"
    # 真机开启 DAPO 四件套
    assert cfg.algo.clip_higher is True
    assert cfg.algo.dynamic_sampling is True
    assert cfg.algo.token_level_loss is True
    assert cfg.algo.kl_coef > 0
    assert cfg.rollout.max_steps == 100    # tau2 官方
    assert cfg.credit.gamma == 0.95


def test_configs_share_structure():
    """离线与真机配置结构一致(同一 Config)，只值不同 —— 迁移零改动的保证。"""
    off = load_config(CONFIGS / "offline.yaml")
    real = load_config(CONFIGS / "real_qwen3_4b.yaml")
    assert type(off) is type(real)
    assert type(off.algo) is type(real.algo)
    # 关键：算法开关字段完全一致
    assert set(vars(off.algo)) == set(vars(real.algo))


def test_unknown_runtime_section_ignored():
    """真机 YAML 的 runtime 段(离线 Config 没有)被安全忽略，不报错。"""
    cfg = load_config(CONFIGS / "real_qwen3_4b.yaml")
    assert not hasattr(cfg, "runtime")   # 未知段不污染 Config
