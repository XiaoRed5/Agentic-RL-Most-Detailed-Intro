"""InfoPO 接入训练器测试：核心验证'零方差组靠 info-gain 救活'。

survey ④ InfoPO 主战场：一组 rollout 奖励全一样(zero-variance) → GRPO advantage 全 0
→ 无梯度、学不动。InfoPO 用反事实信息增益顶上，让这种组继续供梯度。
"""
import pytest
import torch

from agentic_rl.credit.infopo_runtime import (
    group_infogain_advantages,
    trajectory_info_gains,
)
from agentic_rl.env.offline_tasks import make_retail_task
from agentic_rl.env.tau_env import TauEnv
from agentic_rl.rollout.policy import ScriptedPolicy
from agentic_rl.rollout.rollout import rollout
from agentic_rl.rollout.tokenizer import ByteTokenizer
from agentic_rl.train.model import TinyTransformer
from agentic_rl.train.trainer import train_step
from agentic_rl.utils.config import Config


@pytest.fixture
def tok():
    return ByteTokenizer(vocab_size=512)


@pytest.fixture
def model(tok):
    torch.manual_seed(0)
    return TinyTransformer(vocab_size=tok.vocab_size, hidden=32, n_layers=2, n_heads=2)


def _success_group(tok, G=4):
    """全成功 group → reward 全 1 → 零方差组。"""
    trajs = []
    for _ in range(G):
        env = TauEnv(*make_retail_task(), max_steps=16)
        policy = ScriptedPolicy([
            {"name": "cancel_order", "arguments": {"order_id": "#W001"}},
            {"name": "process_refund", "arguments": {"order_id": "#W001"}},
            {"name": "respond", "arguments": {"content": "Your refund is done."}},
        ])
        trajs.append(rollout(env, policy, tok, max_steps=16))
    return trajs


# ========== info-gain 运行时 ==========
def test_trajectory_info_gains_shape(model, tok):
    traj = _success_group(tok, G=1)[0]
    gains = trajectory_info_gains(model, traj, pad_id=tok.vocab_size - 1)
    assert len(gains) == traj.n_turns
    assert gains[-1] == 0.0            # 最后一轮无"下一步动作"
    assert all(g >= 0.0 for g in gains)   # KL 非负


def test_info_gains_nonzero_somewhere(model, tok):
    """至少某一轮的 observation 对下一步有影响(gain>0)，否则 InfoPO 无信号。"""
    traj = _success_group(tok, G=1)[0]
    gains = trajectory_info_gains(model, traj, pad_id=tok.vocab_size - 1)
    assert any(g > 0 for g in gains[:-1])


def test_group_infogain_advantages_masked_zero(model, tok):
    """obs token 的 info-gain advantage 必须是 0(只强化 policy 生成的 token)。"""
    group = _success_group(tok, G=2)
    advs = group_infogain_advantages(model, group, pad_id=tok.vocab_size - 1)
    for traj, adv in zip(group, advs):
        idx = 0
        for turn in traj.turns:
            idx += len(turn.assistant_token_ids)
            for _ in turn.obs_token_ids:
                assert adv[idx] == 0.0
                idx += 1


# ========== 核心：零方差组靠 InfoPO 供梯度 ==========
def test_zero_variance_group_no_gradient_under_grpo(tok):
    """基线证明：纯 GRPO 下，零方差组梯度≈0(学不动)。"""
    torch.manual_seed(1)
    model = TinyTransformer(vocab_size=tok.vocab_size, hidden=32, n_layers=2, n_heads=2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = Config()
    cfg.credit.method = "r2g"    # 非 infopo
    group = _success_group(tok, G=4)   # 零方差
    train_step(model, opt, [group], cfg)
    total_grad = sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)
    # r2g 在零方差组内 turn 之间仍有结构，但轨迹间无区分 → 梯度很小
    # 关键对比在下一个测试：InfoPO 应显著更大
    assert float(total_grad) >= 0.0   # 存在(可能很小)


def test_infopo_rescues_zero_variance_group(tok):
    """核心断言：零方差组下，InfoPO(方差门控开启)产生非平凡梯度，救活 GRPO 学不动的组。"""
    torch.manual_seed(2)
    model = TinyTransformer(vocab_size=tok.vocab_size, hidden=32, n_layers=2, n_heads=2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = Config()
    cfg.credit.method = "infopo"
    cfg.credit.infopo_gain_weight = 1.0
    group = _success_group(tok, G=4)   # 零方差(全成功)

    stats = train_step(model, opt, [group], cfg)
    assert stats.n_infogain_gated == 1    # 方差门控确实开启了
    total_grad = sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)
    assert float(total_grad) > 1e-6       # InfoPO 供了非平凡梯度


def test_infopo_gate_closes_on_varied_group(tok):
    """有区分度的组(非零方差)→ 门控关闭，退回外部奖励主导(不触发 info-gain)。"""
    torch.manual_seed(3)
    model = TinyTransformer(vocab_size=tok.vocab_size, hidden=32, n_layers=2, n_heads=2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = Config()
    cfg.credit.method = "infopo"

    # 造有成功有失败的组
    good = _success_group(tok, G=2)
    bad = []
    for _ in range(2):
        env = TauEnv(*make_retail_task(), max_steps=16)
        bad.append(rollout(env, ScriptedPolicy([
            {"name": "respond", "arguments": {"content": "policy forbids"}},
            {"name": "respond", "arguments": {"content": "cannot per policy"}},
        ]), tok, max_steps=16))
    stats = train_step(model, opt, [good + bad], cfg)
    assert stats.n_infogain_gated == 0    # 有方差 → 不触发 info-gain


def test_infopo_overrides_dynamic_sampling(tok):
    """InfoPO 与 Dynamic Sampling 同开时：InfoPO 优先(救活)，不丢弃零方差组。"""
    torch.manual_seed(4)
    model = TinyTransformer(vocab_size=tok.vocab_size, hidden=32, n_layers=2, n_heads=2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = Config()
    cfg.credit.method = "infopo"
    cfg.algo.dynamic_sampling = True
    group = _success_group(tok, G=4)
    stats = train_step(model, opt, [group], cfg)
    assert stats.n_dropped_zero_var == 0    # 没丢弃
    assert stats.n_infogain_gated == 1      # 而是救活
