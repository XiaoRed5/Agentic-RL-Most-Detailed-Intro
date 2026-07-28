"""端到端训练单测：tiny model 上真跑 GRPO，验证整条闭环可导、可学。

这是整个项目的收官测试：rollout → credit → shaping → advantage → PPO loss → backward
全链路在 CPU 上真实跑通，且断言"梯度真的在流、loss 有限、能过拟合提升"。
"""
import random

import pytest
import torch

from agentic_rl.env.offline_tasks import make_retail_task, make_telecom_task
from agentic_rl.env.tau_env import TauEnv
from agentic_rl.rollout.policy import ScriptedPolicy
from agentic_rl.rollout.rollout import rollout
from agentic_rl.rollout.tokenizer import ByteTokenizer
from agentic_rl.train.model import TinyTransformer, token_logprobs
from agentic_rl.train.trainer import train_step
from agentic_rl.utils.config import Config


@pytest.fixture
def tok():
    return ByteTokenizer(vocab_size=512)


@pytest.fixture
def model(tok):
    torch.manual_seed(0)
    return TinyTransformer(vocab_size=tok.vocab_size, hidden=32, n_layers=2,
                           n_heads=2, max_len=2048, with_value=True)


def _make_group(tok, success=True, G=4):
    """造一个 group：G 条 retail rollout。success=True 时都成功(reward=1)。"""
    trajs = []
    for _ in range(G):
        task, tools = make_retail_task()
        env = TauEnv(task, tools, max_steps=16)
        if success:
            policy = ScriptedPolicy([
                {"name": "cancel_order", "arguments": {"order_id": "#W001"}},
                {"name": "process_refund", "arguments": {"order_id": "#W001"}},
                {"name": "respond", "arguments": {"content": "Your refund is done."}},
            ])
        else:
            policy = ScriptedPolicy([
                {"name": "respond", "arguments": {"content": "Sorry, policy forbids this."}},
                {"name": "respond", "arguments": {"content": "I cannot per policy."}},
            ])
        trajs.append(rollout(env, policy, tok, max_steps=16))
    return trajs


# ========== 模型基本可导 ==========
def test_model_forward_shapes(model, tok):
    ids = torch.tensor([tok.encode("hello world")])
    logits, values = model(ids)
    assert logits.shape[0] == 1 and logits.shape[-1] == tok.vocab_size
    assert values.shape == ids.shape


def test_token_logprobs_shape(model, tok):
    ids = torch.tensor([tok.encode("abcdef")])
    logits, _ = model(ids)
    lp = token_logprobs(logits, ids)
    assert lp.shape == (1, ids.shape[1] - 1)
    assert torch.all(lp <= 0)   # log 概率 <= 0


# ========== 单步训练：loss 有限、梯度在流 ==========
def test_train_step_finite_loss_and_grad(model, tok):
    cfg = Config()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # 一个有区分度的 group(有成功有失败 → 非零方差)
    group = _make_group(tok, success=True, G=2) + _make_group(tok, success=False, G=2)
    stats = train_step(model, opt, [group], cfg)
    assert not torch.isnan(torch.tensor(stats.loss))
    assert abs(stats.loss) < 1e4
    # 梯度真的在流
    total_grad = sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)
    assert float(total_grad) > 0


def test_zero_variance_group_dropped_with_dynamic_sampling(model, tok):
    """DAPO Dynamic Sampling 开启时，全成功(零方差)的 group 被丢弃。"""
    cfg = Config()
    cfg.algo.dynamic_sampling = True
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    group = _make_group(tok, success=True, G=4)   # 全 reward=1 → 零方差
    stats = train_step(model, opt, [group], cfg)
    assert stats.n_dropped_zero_var == 1


def test_zero_variance_kept_without_dynamic_sampling(model, tok):
    cfg = Config()
    cfg.algo.dynamic_sampling = False
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    group = _make_group(tok, success=True, G=4)
    stats = train_step(model, opt, [group], cfg)
    assert stats.n_dropped_zero_var == 0


# ========== 多步训练：能过拟合(学到偏好成功轨迹) ==========
def test_overfit_increases_success_logprob(tok):
    """核心 learnability 测试：反复对'成功轨迹 advantage>0 / 失败<0'做 GRPO，
    成功轨迹的 token logprob 应上升(模型学会更可能生成成功轨迹的 token)。"""
    torch.manual_seed(0)
    random.seed(0)
    model = TinyTransformer(vocab_size=tok.vocab_size, hidden=32, n_layers=2, n_heads=2)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    cfg = Config()

    # 固定一条成功轨迹，测其 response token 的平均 logprob 变化
    task, tools = make_retail_task()
    env = TauEnv(task, tools, max_steps=16)
    success_policy = lambda: ScriptedPolicy([
        {"name": "cancel_order", "arguments": {"order_id": "#W001"}},
        {"name": "process_refund", "arguments": {"order_id": "#W001"}},
        {"name": "respond", "arguments": {"content": "Your refund is done."}},
    ])
    probe = rollout(TauEnv(*make_retail_task(), max_steps=16), success_policy(), tok)

    def mean_success_logprob():
        ids = torch.tensor([probe.all_token_ids()])
        logits, _ = model(ids)
        lp = token_logprobs(logits, ids)          # [1, T-1]
        mask = torch.tensor([probe.full_loss_mask()], dtype=torch.float32)[:, 1:]
        return float((lp * mask).sum() / mask.sum().clamp_min(1))

    before = mean_success_logprob()
    for _ in range(30):
        group = _make_group(tok, success=True, G=2) + _make_group(tok, success=False, G=2)
        train_step(model, opt, [group], cfg)
    after = mean_success_logprob()

    # 成功轨迹被正 advantage 强化 → 其 token logprob 应上升
    assert after > before, f"logprob 未上升: {before:.3f} -> {after:.3f}"


def test_credit_methods_all_run_end_to_end(model, tok):
    """三种信用分配打法都能端到端跑通一个 train_step。"""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for method in ["outcome", "r2g", "infopo"]:
        cfg = Config()
        cfg.credit.method = method
        group = _make_group(tok, success=True, G=2) + _make_group(tok, success=False, G=2)
        stats = train_step(model, opt, [group], cfg)
        assert not torch.isnan(torch.tensor(stats.loss)), method


def test_bao_shaping_runs_end_to_end(model, tok):
    """BAO 正则开启时端到端跑通。"""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    cfg = Config()
    cfg.shaping.bao_info_seeking = True
    cfg.shaping.bao_over_thinking = True
    group = _make_group(tok, success=True, G=2) + _make_group(tok, success=False, G=2)
    stats = train_step(model, opt, [group], cfg)
    assert not torch.isnan(torch.tensor(stats.loss))


def test_success_trajectory_gets_positive_advantage(tok):
    """group 级 credit 正确性：成功轨迹的 assistant token 平均 advantage > 失败轨迹。

    这是修 demo 里 logprob 下降 bug 的回归保护：早先 traj_adv*(1+nv) 会把成功轨迹
    早期轮翻负。正确的 group 级 R2G 归一化下，成功轨迹整体应得正信用。
    """
    from agentic_rl.train.trainer import compute_group_token_advantages

    succ = rollout(TauEnv(*make_retail_task(), max_steps=16), ScriptedPolicy([
        {"name": "cancel_order", "arguments": {"order_id": "#W001"}},
        {"name": "process_refund", "arguments": {"order_id": "#W001"}},
        {"name": "respond", "arguments": {"content": "Your refund is done."}},
    ]), tok, max_steps=16)
    fail = rollout(TauEnv(*make_retail_task(), max_steps=16), ScriptedPolicy([
        {"name": "respond", "arguments": {"content": "policy forbids"}},
        {"name": "respond", "arguments": {"content": "cannot per policy"}},
    ]), tok, max_steps=16)

    cfg = Config(); cfg.credit.method = "r2g"
    advs = compute_group_token_advantages([succ, fail], cfg)
    succ_mean = sum(advs[0]) / max(len([a for a in advs[0] if a != 0]), 1)
    fail_mean = sum(advs[1]) / max(len([a for a in advs[1] if a != 0]), 1)
    assert succ_mean > fail_mean
    assert succ_mean > 0    # 成功轨迹整体得正信用


def test_masked_tokens_get_zero_advantage(tok):
    """回归保护：obs token(loss_mask=0)在 token_adv 里必须是 0(不参与强化)。"""
    from agentic_rl.train.trainer import compute_group_token_advantages
    task, tools = make_retail_task()
    env = TauEnv(task, tools, max_steps=16)
    policy = ScriptedPolicy([
        {"name": "cancel_order", "arguments": {"order_id": "#W001"}},
        {"name": "respond", "arguments": {"content": "refund done"}},
    ])
    traj = rollout(env, policy, tok, max_steps=16)
    # 造一个含成功+失败的 group，走 group 级 credit
    fail_env = TauEnv(*make_retail_task(), max_steps=16)
    fail_traj = rollout(fail_env, ScriptedPolicy([
        {"name": "respond", "arguments": {"content": "policy forbids"}},
        {"name": "respond", "arguments": {"content": "cannot per policy"}},
    ]), tok, max_steps=16)
    cfg = Config()
    resp_advs = compute_group_token_advantages([traj, fail_traj], cfg)
    resp_adv = resp_advs[0]
    # 逐 turn 检查：obs 段 advantage 全 0
    idx = 0
    for turn in traj.turns:
        idx += len(turn.assistant_token_ids)
        for _ in turn.obs_token_ids:
            assert resp_adv[idx] == 0.0
            idx += 1
