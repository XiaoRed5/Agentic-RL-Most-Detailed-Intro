"""加固层测试：多步 PPO(ratio≠1 + clip 真正起作用 + ref KL) + PPP effort 解析。"""
import pytest
import torch

from agentic_rl.env.offline_tasks import make_retail_task
from agentic_rl.env.tau_env import TauEnv
from agentic_rl.rollout.policy import ScriptedPolicy
from agentic_rl.rollout.rollout import rollout
from agentic_rl.rollout.tokenizer import ByteTokenizer
from agentic_rl.shaping.shaping import (
    Effort,
    efforts_from_user_replies,
    parse_effort_from_cost,
    ppp_proactivity_reward,
)
from agentic_rl.train.model import TinyTransformer
from agentic_rl.train.trainer import multi_step_ppo_update
from agentic_rl.utils.config import Config


@pytest.fixture
def tok():
    return ByteTokenizer(vocab_size=512)


def _mixed_groups(tok, G=2):
    good, bad = [], []
    for _ in range(G):
        good.append(rollout(TauEnv(*make_retail_task(), max_steps=16), ScriptedPolicy([
            {"name": "cancel_order", "arguments": {"order_id": "#W001"}},
            {"name": "process_refund", "arguments": {"order_id": "#W001"}},
            {"name": "respond", "arguments": {"content": "Your refund is done."}},
        ]), tok, max_steps=16))
        bad.append(rollout(TauEnv(*make_retail_task(), max_steps=16), ScriptedPolicy([
            {"name": "respond", "arguments": {"content": "policy forbids"}},
            {"name": "respond", "arguments": {"content": "cannot per policy"}},
        ]), tok, max_steps=16))
    return [good + bad]


# ========== 多步 PPO ==========
def test_multi_step_ppo_runs_and_ratio_diverges(tok):
    """多步 PPO：第一 epoch ratio=1(clip_frac=0)，后续 epoch ratio≠1(clip 开始生效)。"""
    torch.manual_seed(0)
    model = TinyTransformer(vocab_size=tok.vocab_size, hidden=32, n_layers=2, n_heads=2)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)   # 大 lr 让 ratio 快速偏离
    cfg = Config()
    hist = multi_step_ppo_update(model, opt, _mixed_groups(tok), cfg, ppo_epochs=4)
    assert len(hist) == 4
    # 第一 epoch：old==current → approx_kl≈0
    assert abs(hist[0]["approx_kl"]) < 1e-5
    # 后续 epoch：策略已更新 → 与 old 出现偏离(kl>0)
    assert hist[-1]["approx_kl"] != hist[0]["approx_kl"]
    for h in hist:
        assert not torch.isnan(torch.tensor(h["loss"]))


def test_multi_step_ppo_clip_activates(tok):
    """足够大的更新后，clip_frac 应从 0 变正(证明 clip 真在裁剪)。"""
    torch.manual_seed(1)
    model = TinyTransformer(vocab_size=tok.vocab_size, hidden=32, n_layers=2, n_heads=2)
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    cfg = Config()
    hist = multi_step_ppo_update(model, opt, _mixed_groups(tok), cfg, ppo_epochs=6)
    assert hist[0]["clip_frac"] == pytest.approx(0.0)   # 首轮无裁剪
    assert max(h["clip_frac"] for h in hist) > 0.0      # 后续有裁剪


def test_ref_kl_penalty_reduces_drift(tok):
    """加 ref KL 惩罚 → 相比无 KL，策略对 ref 的漂移更小(用稳定 lr 避免 SGD 过冲)。

    漂移用'当前 policy 对 ref 的真实 KL'度量(而非对 old 的 approx_kl)。"""
    def run(kl_coef):
        torch.manual_seed(7)
        m = TinyTransformer(vocab_size=tok.vocab_size, hidden=32, n_layers=2, n_heads=2)
        ref = TinyTransformer(vocab_size=tok.vocab_size, hidden=32, n_layers=2, n_heads=2)
        ref.load_state_dict(m.state_dict())
        opt = torch.optim.SGD(m.parameters(), lr=0.05)   # 稳定 lr，避免 KL 梯度过冲
        cfg = Config(); cfg.algo.kl_coef = kl_coef
        groups = _mixed_groups(tok)
        multi_step_ppo_update(m, opt, groups, cfg, ppo_epochs=5, ref_model=ref)
        # 度量：训练后 policy 相对 ref 的 KL(在同一批 token 上)
        from agentic_rl.algo.loss import kl_penalty
        from agentic_rl.train.model import token_logprobs
        from agentic_rl.train.trainer import build_batch_advantages
        ids, mask, _ = build_batch_advantages(groups, m, cfg)
        with torch.no_grad():
            lp = token_logprobs(m(ids)[0], ids)
            rlp = token_logprobs(ref(ids)[0], ids)
            return float(kl_penalty(lp.reshape(-1), rlp.reshape(-1), mask[:, 1:].reshape(-1)))
    drift_no_kl = run(0.0)
    drift_with_kl = run(2.0)
    assert drift_with_kl < drift_no_kl   # KL 惩罚确实抑制了对 ref 的漂移


# ========== PPP effort 解析([Cost N]) ==========
def test_parse_cost_low():
    assert parse_effort_from_cost("Sure, it's the login page. [Cost 1]") == Effort.LOW


def test_parse_cost_medium():
    assert parse_effort_from_cost("I don't know. [Cost 3]") == Effort.MEDIUM


def test_parse_cost_high():
    """Cost 5 = 被迫掏出文件路径/函数名 → high(甩活给用户)。"""
    assert parse_effort_from_cost("The file is auth/login.py, function verify_user. [Cost 5]") == Effort.HIGH


def test_parse_no_cost_tag():
    assert parse_effort_from_cost("Okay, thanks.") is None


def test_efforts_from_replies_skips_untagged():
    replies = ["It's the login page [Cost 1]", "no tag here", "which file? [Cost 4]"]
    efforts = efforts_from_user_replies(replies)
    assert efforts == [Effort.LOW, Effort.HIGH]


def test_effort_feeds_proactivity_reward():
    """端到端：解析出的 effort 直接喂 PPP proactivity reward。"""
    replies = ["login page [Cost 1]", "which function? [Cost 5]"]
    efforts = efforts_from_user_replies(replies)
    r = ppp_proactivity_reward(efforts)
    # 一个 low 一个 high → 非全 low 无 bonus，一个 high -0.5
    assert r == pytest.approx(-0.5)
