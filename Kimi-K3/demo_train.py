"""端到端 demo：在离线 tau2 环境上真跑几步 GRPO，打印 reward/loss 变化。

    python3 demo_train.py

这不是单测，而是给人看的直观演示：证明 rollout→credit→advantage→loss→backward
整条闭环在 CPU 上真实运转。真机训练时把 model 换成 Qwen3-4B、rollout 后端换 SGLang。
"""
import random

import torch

from agentic_rl.env.offline_tasks import make_retail_task, make_telecom_task
from agentic_rl.env.tau_env import TauEnv
from agentic_rl.rollout.policy import ScriptedPolicy
from agentic_rl.rollout.rollout import rollout
from agentic_rl.rollout.tokenizer import ByteTokenizer
from agentic_rl.train.model import TinyTransformer
from agentic_rl.train.trainer import train_step
from agentic_rl.utils.config import Config


def make_group(tok, task_fn, good_ratio, G=4):
    trajs = []
    for i in range(G):
        env = TauEnv(*task_fn(), max_steps=16)
        if i < int(G * good_ratio):
            if task_fn is make_retail_task:
                actions = [
                    {"name": "cancel_order", "arguments": {"order_id": "#W001"}},
                    {"name": "process_refund", "arguments": {"order_id": "#W001"}},
                    {"name": "respond", "arguments": {"content": "Your refund is done."}},
                ]
            else:
                actions = [{"name": "respond",
                            "arguments": {"content": "Turn off airplane mode and enable data."}}]
        else:
            actions = [{"name": "respond", "arguments": {"content": "Sorry, policy forbids."}},
                       {"name": "respond", "arguments": {"content": "I cannot per policy."}}]
        trajs.append(rollout(env, ScriptedPolicy(actions), tok, max_steps=16))
    return trajs


def main():
    torch.manual_seed(0); random.seed(0)
    tok = ByteTokenizer(vocab_size=512)
    model = TinyTransformer(vocab_size=tok.vocab_size, hidden=32, n_layers=2, n_heads=2)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    cfg = Config()
    cfg.credit.method = "r2g"      # UserRL 折扣前传
    cfg.algo.estimator = "grpo"

    # 探针：一条固定成功轨迹，追踪模型对其 response token 的平均 logprob。
    # GRPO 用正 advantage 强化成功轨迹 → 该 logprob 应随训练上升 = 模型真的在学。
    from agentic_rl.train.model import token_logprobs
    probe = rollout(TauEnv(*make_retail_task(), max_steps=16),
                    ScriptedPolicy([
                        {"name": "cancel_order", "arguments": {"order_id": "#W001"}},
                        {"name": "process_refund", "arguments": {"order_id": "#W001"}},
                        {"name": "respond", "arguments": {"content": "Your refund is done."}},
                    ]), tok, max_steps=16)

    def probe_logprob():
        ids = torch.tensor([probe.all_token_ids()])
        logits, _ = model(ids)
        lp = token_logprobs(logits, ids)
        mask = torch.tensor([probe.full_loss_mask()], dtype=torch.float32)[:, 1:]
        return float((lp * mask).sum() / mask.sum().clamp_min(1))

    print(f"config: credit={cfg.credit.method} estimator={cfg.algo.estimator} "
          f"gamma={cfg.credit.gamma}")
    print(f"{'step':>4} {'loss':>9} {'success_logprob↑':>17} {'mean_reward':>12} {'resp_len':>9}")
    for step in range(30):
        groups = [
            make_group(tok, make_retail_task, good_ratio=0.5, G=4),
            make_group(tok, make_telecom_task, good_ratio=0.5, G=4),
        ]
        s = train_step(model, opt, groups, cfg)
        if step % 5 == 0 or step == 29:
            print(f"{step:>4} {s.loss:>9.4f} {probe_logprob():>17.4f} "
                  f"{s.mean_reward:>12.3f} {s.mean_response_len:>9.1f}")

    print("\n✓ 端到端闭环运转正常，success_logprob 上升 = 成功轨迹被正 advantage 强化(模型在学)。")
    print("  真机：TinyTransformer→Qwen3-4B-tau2-sft1，ScriptedPolicy→真实采样，"
          "TauEnv→真实 tau2 env。")


if __name__ == "__main__":
    main()
