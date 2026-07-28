# 从离线单测迁移到真机训练

本文档说明如何把已 100% 离线单测通过的代码，切换到真机(GPU + 真实 tau2 + Qwen3-4B)。
**核心承诺：RL 算法逻辑零改动，只替换三个"接口边界"组件 + 换配置文件。**

## 前置环境

```bash
# tau2-bench 要求 Python>=3.12(离线单测环境是 3.9，故离线用 mock env)
python --version   # >= 3.12
pip install git+https://github.com/sierra-research/tau2-bench.git
pip install transformers accelerate vllm   # 或 sglang
# 下载冷启动模型(免 SFT)
hf download Jarrodbarnes/Qwen3-4B-tau2-sft1
```

## 三个替换点(边界组件)

| 离线(单测) | 真机 | 替换文件 |
|---|---|---|
| `TinyTransformer` | Qwen3-4B (HF) | `train/model.py` → 用 `AutoModelForCausalLM` |
| `ScriptedPolicy` / `TinyModelPolicy` | `HFPolicy` | `rollout/hf_policy.py`(接口已就位) |
| `TauEnv`(mock) | `RealTau2Env` | `env/tau2_adapter.py`(接口已就位) |
| `ByteTokenizer` | Qwen3 tokenizer | `rollout/hf_policy.py: load_hf_tokenizer` |

**不变的**：`rollout()`、`train_step()`、`multi_step_ppo_update()`、所有 `algo/`、
`credit/`、`shaping/` 模块 —— 它们只依赖 `Sample`/`Trajectory` 契约，与后端无关。

## 配置切换

```python
from agentic_rl.utils.config_loader import load_config
cfg = load_config("configs/real_qwen3_4b.yaml")   # 离线是 configs/offline.yaml
```

两个 YAML 结构完全一致(`test_10_config.py::test_configs_share_structure` 保证)，
真机版把 DAPO 四件套、KL 锚点、tau2 官方 max_steps=100 等打开。

## 复现路线(对齐 survey 三层)

1. **Baseline (UserRL 式 GRPO)**：`credit.method=r2g` + `estimator=grpo`。
   先不开 shaping，跑出 baseline，观察 pass@k 与 User-Involvement Rate。
   效果参照：`Jarrodbarnes/Qwen3-4B-tau2-grpo-v1`(pass@4 59%)。

2. **叠 DAPO**：打开 `clip_higher / dynamic_sampling / token_level_loss / overlong_shaping`。

3. **BAO 治副作用**：观察到 agent"话太多/想太多"后，打开 `bao_info_seeking` +
   `bao_over_thinking`，调 `lambda_ans / lambda_think`(论文未公开数值，需自调)。

4. **InfoPO(可选)**：`credit.method=infopo`，让零方差组靠反事实 info-gain 供梯度。
   与 `dynamic_sampling` 是对零方差组的对立处理，InfoPO 优先救活。

5. **PPP(可选)**：把 tau2 user-sim 改成吐 `[Cost N]` 标签，用 `shaping.parse_effort_from_cost`
   解析 effort，接入三目标 reward。

## 接 Slime

真机训练后端用 THUDM/slime(Megatron-LM + SGLang)。我们的 `Sample` 字段与 slime 的
`slime.utils.types.Sample` 对齐(`index/prompt/tokens/response/reward/loss_mask/status/
response_length/metadata`)，把 `rollout()` 产出的 Trajectory 经 `Sample.from_trajectory`
转换后即可喂给 slime 的训练 actor。参考 slime `examples/tau-bench/generate_with_tau.py`
的 `res_to_sample` —— 我们的实现与之同构。

## 验证迁移正确性

真机接入后，`tests/test_08_tau2_integration.py` 里 2 个 `skipif` 测试会自动激活，
用真实 tau2 evaluator 交叉验证我们的 reward 聚合。当前离线已用真实 gold reward_info
全部对拍通过(最强证明)。
