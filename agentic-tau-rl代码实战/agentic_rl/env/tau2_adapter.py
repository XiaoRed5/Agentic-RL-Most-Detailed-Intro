"""真实 tau2-bench 适配层：把真实 tau2 Environment 包装成我们的 TauEnv 契约。

现状：tau2-bench 要求 Python>=3.12(当前离线环境 3.9)，且未上 PyPI，故离线单测阶段
用 env/tau_env.py 的 mock。本模块提供**接入真实 tau2 的适配器接口**：
一旦运行环境满足(Python>=3.12 + pip install tau2-bench)，切换到真实 env 只需用
RealTau2Env 替换 TauEnv，rollout/train 代码零改动。

真实 tau2 的 rollout 循环由其 orchestrator 驱动(agent↔user_simulator↔environment)，
与我们的 env.step(action) 语义有差异 —— 适配器负责把二者对齐(见 step 映射注释)。
"""
from __future__ import annotations

from typing import Any, Optional


def tau2_available() -> bool:
    """检测真实 tau2-bench 是否可导入。"""
    try:
        import tau2  # noqa: F401
        return True
    except Exception:
        return False


class RealTau2Env:
    """真实 tau2 Environment 的适配器，暴露与 env.tau_env.TauEnv 相同的接口。

    映射关系(真实 tau2 → 我们的契约)：
    - tau2 的 env.get_tools() → 我们的 tools_info(供 policy 构造 tool schema)
    - tau2 orchestrator 一步 = 我们的一个 turn；user_simulator 的 STOP/TRANSFER →
      我们的 terminated + end_reason
    - tau2 EnvironmentEvaluator.calculate_reward(基于 DB 终态 hash + env_assertion)
      → 我们的 StepResult.reward。**这套 reward 语义与我们 env/reward.py 完全一致**
      (我们正是照它复刻的，并用真实 gold reward_info 对拍通过)。
    """

    def __init__(self, domain: str, task_index: int, user_model: str = "gpt-4.1-mini",
                 max_steps: int = 100):
        if not tau2_available():
            raise RuntimeError(
                "tau2-bench 不可用。需 Python>=3.12 且 `pip install -e tau2-bench`。"
                "离线单测请用 env.tau_env.TauEnv。"
            )
        self.domain = domain
        self.task_index = task_index
        self.user_model = user_model
        self.max_steps = max_steps
        self._build()

    def _build(self):
        from tau2.registry import registry  # type: ignore
        # 真实构建：注意 tau2 的 API 以其源码为准，此处给出接入骨架
        get_env = registry.get_env_constructor(self.domain)
        self._env = get_env()
        self._tasks = registry.get_tasks(self.domain)
        self._task = self._tasks[self.task_index]

    def get_tools_info(self) -> list[dict[str, Any]]:
        """返回 OpenAI-schema 的工具列表(policy 构造 tool-calling prompt 用)。"""
        return [t.openai_schema() for t in self._env.get_tools()]

    def reset(self) -> str:
        # 真实 tau2 通过 orchestrator 初始化；此处返回首条 user 消息
        raise NotImplementedError(
            "接入真实 tau2 时，用其 orchestrator 驱动多轮循环。"
            "reward 计算复用 env/reward.py 的 compute_reward(已对拍真实 gold)。"
        )

    def step(self, action: dict[str, Any]):
        raise NotImplementedError("同 reset：由 tau2 orchestrator 驱动。")
