"""核心数据类型：对齐 slime 的 Sample 契约，同时承载 tau2 轨迹的多轮结构。

设计原则：
- Turn 是信用分配(credit assignment)的最小单位。UserRL 的 R2G、BAO 的 turn-level
  正则、InfoPO 的反事实增益都在 Turn 粒度上工作，所以 Turn 必须携带足够信息。
- Trajectory 拍平成 token 序列后就是 slime 训练需要的 (tokens, loss_mask, reward)。
  loss_mask 是 agentic RL 最易错的地方：只有 policy 自己生成的 token 参与 loss，
  环境/工具/用户返回的 token 一律 mask 掉(=0)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ActionType(str, Enum):
    """agent 每一轮做的动作类型。用于 BAO 的 Information-Seeking 正则
    (判断是否连续两轮都在'问用户')和 credit assignment 的动作归类。"""

    RESPOND = "respond"      # 对用户说话/提问 —— 属于 BAO 的 𝒜_u(user-involved 动作集)
    TOOL_CALL = "tool_call"  # 调用后台工具(改数据库状态) —— 真正干活
    ANSWER = "answer"        # 提交最终答案(部分 gym 有显式 answer 动作)
    OTHER = "other"


class Status(str, Enum):
    """与 slime example 的 status 映射对齐。"""

    COMPLETED = "completed"   # 任务自然结束(user 发 STOP/TRANSFER，或答对/答错)
    TRUNCATED = "truncated"   # 到达 max_steps 被截断
    ABORTED = "aborted"       # 解析失败/异常中断


@dataclass
class Turn:
    """一个 agent 决策轮：从'轮到 agent'到'agent 交还控制权'之间的一段。

    一个 Turn 通常 = 一条 assistant 消息(可能含 tool_call) + 紧随其后的
    环境/工具/用户反馈。token 层面：assistant 部分 loss_mask=1，反馈部分=0。
    """

    index: int                                   # 该轮在轨迹中的序号(从 0 起)
    action_type: ActionType
    action_name: Optional[str] = None            # 工具名 / "respond" / "answer"
    action_args: dict[str, Any] = field(default_factory=dict)

    # token 级 payload(离线单测阶段可为空，用 tokenizer 现场填)
    assistant_token_ids: list[int] = field(default_factory=list)  # policy 生成的 token
    obs_token_ids: list[int] = field(default_factory=list)        # 环境/用户返回的 token

    # reward：区分即时(env 当轮给的)与塑形后(R2G/BAO 处理后的)
    immediate_reward: float = 0.0     # 环境在这一轮给出的原始即时奖励
    shaped_reward: float = 0.0        # 经 turn-level shaping(BAO 正则等)后的奖励
    r2g: float = 0.0                  # Reward-to-Go(UserRL 折扣前传结果)
    advantage: float = 0.0            # 最终喂给 loss 的 advantage

    # 诊断/归因用
    is_user_involved: bool = False    # 是否属于 𝒜_u(问用户/请求核验) —— BAO 正则用
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def n_assistant_tokens(self) -> int:
        return len(self.assistant_token_ids)

    def token_ids(self) -> list[int]:
        """该轮完整 token 序列：assistant 生成在前，环境反馈在后。"""
        return self.assistant_token_ids + self.obs_token_ids

    def loss_mask(self) -> list[int]:
        """该轮 loss mask：assistant token=1(算 loss)，环境反馈 token=0(不算)。"""
        return [1] * len(self.assistant_token_ids) + [0] * len(self.obs_token_ids)


@dataclass
class Trajectory:
    """一条完整的多轮 agent-环境交互轨迹。

    拍平成 token 序列后即 slime 训练所需：prompt_token_ids 是初始 prompt(全 mask=0)，
    turns 拍平后是 response 部分(assistant=1 / env=0)。
    """

    task_id: str
    domain: str                                  # retail / airline / telecom
    prompt_token_ids: list[int] = field(default_factory=list)  # 初始 system+首轮 user
    turns: list[Turn] = field(default_factory=list)

    # 轨迹级 reward(trajectory-level scoring，与 turn-level shaping 解耦 —— UserRL 思想)
    outcome_reward: float = 0.0                   # 最终成败(DB终态匹配等)，∈{0,1} 或 partial
    reward_info: dict[str, Any] = field(default_factory=dict)  # 对齐 tau2 reward_info

    status: Status = Status.COMPLETED
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---- 拍平接口：credit / algo / train 都从这里取 token 级张量 ----
    def response_token_ids(self) -> list[int]:
        out: list[int] = []
        for t in self.turns:
            out.extend(t.token_ids())
        return out

    def response_loss_mask(self) -> list[int]:
        out: list[int] = []
        for t in self.turns:
            out.extend(t.loss_mask())
        return out

    def all_token_ids(self) -> list[int]:
        return list(self.prompt_token_ids) + self.response_token_ids()

    def full_loss_mask(self) -> list[int]:
        """整条序列的 loss mask：prompt 部分恒为 0，response 部分按 turn 规则。"""
        return [0] * len(self.prompt_token_ids) + self.response_loss_mask()

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    @property
    def response_length(self) -> int:
        return len(self.response_token_ids())

    def num_user_involved_turns(self) -> int:
        return sum(1 for t in self.turns if t.is_user_involved)


@dataclass
class Sample:
    """与 slime.utils.types.Sample 字段对齐的最小契约。

    离线自建实现与真机 slime 之间的桥：rollout 产出 Sample，训练消费 Sample。
    迁到真机时只需把这个 dataclass 换成 slime 的 Sample，字段名一致即可。
    """

    index: int
    prompt: str
    tokens: list[int]                 # prompt + response 的完整 token 序列
    response: str
    reward: float
    loss_mask: list[int]              # 只覆盖 response 部分(与 slime example 一致)
    status: str
    response_length: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_trajectory(cls, traj: Trajectory, index: int, prompt_text: str = "",
                        response_text: str = "") -> "Sample":
        return cls(
            index=index,
            prompt=prompt_text,
            tokens=traj.all_token_ids(),
            response=response_text,
            reward=traj.outcome_reward,
            loss_mask=traj.response_loss_mask(),   # slime: loss_mask 只含 response 部分
            status=traj.status.value,
            response_length=traj.response_length,
            metadata=traj.metadata,
        )
