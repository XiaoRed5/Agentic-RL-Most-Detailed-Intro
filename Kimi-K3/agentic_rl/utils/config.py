"""全局配置：把所有可切换的算法开关集中在一处。

设计目标：离线单测(tiny model, CPU) 与 真机训练(Qwen3-4B, GPU) 共用同一份
配置结构，切换只改字段值，不改代码。对齐 survey 里三层 + 两条同期解法的所有旋钮。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # 离线单测用 tiny model；真机换成 "Jarrodbarnes/Qwen3-4B-tau2-sft1"
    name: str = "tiny"
    vocab_size: int = 256        # tiny tokenizer 词表(离线用)
    hidden_size: int = 32
    n_layers: int = 2
    n_heads: int = 2
    max_seq_len: int = 2048


@dataclass
class CreditConfig:
    """信用分配(credit assignment) —— survey 的核心深水区。三种打法可切换。"""

    # method: "outcome" | "r2g" | "infopo"
    #   outcome —— 只用最终奖励，中间轮全 0(最朴素，Search-R1 式)
    #   r2g     —— UserRL 的 Reward-to-Go 折扣前传
    #   infopo  —— InfoPO 的反事实信息增益 + 方差门控
    method: str = "r2g"

    # --- R2G (UserRL) ---
    gamma: float = 0.8                # 折扣因子，survey TravelGym 算例用 0.8

    # --- InfoPO ---
    infopo_variance_threshold: float = 1e-6   # 组内奖励方差低于此值 → 判定"零方差组"
    infopo_gain_weight: float = 1.0           # info-gain advantage 的放大权重
    infopo_kl_coef: float = 0.1               # token 级 KL 正则系数


@dataclass
class ShapingConfig:
    """行为塑形正则 —— BAO(减法) + PPP(加法)。默认全关，单测逐个打开。"""

    # BAO 正则①：Information-Seeking，连续两轮都问用户则罚
    bao_info_seeking: bool = False
    lambda_ans: float = 0.1           # -λ_ans(论文未公开数值，此为占位默认)

    # BAO 正则②：Over-Thinking，失败且提前终止则罚 -λ_think·(T-T')/T'
    bao_over_thinking: bool = False
    lambda_think: float = 0.1
    turn_budget: int = 16             # T = 固定交互预算(轮数)，BAO 用 16

    # PPP 三目标：R = R_Prod + R_Proact + R_Pers(无权重相加)
    ppp_enabled: bool = False
    ppp_proact_low_bonus: float = 0.05    # 所有提问都 low-effort 的整体奖励
    ppp_proact_medium: float = -0.1       # 每个 medium-effort 提问
    ppp_proact_high: float = -0.5         # 每个 high-effort 提问(甩活给用户)
    ppp_pers_bonus: float = 0.05          # 满足一条偏好


@dataclass
class AlgoConfig:
    """advantage estimator + loss。GRPO/RLOO/PPO 可切换，DAPO 四件套可逐个开关。"""

    # estimator: "grpo" | "rloo" | "ppo_gae"
    estimator: str = "grpo"
    group_size: int = 8               # GRPO/RLOO 每个 prompt 采样组大小

    # PPO / GAE
    gae_lambda: float = 0.95
    gae_gamma: float = 1.0

    # PPO clip
    clip_eps: float = 0.2
    # DAPO 四件套
    clip_higher: bool = False         # ① Clip-Higher：正负 clip 用不同 eps
    clip_eps_high: float = 0.28
    dynamic_sampling: bool = False    # ② Dynamic Sampling：丢弃零方差组
    token_level_loss: bool = False    # ③ Token-Level Loss：按 token 平均而非按序列
    overlong_shaping: bool = False    # ④ Overlong Reward Shaping：超长软惩罚
    overlong_buffer: int = 512
    max_response_len: int = 1024

    kl_coef: float = 0.0              # 与 ref policy 的 KL 惩罚(GRPO 常用 0)
    entropy_coef: float = 0.0


@dataclass
class RolloutConfig:
    max_steps: int = 16               # 一条轨迹最多多少个 agent 决策轮
    temperature: float = 1.0


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    credit: CreditConfig = field(default_factory=CreditConfig)
    shaping: ShapingConfig = field(default_factory=ShapingConfig)
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    seed: int = 0


def default_offline_config() -> Config:
    """离线单测默认配置：tiny model + R2G + GRPO。"""
    return Config()
