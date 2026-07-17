"""真机 policy 适配器接口：把 HF Qwen3-4B 包装成与离线 Policy 相同的契约。

离线 rollout 用 ScriptedPolicy / TinyModelPolicy；真机换成 HFPolicy，rollout() 代码零改动。
本文件提供接口骨架 —— 真机安装 transformers + vllm/sglang 后填充实现即可。

关键：真机的 logprobs_of / act 接口签名与离线一致，train_step 消费的 Sample 契约不变。
"""
from __future__ import annotations

from typing import Any, Optional


class HFPolicy:
    """HuggingFace 因果模型的 policy 适配器。

    真机加载 Jarrodbarnes/Qwen3-4B-tau2-sft1，用其原生 <tool_call> 格式生成。
    """

    def __init__(self, model_name: str = "Jarrodbarnes/Qwen3-4B-tau2-sft1",
                 device: str = "cuda", max_new_tokens: int = 512,
                 temperature: float = 0.8):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._tok = None

    def load(self):
        """惰性加载(离线单测不触发)。"""
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype="auto", device_map=self.device,
        )

    def act(self, messages: list[dict[str, Any]]) -> tuple[Optional[dict[str, Any]], str]:
        """与离线 Policy.act 同签名：给定对话历史，生成动作 + 原始文本。

        真机流程：apply_chat_template(messages, tools=...) → generate →
        用 rollout.policy.parse_tool_call 解析出 {name, arguments}。
        """
        if self._model is None:
            self.load()
        from agentic_rl.rollout.policy import parse_tool_call
        prompt = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tok(prompt, return_tensors="pt").to(self.device)
        out = self._model.generate(
            **inputs, max_new_tokens=self.max_new_tokens,
            temperature=self.temperature, do_sample=self.temperature > 0,
        )
        gen = self._tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        action = parse_tool_call(gen)
        return action, gen

    def logprobs_of(self, token_ids: list[int]):
        """返回每个 token 的 log π(供 PPO ratio 计算)。真机由 model forward 得到。"""
        raise NotImplementedError("真机接入：model(input_ids) → log_softmax → gather。")


# 真机 tokenizer 也换成 HF：
def load_hf_tokenizer(model_name: str = "Jarrodbarnes/Qwen3-4B-tau2-sft1"):
    from transformers import AutoTokenizer  # type: ignore
    return AutoTokenizer.from_pretrained(model_name)
