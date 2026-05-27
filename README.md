# Agentic RL 信用分配

这是一份面向算法工程师的 Agentic RL 信用分配入门报告，聚焦长程多步智能体任务中如何把最终 outcome reward 分配到中间步骤。

## 内容结构

- Agentic RL 中 credit assignment 的基本问题
- 三类 step-level 信用分配方法
  - State-anchored Stepwise：GiGPO、HGPO
  - Process / Progress Reward：SPA-RL、AgentPRM
  - Intrinsic Signal：ARPO、IGPO
- 关键公式、直觉例子和方法对比

## 在线阅读

打开 `index.html` 即可阅读完整报告。

## 本地预览

```bash
python3 -m http.server 8000
```

然后访问：

```text
http://localhost:8000
```
