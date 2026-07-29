# Agentic RL 入门

又是小red帮学妹找基座实习的一天。

我目前在美团M17做基座agent后训练，24年上交EE毕业转AI，帮助像曾经的我一样想转行的朋友！

欢迎关注我的小红书：[小red](https://xhslink.com/m/2dKOxmxjYcC)，后续会继续更新 Agentic RL、基座模型后训练和算法工程师转行相关内容。

## 三个项目方向

本仓库目前沿着三个方向持续更新：实验可视化、Agentic RL 代码实战，以及 Agentic RL 技术论文阅读。

### 1️⃣ Agent Forge：Agent / RL 实验可视化

这是本仓库重点推荐的可视化项目：把两组 Agent / RL 实验的训练趋势、离线评测、工具调用成本和证据强度放进同一个交互式看板。

**[进入可视化项目](<./Agent Forge：Agent ／ RL 实验可视化/>)** · **[查看完整使用说明](<./Agent Forge：Agent ／ RL 实验可视化/docs/USAGE.md>)** · **[下载数据模板](<./Agent Forge：Agent ／ RL 实验可视化/data/experiment.example.json>)**

### 2️⃣ agentic-tau-rl 代码实战

一套可运行、可单测的 Agentic RL 实现，覆盖多轮 rollout、信用分配、策略优化和行为塑形，并提供从离线验证迁移到真实模型训练的完整说明。

**[进入代码实战项目](<./agentic-tau-rl代码实战/>)** · **[阅读技术报告](<./agentic-tau-rl代码实战/技术报告.html>)** · **[查看真机迁移指南](<./agentic-tau-rl代码实战/docs/MIGRATION.md>)**

### 3️⃣ Agentic RL 技术论文阅读

从基础概念、信用分配和多轮工具调用，到 LongCat、GLM 与 Kimi 等代表性工作，持续整理 Agentic RL 论文与技术实践。

**[进入技术论文在线阅读](https://xiaored5.github.io/Agentic-RL-Most-Detailed-Intro/)**

- [Agentic RL入门1：基础、代码](https://xiaored5.github.io/Agentic-RL-Most-Detailed-Intro/Agentic%20RL%E5%85%A5%E9%97%A81%EF%BC%9A%E5%9F%BA%E7%A1%80%E3%80%81%E4%BB%A3%E7%A0%81.html)
- [Agentic RL入门2：信用分配](https://xiaored5.github.io/Agentic-RL-Most-Detailed-Intro/Agentic%20RL%E5%85%A5%E9%97%A82%EF%BC%9A%E4%BF%A1%E7%94%A8%E5%88%86%E9%85%8D.html)
- [Agentic RL入门3：transformer 架构](https://xiaored5.github.io/Agentic-RL-Most-Detailed-Intro/Agentic%20RL%E5%85%A5%E9%97%A83%EF%BC%9Atransformer%E6%9E%B6%E6%9E%84.html)
- [Agentic RL入门4：Credit 错分与算法接口](https://xiaored5.github.io/Agentic-RL-Most-Detailed-Intro/Agentic%20RL%E5%85%A5%E9%97%A84%EF%BC%9ACredit%20%E9%94%99%E5%88%86%E4%B8%8E%E7%AE%97%E6%B3%95%E6%8E%A5%E5%8F%A3.html)
- [Agentic RL入门5：Skill-based Agentic RL](https://xiaored5.github.io/Agentic-RL-Most-Detailed-Intro/Agentic%20RL%E5%85%A5%E9%97%A85%EF%BC%9ASkill-based%20Agentic%20RL.html)
- [Agentic RL入门6：多轮工具调用](https://xiaored5.github.io/Agentic-RL-Most-Detailed-Intro/Agentic%20RL%E5%85%A5%E9%97%A86%EF%BC%9A%E5%A4%9A%E8%BD%AE%E5%B7%A5%E5%85%B7%E8%B0%83%E7%94%A8.html)
- [如何在 5 万张国产芯片训练出 1.6T 万亿参数模型？](https://xiaored5.github.io/Agentic-RL-Most-Detailed-Intro/longcat-2.0/如何在%205%20万张国产芯片训练出%201.6T%20万亿参数模型？.html)
- [GLM-5.2 长程任务的 RL 怎么做？](https://xiaored5.github.io/Agentic-RL-Most-Detailed-Intro/glm-5.2-longhorizon-rl/GLM-5.2%20%E9%95%BF%E7%A8%8B%E4%BB%BB%E5%8A%A1%E7%9A%84%20RL%20%E6%80%8E%E4%B9%88%E5%81%9A%EF%BC%9F.html)
- [Kimi-K3 为什么这么强？](https://xiaored5.github.io/Agentic-RL-Most-Detailed-Intro/Kimi-K3%20%E4%B8%BA%E4%BB%80%E4%B9%88%E8%BF%99%E4%B9%88%E5%BC%BA%EF%BC%9F.html)
- [Kimi K3 后训练：9 个专家蒸馏成 1 个](https://xiaored5.github.io/Agentic-RL-Most-Detailed-Intro/kimi-k3-report-posttraining-xhs.html)

## 本地预览

```bash
python3 -m http.server 8000
```

然后访问：

```text
http://localhost:8000
```
