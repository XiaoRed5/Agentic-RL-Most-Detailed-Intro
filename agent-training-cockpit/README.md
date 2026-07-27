# Agent Forge · 训练驾驶舱

一个面向 Agent 训练与评估工作的可视化网页 Demo，用来集中查看训练表现、任务轨迹、错误类型和版本变化。

![Agent Forge 训练驾驶舱](public/og.png)

## 功能

- 训练总览：成功率、平均奖励、工具成功率和单任务成本
- 轨迹回放：查看任务状态、耗时、成本和失败原因
- 错误地图：汇总工具选择、重复循环、遗漏约束等常见问题
- 版本对比：比较不同 Agent 版本的能力变化
- 响应式界面：支持桌面端和移动端

当前页面使用模拟数据，适合作为 Agentic RL 评测平台、训练监控台或内部实验看板的前端原型。

## 本地运行

需要 Node.js 22.13 或更高版本。

```bash
npm install
npm run dev
```

浏览器打开终端中显示的本地地址即可。

## 构建与检查

```bash
npm run build
npm test
npm run lint
```

## 主要技术

- Next.js 16
- React 19
- TypeScript
- Tailwind CSS 4
- vinext / Vite

页面主体位于 `app/page.tsx`，样式位于 `app/globals.css`。
