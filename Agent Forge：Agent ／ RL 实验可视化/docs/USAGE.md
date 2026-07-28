# 如何导入自己的实验数据

这个看板支持比较一组基线实验和一组候选实验。最简单的用法是在浏览器中导入 JSON，不需要修改页面代码。

## 1. 启动项目

需要 Node.js `>=22.13.0`。

```bash
npm install
npm run dev
```

打开 <http://localhost:3000>，进入左侧的“导入数据”页面。

## 2. 下载并填写模板

在“导入数据”页面点击“下载 JSON 模板”，或者直接复制：

```text
data/experiment.example.json
```

模板的核心结构如下：

```json
{
  "version": 1,
  "meta": {
    "name": "my-experiment",
    "baselineLabel": "baseline",
    "candidateLabel": "candidate",
    "defaultMode": "exact",
    "valueNote": "仅在本机查看"
  },
  "verdict": {
    "title": "一句话结论",
    "description": "说明收益、成本和不确定性",
    "status": "继续验证",
    "nextStep": "下一步实验建议"
  },
  "metrics": [
    {
      "id": "success-rate",
      "label": "成功率",
      "baseline": 40.0,
      "candidate": 42.0,
      "unit": "%",
      "precision": 1,
      "higherIsBetter": true,
      "qualitative": "略有改善",
      "deltaLabel": "方向为正",
      "note": "候选组略高"
    }
  ],
  "trends": [
    {
      "id": "reward",
      "label": "Reward",
      "title": "Reward 趋势",
      "unit": "",
      "precision": 3,
      "baseline": [0.1, 0.2, 0.3],
      "candidate": [0.1, 0.25, 0.35],
      "relation": "候选末段略高",
      "note": "过程仍有波动"
    }
  ],
  "evidence": [
    {
      "title": "评测方向改善",
      "note": "仍需重复实验",
      "tag": "收益",
      "tone": "positive"
    }
  ]
}
```

### 字段说明

| 区域 | 用途 | 重要规则 |
| --- | --- | --- |
| `meta` | 实验名称、两组标签、默认显示模式 | `defaultMode` 只能是 `qualitative` 或 `exact` |
| `verdict` | 首页结论和下一步建议 | 使用可直接汇报的一句话 |
| `metrics` | 成功率、工具调用、轮数、耗时等汇总指标 | 最多 8 项；数值应已经转换成希望展示的单位 |
| `trends` | Reward、Advantage、熵、训练耗时等序列 | 最多 8 组；两条数组长度必须一致 |
| `evidence` | 支持或反对候选方案的证据 | `tone` 可用 `positive`、`negative`、`neutral` |

`higherIsBetter` 表示指标越大是否越好。例如成功率填 `true`，工具调用和耗时通常填 `false`。

## 3. 导入并查看

在“导入数据”页面选择填写好的 JSON。校验成功后：

- 结论总览会展示汇总指标和证据；
- 趋势对比会为每个 `trends` 项生成一组曲线；
- 评测画像会比较基线与候选的指标；
- 定性图集会把每组趋势拆开查看；
- 右上角可以在“脱敏”和“精确”之间切换。

导入的数据只保存在当前浏览器的本地存储中，不会发送到服务器。点击“恢复公开示例”可以清除。

## 4. 导入前校验

```bash
npm run data:validate -- /你的路径/experiment.json
```

出现“数据格式正确”后再导入。常见错误包括：

- 基线与候选曲线长度不同；
- 数组中存在字符串、`null`、`NaN` 或无穷值；
- `id` 重复；
- `precision` 不在 0 到 6 之间；
- 指标、趋势或证据为空。

## 隐私提醒

- 浏览器导入不会上传文件，但浏览器会在本机保存一份，适合个人分析。
- 不要把包含真实实验 ID、内部路径、用户提示词或轨迹文本的 JSON 提交到公开仓库。
- 需要截图或分享时，先切换到“脱敏”模式。
- 如果要修改仓库自带示例，请只使用合成或归一化数据。
