"use client";

import { ChangeEvent, useEffect, useMemo, useState } from "react";
import {
  demoExperiment,
  demoExperimentSource,
  DisplayMode,
  ExperimentData,
  ExperimentMetric,
  parseExperimentData,
} from "./experiment-data";

type View = "overview" | "curves" | "evaluation" | "figures" | "guide";

const STORAGE_KEY = "agent-forge-experiment-v1";
const COLORS = {
  baseline: "#8f9bad",
  candidate: "#ff795a",
  mint: "#63dab1",
};

const navItems: { id: View; label: string; short: string }[] = [
  { id: "guide", label: "导入数据", short: "入" },
  { id: "overview", label: "结论总览", short: "总" },
  { id: "curves", label: "趋势对比", short: "线" },
  { id: "evaluation", label: "评测画像", short: "测" },
  { id: "figures", label: "定性图集", short: "图" },
];

const viewCopy: Record<View, { eyebrow: string; title: string; subtitle: string }> = {
  overview: {
    eyebrow: "EXPERIMENT DECISION",
    title: "两组实验，核心判断是什么？",
    subtitle: "把训练效果、评测收益和交互成本放在同一张桌上看。",
  },
  curves: {
    eyebrow: "TRAINING SIGNALS",
    title: "趋势对比",
    subtitle: "同一指标、同一坐标系，直接比较基线与候选。",
  },
  evaluation: {
    eyebrow: "EVALUATION PROFILE",
    title: "评测画像",
    subtitle: "同时查看效果、成本与证据强度。",
  },
  figures: {
    eyebrow: "EXPERIMENT FIGURES",
    title: "定性图集",
    subtitle: "逐项查看每组趋势及其相对关系。",
  },
  guide: {
    eyebrow: "USE YOUR OWN DATA",
    title: "导入自己的实验",
    subtitle: "下载模板、填写 JSON、在本机导入，不需要修改页面代码。",
  },
};

function formatValue(value: number, unit: string, precision: number) {
  const formatted = new Intl.NumberFormat("zh-CN", {
    minimumFractionDigits: precision,
    maximumFractionDigits: precision,
  }).format(value);
  if (!unit) return formatted;
  return unit === "%" ? `${formatted}%` : `${formatted} ${unit}`;
}

function formatDelta(metric: ExperimentMetric) {
  const difference = metric.candidate - metric.baseline;
  const prefix = difference > 0 ? "+" : "";
  return `${prefix}${formatValue(difference, metric.unit, metric.precision)}`;
}

function metricImproved(metric: ExperimentMetric) {
  return metric.higherIsBetter
    ? metric.candidate >= metric.baseline
    : metric.candidate <= metric.baseline;
}

function metricWidths(metric: ExperimentMetric) {
  const ceiling = Math.max(Math.abs(metric.baseline), Math.abs(metric.candidate), 1);
  const width = (value: number) => `${Math.max(12, (Math.abs(value) / ceiling) * 100)}%`;
  return { baseline: width(metric.baseline), candidate: width(metric.candidate) };
}

function polyline(points: number[], width: number, height: number, pad: number, min: number, max: number) {
  const innerWidth = width - pad * 2;
  const innerHeight = height - pad * 2;
  const range = Math.max(max - min, 1);
  return points
    .map((value, index) => {
      const x = pad + (index / Math.max(points.length - 1, 1)) * innerWidth;
      const y = pad + (1 - (value - min) / range) * innerHeight;
      return `${x},${y}`;
    })
    .join(" ");
}

function TrendChart({
  baseline,
  candidate,
  compact = false,
}: {
  baseline: number[];
  candidate: number[];
  compact?: boolean;
}) {
  const width = 820;
  const height = compact ? 210 : 300;
  const pad = compact ? 24 : 34;
  const values = [...baseline, ...candidate];
  const rawMin = Math.min(...values);
  const rawMax = Math.max(...values);
  const margin = Math.max((rawMax - rawMin) * 0.12, Math.abs(rawMax || 1) * 0.02, 1);
  const min = rawMin - margin;
  const max = rawMax + margin;

  return (
    <div className={compact ? "chart-wrap compact-chart" : "chart-wrap"}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="基线与候选趋势对比">
        {[0.25, 0.5, 0.75].map((ratio) => (
          <line
            key={ratio}
            x1={pad}
            x2={width - pad}
            y1={pad + (height - pad * 2) * ratio}
            y2={pad + (height - pad * 2) * ratio}
            className="grid-line"
          />
        ))}
        <text x={pad} y={height - 8} className="axis-text">开始</text>
        <text x={width / 2} y={height - 8} textAnchor="middle" className="axis-text">中段</text>
        <text x={width - pad} y={height - 8} textAnchor="end" className="axis-text">末段</text>
        <polyline
          points={polyline(baseline, width, height, pad, min, max)}
          fill="none"
          stroke={COLORS.baseline}
          strokeWidth="4"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <polyline
          points={polyline(candidate, width, height, pad, min, max)}
          fill="none"
          stroke={COLORS.candidate}
          strokeWidth="4"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </div>
  );
}

function Sidebar({
  view,
  setView,
  experiment,
  mode,
  sourceLabel,
}: {
  view: View;
  setView: (view: View) => void;
  experiment: ExperimentData;
  mode: DisplayMode;
  sourceLabel: string;
}) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark">A</span>
        <div><strong>Agent Forge</strong><small>实验分析台</small></div>
      </div>
      <nav className="main-nav" aria-label="主要导航">
        <p className="nav-label">当前实验</p>
        {navItems.map((item) => (
          <button
            key={item.id}
            className={view === item.id ? "nav-item active" : "nav-item"}
            onClick={() => setView(item.id)}
            aria-current={view === item.id ? "page" : undefined}
          >
            <span className="nav-icon">{item.short}</span>
            <span>{item.label}</span>
          </button>
        ))}
      </nav>
      <div className="side-run">
        <div className="run-status"><span /> {mode === "qualitative" ? "数值已隐藏" : "精确值仅在本机"}</div>
        <strong>{experiment.meta.name}</strong>
        <p>{experiment.trends.length} 组趋势 · {experiment.metrics.length} 个指标</p>
        <p>{sourceLabel}</p>
      </div>
      <div className="side-footer">
        <span className="avatar">DE</span>
        <div><strong>local workspace</strong><small>数据不会发送到服务器</small></div>
      </div>
    </aside>
  );
}

function Topbar({
  view,
  mode,
  setMode,
}: {
  view: View;
  mode: DisplayMode;
  setMode: (mode: DisplayMode) => void;
}) {
  const copy = viewCopy[view];
  return (
    <header className="topbar">
      <div>
        <p className="eyebrow">{copy.eyebrow}</p>
        <h1>{copy.title}</h1>
        <p>{copy.subtitle}</p>
      </div>
      <div className="top-actions">
        <span className="data-chip"><i /> 本机数据</span>
        <div className="mode-switch" aria-label="数值显示方式">
          <button className={mode === "qualitative" ? "active" : ""} onClick={() => setMode("qualitative")}>
            脱敏
          </button>
          <button className={mode === "exact" ? "active" : ""} onClick={() => setMode("exact")}>
            精确
          </button>
        </div>
      </div>
    </header>
  );
}

function ExperimentLegend({ experiment }: { experiment: ExperimentData }) {
  return (
    <div className="legend">
      <span><i style={{ background: COLORS.baseline }} /> {experiment.meta.baselineLabel}</span>
      <span><i style={{ background: COLORS.candidate }} /> {experiment.meta.candidateLabel}</span>
    </div>
  );
}

function MetricValue({ metric, mode }: { metric: ExperimentMetric; mode: DisplayMode }) {
  return (
    <>
      <strong className="qualitative-value">
        {mode === "qualitative"
          ? metric.qualitative
          : formatValue(metric.candidate, metric.unit, metric.precision)}
      </strong>
      <div className="metric-foot">
        <b className={metricImproved(metric) ? "gain" : "cost-up"}>
          {mode === "qualitative" ? metric.deltaLabel : formatDelta(metric)}
        </b>
        <small>
          {mode === "qualitative"
            ? metric.note
            : `基线 ${formatValue(metric.baseline, metric.unit, metric.precision)}`}
        </small>
      </div>
    </>
  );
}

function Overview({ experiment, mode }: { experiment: ExperimentData; mode: DisplayMode }) {
  const primaryTrend = experiment.trends[0];
  const tones = ["mint", "orange", "violet", "blue"];
  return (
    <div className="view-stack">
      <section className="verdict-banner">
        <div className="verdict-mark">≋</div>
        <div>
          <p className="eyebrow">核心结论</p>
          <h2>{experiment.verdict.title}</h2>
          <p>{experiment.verdict.description}</p>
        </div>
        <span className="verdict-pill">{experiment.verdict.status}</span>
      </section>

      <section className="metric-grid">
        {experiment.metrics.slice(0, 4).map((metric, index) => (
          <article className={`metric-card tone-${tones[index % tones.length]}`} key={metric.id}>
            <div className="metric-label"><span>{metric.label}</span><i /></div>
            <MetricValue metric={metric} mode={mode} />
          </article>
        ))}
      </section>

      <section className="overview-grid">
        <article className="panel main-chart-panel">
          <div className="panel-heading">
            <div><p className="eyebrow">TRAINING TREND</p><h3>{primaryTrend.title}</h3></div>
            <ExperimentLegend experiment={experiment} />
          </div>
          <TrendChart baseline={primaryTrend.baseline} candidate={primaryTrend.candidate} />
          <div className="chart-callout">
            <span>↗</span>
            <p><strong>{primaryTrend.relation}。</strong> {primaryTrend.note}</p>
          </div>
        </article>

        <article className="panel evidence-panel">
          <div className="panel-heading"><div><p className="eyebrow">EVIDENCE BALANCE</p><h3>证据天平</h3></div></div>
          <div className="evidence-list">
            {experiment.evidence.map((entry, index) => (
              <div className={`evidence-row ${entry.tone}`} key={`${entry.title}-${index}`}>
                <span>{String(index + 1).padStart(2, "0")}</span>
                <div><strong>{entry.title}</strong><p>{entry.note}</p></div>
                <b>{entry.tag}</b>
              </div>
            ))}
          </div>
          <div className="next-step">
            <span>下一步</span>
            <p>{experiment.verdict.nextStep}</p>
          </div>
        </article>
      </section>

      <section className="panel experiment-strip">
        <article>
          <i style={{ background: COLORS.baseline }} />
          <div><span>对照组</span><strong>{experiment.meta.baselineLabel}</strong><small>{experiment.meta.valueNote}</small></div>
        </article>
        <article>
          <i style={{ background: COLORS.candidate }} />
          <div><span>候选组</span><strong>{experiment.meta.candidateLabel}</strong><small>{experiment.meta.valueNote}</small></div>
        </article>
      </section>
    </div>
  );
}

function CurvesView({ experiment, mode }: { experiment: ExperimentData; mode: DisplayMode }) {
  const [selectedId, setSelectedId] = useState(experiment.trends[0].id);
  const selected = experiment.trends.find((item) => item.id === selectedId) ?? experiment.trends[0];
  const baselineEnd = selected.baseline.at(-1) ?? 0;
  const candidateEnd = selected.candidate.at(-1) ?? 0;

  return (
    <div className="view-stack">
      <section className="curve-tabs" aria-label="趋势选择">
        {experiment.trends.map((item) => (
          <button key={item.id} className={item.id === selected.id ? "active" : ""} onClick={() => setSelectedId(item.id)}>
            {item.label}
          </button>
        ))}
      </section>

      <section className="panel curve-panel">
        <div className="panel-heading">
          <div><p className="eyebrow">OVERLAY VIEW</p><h3>{selected.title}</h3></div>
          <ExperimentLegend experiment={experiment} />
        </div>
        <TrendChart baseline={selected.baseline} candidate={selected.candidate} />
        <div className="curve-summary qualitative-summary">
          <div>
            <span>{experiment.meta.baselineLabel}末值</span>
            <strong>{mode === "exact" ? formatValue(baselineEnd, selected.unit, selected.precision) : "参考走势"}</strong>
          </div>
          <div>
            <span>{experiment.meta.candidateLabel}末值</span>
            <strong>{mode === "exact" ? formatValue(candidateEnd, selected.unit, selected.precision) : selected.relation}</strong>
          </div>
          <div><span>显示粒度</span><strong className="orange-value">{mode === "exact" ? "精确" : "仅定性"}</strong></div>
          <p>{selected.note}</p>
        </div>
      </section>

      <section className="signal-grid">
        {experiment.trends.slice(0, 3).map((trend) => {
          const finalValue = trend.candidate.at(-1) ?? 0;
          return (
            <article className="panel" key={trend.id}>
              <p className="eyebrow">{trend.label.toUpperCase()}</p>
              <h3>{trend.relation}</h3>
              <div className="signal-number signal-text">
                {mode === "exact" ? formatValue(finalValue, trend.unit, trend.precision) : "方向已记录"}
              </div>
              <p>{trend.note}</p>
            </article>
          );
        })}
      </section>
    </div>
  );
}

function MetricBar({
  metric,
  experiment,
  mode,
}: {
  metric: ExperimentMetric;
  experiment: ExperimentData;
  mode: DisplayMode;
}) {
  const widths = metricWidths(metric);
  return (
    <div className="compare-bar">
      <div className="compare-bar-head">
        <strong>{metric.label}</strong>
        <span className={metricImproved(metric) ? "gain" : "cost-up"}>
          {mode === "exact" ? formatDelta(metric) : metric.qualitative}
        </span>
      </div>
      <div className="bar-line">
        <span>基线</span>
        <div><i style={{ width: widths.baseline, background: COLORS.baseline }} /></div>
        <b>{mode === "exact" ? formatValue(metric.baseline, metric.unit, metric.precision) : experiment.meta.baselineLabel}</b>
      </div>
      <div className="bar-line">
        <span>候选</span>
        <div><i style={{ width: widths.candidate, background: COLORS.candidate }} /></div>
        <b>{mode === "exact" ? formatValue(metric.candidate, metric.unit, metric.precision) : metric.qualitative}</b>
      </div>
    </div>
  );
}

function EvaluationView({ experiment, mode }: { experiment: ExperimentData; mode: DisplayMode }) {
  const primaryMetric = experiment.metrics[0];
  const positive = experiment.evidence.filter((item) => item.tone === "positive").length;
  const neutral = experiment.evidence.filter((item) => item.tone === "neutral").length;
  const negative = experiment.evidence.filter((item) => item.tone === "negative").length;

  return (
    <div className="view-stack">
      <section className="eval-hero">
        <div>
          <p className="eyebrow">PAIRED EXPERIMENT VIEW</p>
          <h2>{experiment.verdict.title}</h2>
          <p>{experiment.verdict.description}</p>
        </div>
        <div className="eval-score">
          <span>{primaryMetric.label}</span>
          <strong className="textual-score">
            {mode === "exact"
              ? formatValue(primaryMetric.candidate, primaryMetric.unit, primaryMetric.precision)
              : primaryMetric.qualitative}
          </strong>
          <small>{experiment.verdict.status}</small>
        </div>
      </section>

      <section className="eval-grid">
        <article className="panel eval-bars-panel">
          <div className="panel-heading">
            <div><p className="eyebrow">QUALITY & COST</p><h3>效果与交互成本</h3></div>
            <ExperimentLegend experiment={experiment} />
          </div>
          <div className="compare-bars">
            {experiment.metrics.map((metric) => (
              <MetricBar key={metric.id} metric={metric} experiment={experiment} mode={mode} />
            ))}
          </div>
        </article>

        <article className="panel task-balance">
          <div className="panel-heading"><div><p className="eyebrow">EVIDENCE VIEW</p><h3>证据分布</h3></div></div>
          <div className="balance-rail qualitative-balance">
            <div className="win-block"><strong>{mode === "exact" ? positive : "若干"}</strong><span>正向证据</span></div>
            <div className="tie-block"><strong>{mode === "exact" ? neutral : "保留"}</strong><span>中性证据</span></div>
            <div className="loss-block"><strong>{mode === "exact" ? negative : "若干"}</strong><span>成本证据</span></div>
          </div>
          <div className="confidence-card">
            <span>判断提醒</span><h4>{experiment.verdict.status}</h4>
            <div className="confidence-line qualitative-confidence"><i className="zero" /><i className="range" /></div>
            <div className="confidence-labels"><span>可能回退</span><b>中性</b><span>可能提升</span></div>
            <p>{experiment.verdict.nextStep}</p>
          </div>
        </article>
      </section>

      <section className="panel eval-table-panel">
        <div className="panel-heading">
          <div><p className="eyebrow">METRIC SUMMARY</p><h3>实验指标摘要</h3></div>
          <span className="source-chip">{mode === "exact" ? "local exact values" : "qualitative view"}</span>
        </div>
        <div className="eval-table">
          <div className="eval-row head"><span>指标</span><span>{experiment.meta.baselineLabel}</span><span>{experiment.meta.candidateLabel}</span><span>核心结论</span></div>
          {experiment.metrics.map((metric) => (
            <div className="eval-row" key={metric.id}>
              <strong>{metric.label}</strong>
              <span>{mode === "exact" ? formatValue(metric.baseline, metric.unit, metric.precision) : "参考"}</span>
              <span>{mode === "exact" ? formatValue(metric.candidate, metric.unit, metric.precision) : metric.qualitative}</span>
              <b className={metricImproved(metric) ? "gain" : "cost-up"}>{metric.deltaLabel}</b>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function FiguresView({ experiment, mode }: { experiment: ExperimentData; mode: DisplayMode }) {
  const [selectedId, setSelectedId] = useState(experiment.trends[0].id);
  const selected = useMemo(
    () => experiment.trends.find((item) => item.id === selectedId) ?? experiment.trends[0],
    [experiment, selectedId],
  );

  return (
    <div className="view-stack">
      <section className="figure-tabs" aria-label="图集分类">
        {experiment.trends.map((item) => (
          <button key={item.id} className={selected.id === item.id ? "active" : ""} onClick={() => setSelectedId(item.id)}>
            {item.label}
          </button>
        ))}
      </section>

      <section className="figure-grid">
        <article className="panel figure-card qualitative-figure">
          <div className="figure-card-head">
            <div><span className="baseline-dot" />{experiment.meta.baselineLabel}</div>
            <strong>{mode === "exact" ? "LOCAL DATA" : "SCHEMATIC"}</strong>
          </div>
          <div className="qualitative-plot">
            <p className="eyebrow">{selected.title}</p>
            <TrendChart baseline={selected.baseline} candidate={selected.baseline} compact />
            <div className="plot-label">
              <span>末段值</span>
              <strong>
                {mode === "exact"
                  ? formatValue(selected.baseline.at(-1) ?? 0, selected.unit, selected.precision)
                  : "参考关系"}
              </strong>
            </div>
          </div>
          <p>{mode === "exact" ? "数值来自本机导入文件。" : "当前隐藏坐标值与运行标识。"}</p>
        </article>

        <article className="panel figure-card qualitative-figure candidate-figure">
          <div className="figure-card-head">
            <div><span className="candidate-dot" />{experiment.meta.candidateLabel}</div>
            <strong>{mode === "exact" ? "LOCAL DATA" : "SCHEMATIC"}</strong>
          </div>
          <div className="qualitative-plot">
            <p className="eyebrow">{selected.title}</p>
            <TrendChart baseline={selected.candidate} candidate={selected.candidate} compact />
            <div className="plot-label">
              <span>相对基线</span>
              <strong>
                {mode === "exact"
                  ? formatValue(selected.candidate.at(-1) ?? 0, selected.unit, selected.precision)
                  : selected.relation}
              </strong>
            </div>
          </div>
          <p>{selected.note}</p>
        </article>
      </section>

      <section className="figure-note">
        <span>显示说明</span>
        <p>
          {mode === "exact"
            ? "当前显示本机导入的精确值；分享截图前建议切换到脱敏模式。"
            : "当前只表达趋势关系；切换到精确模式可查看本机数据值。"}
        </p>
      </section>
    </div>
  );
}

function GuideView({
  experiment,
  sourceLabel,
  onImport,
  onReset,
  setView,
}: {
  experiment: ExperimentData;
  sourceLabel: string;
  onImport: (file: File) => Promise<void>;
  onReset: () => void;
  setView: (view: View) => void;
}) {
  const [message, setMessage] = useState("请选择符合模板格式的 JSON 文件。");
  const [status, setStatus] = useState<"idle" | "success" | "error">("idle");

  const handleFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      await onImport(file);
      setMessage(`已载入 ${file.name}，看板已更新。`);
      setStatus("success");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "无法读取这个文件");
      setStatus("error");
    } finally {
      event.target.value = "";
    }
  };

  const downloadTemplate = () => {
    const blob = new Blob([`${JSON.stringify(demoExperimentSource, null, 2)}\n`], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "experiment.example.json";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const reset = () => {
    onReset();
    setMessage("已恢复仓库自带的公开示例。");
    setStatus("success");
  };

  return (
    <div className="view-stack guide-view">
      <section className="guide-hero">
        <div>
          <p className="eyebrow">ZERO-CODE IMPORT</p>
          <h2>准备一个 JSON，就能比较自己的两组实验。</h2>
          <p>文件在浏览器本机读取并保存，不会发送到服务器。导入后所有页面会立即使用你的指标和曲线。</p>
        </div>
        <div className="privacy-seal"><span>LOCAL</span><strong>本机处理</strong><small>不上传数据</small></div>
      </section>

      <section className="guide-steps">
        {[
          ["01", "下载模板", "保留字段名，把示例内容替换成你的实验信息。"],
          ["02", "填写数据", "加入汇总指标、趋势数组、结论和证据。"],
          ["03", "本机导入", "选择 JSON 后即可查看总览、曲线和评测画像。"],
        ].map(([index, title, note]) => (
          <article className="panel guide-step" key={index}>
            <span>{index}</span><div><h3>{title}</h3><p>{note}</p></div>
          </article>
        ))}
      </section>

      <section className="guide-grid">
        <article className="panel import-panel">
          <div className="panel-heading">
            <div><p className="eyebrow">IMPORT JSON</p><h3>导入实验文件</h3></div>
            <span className="source-chip">{sourceLabel}</span>
          </div>
          <label className="file-drop">
            <input type="file" accept=".json,application/json" onChange={handleFile} />
            <span className="file-drop-icon">＋</span>
            <strong>选择本地 JSON</strong>
            <small>最大 1 MB；数据只保存在当前浏览器</small>
          </label>
          <div className={`import-message ${status}`}>
            <span>{status === "error" ? "!" : status === "success" ? "✓" : "i"}</span>
            <p>{message}</p>
          </div>
          <div className="guide-actions">
            <button className="primary-action" onClick={downloadTemplate}>下载 JSON 模板</button>
            <button className="secondary-action" onClick={() => setView("overview")}>查看当前看板</button>
            <button className="secondary-action" onClick={reset}>恢复公开示例</button>
          </div>
        </article>

        <article className="panel schema-panel">
          <div className="panel-heading"><div><p className="eyebrow">DATA SCHEMA</p><h3>需要填写什么？</h3></div></div>
          <div className="schema-list">
            <div><code>meta</code><span>实验名称、两组标签、默认显示模式</span></div>
            <div><code>verdict</code><span>一句话结论、说明、状态与下一步</span></div>
            <div><code>metrics</code><span>成功率、工具调用、轮数、耗时等汇总值</span></div>
            <div><code>trends</code><span>Reward、Advantage、熵等同长度序列</span></div>
            <div><code>evidence</code><span>支持收益、成本或中性判断的证据</span></div>
          </div>
          <div className="schema-status">
            <span>当前数据</span>
            <strong>{experiment.metrics.length} 个指标 · {experiment.trends.length} 组趋势</strong>
            <small>{experiment.meta.valueNote}</small>
          </div>
        </article>
      </section>

      <section className="panel command-panel">
        <div><p className="eyebrow">OPTIONAL VALIDATION</p><h3>也可以先在命令行校验</h3></div>
        <code>npm run data:validate -- /你的路径/experiment.json</code>
        <p>完整字段说明见 <strong>docs/USAGE.md</strong>。</p>
      </section>
    </div>
  );
}

export default function Home() {
  const [view, setView] = useState<View>("overview");
  const [experiment, setExperiment] = useState<ExperimentData>(demoExperiment);
  const [mode, setMode] = useState<DisplayMode>(demoExperiment.meta.defaultMode);
  const [sourceLabel, setSourceLabel] = useState("仓库公开示例");

  useEffect(() => {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (!stored) return;
    try {
      const parsed = parseExperimentData(JSON.parse(stored));
      queueMicrotask(() => {
        setExperiment(parsed);
        setMode(parsed.meta.defaultMode);
        setSourceLabel("浏览器本机数据");
      });
    } catch {
      window.localStorage.removeItem(STORAGE_KEY);
    }
  }, []);

  const importFile = async (file: File) => {
    if (file.size > 1024 * 1024) {
      throw new Error("文件超过 1 MB，请先精简曲线点数或字段内容。");
    }
    const parsed = parseExperimentData(JSON.parse(await file.text()));
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(parsed));
    setExperiment(parsed);
    setMode(parsed.meta.defaultMode);
    setSourceLabel(`本机文件：${file.name}`);
  };

  const reset = () => {
    window.localStorage.removeItem(STORAGE_KEY);
    setExperiment(demoExperiment);
    setMode(demoExperiment.meta.defaultMode);
    setSourceLabel("仓库公开示例");
  };

  return (
    <main className="app-shell">
      <Sidebar
        view={view}
        setView={setView}
        experiment={experiment}
        mode={mode}
        sourceLabel={sourceLabel}
      />
      <div className="content-shell">
        <Topbar view={view} mode={mode} setMode={setMode} />
        <div className="content">
          {view === "overview" && <Overview experiment={experiment} mode={mode} />}
          {view === "curves" && <CurvesView experiment={experiment} mode={mode} />}
          {view === "evaluation" && <EvaluationView experiment={experiment} mode={mode} />}
          {view === "figures" && <FiguresView experiment={experiment} mode={mode} />}
          {view === "guide" && (
            <GuideView
              experiment={experiment}
              sourceLabel={sourceLabel}
              onImport={importFile}
              onReset={reset}
              setView={setView}
            />
          )}
        </div>
      </div>
    </main>
  );
}
