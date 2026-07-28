"use client";

import { useMemo, useState } from "react";

type View = "overview" | "curves" | "evaluation" | "figures";
type TrendKey = "score" | "smooth" | "signal" | "entropy" | "time";

const COLORS = {
  baseline: "#8f9bad",
  candidate: "#ff795a",
  mint: "#63dab1",
};

const navItems: { id: View; label: string; short: string }[] = [
  { id: "overview", label: "结论总览", short: "总" },
  { id: "curves", label: "趋势对比", short: "线" },
  { id: "evaluation", label: "评测画像", short: "测" },
  { id: "figures", label: "定性图集", short: "图" },
];

const trendOptions: {
  id: TrendKey;
  label: string;
  title: string;
  baseline: number[];
  candidate: number[];
  relation: string;
  note: string;
}[] = [
  {
    id: "score",
    label: "训练表现",
    title: "整体表现趋势",
    baseline: [42, 44, 45, 41, 46, 52, 48, 55, 53, 58, 57, 63],
    candidate: [43, 46, 47, 42, 50, 58, 51, 60, 57, 65, 62, 69],
    relation: "候选末段略高",
    note: "保留“候选方案后段略有改善、过程仍有波动”这一关系，不展示真实训练节点或指标值。",
  },
  {
    id: "smooth",
    label: "连续反馈",
    title: "连续反馈趋势",
    baseline: [48, 50, 54, 53, 58, 59, 61, 64, 66, 67, 70, 72],
    candidate: [46, 50, 52, 55, 57, 60, 60, 63, 65, 66, 69, 71],
    relation: "两组基本接近",
    note: "公开版只表达两组走势相近，不保留奖励、比例或训练阶段的具体数值。",
  },
  {
    id: "signal",
    label: "加权信号",
    title: "加权训练信号",
    baseline: [49, 52, 48, 55, 53, 58, 55, 60, 58, 61, 60, 63],
    candidate: [58, 61, 57, 64, 62, 67, 64, 69, 66, 71, 69, 73],
    relation: "候选信号更强",
    note: "仅保留“候选加权通道更强”的核心判断，不展示 Advantage、权重或幅度。",
  },
  {
    id: "entropy",
    label: "策略探索",
    title: "策略探索趋势",
    baseline: [70, 68, 67, 65, 63, 61, 60, 58, 57, 55, 54, 53],
    candidate: [72, 69, 67, 64, 62, 59, 57, 55, 53, 51, 49, 47],
    relation: "候选收敛更快",
    note: "示意图表达候选策略更快变得确定，同时保留“需关注探索是否过早收缩”的风险。",
  },
  {
    id: "time",
    label: "训练开销",
    title: "相对计算开销",
    baseline: [49, 51, 50, 52, 49, 51, 50, 52, 51, 50, 52, 51],
    candidate: [55, 57, 54, 58, 55, 57, 56, 59, 57, 56, 58, 57],
    relation: "候选开销略高",
    note: "只表达候选方案需要更多计算，不保留秒数、阶段编号或相对百分比。",
  },
];

const figureGroups = [
  { id: "quality", label: "效果趋势", trend: "score" as TrendKey },
  { id: "feedback", label: "连续反馈", trend: "smooth" as TrendKey },
  { id: "signal", label: "加权信号", trend: "signal" as TrendKey },
  { id: "exploration", label: "策略探索", trend: "entropy" as TrendKey },
  { id: "compute", label: "计算开销", trend: "time" as TrendKey },
];

function polyline(points: number[], width: number, height: number, pad = 18) {
  const innerWidth = width - pad * 2;
  const innerHeight = height - pad * 2;
  const min = 30;
  const max = 80;
  return points
    .map((value, index) => {
      const x = pad + (index / Math.max(points.length - 1, 1)) * innerWidth;
      const y = pad + (1 - (value - min) / (max - min)) * innerHeight;
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
  return (
    <div className={compact ? "chart-wrap compact-chart" : "chart-wrap"}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="脱敏后的两组定性趋势对比">
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
        <text x={pad} y={height - 8} className="axis-text">早期</text>
        <text x={width / 2} y={height - 8} textAnchor="middle" className="axis-text">中期</text>
        <text x={width - pad} y={height - 8} textAnchor="end" className="axis-text">后期</text>
        <polyline
          points={polyline(baseline, width, height, pad)}
          fill="none"
          stroke={COLORS.baseline}
          strokeWidth="4"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <polyline
          points={polyline(candidate, width, height, pad)}
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

function Sidebar({ view, setView }: { view: View; setView: (view: View) => void }) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark">A</span>
        <div><strong>Agent Forge</strong><small>定性分析 Demo</small></div>
      </div>
      <nav className="main-nav" aria-label="主要导航">
        <p className="nav-label">公开演示</p>
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
        <div className="run-status"><span /> 数值已隐藏</div>
        <strong>qualitative demo</strong>
        <p>两组匿名对照 · 聚合趋势</p>
        <p>不含原始指标与轨迹</p>
      </div>
      <div className="side-footer">
        <span className="avatar">DE</span>
        <div><strong>public-agent-demo</strong><small>公开演示工作区</small></div>
      </div>
    </aside>
  );
}

const viewCopy: Record<View, { eyebrow: string; title: string; subtitle: string }> = {
  overview: {
    eyebrow: "QUALITATIVE DECISION",
    title: "两组实验，核心判断是什么？",
    subtitle: "只保留方向和权衡，不公开训练节点、指标或调用次数。",
  },
  curves: {
    eyebrow: "ANONYMIZED TRENDS",
    title: "趋势对比",
    subtitle: "所有曲线均为示意形状，只表达相对关系。",
  },
  evaluation: {
    eyebrow: "QUALITATIVE EVALUATION",
    title: "评测画像",
    subtitle: "隐藏样本规模与具体值，只展示收益、成本和不确定性。",
  },
  figures: {
    eyebrow: "SCHEMATIC FIGURES",
    title: "定性图集",
    subtitle: "用示意图替代带坐标和数值的原始图。",
  },
};

function Topbar({ view }: { view: View }) {
  const copy = viewCopy[view];
  return (
    <header className="topbar">
      <div>
        <p className="eyebrow">{copy.eyebrow}</p>
        <h1>{copy.title}</h1>
        <p>{copy.subtitle}</p>
      </div>
      <div className="top-actions">
        <span className="data-chip"><i /> 定性演示</span>
        <span className="date-chip">NO RAW VALUES</span>
      </div>
    </header>
  );
}

function ExperimentLegend() {
  return (
    <div className="legend">
      <span><i style={{ background: COLORS.baseline }} /> 匿名基线</span>
      <span><i style={{ background: COLORS.candidate }} /> 匿名候选</span>
    </div>
  );
}

function Overview() {
  const score = trendOptions[0];
  return (
    <div className="view-stack">
      <section className="verdict-banner">
        <div className="verdict-mark">≋</div>
        <div>
          <p className="eyebrow">核心结论</p>
          <h2>候选方案略有收益，但额外开销更高，当前仍不足以证明“更划算”。</h2>
          <p>公开版只保留这一决策关系。具体训练阶段、奖励、加权信号、样本规模、工具调用和耗时均已隐藏。</p>
        </div>
        <span className="verdict-pill">继续验证</span>
      </section>

      <section className="metric-grid">
        {[
          { label: "离线评测", value: "略有改善", delta: "方向为正", note: "不公开样本量与得分", tone: "mint" },
          { label: "训练末段", value: "候选略高", delta: "存在波动", note: "不公开节点与奖励", tone: "orange" },
          { label: "工具调用", value: "明显增加", delta: "成本上升", note: "不公开次数与比例", tone: "violet", negative: true },
          { label: "交互长度", value: "有所增加", delta: "成本上升", note: "不公开轮数与比例", tone: "blue", negative: true },
        ].map((metric) => (
          <article className={`metric-card tone-${metric.tone}`} key={metric.label}>
            <div className="metric-label"><span>{metric.label}</span><i /></div>
            <strong className="qualitative-value">{metric.value}</strong>
            <div className="metric-foot">
              <b className={metric.negative ? "cost-up" : "gain"}>{metric.delta}</b>
              <small>{metric.note}</small>
            </div>
          </article>
        ))}
      </section>

      <section className="overview-grid">
        <article className="panel main-chart-panel">
          <div className="panel-heading">
            <div><p className="eyebrow">TRAINING TREND</p><h3>整体表现走势</h3></div>
            <ExperimentLegend />
          </div>
          <TrendChart baseline={score.baseline} candidate={score.candidate} />
          <div className="chart-callout">
            <span>↗</span>
            <p><strong>候选末段略高，但过程仍有波动。</strong> 曲线只表达关系，不映射真实节点、刻度或指标值。</p>
          </div>
        </article>

        <article className="panel evidence-panel">
          <div className="panel-heading"><div><p className="eyebrow">EVIDENCE BALANCE</p><h3>证据天平</h3></div></div>
          <div className="evidence-list">
            {[
              ["01", "评测方向小幅改善", "尚需重复验证", "收益", "positive"],
              ["02", "训练末段略高", "过程中存在波动", "收益", "positive"],
              ["03", "工具使用明显增加", "交互成本更高", "成本", "negative"],
              ["04", "训练与推理开销增加", "效率存在回退", "成本", "negative"],
            ].map(([index, title, note, tag, tone]) => (
              <div className={`evidence-row ${tone}`} key={index}>
                <span>{index}</span>
                <div><strong>{title}</strong><p>{note}</p></div>
                <b>{tag}</b>
              </div>
            ))}
          </div>
          <div className="next-step">
            <span>下一步</span>
            <p>使用相同配置做多次独立复现；只有收益方向稳定，才值得继续优化工具调用和计算开销。</p>
          </div>
        </article>
      </section>

      <section className="panel experiment-strip">
        <article><i style={{ background: COLORS.baseline }} /><div><span>对照组</span><strong>匿名基线</strong><small>隐藏运行标识与配置细节</small></div></article>
        <article><i style={{ background: COLORS.candidate }} /><div><span>候选组</span><strong>匿名候选</strong><small>隐藏运行标识与配置细节</small></div></article>
      </section>
    </div>
  );
}

function CurvesView() {
  const [selectedId, setSelectedId] = useState<TrendKey>("score");
  const selected = trendOptions.find((item) => item.id === selectedId) ?? trendOptions[0];
  return (
    <div className="view-stack">
      <section className="curve-tabs" aria-label="趋势选择">
        {trendOptions.map((item) => (
          <button key={item.id} className={item.id === selectedId ? "active" : ""} onClick={() => setSelectedId(item.id)}>
            {item.label}
          </button>
        ))}
      </section>

      <section className="panel curve-panel">
        <div className="panel-heading">
          <div><p className="eyebrow">SCHEMATIC OVERLAY</p><h3>{selected.title}</h3></div>
          <ExperimentLegend />
        </div>
        <TrendChart baseline={selected.baseline} candidate={selected.candidate} />
        <div className="curve-summary qualitative-summary">
          <div><span>匿名基线</span><strong>参考走势</strong></div>
          <div><span>匿名候选</span><strong>{selected.relation}</strong></div>
          <div><span>公开粒度</span><strong className="orange-value">仅定性</strong></div>
          <p>{selected.note}</p>
        </div>
      </section>

      <section className="signal-grid">
        <article className="panel">
          <p className="eyebrow">WEIGHTED SIGNAL</p><h3>候选加权通道更强</h3>
          <div className="signal-number signal-text">方向增强</div>
          <p>保留加权机制已经生效这一判断；具体优势值、权重和分量均不公开。</p>
        </article>
        <article className="panel">
          <p className="eyebrow">STABILITY</p><h3>未见明显异常模式</h3>
          <div className="signal-number mint-value signal-text">整体稳定</div>
          <p>只说明未观察到截断或重复风险，不公开计数、比例或逐条记录。</p>
        </article>
        <article className="panel">
          <p className="eyebrow">COMPUTE</p><h3>候选训练开销略高</h3>
          <div className="signal-number signal-text">成本增加</div>
          <p>保留成本方向，不公开每阶段耗时、相对比例或硬件效率细节。</p>
        </article>
      </section>
    </div>
  );
}

function QualitativeBar({
  label,
  baselineWidth,
  candidateWidth,
  candidateLabel,
  lowerIsBetter = false,
}: {
  label: string;
  baselineWidth: string;
  candidateWidth: string;
  candidateLabel: string;
  lowerIsBetter?: boolean;
}) {
  return (
    <div className="compare-bar">
      <div className="compare-bar-head">
        <strong>{label}</strong>
        <span className={lowerIsBetter ? "cost-up" : "gain"}>{candidateLabel}</span>
      </div>
      <div className="bar-line">
        <span>基线</span><div><i style={{ width: baselineWidth, background: COLORS.baseline }} /></div><b>参考</b>
      </div>
      <div className="bar-line">
        <span>候选</span><div><i style={{ width: candidateWidth, background: COLORS.candidate }} /></div><b>{candidateLabel}</b>
      </div>
    </div>
  );
}

function EvaluationView() {
  return (
    <div className="view-stack">
      <section className="eval-hero">
        <div>
          <p className="eyebrow">ANONYMIZED TASK VIEW</p>
          <h2>候选评测方向略好，但差异仍落在不确定区间内。</h2>
          <p>公开版隐藏任务数、通过数、均值和区间边界，只保留“收益小、成本高、证据不足”的判断。</p>
        </div>
        <div className="eval-score"><span>评测方向</span><strong className="textual-score">小幅向好</strong><small>仍需独立复现</small></div>
      </section>

      <section className="eval-grid">
        <article className="panel eval-bars-panel">
          <div className="panel-heading"><div><p className="eyebrow">QUALITY & COST</p><h3>效果与交互成本</h3></div><ExperimentLegend /></div>
          <div className="compare-bars">
            <QualitativeBar label="评测表现" baselineWidth="62%" candidateWidth="67%" candidateLabel="略高" />
            <QualitativeBar label="平均工具调用" baselineWidth="58%" candidateWidth="79%" candidateLabel="明显增加" lowerIsBetter />
            <QualitativeBar label="平均交互长度" baselineWidth="61%" candidateWidth="76%" candidateLabel="有所增加" lowerIsBetter />
            <QualitativeBar label="用户模拟负担" baselineWidth="70%" candidateWidth="64%" candidateLabel="略低" />
          </div>
        </article>

        <article className="panel task-balance">
          <div className="panel-heading"><div><p className="eyebrow">TASK-LEVEL VIEW</p><h3>逐任务关系</h3></div></div>
          <div className="balance-rail qualitative-balance">
            <div className="win-block"><strong>略多</strong><span>候选更好</span></div>
            <div className="tie-block"><strong>接近</strong><span>表现持平</span></div>
            <div className="loss-block"><strong>接近</strong><span>候选更差</span></div>
          </div>
          <div className="confidence-card">
            <span>统计提醒</span><h4>不确定区间跨过中性线</h4>
            <div className="confidence-line qualitative-confidence"><i className="zero" /><i className="range" /></div>
            <div className="confidence-labels"><span>可能回退</span><b>中性</b><span>可能提升</span></div>
            <p>单次小幅改善仍可能来自随机波动。公开版不展示区间端点或样本规模。</p>
          </div>
        </article>
      </section>

      <section className="panel eval-table-panel">
        <div className="panel-heading"><div><p className="eyebrow">QUALITATIVE SUMMARY</p><h3>脱敏评测摘要</h3></div><span className="source-chip">no raw values</span></div>
        <div className="eval-table">
          <div className="eval-row head"><span>指标</span><span>匿名基线</span><span>匿名候选</span><span>核心结论</span></div>
          <div className="eval-row"><strong>评测表现</strong><span>参考</span><span>略高</span><b className="gain">小幅收益</b></div>
          <div className="eval-row"><strong>工具使用</strong><span>参考</span><span>明显更多</span><b className="cost-up">成本增加</b></div>
          <div className="eval-row"><strong>交互长度</strong><span>参考</span><span>有所增加</span><b className="cost-up">效率回退</b></div>
          <div className="eval-row"><strong>证据强度</strong><span>单次实验</span><span>单次实验</span><b>不足以下结论</b></div>
        </div>
      </section>
    </div>
  );
}

function FiguresView() {
  const [selectedId, setSelectedId] = useState("quality");
  const selectedGroup = useMemo(
    () => figureGroups.find((item) => item.id === selectedId) ?? figureGroups[0],
    [selectedId],
  );
  const selectedTrend = trendOptions.find((item) => item.id === selectedGroup.trend) ?? trendOptions[0];
  return (
    <div className="view-stack">
      <section className="figure-tabs" aria-label="图集分类">
        {figureGroups.map((item) => (
          <button key={item.id} className={selectedId === item.id ? "active" : ""} onClick={() => setSelectedId(item.id)}>
            {item.label}
          </button>
        ))}
      </section>

      <section className="figure-grid">
        <article className="panel figure-card qualitative-figure">
          <div className="figure-card-head"><div><span className="baseline-dot" />匿名基线</div><strong>SCHEMATIC</strong></div>
          <div className="qualitative-plot">
            <p className="eyebrow">{selectedTrend.title}</p>
            <TrendChart baseline={selectedTrend.baseline} candidate={selectedTrend.baseline} compact />
            <div className="plot-label"><span>仅显示走势</span><strong>参考关系</strong></div>
          </div>
          <p>无坐标值、无训练节点、无运行标识。</p>
        </article>
        <article className="panel figure-card qualitative-figure candidate-figure">
          <div className="figure-card-head"><div><span className="candidate-dot" />匿名候选</div><strong>SCHEMATIC</strong></div>
          <div className="qualitative-plot">
            <p className="eyebrow">{selectedTrend.title}</p>
            <TrendChart baseline={selectedTrend.candidate} candidate={selectedTrend.candidate} compact />
            <div className="plot-label"><span>相对基线</span><strong>{selectedTrend.relation}</strong></div>
          </div>
          <p>无坐标值、无训练节点、无运行标识。</p>
        </article>
      </section>

      <section className="figure-note">
        <span>隐私说明</span>
        <p>仓库不再包含指标 JSON、带数值的训练图、真实或扰动后的精确值。这里的曲线是人工示意形状，只用于表达核心结论。</p>
      </section>
    </div>
  );
}

export default function Home() {
  const [view, setView] = useState<View>("overview");
  return (
    <main className="app-shell">
      <Sidebar view={view} setView={setView} />
      <div className="content-shell">
        <Topbar view={view} />
        <div className="content">
          {view === "overview" && <Overview />}
          {view === "curves" && <CurvesView />}
          {view === "evaluation" && <EvaluationView />}
          {view === "figures" && <FiguresView />}
        </div>
      </div>
    </main>
  );
}
