"use client";

import { useMemo, useState } from "react";

type View = "overview" | "traces" | "errors" | "compare";
type TraceStatus = "成功" | "失败" | "需复核";

type Trace = {
  id: string;
  task: string;
  category: string;
  status: TraceStatus;
  reward: number;
  duration: string;
  cost: string;
  tools: number;
  note: string;
};

const traces: Trace[] = [
  {
    id: "TR-2841",
    task: "整理竞品功能并输出对比表",
    category: "深度研究",
    status: "成功",
    reward: 0.92,
    duration: "1m 42s",
    cost: "¥0.38",
    tools: 8,
    note: "完整覆盖 6 个目标产品，所有关键结论均有来源。",
  },
  {
    id: "TR-2839",
    task: "查找订单异常并生成退款建议",
    category: "客服运营",
    status: "失败",
    reward: 0.31,
    duration: "2m 18s",
    cost: "¥0.61",
    tools: 12,
    note: "重复调用订单查询工具，未在第三次失败后切换策略。",
  },
  {
    id: "TR-2837",
    task: "分析实验日志并定位性能回退",
    category: "代码分析",
    status: "需复核",
    reward: 0.66,
    duration: "3m 05s",
    cost: "¥0.74",
    tools: 15,
    note: "定位方向正确，但结论依赖一个尚未验证的假设。",
  },
  {
    id: "TR-2834",
    task: "从会议记录中生成行动清单",
    category: "办公协作",
    status: "成功",
    reward: 0.88,
    duration: "38s",
    cost: "¥0.12",
    tools: 2,
    note: "负责人、截止日期和依赖关系抽取准确。",
  },
  {
    id: "TR-2831",
    task: "根据用户画像制定七日触达计划",
    category: "营销策划",
    status: "失败",
    reward: 0.44,
    duration: "1m 56s",
    cost: "¥0.47",
    tools: 7,
    note: "计划完整，但忽略了用户明确提出的触达频率限制。",
  },
];

const navItems: { id: View; label: string; short: string; badge?: string }[] = [
  { id: "overview", label: "训练总览", short: "总" },
  { id: "traces", label: "轨迹回放", short: "轨", badge: "3" },
  { id: "errors", label: "错误地图", short: "错", badge: "12" },
  { id: "compare", label: "版本对比", short: "比" },
];

const metricCards = [
  { label: "任务成功率", value: "78.4%", delta: "+6.8%", hint: "较 v2.7", tone: "mint" },
  { label: "平均奖励", value: "0.72", delta: "+0.09", hint: "近 1,248 条轨迹", tone: "orange" },
  { label: "工具成功率", value: "91.3%", delta: "+2.1%", hint: "仍是主要瓶颈", tone: "blue" },
  { label: "单任务成本", value: "¥0.42", delta: "-14.2%", hint: "平均节省 ¥0.07", tone: "violet" },
];

const scoreRows = [
  { label: "任务理解", value: 88, delta: "+4" },
  { label: "计划质量", value: 82, delta: "+7" },
  { label: "工具选择", value: 69, delta: "+3" },
  { label: "错误恢复", value: 61, delta: "+11" },
  { label: "结果可信度", value: 85, delta: "+5" },
];

const errorTypes = [
  { label: "工具选择错误", count: 38, pct: 100, trend: "-8%" },
  { label: "重复循环", count: 27, pct: 71, trend: "-21%" },
  { label: "遗漏约束", count: 22, pct: 58, trend: "+4%" },
  { label: "提前结束", count: 17, pct: 45, trend: "-6%" },
  { label: "事实无依据", count: 11, pct: 29, trend: "-18%" },
];

function StatusPill({ status }: { status: TraceStatus }) {
  return <span className={`status status-${status}`}>{status}</span>;
}

function Sidebar({ view, setView }: { view: View; setView: (view: View) => void }) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark">A</span>
        <div>
          <strong>Agent Forge</strong>
          <small>训练驾驶舱</small>
        </div>
      </div>

      <nav className="main-nav" aria-label="主要导航">
        <p className="nav-label">工作台</p>
        {navItems.map((item) => (
          <button
            key={item.id}
            className={view === item.id ? "nav-item active" : "nav-item"}
            onClick={() => setView(item.id)}
            aria-current={view === item.id ? "page" : undefined}
          >
            <span className="nav-icon">{item.short}</span>
            <span>{item.label}</span>
            {item.badge && <span className="nav-badge">{item.badge}</span>}
          </button>
        ))}
      </nav>

      <div className="side-model">
        <div className="model-line">
          <span className="pulse-dot" />
          <span>当前训练版本</span>
        </div>
        <strong>agent-research-v2.8</strong>
        <div className="model-meta">
          <span>1,248 轨迹</span>
          <span>运行中</span>
        </div>
      </div>

      <div className="side-footer">
        <span className="avatar">HB</span>
        <div>
          <strong>Agent Lab</strong>
          <small>研究工作区</small>
        </div>
        <button className="more-button" aria-label="打开工作区菜单">•••</button>
      </div>
    </aside>
  );
}

function Topbar({ view }: { view: View }) {
  const titles: Record<View, { title: string; subtitle: string }> = {
    overview: { title: "训练总览", subtitle: "看清 Agent 今天学会了什么，还卡在哪里" },
    traces: { title: "轨迹回放", subtitle: "像看录像一样，逐步还原 Agent 的每次行动" },
    errors: { title: "错误地图", subtitle: "把零散失败整理成下一轮训练方向" },
    compare: { title: "版本对比", subtitle: "判断新版本是真进步，还是只擅长了少数题目" },
  };
  return (
    <header className="topbar">
      <div>
        <p className="eyebrow">AGENT TRAINING / JUL 24</p>
        <h1>{titles[view].title}</h1>
        <p>{titles[view].subtitle}</p>
      </div>
      <div className="top-actions">
        <button className="selector">
          <span className="selector-dot" />
          agent-research-v2.8
          <span className="chevron">⌄</span>
        </button>
        <button className="icon-button" aria-label="查看通知">
          <span>3</span>
          铃
        </button>
      </div>
    </header>
  );
}

function Overview({
  openTrace,
}: {
  openTrace: (trace: Trace) => void;
}) {
  return (
    <div className="view-stack">
      <section className="insight-banner">
        <div className="insight-icon">↗</div>
        <div>
          <p className="eyebrow">本轮关键发现</p>
          <h2>整体表现稳步上升，错误恢复能力进步最明显</h2>
          <p>v2.8 在复杂任务中少走了 18% 的重复步骤，但“工具选择错误”仍占失败案例的三分之一。</p>
        </div>
        <button onClick={() => openTrace(traces[1])}>查看典型失败 <span>→</span></button>
      </section>

      <section className="metric-grid" aria-label="核心训练指标">
        {metricCards.map((metric) => (
          <article className="metric-card" key={metric.label}>
            <div className="metric-top">
              <span>{metric.label}</span>
              <span className={`metric-spark spark-${metric.tone}`} />
            </div>
            <strong>{metric.value}</strong>
            <div className="metric-foot">
              <span className={metric.delta.startsWith("-") && metric.label !== "单任务成本" ? "down" : "up"}>
                {metric.delta}
              </span>
              <small>{metric.hint}</small>
            </div>
          </article>
        ))}
      </section>

      <section className="overview-grid">
        <article className="panel performance-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">TRAINING PULSE</p>
              <h3>最近 14 轮表现</h3>
            </div>
            <div className="legend">
              <span><i className="legend-current" /> v2.8</span>
              <span><i className="legend-prev" /> v2.7</span>
            </div>
          </div>
          <div className="chart-summary">
            <strong>78.4%</strong>
            <span>当前成功率</span>
          </div>
          <div className="line-chart" aria-label="成功率从 66% 上升到 78%">
            <div className="chart-grid-line line-one" />
            <div className="chart-grid-line line-two" />
            <div className="chart-grid-line line-three" />
            {[22, 32, 28, 43, 39, 54, 50, 63, 59, 70, 66, 77, 72, 84].map((height, index, list) => {
              const next = list[index + 1] ?? height;
              const angle = Math.atan2(next - height, 100 / (list.length - 1)) * (180 / Math.PI);
              const length = Math.sqrt(Math.pow(100 / (list.length - 1), 2) + Math.pow(next - height, 2));
              return (
                <div key={index} className="chart-point" style={{ left: `${index * (100 / (list.length - 1))}%`, bottom: `${height}%` }}>
                  {index < list.length - 1 && (
                    <span className="chart-segment" style={{ width: `${length}%`, transform: `rotate(${-angle}deg)` }} />
                  )}
                </div>
              );
            })}
            <div className="chart-axis-labels">
              <span>07/11</span><span>07/15</span><span>07/19</span><span>今天</span>
            </div>
          </div>
          <div className="chart-note"><span /> 第 9 轮加入“连续工具失败后强制重规划”策略</div>
        </article>

        <article className="panel score-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">CAPABILITY</p>
              <h3>能力得分</h3>
            </div>
            <button className="text-button">查看评分规则</button>
          </div>
          <div className="score-list">
            {scoreRows.map((row) => (
              <div className="score-row" key={row.label}>
                <div className="score-meta">
                  <span>{row.label}</span>
                  <strong>{row.value}</strong>
                  <small>{row.delta}</small>
                </div>
                <div className="score-track"><span style={{ width: `${row.value}%` }} /></div>
              </div>
            ))}
          </div>
          <div className="score-callout">
            <span>!</span>
            <p><strong>建议优先优化错误恢复</strong><br />它对复杂任务成功率的影响最大。</p>
          </div>
        </article>
      </section>

      <section className="panel recent-panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">RECENT TRACES</p>
            <h3>最近任务轨迹</h3>
          </div>
          <button className="text-button" onClick={() => openTrace(traces[0])}>查看全部轨迹 →</button>
        </div>
        <div className="trace-table">
          <div className="trace-table-head">
            <span>任务</span><span>状态</span><span>奖励</span><span>耗时</span><span>成本</span><span />
          </div>
          {traces.slice(0, 4).map((trace) => (
            <button className="trace-table-row" key={trace.id} onClick={() => openTrace(trace)}>
              <span className="task-cell"><i>{trace.id.slice(-2)}</i><span><strong>{trace.task}</strong><small>{trace.id} · {trace.category}</small></span></span>
              <StatusPill status={trace.status} />
              <strong className="reward-cell">{trace.reward.toFixed(2)}</strong>
              <span>{trace.duration}</span>
              <span>{trace.cost}</span>
              <span className="row-arrow">→</span>
            </button>
          ))}
        </div>
      </section>
    </div>
  );
}

const trajectorySteps = [
  { type: "理解任务", time: "00:00", title: "识别用户目标与约束", detail: "目标：定位退款异常；约束：不得承诺具体到账时间。", state: "good" },
  { type: "制定计划", time: "00:04", title: "先查订单，再核对支付流水", detail: "计划包含 3 个步骤，工具选择合理。", state: "good" },
  { type: "工具调用", time: "00:12", title: "调用 order.lookup", detail: "返回：订单服务暂时不可用（503）。", state: "warn" },
  { type: "工具调用", time: "00:36", title: "重复调用 order.lookup", detail: "相同参数再次返回 503，未改变查询策略。", state: "bad" },
  { type: "工具调用", time: "01:02", title: "第三次重复调用", detail: "触发循环检测，但 Agent 忽略了重规划提示。", state: "bad" },
  { type: "最终回答", time: "02:18", title: "给出不完整的退款建议", detail: "缺少支付流水证据，结论可信度不足。", state: "warn" },
];

function TracesView({ selected, setSelected }: { selected: Trace; setSelected: (trace: Trace) => void }) {
  const [filter, setFilter] = useState<"全部" | TraceStatus>("全部");
  const [saved, setSaved] = useState(false);
  const filtered = useMemo(
    () => traces.filter((trace) => filter === "全部" || trace.status === filter),
    [filter],
  );

  return (
    <div className="trace-workspace">
      <section className="trace-list-panel panel">
        <div className="trace-list-top">
          <div>
            <p className="eyebrow">1,248 TRACES</p>
            <h3>任务轨迹</h3>
          </div>
          <button className="filter-button">筛选 ⌄</button>
        </div>
        <div className="filter-tabs" role="tablist" aria-label="轨迹状态筛选">
          {(["全部", "成功", "失败", "需复核"] as const).map((item) => (
            <button key={item} className={filter === item ? "active" : ""} onClick={() => setFilter(item)}>{item}</button>
          ))}
        </div>
        <div className="trace-card-list">
          {filtered.map((trace) => (
            <button key={trace.id} className={selected.id === trace.id ? "trace-card selected" : "trace-card"} onClick={() => { setSelected(trace); setSaved(false); }}>
              <div><StatusPill status={trace.status} /><small>{trace.id}</small></div>
              <strong>{trace.task}</strong>
              <p>{trace.note}</p>
              <div className="trace-card-meta"><span>奖励 {trace.reward}</span><span>{trace.duration}</span><span>{trace.cost}</span></div>
            </button>
          ))}
        </div>
      </section>

      <section className="trace-detail panel">
        <div className="trace-detail-head">
          <div>
            <div className="detail-id"><StatusPill status={selected.status} /><span>{selected.id}</span><span>{selected.category}</span></div>
            <h2>{selected.task}</h2>
            <p>{selected.note}</p>
          </div>
          <button className={saved ? "primary-button saved" : "primary-button"} onClick={() => setSaved(!saved)}>
            {saved ? "✓ 已加入训练样本" : "+ 加入训练样本"}
          </button>
        </div>

        <div className="detail-metrics">
          <div><span>奖励分</span><strong>{selected.reward.toFixed(2)}</strong></div>
          <div><span>总耗时</span><strong>{selected.duration}</strong></div>
          <div><span>工具调用</span><strong>{selected.tools} 次</strong></div>
          <div><span>本次成本</span><strong>{selected.cost}</strong></div>
        </div>

        <div className="playback-heading">
          <div><p className="eyebrow">STEP BY STEP</p><h3>行动回放</h3></div>
          <button className="play-button">▶ 自动播放</button>
        </div>
        <div className="timeline">
          {trajectorySteps.map((step, index) => (
            <article className={`timeline-step step-${step.state}`} key={`${step.time}-${step.title}`}>
              <div className="step-index">{index + 1}</div>
              <div className="step-copy">
                <div><span>{step.type}</span><time>{step.time}</time></div>
                <h4>{step.title}</h4>
                <p>{step.detail}</p>
              </div>
              {step.state === "bad" && <span className="issue-tag">关键问题</span>}
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}

function ErrorsView() {
  const [queued, setQueued] = useState<string[]>(["遗漏约束"]);
  const toggleQueue = (item: string) => setQueued((items) => items.includes(item) ? items.filter((x) => x !== item) : [...items, item]);
  return (
    <div className="view-stack">
      <section className="error-hero">
        <div>
          <p className="eyebrow">FAILURE ATLAS / 115 CASES</p>
          <h2>失败不是一个数字，而是一组可以被修复的模式。</h2>
          <p>过去 7 天的失败轨迹已自动聚类为 5 类，其中 63% 可以通过改进工具策略和约束检查解决。</p>
        </div>
        <div className="error-score">
          <span>可修复失败</span>
          <strong>63%</strong>
          <small>预计可带来 +8.2% 成功率</small>
        </div>
      </section>

      <section className="error-grid">
        <article className="panel error-bars-panel">
          <div className="panel-heading">
            <div><p className="eyebrow">ERROR DISTRIBUTION</p><h3>错误类型分布</h3></div>
            <span className="date-chip">最近 7 天</span>
          </div>
          <div className="error-bars">
            {errorTypes.map((error, index) => (
              <div className="error-bar-row" key={error.label}>
                <span className="error-rank">0{index + 1}</span>
                <div>
                  <div className="error-bar-meta"><strong>{error.label}</strong><span>{error.count} 条</span><small className={error.trend.startsWith("+") ? "trend-bad" : ""}>{error.trend}</small></div>
                  <div className="error-track"><span style={{ width: `${error.pct}%` }} /></div>
                </div>
              </div>
            ))}
          </div>
        </article>

        <article className="panel diagnosis-panel">
          <div className="panel-heading">
            <div><p className="eyebrow">DIAGNOSIS</p><h3>本周诊断</h3></div>
          </div>
          <div className="diagnosis-ring"><span>38</span><small>工具错误</small></div>
          <h4>Agent 知道要做什么，但经常选错“怎么做”</h4>
          <p>错误主要集中在工具功能相似、返回值为空，以及连续失败后的策略切换。</p>
          <div className="diagnosis-tip"><span>建议</span><p>增加工具说明对比样本，并加入“连续失败两次必须重规划”的训练规则。</p></div>
        </article>
      </section>

      <section className="panel fix-queue">
        <div className="panel-heading">
          <div><p className="eyebrow">FIX QUEUE</p><h3>高价值修复队列</h3></div>
          <span className="queue-count">{queued.length} 项已加入</span>
        </div>
        <div className="fix-list">
          {[
            { title: "工具连续失败后没有切换策略", cases: "27 条失败轨迹", gain: "+3.4%", label: "重复循环" },
            { title: "最终回答遗漏用户的硬性限制", cases: "22 条失败轨迹", gain: "+2.8%", label: "遗漏约束" },
            { title: "证据不足时仍输出确定性结论", cases: "11 条失败轨迹", gain: "+1.6%", label: "事实无依据" },
          ].map((item, index) => (
            <article key={item.title}>
              <span className="fix-number">{index + 1}</span>
              <div><strong>{item.title}</strong><p>{item.cases} · 预计成功率提升 <b>{item.gain}</b></p></div>
              <button className={queued.includes(item.label) ? "queued" : ""} onClick={() => toggleQueue(item.label)}>
                {queued.includes(item.label) ? "✓ 已加入" : "+ 加入队列"}
              </button>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}

function CompareView() {
  const dimensions = [
    { label: "总体成功率", before: 71.6, after: 78.4, suffix: "%" },
    { label: "复杂任务", before: 58.2, after: 69.7, suffix: "%" },
    { label: "工具调用成功", before: 89.2, after: 91.3, suffix: "%" },
    { label: "平均奖励", before: 0.63, after: 0.72, suffix: "" },
  ];
  return (
    <div className="view-stack">
      <section className="compare-top">
        <div className="version-card old-version">
          <span>基准版本</span><h2>agent-research-v2.7</h2><p>2026.07.11 · 1,104 条评测</p>
        </div>
        <div className="versus">VS</div>
        <div className="version-card new-version">
          <span>候选版本</span><h2>agent-research-v2.8</h2><p>2026.07.24 · 1,248 条评测</p>
          <i>推荐发布</i>
        </div>
      </section>

      <section className="compare-summary">
        <div><span className="summary-icon">↗</span><p><strong>v2.8 总体更优</strong><br />14 项核心指标中，11 项提升、2 项持平、1 项轻微回退。</p></div>
        <button className="primary-button">生成对比报告</button>
      </section>

      <section className="compare-grid">
        <article className="panel compare-metrics">
          <div className="panel-heading"><div><p className="eyebrow">HEAD TO HEAD</p><h3>核心指标对比</h3></div></div>
          <div className="compare-table-head"><span>指标</span><span>v2.7</span><span>v2.8</span><span>变化</span></div>
          {dimensions.map((item) => {
            const diff = item.after - item.before;
            return (
              <div className="compare-row" key={item.label}>
                <strong>{item.label}</strong>
                <span>{item.before}{item.suffix}</span>
                <span className="after-value">{item.after}{item.suffix}</span>
                <span className="delta-value">+{diff.toFixed(item.suffix ? 1 : 2)}{item.suffix}</span>
              </div>
            );
          })}
          <div className="cost-compare">
            <div><span>每 100 个任务成本</span><strong>¥49.00</strong><small>v2.7</small></div>
            <span className="cost-arrow">→</span>
            <div><span>每 100 个任务成本</span><strong>¥42.00</strong><small>v2.8 · 节省 14.2%</small></div>
          </div>
        </article>

        <article className="panel benchmark-panel">
          <div className="panel-heading"><div><p className="eyebrow">BENCHMARK SETS</p><h3>分场景表现</h3></div></div>
          {[
            { name: "深度研究", before: 68, after: 82 },
            { name: "代码与数据", before: 73, after: 79 },
            { name: "客服运营", before: 76, after: 81 },
            { name: "办公协作", before: 84, after: 87 },
            { name: "营销策划", before: 71, after: 68 },
          ].map((row) => (
            <div className="benchmark-row" key={row.name}>
              <div><strong>{row.name}</strong><span className={row.after >= row.before ? "positive" : "negative"}>{row.after >= row.before ? "+" : ""}{row.after - row.before}</span></div>
              <div className="dual-track">
                <span className="before-bar" style={{ width: `${row.before}%` }} />
                <span className="after-bar" style={{ width: `${row.after}%` }} />
              </div>
              <div className="benchmark-values"><span>{row.before}</span><strong>{row.after}</strong></div>
            </div>
          ))}
          <div className="benchmark-warning"><span>!</span><p><strong>营销策划回退 3 分</strong><br />主要原因是新版对创意开放性限制过强。</p></div>
        </article>
      </section>

      <section className="panel sample-diff">
        <div className="panel-heading"><div><p className="eyebrow">SAMPLE DIFF</p><h3>典型案例变化</h3></div><button className="text-button">浏览全部 143 个变化样本 →</button></div>
        <div className="sample-columns">
          <article><span className="version-label">v2.7 · 失败</span><h4>遇到工具报错后重复尝试 5 次</h4><p>“订单查询暂时失败，我将再次尝试……”</p><small>没有识别重复失败，也没有采用备用查询方法。</small></article>
          <div className="sample-divider">→</div>
          <article><span className="version-label improved">v2.8 · 成功</span><h4>第二次失败后主动切换验证路径</h4><p>“订单服务暂不可用，我将改用支付流水核对……”</p><small>减少 3 次无效调用，完成时间缩短 46 秒。</small></article>
        </div>
      </section>
    </div>
  );
}

export default function Home() {
  const [view, setView] = useState<View>("overview");
  const [selectedTrace, setSelectedTrace] = useState<Trace>(traces[1]);

  const openTrace = (trace: Trace) => {
    setSelectedTrace(trace);
    setView("traces");
  };

  return (
    <main className="app-shell">
      <Sidebar view={view} setView={setView} />
      <div className="content-shell">
        <Topbar view={view} />
        <div className="content">
          {view === "overview" && <Overview openTrace={openTrace} />}
          {view === "traces" && <TracesView selected={selectedTrace} setSelected={setSelectedTrace} />}
          {view === "errors" && <ErrorsView />}
          {view === "compare" && <CompareView />}
        </div>
      </div>
    </main>
  );
}
