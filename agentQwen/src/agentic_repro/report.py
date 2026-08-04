from __future__ import annotations

import html
import json
from pathlib import Path


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _bar(value: float, label: str) -> str:
    width = max(0.0, min(100.0, value * 100))
    return f'<div class="bar-row"><span>{html.escape(label)}</span><div class="bar-track"><i style="width:{width:.1f}%"></i></div><b>{value:.3f}</b></div>'


def _trace_card(item: dict) -> str:
    traj = item["trajectory"]
    turns = item["model_turns"]
    rows = "".join(
        "<tr>"
        f"<td>{turn['step']}</td>"
        f"<td><code>{html.escape(str(turn.get('parsed_action')))}</code></td>"
        f"<td>{html.escape(turn.get('reason', ''))}</td>"
        f"<td>{turn['seconds']:.2f}s</td>"
        "</tr>"
        for turn in turns
    )
    raw = "\n\n".join(f"Step {turn['step']}: {turn['raw_output']}" for turn in turns)
    status = "通过" if traj["success"] else "未通过"
    return f"""
    <section class="trace-card">
      <div class="trace-head"><div><span class="kicker">{html.escape(traj['scenario'])}</span><h4>{html.escape(' → '.join(traj['actions']) or 'No parsed action')}</h4></div><span class="status {'pass' if traj['success'] else 'fail'}">{status} · R={traj['reward']:.2f}</span></div>
      <table><thead><tr><th>Step</th><th>Action</th><th>Reason</th><th>Latency</th></tr></thead><tbody>{rows}</tbody></table>
      <details><summary>查看模型原始输出</summary><pre><code>{html.escape(raw)}</code></pre></details>
    </section>"""


def build_report(config: dict, project_dir: Path) -> Path:
    artifacts = project_dir / config["paths"]["artifacts"]
    report_dir = project_dir / config["paths"]["report"]
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((artifacts / "metrics.json").read_text(encoding="utf-8"))
    verify = json.loads((artifacts / "verification.json").read_text(encoding="utf-8"))
    qwen = json.loads((artifacts / "qwen3_8b_inference.json").read_text(encoding="utf-8"))

    baseline = metrics["baseline"]["overall"]
    final = metrics["final"]["overall"]
    delta = final["mean_reward"] - baseline["mean_reward"]
    rounds = metrics["rounds"]
    round_bars = "".join(_bar(item["mean_reward"], f"Round {item['round']} · levels {item['train_levels']}") for item in rounds)
    level_rows = "".join(
        f"<tr><td>Level {level}</td><td>{data['mean_reward']:.3f}</td><td>{_pct(data['success_rate'])}</td><td>{_pct(data['safety_rate'])}</td><td>{data['mean_steps']:.2f}</td></tr>"
        for level, data in metrics["final"]["by_level"].items()
    )
    check_rows = "".join(
        f"<tr><td>{html.escape(check['name'])}</td><td><span class=\"status {check['status'].lower()}\">{check['status']}</span></td><td>{html.escape(check['detail'])}</td></tr>"
        for check in verify["checks"]
    )
    if qwen.get("status") == "completed":
        qwen_cards = "".join(_trace_card(item) for item in qwen["results"])
        qwen_summary = f"""
          <div class="metric"><strong>{_pct(qwen['success_rate'])}</strong><span>3 场景成功率</span></div>
          <div class="metric"><strong>{qwen['mean_reward']:.3f}</strong><span>平均 rubric reward</span></div>
          <div class="metric"><strong>{qwen.get('peak_memory_gib', 'n/a')} GiB</strong><span>MLX 峰值内存</span></div>
          <div class="metric"><strong>{qwen['load_seconds']:.1f}s</strong><span>模型加载耗时</span></div>"""
        qwen_meta = f"本地模型：{html.escape(qwen['model_id'])}；后端：MLX；量化：{qwen['quantization']}；网络 fallback：{qwen['network_fallback']}。"
    else:
        qwen_cards = f'<blockquote>Qwen3-8B inference 未运行：{html.escape(qwen.get("reason", "unknown"))}</blockquote>'
        qwen_summary = '<div class="metric"><strong>SKIPPED</strong><span>Qwen3-8B inference</span></div>'
        qwen_meta = "未提供本地权重。"

    content = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AgenticQwen 可审计复现报告</title>
<meta name="description" content="从论文解析、Planner/Executor/Verifier 到本地 Qwen3-8B inference 的 AgenticQwen 复现报告。">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js" onload="renderMathInElement(document.body,{{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}],throwOnError:false}});"></script>
<style>
:root{{--ink:#28231f;--muted:#756b62;--line:#e7ded2;--paper:#fffdf9;--wash:#f5eee5;--accent:#a8472d;--accent-dark:#713322;--green:#2f6f55;--red:#a12d2d;--blue:#284c75;--shadow:0 18px 55px rgba(80,47,27,.09)}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;color:var(--ink);background:var(--wash);font:16px/1.9 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",sans-serif}}
.site-header{{position:sticky;top:0;z-index:8;border-bottom:1px solid rgba(130,91,58,.14);background:rgba(255,253,249,.92);backdrop-filter:blur(14px)}}.nav-shell,.page-shell{{width:min(1120px,calc(100% - 36px));margin:auto}}.nav-shell{{display:flex;justify-content:space-between;align-items:center;gap:20px;min-height:68px}}.brand{{color:var(--ink);font:700 18px/1.2 Georgia,"Songti SC",serif;text-decoration:none}}.brand span{{color:var(--accent)}}nav{{display:flex;flex-wrap:wrap;gap:18px}}nav a{{color:var(--muted);font-size:13px;text-decoration:none}}nav a:hover{{color:var(--accent)}}
.page-shell{{padding:58px 0 100px}}.article{{max-width:900px;margin:auto;padding:58px clamp(24px,6vw,76px) 76px;background:var(--paper);box-shadow:var(--shadow)}}h1,h2,h3,h4{{color:var(--ink);font-family:Georgia,"Songti SC","STSong",serif;font-weight:700;line-height:1.28;letter-spacing:-.025em}}h1{{margin:0 0 18px;font-size:clamp(36px,5vw,58px)}}h2{{margin:68px 0 18px;padding-top:12px;font-size:clamp(25px,3vw,34px);border-top:1px solid var(--line)}}h3{{margin:38px 0 10px;font-size:22px}}h4{{margin:0;font-size:18px}}p{{margin:0 0 18px}}a{{color:var(--accent);text-underline-offset:3px}}strong{{color:var(--accent-dark)}}.lead{{color:var(--muted);font-size:18px}}.eyebrow,.kicker{{color:var(--accent);font-size:12px;font-weight:800;letter-spacing:.12em;text-transform:uppercase}}.meta{{display:flex;flex-wrap:wrap;gap:8px;margin:24px 0 0}}.meta span{{border:1px solid var(--line);padding:4px 9px;border-radius:999px;color:var(--muted);font-size:12px}}
.metric-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:28px 0}}.metric{{min-height:112px;padding:16px;border:1px solid var(--line);border-top:3px solid var(--accent);background:#fff}}.metric strong{{display:block;color:var(--ink);font:700 27px/1.2 Georgia,serif}}.metric span{{display:block;margin-top:8px;color:var(--muted);font-size:12px;line-height:1.45}}blockquote{{margin:26px 0;padding:16px 20px;border-left:4px solid var(--accent);color:#604d3e;background:#fbf1e7}}.note{{border:1px solid var(--line);padding:18px 20px;background:#faf7f2}}.note.blue{{border-left:4px solid var(--blue)}}
.figure-box{{margin:30px 0;padding:14px;border:1px solid var(--line);background:#fff;text-align:center}}.figure-box img{{display:block;max-width:100%;height:auto;margin:auto}}.caption{{margin-top:10px;color:var(--muted);font:italic 12px/1.65 Georgia,"Songti SC",serif}}table{{width:100%;margin:25px 0 30px;border-collapse:collapse;font-size:14px}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{color:var(--accent-dark);background:#fbf4eb}}code{{padding:2px 5px;border-radius:4px;background:#f2e9df;font-size:.9em}}pre{{overflow:auto;margin:18px 0;padding:18px 20px;border-radius:10px;background:#28231f;color:#f8efe5;line-height:1.6}}pre code{{padding:0;background:none}}.bar-row{{display:grid;grid-template-columns:190px 1fr 58px;align-items:center;gap:12px;margin:12px 0;font-size:13px}}.bar-track{{height:10px;overflow:hidden;border-radius:99px;background:#eee6dc}}.bar-track i{{display:block;height:100%;border-radius:99px;background:var(--accent)}}.bar-row b{{text-align:right}}.status{{display:inline-block;padding:2px 8px;border:1px solid currentColor;border-radius:99px;font-size:11px;font-weight:800}}.status.pass{{color:var(--green)}}.status.warn{{color:#a36b13}}.status.fail{{color:var(--red)}}
.trace-card{{margin:22px 0;padding:20px;border:1px solid var(--line);background:#fff}}.trace-head{{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}}details summary{{cursor:pointer;color:var(--accent);font-size:13px}}.flow{{display:grid;grid-template-columns:repeat(3,1fr);gap:0;margin:28px 0}}.flow>div{{position:relative;padding:20px;border:1px solid var(--line);background:#fff}}.flow>div+div{{border-left:0}}.flow b{{display:block;font:700 20px Georgia,serif}}.flow span{{color:var(--muted);font-size:13px}}.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}ul,ol{{margin:0 0 22px;padding-left:1.5em}}li+li{{margin-top:5px}}.site-footer{{padding:26px 0 40px;color:var(--muted);font-size:13px;text-align:center}}
@media(max-width:760px){{.metric-grid{{grid-template-columns:1fr 1fr}}.flow,.two-col{{grid-template-columns:1fr}}.flow>div+div{{border-left:1px solid var(--line);border-top:0}}.bar-row{{grid-template-columns:120px 1fr 50px}}.article{{padding:34px 20px 48px}}.page-shell{{padding-top:28px}}.nav-shell{{align-items:flex-start;flex-direction:column;justify-content:center;min-height:92px}}table{{display:block;overflow-x:auto;white-space:nowrap}}}}
</style>
</head>
<body>
<header class="site-header"><div class="nav-shell"><a class="brand" href="#top">AgenticQwen<span>/</span>repro</a><nav><a href="#method">论文方法</a><a href="#pipeline">复现流水线</a><a href="#qwen">本地推理</a><a href="#audit">验证账本</a></nav></div></header>
<main class="page-shell"><article class="article" id="top">
<div class="eyebrow">Paper → Plan → Execute → Verify → Report → Slides</div>
<h1>AgenticQwen 可审计复现</h1>
<p class="lead">从论文拆解到行为树数据飞轮、GRPO-style smoke 训练，再到 M4 Pro 上的 Qwen3-8B 本地多轮工具调用。每一个数字都标注来源层级，避免把结构复现、模型推理和论文规模训练混为一谈。</p>
<div class="meta"><span>arXiv:2604.21590</span><span>Apple M4 Pro · 24GB</span><span>Seed {metrics['seed']}</span><span>Verifier {verify['overall_status']}</span></div>
<div class="metric-grid">
  <div class="metric"><strong>{baseline['mean_reward']:.3f}</strong><span>Smoke baseline reward</span></div>
  <div class="metric"><strong>{final['mean_reward']:.3f}</strong><span>Round 3 smoke reward</span></div>
  <div class="metric"><strong>+{delta:.3f}</strong><span>固定 eval suite 提升</span></div>
  <div class="metric"><strong>{verify['summary']['pass']}/{len(verify['checks'])}</strong><span>Verifier 通过项</span></div>
</div>
<blockquote><strong>结论先行：</strong>本地工程完成了论文关键机制的结构性复现，并用真实 Qwen3-8B 跑了相同环境的 inference；它没有声称在 24GB Mac 上复现论文的 100K 数据、多卡 RL 或 TAU-2/BFCL-V4 指标。</blockquote>

<h2 id="paper">一、论文身份与复现对象</h2>
<p>输入中显示文本为 <code>2604.18292</code>，但超链接实际指向 <code>2604.21590</code>。前者是 Agent-World，后者才是 <em>AgenticQwen: Training Small Agentic Language Models with Dual Data Flywheels for Industrial-Scale Tool Use</em>，且与“reasoning RL + agentic RL + 双数据飞轮”的描述一致，因此本项目锁定后者。</p>
<p>论文要解决的不是所有 agent 任务，而是工业场景里高频、标准化、对时延和成本敏感的 tool-use 工作流。核心判断是：小模型若接受针对性的多轮 agentic RL，可以在部分场景逼近大模型，同时降低推理成本。</p>

<h2 id="method">二、方法拆解：双数据飞轮</h2>
<div class="figure-box"><img src="images/figure1_dual_flywheel.png" alt="AgenticQwen dual data flywheels"><div class="caption"><b>Figure 1.</b> 论文原图。上半部分从错误样本构造更难且可验证的 reasoning 数据；下半部分把线性 workflow 逐轮扩成行为树，并从分支反向生成任务。</div></div>
<h3>2.1 Reasoning flywheel</h3>
<p>每轮收集模型失败题，经 self-instruct 改数值、加约束或引入更深理论，再用 persona injection 改写成物理、化学等应用语境。候选题由强模型独立求解三次，仅保留答案一致的样本。这里的关键不是“更多合成数据”，而是<strong>错误驱动、结构多样、答案可验证</strong>。</p>
<h3>2.2 Agentic flywheel</h3>
<p>Agentic 数据先从线性路径 $A_{{query}}\rightarrow B_{{book}}\rightarrow C_{{confirm}}$ 起步；每轮 RL 后加入环境条件，形成 sold-out、delay、membership 等分支。随后做 branch-to-task inversion：选择一个分支 $b$，反推使它成为最优路径的环境状态 $s_b$、用户指令 $u_b$ 和 agent SOP $a_b$。</p>
<div class="note blue">本地映射：available → 订机票；sold-out → 查高铁并预订；delayed → 查会员等级，Gold 发现金、Standard 发券；Standard 用户仍索要现金时形成 adversarial trap。</div>
<h3>2.3 Rubric reward 与 GRPO</h3>
<p>论文把任务拆成可验证 subgoals，reward 是完成比例：</p>
<div class="note">$$R(\tau)=\frac{{1}}{{M}}\sum_{{m=1}}^M \mathbf{{1}}[g_m(\tau)=\mathrm{{done}}],\quad R\in[0,1]$$</div>
<p>本地 smoke 版保留 group-relative 核心：对同一任务采样一组轨迹，用组内均值和标准差归一化 advantage，再更新透明 softmax policy。它用于验证闭环，不等价于论文的 veRL 分布式语言模型训练。</p>

<h2 id="pipeline">三、Planner → Executor → Verifier</h2>
<div class="flow"><div><span>01 · SPEC</span><b>Planner</b><span>锁定论文、复现层级、DAG、输出与验收条件。</span></div><div><span>02 · RUN</span><b>Executor</b><span>生成任务、训练 smoke policy、执行 Qwen3-8B rollout。</span></div><div><span>03 · AUDIT</span><b>Verifier</b><span>检查 reward、determinism、provenance、trace 和 claim 边界。</span></div></div>
<table><thead><tr><th>层级</th><th>本项目证据</th><th>可下结论</th><th>不能下结论</th></tr></thead><tbody>
<tr><td>L0 结构</td><td>可执行 state machine + 行为树</td><td>分支、陷阱、rubric 映射成立</td><td>论文 benchmark 已复现</td></tr>
<tr><td>L1 算法 smoke</td><td>固定 seed 的 group-relative policy gradient</td><td>训练闭环与学习信号可运行</td><td>等价于 8B GRPO</td></tr>
<tr><td>L2 模型 inference</td><td>Qwen3-8B 4-bit MLX 原始输出与工具事件</td><td>本地模型能否完成三类工具路径</td><td>模型经过本项目 RL</td></tr>
<tr><td>L3 论文规模</td><td>paper-scale config + upstream 代码映射</td><td>资源与步骤清晰</td><td>尚未占用多卡算力执行</td></tr>
</tbody></table>

<h2 id="smoke">四、Smoke RL 实验</h2>
<p>训练共 {len(rounds)} 轮。Round 1 只见线性任务；Round 2 加入 sold-out 分支；Round 3 加入 delayed、membership 与 adversarial cash claim。评测始终使用固定的三层任务集，防止用训练难度变化掩盖退化。</p>
<div class="note">{round_bars}</div>
<table><thead><tr><th>Eval level</th><th>Mean reward</th><th>Success</th><th>Safety</th><th>Mean steps</th></tr></thead><tbody>{level_rows}</tbody></table>
<p>结果只说明这个最小实现形成了可检测的学习信号。由于 policy 是透明线性 softmax，且环境规模很小，它更像“算法接线测试”，不该与论文的 8B/30B checkpoint 比较。</p>

<h2 id="qwen">五、Qwen3-8B 本地 inference</h2>
<p>{qwen_meta}</p>
<div class="metric-grid">{qwen_summary}</div>
{qwen_cards}

<h2 id="paper-results">六、论文结果与本地结果不能混读</h2>
<div class="figure-box"><img src="images/figure2_flywheel_results.png" alt="Paper flywheel results"><div class="caption"><b>Figure 2.</b> 论文原图显示 Qwen3-8B 与 Qwen3-30B-A3B 从 Round 0 到 Round 3 的多个 benchmark 子集持续上升；这是论文结果，不是本机测量。</div></div>
<table><thead><tr><th>Model</th><th>TAU-2 Airline</th><th>Telecom</th><th>Retail</th><th>BFCL Base</th><th>Paper Avg.</th></tr></thead><tbody>
<tr><td>Qwen3-8B</td><td>14.5</td><td>7.9</td><td>31.6</td><td>35.5</td><td>23.8</td></tr>
<tr><td><strong>AgenticQwen-8B</strong></td><td>40.5</td><td>53.5</td><td>60.3</td><td>56.0</td><td>47.4</td></tr>
<tr><td>Qwen3-30B-A3B</td><td>32.0</td><td>31.6</td><td>55.3</td><td>47.0</td><td>36.2</td></tr>
<tr><td><strong>AgenticQwen-30B-A3B</strong></td><td>42.0</td><td>52.6</td><td>60.5</td><td>60.0</td><td>50.2</td></tr>
</tbody></table>
<div class="figure-box"><img src="images/figure3_case_study.png" alt="Paper industrial analytics case"><div class="caption"><b>Figure 3.</b> 论文中的企业数据分析案例：agent 分解任务并编排 SQL、JSON 与 PDF/RAG 工具。</div></div>

<h2 id="audit">七、Verifier 账本</h2>
<table><thead><tr><th>Check</th><th>Status</th><th>Evidence</th></tr></thead><tbody>{check_rows}</tbody></table>
<p>Verifier 的核心原则是“产物存在”不等于“复现成立”。它额外检查 reward 边界、固定 seed 重跑、难度层覆盖，以及 Qwen3-8B 是否确实从本地 MLX 权重执行、有没有云端 fallback、是否保留 raw output。</p>

<h2 id="next">八、走向论文级复现</h2>
<ol><li>接入官方 <code>tool_use_data_synthesis</code>，用 Qwen3-235B 本地服务生成与过滤约 100K 任务。</li><li>冻结 simulator、prompt、reward judge 与数据 hash；建立 held-out TAU-2/BFCL-V4 评测。</li><li>在多卡 GPU 上用 veRL + SGLang 对 Qwen3-8B 运行三轮 GRPO，每轮从失败轨迹扩展 behavior tree。</li><li>报告每轮 checkpoint、数据版本、随机种子、吞吐、显存、失败类型和置信区间。</li></ol>
<pre><code># 本机已验证的一键入口
QWEN_MODEL_PATH=/absolute/path/to/Qwen3-8B-4bit ./run.sh

# 不重复跑 inference，只看账本
python3 -m json.tool artifacts/verification.json</code></pre>

<h2 id="sources">九、来源</h2>
<ul><li><a href="https://arxiv.org/abs/2604.21590">AgenticQwen paper</a></li><li><a href="https://github.com/haruhi-sudo/data_synth_and_rl">Official data synthesis and RL code</a></li><li><a href="https://huggingface.co/Qwen/Qwen3-8B">Qwen3-8B model card</a></li><li><a href="https://huggingface.co/mlx-community/Qwen3-8B-4bit">MLX 4-bit conversion used for local inference</a></li><li><a href="https://yyhdbl.github.io">报告视觉参考</a></li></ul>
</article></main><footer class="site-footer"><div class="page-shell">Generated from an auditable local run · paper claims and local measurements are kept separate.</div></footer>
</body></html>"""
    out = report_dir / "index.html"
    out.write_text(content, encoding="utf-8")
    return out

