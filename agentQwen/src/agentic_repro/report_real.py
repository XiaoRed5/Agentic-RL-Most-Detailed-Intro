from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def _load(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def _short(value: str, size: int = 12) -> str:
    return value[:size] + "…"


def _status_rows(items: list[dict[str, str]]) -> str:
    return "".join(
        "<tr>"
        f"<td><strong>{html.escape(item['area'])}</strong></td>"
        f"<td><span class='status {item['status'].lower().replace('_', '-')}'>{html.escape(item['status'])}</span></td>"
        f"<td>{html.escape(item['evidence'])}</td>"
        f"<td>{html.escape(item['completion'])}</td>"
        "</tr>"
        for item in items
    )


def _verification_rows(items: list[dict[str, str]]) -> str:
    return "".join(
        "<tr>"
        f"<td>{html.escape(item['name'])}</td>"
        f"<td><span class='status {item['status'].lower()}'>{html.escape(item['status'])}</span></td>"
        f"<td><code>{html.escape(item['detail'])}</code></td>"
        "</tr>"
        for item in items
    )


def _trajectory_events(items: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    previous_state: dict[str, Any] = {}
    for item in items:
        state_after = item.get("state_after") or previous_state
        changed = []
        if isinstance(state_after, dict):
            for key, value in state_after.items():
                if key == "tool_sequence":
                    continue
                if previous_state.get(key) != value:
                    changed.append(
                        f"<span><code>{html.escape(str(key))}</code> → {html.escape(str(value))}</span>"
                    )
            previous_state = state_after
        content = html.escape(str(item.get("content", ""))).replace("\n", "<br>")
        reward = item.get("reward_delta")
        reward_html = (
            f"<span class='event-reward'>reward Δ {float(reward):+.2f}</span>"
            if isinstance(reward, (int, float)) and reward
            else ""
        )
        state_html = (
            "<div class='state-delta'><b>State change</b>" + "".join(changed) + "</div>"
            if changed
            else ""
        )
        tool_name = item.get("tool_name")
        tool_html = (
            f"<span class='event-tool'>{html.escape(str(tool_name))}</span>"
            if tool_name
            else ""
        )
        rows.append(
            f"<article class='trajectory-event role-{html.escape(item['role'])}' data-role='{html.escape(item['role'])}'>"
            "<div class='event-index'>"
            f"{int(item['event_id']):02d}</div><div class='event-main'><header>"
            f"<span class='event-role'>{html.escape(item['role'])}</span>"
            f"<span class='event-type'>{html.escape(item['event_type'])}</span>{tool_html}{reward_html}"
            f"</header><div class='event-content'>{content}</div>{state_html}</div></article>"
        )
    return "".join(rows)


def _trajectory_check_rows(items: list[dict[str, Any]]) -> str:
    return "".join(
        "<tr>"
        f"<td><code>{html.escape(item['name'])}</code></td>"
        f"<td><span class='status {'pass' if item['passed'] else 'fail'}'>{'PASS' if item['passed'] else 'FAIL'}</span></td>"
        f"<td>{html.escape(item['detail'])}</td>"
        "</tr>"
        for item in items
    )


def _reward_rows(items: list[dict[str, Any]]) -> str:
    return "".join(
        "<tr>"
        f"<td>{int(item['turn'])}</td>"
        f"<td><code>{html.escape(item['event'])}</code></td>"
        f"<td class='{'positive' if float(item['value']) > 0 else 'negative-text'}'>{float(item['value']):+.2f}</td>"
        f"<td>{html.escape(item['reason'])}</td>"
        "</tr>"
        for item in items
    )


def build(config_path: Path) -> Path:
    project_dir = config_path.parent.parent.resolve()
    artifacts = project_dir / "artifacts" / "real_qwen3_8b"
    summary = _load(artifacts / "summary.json")
    verification = _load(artifacts / "verification.json")
    if not summary or not verification:
        raise RuntimeError("Run real_grpo and verify_real before building the report")

    benchmark_manifest = _load(
        project_dir / "artifacts" / "benchmarks" / "planned_manifest.json",
        {"status": "NOT_PREPARED", "scope": {}},
    )
    reward_diagnostic = _load(
        project_dir / "artifacts" / "ablations" / "offline_reward_diagnostic.json",
        {},
    )
    ablation_matrix = _load(project_dir / "configs" / "ablation_matrix.json", {})
    long_trajectory = _load(
        project_dir
        / "artifacts"
        / "long_horizon"
        / "trajectory_qwen3.7_flash.json",
        {},
    )
    if not long_trajectory:
        raise RuntimeError("Run run_long_horizon_demo.sh before building report v3")

    metrics = summary["metrics"]
    train = metrics["train"]
    holdout = metrics["holdout"]
    unseen = metrics["unseen"]
    attempted = summary["training"]["attempted_groups"]
    updated = summary["training"]["updated_groups"]
    saturated = attempted - updated
    saturation_rate = saturated / attempted
    diagnostic_before = reward_diagnostic.get("outcome_nonzero_variance_groups", updated)
    diagnostic_after = reward_diagnostic.get("shaped_nonzero_variance_groups", updated)
    trajectory_runtime = long_trajectory["runtime"]
    trajectory_verification = long_trajectory["verification"]
    trajectory_events_html = _trajectory_events(long_trajectory["events"])
    trajectory_check_html = _trajectory_check_rows(trajectory_verification["checks"])
    trajectory_reward_html = _reward_rows(trajectory_verification["process_ledger"])

    self_check = [
        {
            "area": "论文、源码与主张锁定",
            "status": "COMPLETE",
            "evidence": "arXiv 2604.21590 PDF + LaTeX；论文表 1 与算法逐项提取",
            "completion": "—",
        },
        {
            "area": "真实 Qwen3-8B 权重加载",
            "status": "COMPLETE",
            "evidence": "8.19B parameters；4.35 GB MLX 4-bit；SHA-256",
            "completion": "—",
        },
        {
            "area": "官方 AgenticQwen 数据 provenance",
            "status": "COMPLETE",
            "evidence": "37,401 rows；parquet SHA-256；固定 seed 与 task IDs",
            "completion": "—",
        },
        {
            "area": "真实 LoRA-GRPO 参数更新",
            "status": "COMPLETE",
            "evidence": f"LoRA-B 0 → {summary['training']['lora_b_norm_after']:.3f}；adapter hash 改变",
            "completion": "—",
        },
        {
            "area": "独立 checkpoint 重放",
            "status": "COMPLETE",
            "evidence": f"fresh process；{verification['summary']['pass']} PASS / {verification['summary']['fail']} FAIL",
            "completion": "—",
        },
        {
            "area": "真实 API 多轮工具轨迹",
            "status": "COMPLETE",
            "evidence": (
                f"Qwen3.7-Flash；{trajectory_runtime['agent_turns']} agent turns；"
                f"{trajectory_runtime['tool_calls']} tool calls；9/9 verifier PASS"
            ),
            "completion": "—",
        },
        {
            "area": "Agentic 工具动作训练",
            "status": "PARTIAL_RUN",
            "evidence": "官方任务 + normal/hack path，但策略仅输出四选一 action token",
            "completion": "改为 unrestricted JSON tool call，并训练 arguments 与多步状态",
        },
        {
            "area": "GRPO group rollout",
            "status": "PARTIAL_RUN",
            "evidence": f"{attempted} groups × 16 samples；{updated} 组可更新；{saturated} 组饱和",
            "completion": "在真实多轮轨迹上重跑并报告每 checkpoint 的 group variance",
        },
        {
            "area": "OOD / 泛化评测",
            "status": "PARTIAL_RUN",
            "evidence": "4 unseen tasks；accuracy 25% → 25%",
            "completion": "扩大为 covered / uncovered / unseen 分层且至少 3 seeds",
        },
        {
            "area": "失败轨迹与 saturation 诊断",
            "status": "PARTIAL_RUN",
            "evidence": f"24 组完整 JSONL；离线 PRM 诊断 {diagnostic_before} → {diagnostic_after} 激活组",
            "completion": "补 50/100/…/300 checkpoint 曲线与失败 taxonomy",
        },
        {
            "area": "一键环境与资产自举",
            "status": "PARTIAL_RUN",
            "evidence": "训练、验证、报告、PPT 可一键；model/data 仍假设已缓存",
            "completion": "加入断点下载、lockfile、disk/memory preflight 与 resume",
        },
        {
            "area": "BFCL-V4 Multi-Turn evaluator",
            "status": "CODE_READY",
            "evidence": "官方 bfcl-eval 2026.3.23；4 categories；base/adapter；partial/full commands",
            "completion": "执行 800 tasks 并保存 result/score/inference logs",
        },
        {
            "area": "TAU-2 evaluator",
            "status": "CODE_READY",
            "evidence": "官方 v0.2.0；airline/retail/telecom；smoke 与 Avg@4 full profile",
            "completion": "配置论文一致 user simulator 后执行全部 tasks × 4",
        },
        {
            "area": "PRM-Lite process reward",
            "status": "CODE_READY",
            "evidence": "15 条可解释规则、[-0.5,0.5] clip、逐规则 fired ledger、单测",
            "completion": "接到 multi-turn rollout 在线训练；做 anti-hacking audit",
        },
        {
            "area": "LATA / turn-discount credit",
            "status": "CODE_READY",
            "evidence": "A/L、A/√L、normalized turn-discount kernels + 单测",
            "completion": "必须在多 token、多 turn response mask 上训练才可验证",
        },
        {
            "area": "Vanilla / PRM / LATA / Joint 消融",
            "status": "CODE_READY",
            "evidence": "5 experiments × 3 seeds × 6 checkpoints 的固定矩阵",
            "completion": "运行训练与独立 evaluator；禁止用离线重打分代替精度结果",
        },
        {
            "area": "Long-horizon multi-turn RL environment",
            "status": "PARTIAL_RUN",
            "evidence": "状态化 user/tool sandbox 已运行 1 条完整轨迹；当前 GRPO 训练仍是单步 action-masked",
            "completion": "把 trajectory rollout 接入 response-token GRPO，扩到任务集、group sampling 与 checkpoint eval",
        },
        {
            "area": "论文规模 data flywheel",
            "status": "BLOCKED_RESOURCE",
            "evidence": "12 train vs paper ≈100K；未运行 Round 0–3",
            "completion": "Qwen3-235B synthesis/simulator/judge + multi-round expansion",
        },
        {
            "area": "论文同规模系统",
            "status": "BLOCKED_RESOURCE",
            "evidence": "本机 24GB；没有多机 veRL/SGLang 与 235B serving",
            "completion": "获得多卡 GPU 与同版本推理/训练栈",
        },
    ]
    completion_matrix = {
        "schema_version": 2,
        "reference_style": "https://github.com/qiqihezh/agentic-grpo-longhorizon",
        "status_definitions": {
            "COMPLETE": "实际运行并有独立证据",
            "PARTIAL_RUN": "实际运行，但规模或语义弱于论文",
            "CODE_READY": "代码、配置和 dry-run 已验证，尚无模型结果",
            "BLOCKED_RESOURCE": "需要当前 24GB Mac 之外的模型或算力",
        },
        "items": self_check,
    }
    (artifacts / "completion_matrix.json").write_text(
        json.dumps(completion_matrix, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    counts = {
        status: sum(item["status"] == status for item in self_check)
        for status in ("COMPLETE", "PARTIAL_RUN", "CODE_READY", "BLOCKED_RESOURCE")
    }

    result_rows = []
    for label, key in (
        ("训练任务", "train"),
        ("同任务 · prompt holdout", "holdout"),
        ("完全未见任务", "unseen"),
    ):
        item = metrics[key]
        delta = item["accuracy_after"] - item["accuracy_before"]
        probability_delta = (
            item["mean_correct_probability_after"]
            - item["mean_correct_probability_before"]
        )
        result_rows.append(
            "<tr>"
            f"<td><strong>{label}</strong></td>"
            f"<td>{_pct(item['accuracy_before'])}</td>"
            f"<td>{_pct(item['accuracy_after'])}</td>"
            f"<td class='{'positive' if delta > 0 else 'neutral'}'>{delta * 100:+.1f} pt</td>"
            f"<td>{probability_delta:+.4f}</td>"
            "</tr>"
        )
    result_rows_html = "".join(result_rows)
    self_check_html = _status_rows(self_check)
    verification_html = _verification_rows(verification["checks"])

    ablation_rows = "".join(
        "<tr>"
        f"<td><strong>{html.escape(item['name'])}</strong></td>"
        f"<td>{item['process_reward_weight']}</td>"
        f"<td><code>{html.escape(item['credit_assignment'])}</code></td>"
        f"<td>{html.escape(item['hypothesis'])}</td>"
        "<td><span class='status code-ready'>CODE_READY</span></td>"
        "</tr>"
        for item in ablation_matrix.get("experiments", [])
    )

    benchmark_scope = benchmark_manifest.get("scope", {})
    report_dir = project_dir / "agenticqwen_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    body = rf"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AgenticQwen Long-Horizon RL Lab：完整项目教程</title>
<meta name="description" content="从真实多轮工具轨迹、状态机与 reward，到 Qwen3-8B LoRA-GRPO、失败诊断、消融、评测与面试表达的完整 Agentic RL 项目文档。">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700;900&family=Source+Code+Pro:wght@400;600&family=STIX+Two+Text:ital,wght@0,400;0,600;0,700;1,400&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js" onload="renderMathInElement(document.body,{{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}],throwOnError:false}});"></script>
<style>
:root{{--paper:#fff;--ink:#171717;--muted:#686868;--line:#d7d7d7;--soft:#f6f6f4;--soft2:#efefeb;--accent:#9d261f;--accent-soft:#fbf0ee;--green:#17653a;--amber:#8a5a00;--blue:#275d84;--purple:#6e3f75}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.86 'STIX Two Text','Noto Serif SC','Songti SC','Times New Roman',serif}}a{{color:var(--accent);text-decoration-thickness:1px;text-underline-offset:3px}}code,pre{{font-family:'Source Code Pro','SFMono-Regular',monospace}}code{{font-size:.88em;background:#f0f0ed;padding:2px 5px;overflow-wrap:anywhere}}pre{{margin:18px 0;padding:18px 20px;overflow:auto;background:#191919;color:#f4f4f2;border-radius:0;font-size:12px;line-height:1.65}}pre code{{padding:0;background:none;color:inherit}}
.toc{{position:fixed;left:0;top:0;width:260px;height:100vh;overflow:auto;padding:28px 20px 40px;background:#fafaf8;border-right:1px solid var(--line)}}.toc .wordmark{{font-weight:900;font-size:17px;letter-spacing:-.02em;color:#111;text-decoration:none}}.toc .wordmark span{{color:var(--accent)}}.toc .edition{{margin:4px 0 24px;color:var(--muted);font:10px/1.4 'Source Code Pro',monospace;text-transform:uppercase;letter-spacing:.09em}}.toc h2{{margin:22px 8px 8px;font:700 10px/1.3 'Source Code Pro',monospace;letter-spacing:.12em;text-transform:uppercase;color:#888}}.toc a.nav{{display:block;padding:5px 9px;border-left:2px solid transparent;color:#555;text-decoration:none;font-size:11.5px;line-height:1.45}}.toc a.nav:hover{{border-left-color:var(--accent);background:#f0f0ec;color:#111}}.toc .legend{{margin-top:24px;padding-top:16px;border-top:1px solid var(--line);font-size:10.5px;color:#666}}.toc .legend div+div{{margin-top:6px}}
.main{{margin-left:260px;max-width:1040px;padding:56px 58px 120px}}.eyebrow{{font:600 11px/1.4 'Source Code Pro',monospace;color:var(--accent);letter-spacing:.1em;text-transform:uppercase}}h1{{margin:9px 0 10px;max-width:850px;font-size:42px;line-height:1.2;letter-spacing:-.035em}}.deck{{max-width:850px;margin:0 0 20px;color:#4c4c4c;font-size:18px;line-height:1.72}}.meta{{display:flex;gap:8px;flex-wrap:wrap;margin:20px 0 30px}}.meta span{{padding:3px 8px;border:1px solid var(--line);font:10.5px/1.5 'Source Code Pro',monospace;color:#555;background:#fff}}
h2{{margin:54px 0 16px;padding:0 0 7px;border-bottom:2px solid #171717;font-size:25px;line-height:1.35;letter-spacing:-.02em}}h3{{margin:30px 0 10px;font-size:18px;line-height:1.45}}h4{{margin:22px 0 7px;font-size:15px}}p{{margin:0 0 14px;text-align:justify}}ul,ol{{padding-left:1.4em}}li+li{{margin-top:5px}}.lede{{font-size:16.5px}}.small{{font-size:12px;color:var(--muted)}}
.result-banner{{margin:30px 0;border-top:3px solid var(--accent);border-bottom:1px solid var(--line);padding:18px 0 20px;display:grid;grid-template-columns:1.45fr repeat(3,1fr);gap:18px}}.result-banner .claim{{font-size:16px;line-height:1.6}}.big-number{{display:block;font-size:31px;font-weight:900;line-height:1.15;letter-spacing:-.03em}}.number-label{{display:block;margin-top:5px;color:var(--muted);font-size:11px;line-height:1.45}}.metric-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:18px 0 26px}}.metric{{border:1px solid var(--line);padding:14px 15px;min-height:112px;background:#fff}}.metric .value{{font-size:27px;font-weight:800;line-height:1.2}}.metric .label{{margin-top:7px;color:var(--muted);font-size:11.5px;line-height:1.45}}
.callout{{margin:18px 0;padding:15px 18px;border:1px solid var(--line);border-left:4px solid #222;background:#fafafa}}.callout strong{{font-weight:800}}.callout.negative{{border-left-color:var(--accent);background:var(--accent-soft)}}.callout.positive{{border-left-color:var(--green);background:#f3f8f5}}.callout.code{{border-left-color:var(--blue);background:#f3f7fa}}.callout.warning{{border-left-color:var(--amber);background:#fbf7ec}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:18px 0}}.three-col{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0}}.card{{border:1px solid var(--line);padding:17px;background:#fff}}.card h3{{margin-top:0}}.card .kicker{{font:600 10px/1.4 'Source Code Pro',monospace;color:var(--accent);letter-spacing:.08em;text-transform:uppercase}}.card p:last-child{{margin-bottom:0}}
.figure-box{{margin:22px 0;padding:12px;border:1px solid var(--line);background:#fff;text-align:center}}.figure-box img{{display:block;max-width:100%;height:auto;margin:auto}}.caption{{max-width:820px;margin:9px auto 1px;color:#555;font-size:11.5px;line-height:1.65;font-style:italic;text-align:left}}
table{{width:100%;border-collapse:collapse;margin:16px 0 26px;font-size:12.5px;line-height:1.55}}th{{padding:8px 9px;border-top:2px solid #111;border-bottom:1px solid #111;background:#f3f3f0;text-align:left;vertical-align:bottom}}td{{padding:8px 9px;border-bottom:1px solid #ddd;vertical-align:top}}tbody tr:last-child td{{border-bottom:2px solid #111}}.positive{{color:var(--green);font-weight:800}}.negative-text{{color:var(--accent);font-weight:800}}.neutral{{color:#666;font-weight:700}}.paper-row{{background:#f7f7f5}}.our-row{{background:#fcf2ef}}
.status{{display:inline-block;padding:2px 6px;border:1px solid currentColor;font:700 9px/1.35 'Source Code Pro',monospace;letter-spacing:.03em;white-space:nowrap}}.status.complete,.status.pass{{color:var(--green)}}.status.partial-run{{color:var(--amber)}}.status.code-ready{{color:var(--blue)}}.status.blocked-resource,.status.fail{{color:var(--accent)}}
.pipeline{{display:grid;grid-template-columns:repeat(4,1fr);margin:18px 0 22px}}.pipeline>div{{position:relative;border:1px solid var(--line);padding:15px;min-height:135px}}.pipeline>div+div{{border-left:0}}.pipeline .step{{font:600 10px/1.4 'Source Code Pro',monospace;color:var(--accent)}}.pipeline b{{display:block;margin:5px 0;font-size:16px}}.pipeline p{{margin:0;color:var(--muted);font-size:11.5px;text-align:left}}.equation-note{{margin:-4px 0 18px;color:#555;font-size:12px}}.bars{{border:1px solid var(--line);padding:16px 18px;margin:18px 0}}.bar-row{{display:grid;grid-template-columns:185px 1fr 86px;gap:12px;align-items:center;margin:10px 0;font-size:11.5px}}.bar-track{{height:10px;background:#ecece8}}.bar{{height:100%;background:#999}}.bar.after{{background:var(--accent)}}.bar.active{{background:var(--green)}}
.tree{{margin:18px 0;border:1px solid var(--line);padding:18px;background:#fbfbfa;font:12px/1.85 'Source Code Pro',monospace;white-space:pre-wrap}}details{{margin:12px 0;border:1px solid var(--line);padding:10px 13px}}summary{{cursor:pointer;font-weight:700}}.footer{{margin-top:64px;padding-top:20px;border-top:2px solid #111;color:var(--muted);font-size:11px}}@media(max-width:980px){{.toc{{position:static;width:auto;height:auto;border-right:0;border-bottom:1px solid var(--line)}}.main{{margin-left:0;padding:34px 24px 80px}}.toc .nav-section,.toc .legend{{display:none}}}}@media(max-width:720px){{h1{{font-size:31px}}.result-banner,.metric-grid,.two-col,.three-col,.pipeline{{grid-template-columns:1fr 1fr}}.pipeline>div+div{{border-left:1px solid var(--line)}}.bar-row{{grid-template-columns:120px 1fr 72px}}table{{display:block;overflow-x:auto;white-space:nowrap}}}}@media(max-width:480px){{.result-banner,.metric-grid,.two-col,.three-col,.pipeline{{grid-template-columns:1fr}}}}
.learning-path{{counter-reset:path;display:grid;grid-template-columns:repeat(5,1fr);margin:20px 0 28px;border-top:2px solid #111;border-bottom:1px solid var(--line)}}.learning-path>div{{padding:14px 12px;border-right:1px solid var(--line);min-height:112px}}.learning-path>div:last-child{{border-right:0}}.learning-path b{{display:block;margin:4px 0;font-size:14px}}.learning-path span{{font:10px/1.4 'Source Code Pro',monospace;color:var(--accent)}}.learning-path p{{font-size:11px;line-height:1.55;color:var(--muted);text-align:left}}
.architecture{{display:grid;grid-template-columns:repeat(5,1fr);gap:0;margin:20px 0}}.architecture>div{{position:relative;border:1px solid var(--line);padding:15px;min-height:148px}}.architecture>div+div{{border-left:0}}.architecture>div:not(:last-child)::after{{content:'→';position:absolute;right:-10px;top:54px;z-index:2;background:white;color:var(--accent);font-weight:900}}.architecture b{{display:block;font-size:15px;margin:7px 0}}.architecture p{{font-size:11px;line-height:1.55;color:var(--muted);text-align:left}}.architecture .step{{font:600 9px/1.4 'Source Code Pro',monospace;color:var(--accent)}}
.trajectory-controls{{display:flex;gap:6px;flex-wrap:wrap;margin:16px 0}}.trajectory-controls button{{appearance:none;border:1px solid #999;background:white;color:#333;padding:5px 10px;font:600 10px/1.4 'Source Code Pro',monospace;cursor:pointer}}.trajectory-controls button.active,.trajectory-controls button:hover{{background:#171717;color:white;border-color:#171717}}.trajectory{{position:relative;margin:10px 0 28px}}.trajectory::before{{content:'';position:absolute;left:25px;top:0;bottom:0;border-left:1px solid #bbb}}.trajectory-event{{position:relative;display:grid;grid-template-columns:52px 1fr;gap:12px;margin:0 0 9px}}.trajectory-event[hidden]{{display:none}}.event-index{{position:relative;z-index:1;width:34px;height:34px;margin-left:8px;border:1px solid #777;background:white;display:flex;align-items:center;justify-content:center;font:600 10px/1 'Source Code Pro',monospace}}.role-user .event-index{{border-color:var(--blue);color:var(--blue)}}.role-assistant .event-index{{border-color:var(--accent);color:var(--accent)}}.role-tool .event-index{{border-color:var(--green);color:var(--green)}}.event-main{{border:1px solid var(--line);padding:12px 14px;background:#fff}}.event-main header{{display:flex;align-items:center;gap:7px;flex-wrap:wrap;margin-bottom:7px}}.event-role,.event-type,.event-tool,.event-reward{{font:600 9px/1.35 'Source Code Pro',monospace;text-transform:uppercase;letter-spacing:.04em}}.event-role{{color:#111}}.event-type{{color:#777}}.event-tool{{color:var(--blue)}}.event-reward{{margin-left:auto;color:var(--green)}}.event-content{{font-size:12.5px;line-height:1.65}}.state-delta{{display:flex;gap:7px;flex-wrap:wrap;margin-top:9px;padding-top:8px;border-top:1px dashed #ccc;font-size:10px;color:#555}}.state-delta b{{font-family:'Source Code Pro',monospace;color:#111}}.state-delta span{{background:#f4f4f1;padding:1px 5px}}
.resume-block{{border-top:3px solid var(--accent);border-bottom:1px solid var(--line);padding:18px 0;margin:18px 0}}.resume-block p{{font-size:16px;line-height:1.72}}.qa dt{{font-weight:800;margin-top:15px}}.qa dd{{margin:5px 0 0;padding-left:18px;border-left:2px solid var(--line);color:#555}}.claim-pill{{display:inline-block;border-bottom:2px solid var(--green);font-weight:800}}.formula-walkthrough{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:18px 0}}.formula-walkthrough>div{{border-top:2px solid #111;padding:13px 0}}.formula-walkthrough b{{display:block;margin-bottom:5px}}.formula-walkthrough p{{font-size:12px;color:var(--muted);text-align:left}}
@media(max-width:980px){{.learning-path,.architecture{{grid-template-columns:1fr 1fr}}.architecture>div+div{{border-left:1px solid var(--line)}}.architecture>div::after{{display:none}}}}@media(max-width:720px){{.learning-path,.architecture,.formula-walkthrough{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<aside class="toc">
  <a class="wordmark" href="#top">AgenticQwen<span>/</span>repro</a>
  <div class="edition">Long-horizon learning edition · v3</div>
  <div class="nav-section">
    <h2>Start here</h2>
    <a class="nav" href="#project-card">01 · Project Card</a>
    <a class="nav" href="#primer">02 · Agentic RL Primer</a>
    <a class="nav" href="#architecture">03 · System Architecture</a>
    <a class="nav" href="#trajectory">04 · Full Trajectory</a>
    <a class="nav" href="#reward-state">05 · Reward & State</a>
    <a class="nav" href="#code-map">06 · Code Map</a>
    <h2>Research</h2>
    <a class="nav" href="#key-results">07 · Training Results</a>
    <a class="nav" href="#paper-results">08 · Paper vs Local</a>
    <a class="nav" href="#diagnosis">09 · Failure Diagnosis</a>
    <a class="nav" href="#paper-method">10 · Paper Method</a>
    <a class="nav" href="#implementation">11 · GRPO Implementation</a>
    <a class="nav" href="#ablations">12 · Ablation Matrix</a>
    <h2>Evaluation</h2>
    <a class="nav" href="#benchmarks">13 · Benchmark Code</a>
    <a class="nav" href="#audit">14 · Verification</a>
    <a class="nav" href="#completion">15 · Completion Audit</a>
    <h2>Interview</h2>
    <a class="nav" href="#structure">16 · Project Structure</a>
    <a class="nav" href="#run">17 · One-click Run</a>
    <a class="nav" href="#interview">18 · Resume & Interview</a>
    <a class="nav" href="#limitations">19 · Limitations</a>
    <a class="nav" href="#sources">20 · Sources</a>
  </div>
  <div class="legend">
    <div><span class="status complete">COMPLETE</span> 已运行且有证据</div>
    <div><span class="status partial-run">PARTIAL_RUN</span> 已运行但缩放</div>
    <div><span class="status code-ready">CODE_READY</span> 可执行未产出指标</div>
    <div><span class="status blocked-resource">BLOCKED</span> 需要额外算力</div>
  </div>
</aside>
<main class="main" id="top">
<div class="eyebrow">Environment → Trajectory → Reward → GRPO → Verify → Interview</div>
<h1>AgenticQwen Long-Horizon RL Lab</h1>
<p class="deck">一份从零理解 Agentic RL 的完整项目文档：先拆解一条真实 Qwen3.7-Flash 多轮工具轨迹，再解释状态机、reward、GRPO、group saturation 和 credit assignment；最后回到 Qwen3-8B 本地训练、论文差距、实验设计与面试表达。</p>
<div class="meta">
  <span>paper: arXiv 2604.21590</span><span>host: Apple M4 Pro · 24 GB</span>
  <span>trajectory policy: Qwen3.7-Flash API</span><span>training policy: Qwen3-8B MLX 4-bit</span>
  <span>reference style: agentic-grpo-longhorizon</span>
</div>

<section class="result-banner">
  <div class="claim"><strong>结论先行。</strong>项目已有两条真实证据链：一条是 8-turn 状态化工具轨迹，另一条是 Qwen3-8B LoRA-GRPO 参数更新。它仍不是论文级 long-horizon RL：多轮轨迹尚未进入在线 GRPO，官方 benchmark 尚未运行。</div>
  <div><span class="big-number">{trajectory_runtime['agent_turns']} turns</span><span class="number-label">Qwen3.7-Flash 真实轨迹<br>{trajectory_runtime['tool_calls']} tool calls · 9/9 PASS</span></div>
  <div><span class="big-number">+16.7 pt</span><span class="number-label">Qwen3-8B prompt holdout<br>25.0% → 41.7%</span></div>
  <div><span class="big-number">83.3%</span><span class="number-label">zero-variance groups<br>20 / 24</span></div>
</section>

<h2 id="project-card">01 · Project Card：这个项目到底做了什么</h2>
<div class="two-col">
  <section class="card"><div class="kicker">Problem</div><h3>让小模型学会安全地完成多轮工具任务</h3><p>Agent 不只要“回答对”，还要在状态变化、权限约束和用户确认中选择正确工具、填写参数、恢复错误，并把长期结果归因到前面的决策。</p></section>
  <section class="card"><div class="kicker">What I built</div><h3>环境、轨迹、训练、验证四层闭环</h3><p>状态化客服 sandbox、DashScope function-calling policy、process reward ledger、独立 verifier、本地 Qwen3-8B LoRA-GRPO、PRM/LATA 消融框架与官方 evaluator 接口。</p></section>
</div>
<table><thead><tr><th>Layer</th><th>Actual implementation</th><th>Evidence</th><th>Current boundary</th></tr></thead><tbody>
<tr><td>Long-horizon inference</td><td>Qwen3.7-Flash 驱动完整 JSON tool loop</td><td>{trajectory_runtime['agent_turns']} turns / {trajectory_runtime['tool_calls']} calls / 9 checks</td><td>只有 1 个确定性任务</td></tr>
<tr><td>Stateful environment</td><td>identity、order、charge、policy、confirmation、refund 状态</td><td>逐事件 before/after snapshot</td><td>mini customer-support domain</td></tr>
<tr><td>Agentic RL training</td><td>Qwen3-8B MLX 4-bit action-masked LoRA-GRPO</td><td>LoRA norm、hash、fresh replay</td><td>训练尚未消费多轮 trajectory</td></tr>
<tr><td>Research diagnostics</td><td>group saturation、PRM-Lite、LATA、Turn-Discount</td><td>20/24 saturated；offline PRM 4→4</td><td>机制代码就绪，在线消融未运行</td></tr>
</tbody></table>
<div class="resume-block"><div class="eyebrow">Resume-ready claim</div><p>基于 Qwen3-8B/MLX 构建可审计 Agentic GRPO 实验系统，并实现 Qwen3.7-Flash 驱动的状态化多轮 function-calling 环境；完成 8-turn 安全退款轨迹、独立 verifier、checkpoint 重放与 group saturation 诊断。</p></div>

<h3>建议阅读路径</h3>
<div class="learning-path">
  <div><span>STEP 1</span><b>先看轨迹</b><p>知道 Agent 实际在做什么。</p></div>
  <div><span>STEP 2</span><b>再看状态与 reward</b><p>理解环境如何判对错。</p></div>
  <div><span>STEP 3</span><b>再看 GRPO</b><p>理解多条轨迹如何产生梯度。</p></div>
  <div><span>STEP 4</span><b>再看失败诊断</b><p>理解 saturation 与 credit dilution。</p></div>
  <div><span>STEP 5</span><b>最后讲面试</b><p>把工程选择变成可信叙事。</p></div>
</div>

<h2 id="primer">02 · Agentic RL Primer：从一次回答到一条轨迹</h2>
<p class="lede">普通语言模型优化一个回答；Agentic RL 优化的是一个模型在环境中连续决策后得到的整条 trajectory。</p>
<div class="callout">$$\tau=(s_0,o_0,a_0,r_0,s_1,o_1,a_1,r_1,\dots,s_T),\qquad R(\tau)=R_{{outcome}}+\lambda\sum_t r_{{process,t}}$$</div>
<p><code>s_t</code> 是环境真实状态，<code>o_t</code> 是 Agent 当前能看到的 observation，<code>a_t</code> 是文本或工具动作。工具执行改变状态，最终 verifier 给 outcome reward；过程规则可以在关键 read、confirmation、recovery 或 unsafe write 上提供更密的信号。</p>
<table><thead><tr><th>Concept</th><th>在本项目里的具体含义</th><th>新人常见误区</th></tr></thead><tbody>
<tr><td>State</td><td>是否验证身份、读过付款/政策、用户是否确认、退款是否落库</td><td>把聊天历史当成全部环境状态</td></tr>
<tr><td>Action</td><td>自然语言追问或带参数的 JSON function call</td><td>只评估函数名，不评 arguments 与执行顺序</td></tr>
<tr><td>Observation</td><td>用户消息、tool result、error code</td><td>让模型看到 verifier 的隐藏答案</td></tr>
<tr><td>Trajectory</td><td>3 条用户消息、5 次工具调用、完整状态变化</td><td>把单次 tool call 称为 long-horizon</td></tr>
<tr><td>Reward</td><td>outcome 1.0 + 过程分 0.85</td><td>离线重打分冒充训练提升</td></tr>
<tr><td>Credit assignment</td><td>把最终好坏传回多 turn、多 token 的动作</td><td>默认每个 token 获得同样强度信号</td></tr>
</tbody></table>
<div class="formula-walkthrough">
  <div><b>SFT 学什么</b><p>给定标准轨迹，最大化专家动作 token 的似然；它不会直接比较同一任务的多个失败/成功 rollout。</p></div>
  <div><b>Agentic RL 学什么</b><p>先采样多条完整轨迹，再用环境 reward 比较它们，让更安全、更有效的决策概率上升。</p></div>
</div>

<h2 id="architecture">03 · System Architecture：谁负责决策，谁负责判定</h2>
<div class="architecture">
  <div><span class="step">USER SIM</span><b>提供目标与确认</b><p>隐藏用户档案，只在 Agent 主动询问时提供身份信息。</p></div>
  <div><span class="step">POLICY</span><b>Qwen 选择动作</b><p>生成追问、工具名与 JSON arguments，不直接改环境。</p></div>
  <div><span class="step">ENV / TOOLS</span><b>执行与状态转移</b><p>校验 schema、权限、先读后写、幂等键和退款对象。</p></div>
  <div><span class="step">VERIFIER</span><b>独立判定结果</b><p>读取最终状态，不相信 Agent 自述“已经成功”。</p></div>
  <div><span class="step">TRAINER</span><b>将 reward 变成梯度</b><p>当前 Qwen3-8B trainer 已运行，但还只消费单步 action。</p></div>
</div>
<div class="callout warning"><strong>关键责任边界：</strong>LLM 只能提出工具调用字符串；真正的写操作由环境执行，成功与否由 verifier 从状态判断。把“模型说成功”当成功，是 Agent 项目最危险的伪闭环。</div>

<h2 id="trajectory">04 · Full Trajectory：逐事件回放真实多轮交互</h2>
<div class="metric-grid">
  <div class="metric"><div class="value">{trajectory_runtime['agent_turns']}</div><div class="label">Agent turns<br>每次均调用真实 Qwen API</div></div>
  <div class="metric"><div class="value">{trajectory_runtime['tool_calls']}</div><div class="label">Function calls<br>lookup → refund</div></div>
  <div class="metric"><div class="value">{trajectory_runtime['events']}</div><div class="label">Audited events<br>message / call / result</div></div>
  <div class="metric"><div class="value">{trajectory_runtime['api_latency_seconds']:.2f}s</div><div class="label">API latency<br>{trajectory_runtime['usage'].get('total_tokens', 0):,} total tokens</div></div>
</div>
<p>任务：用户认为机械键盘订单重复扣款，但不知道订单号。Agent 必须先验证身份，再定位订单和重复 charge，读退款政策，得到明确确认后只退重复的一笔。</p>
<div class="trajectory-controls" aria-label="Filter trajectory events">
  <button class="active" type="button" data-filter="all">ALL</button><button type="button" data-filter="user">USER</button><button type="button" data-filter="assistant">ASSISTANT</button><button type="button" data-filter="tool">TOOL</button>
</div>
<div class="trajectory" id="trajectory-viewer">{trajectory_events_html}</div>
<div class="callout positive"><strong>为什么这条轨迹有效：</strong>工具顺序为 <code>lookup_customer → list_orders → get_payment_history → get_refund_policy → create_refund</code>；写操作发生在用户明确确认之后，退款精确指向 <code>CHG-9002</code>，没有越权、重复退款或金额漂移。</div>

<h2 id="reward-state">05 · Reward & State：环境如何把“做对了”变成数字</h2>
<h3>Process reward ledger</h3>
<table><thead><tr><th>Tool turn</th><th>Event</th><th>Reward</th><th>Why</th></tr></thead><tbody>{trajectory_reward_html}</tbody></table>
<div class="callout">$$R(\tau)=1.0+0.3\times0.85=1.255$$</div>
<p class="equation-note">Outcome reward 只看关键状态是否真的完成；process reward 记录身份验证、必要读取、用户确认和正确写操作。这里的 1.255 是轨迹诊断分，不是训练曲线，也没有做范围归一化。</p>
<h3>Independent verifier</h3>
<table><thead><tr><th>Check</th><th>Status</th><th>Evidence meaning</th></tr></thead><tbody>{trajectory_check_html}</tbody></table>
<div class="two-col">
  <section class="card"><div class="kicker">Observable state</div><h3>Agent 看到了什么</h3><p>用户输入、customer/order/charge/policy tool results 和退款回执。</p></section>
  <section class="card"><div class="kicker">Hidden truth</div><h3>Verifier 检查什么</h3><p>最终 refund_id、charge_id、amount、confirmation flag、tool error 与 unsafe attempt。</p></section>
</div>

<h2 id="code-map">06 · Code Map：从消息循环定位到实现</h2>
<table><thead><tr><th>File</th><th>Responsibility</th><th>Interview point</th></tr></thead><tbody>
<tr><td><a href="../src/agentic_repro/long_horizon_env.py"><code>long_horizon_env.py</code></a></td><td>状态机、5 个工具、权限/确认/幂等约束、process reward、verifier</td><td>为什么环境必须独立于 policy model</td></tr>
<tr><td><a href="../src/agentic_repro/dashscope_policy.py"><code>dashscope_policy.py</code></a></td><td>Qwen function-calling client、消息格式、usage/latency、secret boundary</td><td>模型只建议动作，不能直接产生 side effect</td></tr>
<tr><td><a href="../src/agentic_repro/trajectory_runner.py"><code>trajectory_runner.py</code></a></td><td>多轮消息循环、tool dispatch、事件/状态落盘、终止与审计</td><td>如何避免无限循环和“假成功”</td></tr>
<tr><td><a href="../src/agentic_repro/real_grpo.py"><code>real_grpo.py</code></a></td><td>Qwen3-8B MLX LoRA、group rollout、clipped objective</td><td>当前 trainer 与多轮环境之间缺哪一层</td></tr>
<tr><td><a href="../src/agentic_repro/ablations.py"><code>ablations.py</code></a></td><td>PRM-Lite、group advantage、LATA/Turn-Discount kernels</td><td>如何处理 sparse reward 与 length dilution</td></tr>
</tbody></table>
<pre><code># 真实多轮轨迹；key 只通过环境变量注入
DASHSCOPE_API_KEY=&lt;rotated-key&gt; ./run_long_horizon_demo.sh

# 产物
artifacts/long_horizon/trajectory_qwen3.7_flash.json
artifacts/long_horizon/trajectory_qwen3.7_flash.md</code></pre>
<div class="callout negative"><strong>下一条代码链：</strong>把 runner 产出的多个 trajectory 编组，计算 outcome + process reward，再对所有 assistant response tokens 构造 mask 和 advantage；这一步完成后，项目才从“真实多轮 inference + 单步 RL”升级为“真实多轮 Agentic RL training”。</div>

<h2 id="key-results">07 · Training Results：Qwen3-8B 本地更新</h2>
<p class="lede">参考 <a href="https://github.com/qiqihezh/agentic-grpo-longhorizon">agentic-grpo-longhorizon</a> 的叙事方式，先把最重要的结果、负结果和证据放在最前面。</p>
<div class="metric-grid">
  <div class="metric"><div class="value">8.19B</div><div class="label">实际加载参数量；官方 Qwen3-8B MLX 4-bit</div></div>
  <div class="metric"><div class="value">37,401</div><div class="label">官方 AgenticQwen-Data 总行数；固定 seed 抽样</div></div>
  <div class="metric"><div class="value">{summary['training']['optimizer_steps']}</div><div class="label">真实 optimizer steps；不是模拟数值更新</div></div>
  <div class="metric"><div class="value">{verification['summary']['pass']}/{verification['summary']['pass'] + verification['summary']['fail']}</div><div class="label">fresh-process verifier PASS；checkpoint 从磁盘重载</div></div>
</div>
<table>
<thead><tr><th>Evaluation split</th><th>Base</th><th>After adapter</th><th>Δ accuracy</th><th>Δ correct-action probability</th></tr></thead>
<tbody>{result_rows_html}</tbody>
</table>
<div class="callout positive"><strong>证实了什么：</strong>同一批任务在另一种提示语下仍有提升，LoRA adapter 不是空文件，训练后的行为变化可由第二个进程逐题重放。</div>
<div class="callout negative"><strong>没有证实什么：</strong>完全未见任务没有提升，样本量只有 4；因此不能声称获得了论文中的跨域 agentic generalization，更不能拿这组数字与论文的 TAU-2 / BFCL 平均分比较。</div>

<h3>Training progression 与 reward saturation</h3>
<div class="bars">
  <div class="bar-row"><span>Train · base</span><div class="bar-track"><div class="bar" style="width:{train['accuracy_before']*100:.1f}%"></div></div><b>{_pct(train['accuracy_before'])}</b></div>
  <div class="bar-row"><span>Train · adapter</span><div class="bar-track"><div class="bar after" style="width:{train['accuracy_after']*100:.1f}%"></div></div><b>{_pct(train['accuracy_after'])}</b></div>
  <div class="bar-row"><span>Prompt holdout · base</span><div class="bar-track"><div class="bar" style="width:{holdout['accuracy_before']*100:.1f}%"></div></div><b>{_pct(holdout['accuracy_before'])}</b></div>
  <div class="bar-row"><span>Prompt holdout · adapter</span><div class="bar-track"><div class="bar after" style="width:{holdout['accuracy_after']*100:.1f}%"></div></div><b>{_pct(holdout['accuracy_after'])}</b></div>
  <div class="bar-row"><span>Groups with gradient</span><div class="bar-track"><div class="bar active" style="width:{updated/attempted*100:.1f}%"></div></div><b>{updated}/{attempted}</b></div>
  <div class="bar-row"><span>Zero-variance groups</span><div class="bar-track"><div class="bar after" style="width:{saturation_rate*100:.1f}%"></div></div><b>{saturated}/{attempted}</b></div>
</div>

<h2 id="paper-results">08 · Paper Results vs Local Evidence</h2>
<p>论文表 1 报告的是完整交互 benchmark；本地表格报告的是官方训练数据上的四选一 next-action 诊断。两者任务、动作空间、reward、用户模拟器和统计规模都不同，必须分层阅读。</p>
<table>
<thead><tr><th>Model（论文值）</th><th>TAU Airline</th><th>Telecom</th><th>Retail</th><th>BFCL Base</th><th>Miss Func</th><th>Miss Param</th><th>Long Ctx</th><th>Avg.</th></tr></thead>
<tbody>
<tr class="paper-row"><td>Qwen3-235B-A22B</td><td>47.5</td><td>53.2</td><td>68.0</td><td>58.5</td><td>47.5</td><td>35.0</td><td>54.0</td><td>52.0</td></tr>
<tr class="paper-row"><td>Qwen3-30B-A3B</td><td>32.0</td><td>31.6</td><td>55.3</td><td>47.0</td><td>14.0</td><td>28.0</td><td>45.5</td><td>36.2</td></tr>
<tr class="paper-row"><td>Qwen3-8B</td><td>14.5</td><td>7.9</td><td>31.6</td><td>35.5</td><td>35.0</td><td>20.5</td><td>21.5</td><td>23.8</td></tr>
<tr class="our-row"><td><strong>AgenticQwen-8B</strong></td><td><strong>40.5</strong></td><td><strong>53.5</strong></td><td><strong>60.3</strong></td><td><strong>56.0</strong></td><td><strong>47.5</strong></td><td><strong>33.5</strong></td><td><strong>40.5</strong></td><td><strong>47.4</strong></td></tr>
<tr class="our-row"><td><strong>AgenticQwen-30B-A3B</strong></td><td><strong>42.0</strong></td><td><strong>52.6</strong></td><td><strong>60.5</strong></td><td><strong>60.0</strong></td><td><strong>52.0</strong></td><td><strong>29.0</strong></td><td><strong>55.5</strong></td><td><strong>50.2</strong></td></tr>
</tbody>
</table>
<div class="callout warning"><strong>论文主张：</strong>AgenticQwen-8B 从 23.8 提升到 47.4，+23.6 points，接近 235B baseline 的 52.0。<strong>本项目状态：</strong>尚无 TAU-2 / BFCL 实跑分数，不能验证或反驳这项主张。</div>
<div class="figure-box"><img src="images/figure2_flywheel_results.png" alt="Paper-reported data flywheel progression"><div class="caption"><b>Paper Figure 2.</b> 论文报告 Round 0→3 在七类 benchmark 上持续上升。这张图是论文证据，不是本机生成的训练曲线。</div></div>

<h2 id="diagnosis">09 · Problem & Failure Diagnosis</h2>
<div class="three-col">
  <section class="card"><div class="kicker">Root cause 1</div><h3>Group reward saturation</h3><p>binary outcome reward 让同组 16 条 rollout 容易全对或全错，组内标准差为 0，GRPO advantage 全为 0。本次真实观测为 20/24 组。</p></section>
  <section class="card"><div class="kicker">Root cause 2</div><h3>Training / environment gap</h3><p>新轨迹层已经覆盖 JSON arguments、tool response 与 confirmation；但 Qwen3-8B trainer 仍只在四个候选动作中选一个数字，尚未学习这些多轮能力。</p></section>
  <section class="card"><div class="kicker">Root cause 3</div><h3>Generalization deficit</h3><p>同任务 prompt holdout 提升，但 unseen 不变。当前信号更像任务/工具映射记忆，而不是可迁移的状态化 agent policy。</p></section>
</div>
<h3>PRM-Lite 离线反事实诊断：负结果</h3>
<p>项目实现了 8 条 penalty 与 7 条 bonus，并在已保存的 24 个 rollout group 上重新计算 <code>outcome + 0.3 × process_score</code>。这一步只改变 reward，不更新模型。</p>
<table><thead><tr><th>Reward</th><th>Non-zero variance groups</th><th>Zero-variance rate</th><th>Interpretation</th></tr></thead><tbody>
<tr><td>Outcome only</td><td>{diagnostic_before}/{attempted}</td><td>{_pct(1-diagnostic_before/attempted)}</td><td>原始训练观测</td></tr>
<tr><td>Outcome + PRM-Lite（offline）</td><td>{diagnostic_after}/{attempted}</td><td>{_pct(1-diagnostic_after/attempted)}</td><td>没有新增激活组</td></tr>
</tbody></table>
<div class="callout negative"><strong>为什么没有改善：</strong>单步四选一轨迹没有错误恢复、依赖链、不同 read 操作、长 reasoning 等可区分过程事件。这个负结果证明 PRM-Lite 必须进入真实 multi-turn rollout 才有可检验意义；离线重打分不能冒充 Joint ablation。</div>

<h2 id="paper-method">10 · Paper Method: Dual Data Flywheels</h2>
<p>论文不是只提出一个 loss，而是一套“训练—发现失败—合成更难数据—再训练”的 curriculum。reasoning 与 agentic 两条 flywheel 解决不同类型的数据同质化。</p>
<div class="figure-box"><img src="images/figure1_dual_flywheel.png" alt="AgenticQwen dual data flywheels"><div class="caption"><b>Paper Figure 1.</b> Reasoning flywheel 从错误样本生成结构/语境更难的问题；agentic flywheel 将线性 workflow 扩展为行为树，再把分支反演成新的环境、用户和 agent instruction。</div></div>
<div class="two-col">
  <section class="card"><div class="kicker">Reasoning flywheel</div><h3>错误驱动的可验证扩展</h3><ol><li>收集模型失败题。</li><li>Qwen3-235B 用 Self-Instruct 改数值、加约束、叠概念。</li><li>注入 persona，把抽象题改写到物理/化学等语境。</li><li>强模型解三次，仅保留最终答案一致的样本。</li></ol></section>
  <section class="card"><div class="kicker">Agentic flywheel</div><h3>线性路径 → 行为树</h3><ol><li>用 SynthAgent 单路径任务初始化。</li><li>根据 rollout 插入环境条件分支。</li><li>branch-to-task inversion 反推触发分支的状态、用户请求与 SOP。</li><li>mock user 注入误导路径，测试 policy compliance。</li></ol></section>
</div>
<div class="tree">Round k
├─ RL_Train(πθ, Tₖ, Environment, MockUser)
├─ Rollout failures / behaviors
├─ Strong model expands behavior branches Bₖ
├─ BT: branch → (environment state, user instruction, agent instruction)
└─ Tₖ₊₁ = newly inverted branch tasks</div>
<h3>训练信号与规模</h3>
<table><thead><tr><th>Component</th><th>Paper setting</th><th>Why it matters</th></tr></thead><tbody>
<tr><td>Policy backbone</td><td>Qwen3-8B / Qwen3-30B-A3B</td><td>小模型 agentic capability</td></tr>
<tr><td>Simulator / tool / judge</td><td>Qwen3-235B</td><td>用户、工具和 rubric subgoal reward 均由强模型提供</td></tr>
<tr><td>RL</td><td>GRPO-style multi-round training</td><td>按 rollout group 做相对 advantage</td></tr>
<tr><td>Training data</td><td>约 100K</td><td>支持 Round 0–3 的 flywheel 扩展</td></tr>
<tr><td>Reward</td><td>subgoal completion ratio ∈ [0,1]</td><td>比单一 final answer 更适合工作流</td></tr>
</tbody></table>

<h2 id="implementation">11 · GRPO Implementation：从 group reward 到参数更新</h2>
<div class="pipeline">
  <div><span class="step">01 / PLAN</span><b>锁定可证伪 claim</b><p>真实 8B 权重、官方数据、真实 gradient、训练前后差异、fresh-process replay。</p></div>
  <div><span class="step">02 / ROLLOUT</span><b>组采样</b><p>12 tasks × 2 repeats × group size 16；任务 normal path 第一动作作为 rubric。</p></div>
  <div><span class="step">03 / UPDATE</span><b>LoRA-GRPO</b><p>最后两层 Q/V projection，rank 4；PPO clipped surrogate + entropy。</p></div>
  <div><span class="step">04 / VERIFY</span><b>磁盘重载</b><p>独立进程重载 adapter，逐题复算 28 次 decision，并核对 hashes。</p></div>
</div>
<h3>Group-relative advantage</h3>
<div class="callout">$$\hat A_i = \frac{{r_i - \bar r_G}}{{\sqrt{{\operatorname{{Var}}(r_G)}} + \epsilon}}$$</div>
<p class="equation-note">$r_i$ 是第 $i$ 条 rollout 的 0/1 工具动作 reward；若整组全 0 或全 1，方差为 0，本实现显式记录并跳过，而不是制造梯度。</p>
<h3>Clipped policy objective</h3>
<div class="callout">$$\mathcal L = -\mathbb E_i\left[\min\left(\rho_i\hat A_i,\;\operatorname{{clip}}(\rho_i,1-\epsilon_c,1+\epsilon_c)\hat A_i\right)\right] - \beta H(\pi_\theta)$$</div>
<p class="equation-note">$\rho_i=\pi_\theta(a_i|x)/\pi_{{old}}(a_i|x)$；clip 防止一次更新改变过大，entropy 项保留探索。这里只对四个候选 action token 的条件分布求 loss。</p>
<h3>Parameter-change evidence</h3>
<table><thead><tr><th>Evidence</th><th>Before</th><th>After</th><th>Meaning</th></tr></thead><tbody>
<tr><td>LoRA-B L2 norm</td><td>{summary['training']['lora_b_norm_before']:.6f}</td><td>{summary['training']['lora_b_norm_after']:.6f}</td><td>可训练参数离开零初始化</td></tr>
<tr><td>Adapter SHA-256</td><td><code>{_short(summary['training']['adapter_initial_sha256'])}</code></td><td><code>{_short(summary['training']['adapter_final_sha256'])}</code></td><td>checkpoint 内容真实改变</td></tr>
<tr><td>Training time</td><td>—</td><td>{summary['training']['training_seconds']:.1f} s</td><td>本机实际更新耗时</td></tr>
<tr><td>Peak MLX memory</td><td>—</td><td>{summary['runtime']['peak_memory_gib']:.2f} GiB</td><td>24GB Mac 可运行边界</td></tr>
</tbody></table>

<h2 id="ablations">12 · Ablation Matrix</h2>
<p>消融结构直接对齐参考项目的核心问题：Vanilla 是崩溃基线；Turn-Discount 保护早期 turn；PRM-Lite 提供局部质量信号；LATA 用 $A/\sqrt L$ 缓解长回复稀释；Joint 检验“信号源 + 传输路径”是否必须同时存在。</p>
<table><thead><tr><th>Experiment</th><th>Process weight</th><th>Credit assignment</th><th>Hypothesis</th><th>Status</th></tr></thead><tbody>{ablation_rows}</tbody></table>
<h3>PRM-Lite: interpretable rules</h3>
<div class="two-col">
  <section class="card"><h3>P1–P8 penalties</h3><p>placeholder argument、redundant tool、tool error、repeated error、malformed schema、unsafe action、skipped required read、excessive tools。</p></section>
  <section class="card"><h3>B1–B7 bonuses</h3><p>error recovery、data dependency、distinct read、valid schema、policy confirmation、bounded reasoning、task completion。</p></section>
</div>
<h3>LATA 与 Turn-Discount kernels</h3>
<table><thead><tr><th>Mode</th><th>Per-token credit</th><th>Expected effect</th><th>Current validation</th></tr></thead><tbody>
<tr><td>Linear</td><td>$A/L$</td><td>总 credit 固定，但长 response 单 token 信号快速变弱</td><td>kernel test</td></tr>
<tr><td>LATA</td><td>$A/\sqrt L$</td><td>长度增长 4× 时单 token 只减半，不是减到 1/4</td><td>kernel test；未训练</td></tr>
<tr><td>Turn-Discount</td><td>$A\,w_t/L_t$，$w_t\propto\alpha^{{T-1-t}}$</td><td>早期 turn 获得更高 credit，减少末端猜测</td><td>normalized kernel test；未训练</td></tr>
</tbody></table>
<div class="callout warning"><strong>执行纪律：</strong>配置已固定为 5 experiments × 3 seeds × 6 checkpoints；只有 multi-turn response-token mask 接通后才运行。当前单 token action 让 $L=1$，LATA 与 Linear 完全等价，跑出来也没有研究意义。</div>

<h2 id="benchmarks">13 · Official Benchmark Code: Ready, Not Scored</h2>
<p>慢速 benchmark 已按用户指示停止，但代码不是占位符：它会启动 MLX OpenAI-compatible server，分别加载 base / adapter，调用官方 evaluator，保存 result、score、输入日志、server log 和运行 manifest。</p>
<div class="two-col">
  <section class="card"><div class="kicker">BFCL-V4</div><h3>4 × Multi-Turn</h3><p><code>bfcl-eval 2026.3.23</code>；Base / Miss Func / Miss Param / Long Context。smoke 为每类 {benchmark_scope.get('bfcl_tasks_per_category', 5)} 题，共 {benchmark_scope.get('bfcl_total_tasks', 20)} 题；paper profile 为 4×200=800 题，官方 partial/full evaluator 分开。</p></section>
  <section class="card"><div class="kicker">TAU-2</div><h3>3 domains × Avg@4</h3><p>固定官方 <code>v0.2.0</code>；airline / retail / telecom。smoke 为每域 {benchmark_scope.get('tau2_num_tasks_per_domain', 5)} 题 × {benchmark_scope.get('tau2_num_trials', 1)} trial；paper profile 取消 task cap，运行全部任务 × 4。</p></section>
</div>
<table><thead><tr><th>Profile</th><th>Policy</th><th>User simulator</th><th>Evaluator</th><th>Comparability</th></tr></thead><tbody>
<tr><td><code>smoke</code></td><td>local base / adapter</td><td>local Qwen3-8B</td><td>official BFCL + TAU-2</td><td>接口验收；不与论文数字比较</td></tr>
<tr><td><code>paper</code></td><td>local base / adapter</td><td>必须显式配置外部 simulator</td><td>BFCL 800 + TAU all ×4</td><td>只有 simulator、prompt、版本一致才可比较</td></tr>
</tbody></table>
<div class="callout code"><strong>Dry-run 验收：</strong><code>{html.escape(benchmark_manifest.get('status', 'NOT_PREPARED'))}</code>。已展开 2 个模型变体、4 个 BFCL categories、3 个 TAU-2 domains 的完整命令；没有写入任何伪造 score。</div>
<pre><code># 安装固定版本 evaluator
./scripts/setup_benchmarks.sh

# 只展开命令与任务 ID，不启动模型
DRY_RUN=1 PROFILE=smoke ./run_benchmarks.sh

# 正式小规模接口验收
PROFILE=smoke VARIANTS=base,adapter ./run_benchmarks.sh

# 完整 profile（TAU-2 需要显式 user simulator）
TAU2_USER_MODEL=openai/&lt;model&gt; \
TAU2_USER_API_BASE=https://&lt;endpoint&gt;/v1 \
TAU2_USER_API_KEY=&lt;key&gt; \
PROFILE=paper ./run_benchmarks.sh</code></pre>

<h2 id="audit">14 · Independent Verification Ledger</h2>
<p>训练进程退出后，第二个 Python 进程重新加载 base model 与 adapter，并复算 train / prompt-holdout / unseen 共 28 次 decision。验证不信任内存中的模型对象。</p>
<table><thead><tr><th>Check</th><th>Status</th><th>Evidence</th></tr></thead><tbody>{verification_html}</tbody></table>
<details><summary>完整 provenance 与 hashes</summary><ul>
  <li>Model: <code>{html.escape(summary['model']['id'])}</code></li>
  <li>Model SHA-256: <code>{summary['model']['weights_sha256']}</code></li>
  <li>Dataset SHA-256: <code>{summary['dataset']['parquet_sha256']}</code></li>
  <li>Final adapter SHA-256: <code>{summary['training']['adapter_final_sha256']}</code></li>
  <li>Platform: <code>{html.escape(summary['runtime']['platform'])}</code></li>
  <li>Replay time: <code>{verification['fresh_process_replay_seconds']:.1f}s</code></li>
</ul></details>

<h2 id="completion">15 · Completion Audit</h2>
<p>原报告只分 COMPLETE / PARTIAL / MISSING，无法区分“代码已经写好但没跑”和“完全不存在”。新版采用四态模型，让交付边界可机器读取。</p>
<div class="metric-grid">
  <div class="metric"><div class="value">{counts['COMPLETE']}</div><div class="label">COMPLETE<br>真实运行 + 独立证据</div></div>
  <div class="metric"><div class="value">{counts['PARTIAL_RUN']}</div><div class="label">PARTIAL_RUN<br>真实运行但弱于论文</div></div>
  <div class="metric"><div class="value">{counts['CODE_READY']}</div><div class="label">CODE_READY<br>代码/配置/dry-run 完成</div></div>
  <div class="metric"><div class="value">{counts['BLOCKED_RESOURCE']}</div><div class="label">BLOCKED_RESOURCE<br>需要额外模型或 GPU</div></div>
</div>
<table><thead><tr><th>Area</th><th>Status</th><th>Current evidence</th><th>Definition of done</th></tr></thead><tbody>{self_check_html}</tbody></table>
<div class="callout"><strong>机器可读账本：</strong><a href="../artifacts/real_qwen3_8b/completion_matrix.json">completion_matrix.json</a>。任何后续运行都应先更新 evidence，再修改状态；不能只改报告文案。</div>

<h2 id="structure">16 · Project Structure</h2>
<div class="tree">agenticqwen-reproduction/
├── configs/
│   ├── real_qwen3_8b.json          # 已运行的小规模 LoRA-GRPO
│   ├── long_horizon_demo.json      # Qwen3.7-Flash 状态化轨迹
│   ├── benchmarks.json             # BFCL / TAU-2 smoke + paper profiles
│   └── ablation_matrix.json        # 5 exp × 3 seed × 6 checkpoint
├── src/agentic_repro/
│   ├── real_grpo.py                # MLX action-masked LoRA-GRPO
│   ├── long_horizon_env.py         # state / tools / policy constraints / verifier
│   ├── dashscope_policy.py         # Qwen function-calling client
│   ├── trajectory_runner.py        # multi-turn loop + artifact ledger
│   ├── verify_real.py              # fresh-process replay verifier
│   ├── benchmark_runner.py         # MLX server + official evaluators
│   ├── ablations.py                # PRM-Lite / LATA / saturation diagnostic
│   └── report_real.py              # evidence-driven HTML generator
├── artifacts/
│   ├── real_qwen3_8b/              # checkpoint, eval, trajectories, hashes
│   ├── long_horizon/               # 真实 8-turn trajectory JSON / Markdown
│   ├── ablations/                  # offline counterfactual diagnostic
│   └── benchmarks/                 # dry-run manifest / future official scores
├── tests/                          # reward, credit, commands, pipeline tests
├── run_real_qwen3_8b.sh
├── run_long_horizon_demo.sh
├── run_ablations.sh
├── run_benchmarks.sh
└── agenticqwen_report/index.html</div>

<h2 id="run">17 · One-click Runbook</h2>
<h3>真实状态化多轮轨迹</h3>
<pre><code># key 只注入当前进程，不写入 config / artifact
DASHSCOPE_API_KEY=&lt;rotated-key&gt; ./run_long_horizon_demo.sh

# 结果：8 agent turns · 5 tool calls · 9/9 verifier PASS</code></pre>
<h3>已运行闭环</h3>
<pre><code>./run_real_qwen3_8b.sh

# stages:
# 1. real_grpo       2. verify_real
# 3. report_real     4. build PPT</code></pre>
<h3>快速、无模型推理的诊断</h3>
<pre><code>./run_ablations.sh
DRY_RUN=1 ./run_benchmarks.sh
../../work/mlxenv312/bin/python -m pytest -q</code></pre>
<h3>原始证据入口</h3>
<ul>
  <li><a href="../artifacts/real_qwen3_8b/summary.json">summary.json</a> — 实验汇总与 runtime。</li>
  <li><a href="../artifacts/real_qwen3_8b/training_log.jsonl">training_log.jsonl</a> — 24 个 group 的 actions / rewards / loss。</li>
  <li><a href="../artifacts/real_qwen3_8b/baseline_eval.json">baseline_eval.json</a> / <a href="../artifacts/real_qwen3_8b/final_eval.json">final_eval.json</a> — 逐题概率。</li>
  <li><a href="../artifacts/real_qwen3_8b/verification.json">verification.json</a> — fresh-process 验证账本。</li>
  <li><a href="../artifacts/ablations/offline_reward_diagnostic.json">offline_reward_diagnostic.json</a> — PRM-Lite 负结果。</li>
  <li><a href="../artifacts/benchmarks/planned_manifest.json">planned_manifest.json</a> — 官方 benchmark dry-run。</li>
  <li><a href="../artifacts/long_horizon/trajectory_qwen3.7_flash.json">trajectory_qwen3.7_flash.json</a> — 完整事件、状态、reward 与 verifier。</li>
  <li><a href="../artifacts/long_horizon/trajectory_qwen3.7_flash.md">trajectory_qwen3.7_flash.md</a> — 人类可读轨迹。</li>
</ul>

<h2 id="interview">18 · Resume & Interview Kit</h2>
<h3>90 秒项目介绍</h3>
<div class="resume-block"><p>我复现的是 AgenticQwen 背后的 Agentic GRPO 工程链，而不是只调用一次模型。项目分成两层：第一层用 Qwen3.7-Flash 跑一个状态化退款环境，模型经过身份验证、订单与付款读取、政策检查、用户确认和退款写入，完成 8-turn、5-tool 的真实轨迹；环境独立记录状态和 process reward，verifier 不相信模型自报成功。第二层在本地 Qwen3-8B 上运行 LoRA-GRPO，通过 adapter norm、hash 和 fresh-process replay 证明参数真的更新。实验还发现 20/24 rollout group 方差为零，说明 binary outcome reward 会导致 group saturation。因此我实现了 PRM-Lite、LATA 和 Turn-Discount，但明确区分代码就绪与在线消融结果。</p></div>
<h3>可以直接放进简历的 bullets</h3>
<ul>
  <li>基于 Qwen3-8B/MLX 实现 LoRA-GRPO 本地训练闭环，完成 24×16 group rollouts、8 次 optimizer update、adapter hash 审计与独立 checkpoint replay。</li>
  <li>构建 Qwen3.7-Flash 驱动的状态化 multi-turn function-calling sandbox，实现身份验证、先读后写、显式确认、幂等退款和 9 项独立 verifier。</li>
  <li>定位 83.3% rollout group saturation，设计 PRM-Lite、LATA、Turn-Discount 与 Joint 的 5×3×6 消融矩阵，并接入 BFCL-V4/TAU-2 官方 evaluator。</li>
</ul>
<h3>面试高频追问</h3>
<dl class="qa">
  <dt>为什么 verifier 不能由同一个 Agent 自己完成？</dt><dd>否则模型可以通过自然语言宣称成功而不产生真实 side effect。Verifier 必须读环境状态、refund ID、amount 和 policy flags。</dd>
  <dt>为什么这条轨迹不是 RL 结果？</dt><dd>它证明的是 policy inference、工具执行与 reward instrumentation；只有把多个同任务 trajectory 的 reward 用于反向传播，才是 RL training。</dd>
  <dt>为什么 GRPO 会出现 group saturation？</dt><dd>同组 reward 全 0 或全 1 时，组内标准差为 0，标准化 advantage 全为 0，因此没有相对学习信号。</dd>
  <dt>PRM-Lite 为什么离线没有改善？</dt><dd>原始训练轨迹只有一个 action token，没有 recovery、依赖链和多 turn 事件；规则对组内样本给出相同分数，仍然无法打破 tie。</dd>
  <dt>LATA 解决什么问题？</dt><dd>轨迹 advantage 广播到长 response 时，线性 $A/L$ 会快速稀释单 token 信号；$A/\sqrt L$ 保留更强的长回复 credit，但需要在线对照验证。</dd>
  <dt>如何防止重复退款？</dt><dd>环境同时校验目标 charge、精确金额、显式确认和 idempotency key；模型本身无权直接写数据库。</dd>
  <dt>当前最重要的下一步是什么？</dt><dd>批量生成同任务多条多轮 rollout，把 outcome + process reward 接入 response-token mask 的 GRPO，并按 checkpoint 跑成功率、saturation 和 safety。</dd>
</dl>
<div class="callout warning"><strong>面试纪律：</strong>可以说“实现并跑通一条真实多轮轨迹”和“实现并跑通单步 Qwen3-8B GRPO”；不能说“已经完成多轮 GRPO 训练”或“复现论文 47.4”。</div>

<h2 id="limitations">19 · Limitations & Claim Boundary</h2>
<table><thead><tr><th>Claim</th><th>Verdict</th><th>Reason</th></tr></thead><tbody>
<tr><td>“已跑通状态化 multi-turn function calling”</td><td><span class="status complete">SUPPORTED</span></td><td>8 turns / 5 tools / 16 events / 9 checks；状态和 tool side effect 落盘</td></tr>
<tr><td>“本地确实训练了 Qwen3-8B 参数”</td><td><span class="status complete">SUPPORTED</span></td><td>LoRA norm、hash、fresh replay 一致</td></tr>
<tr><td>“训练改善了同任务另一提示”</td><td><span class="status complete">SUPPORTED</span></td><td>25.0% → 41.7%，但 N=12</td></tr>
<tr><td>“训练改善了未见任务”</td><td><span class="status blocked-resource">NOT SUPPORTED</span></td><td>25.0% → 25.0%，N=4</td></tr>
<tr><td>“PRM-Lite 已解决 saturation”</td><td><span class="status blocked-resource">REFUTED HERE</span></td><td>离线诊断 4 → 4 active groups；当前过程信号不足</td></tr>
<tr><td>“已进行多轮 response-token GRPO”</td><td><span class="status partial-run">NOT YET</span></td><td>multi-turn inference 与单步 RL 分别跑通，尚未连接成在线训练</td></tr>
<tr><td>“已复现论文 47.4 平均分”</td><td><span class="status blocked-resource">NOT TESTED</span></td><td>官方 benchmark 未运行</td></tr>
<tr><td>“已实现论文规模 flywheel”</td><td><span class="status blocked-resource">NOT TESTED</span></td><td>12 tasks vs ≈100K；无 235B simulator/judge</td></tr>
</tbody></table>
<p>此外，论文自身也指出小模型在 open-ended 与 very-long-context agentic tasks 上仍有困难；8B/30B 原生约 40K context 会限制 deep-search。论文全程使用 Qwen family 作为 synthesizer、simulator 与 evaluator，也可能引入 model-family bias。</p>

<h2 id="sources">20 · Sources</h2>
<ol>
  <li><a href="https://arxiv.org/abs/2604.21590">AgenticQwen paper (arXiv:2604.21590)</a></li>
  <li><a href="https://huggingface.co/alibaba-pai/AgenticQwen-8B">Official AgenticQwen-8B checkpoint</a> and <a href="https://huggingface.co/collections/alibaba-pai/agenticqwen">model/data collection</a></li>
  <li><a href="https://github.com/haruhi-sudo/data_synth_and_rl">Official data synthesis / RL code</a></li>
  <li><a href="https://github.com/sierra-research/tau2-bench/tree/v0.2.0">Official TAU-2 v0.2.0 evaluator</a></li>
  <li><a href="https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard">Official BFCL evaluator</a></li>
  <li><a href="https://github.com/qiqihezh/agentic-grpo-longhorizon">Agentic-GRPO-LongHorizon — report structure and saturation diagnosis reference</a></li>
  <li><a href="https://github.com/ml-explore/mlx-lm">MLX-LM</a> and <a href="https://huggingface.co/Qwen/Qwen3-8B-MLX-4bit">Qwen3-8B MLX 4-bit</a></li>
  <li><a href="https://help.aliyun.com/en/model-studio/qwen-function-calling">DashScope Qwen Function Calling</a> and <a href="https://help.aliyun.com/en/model-studio/multi-round-conversation">multi-turn conversation contract</a></li>
  <li><a href="https://yyhdbl.github.io">Visual reference</a></li>
</ol>

<div class="footer">Generated only from paper source, official evaluator contracts, and locally persisted artifacts. “Code ready” is never counted as an experimental result.</div>
</main>
<script>
document.querySelectorAll('.trajectory-controls button').forEach((button) => {{
  button.addEventListener('click', () => {{
    const filter = button.dataset.filter;
    document.querySelectorAll('.trajectory-controls button').forEach((item) => item.classList.remove('active'));
    button.classList.add('active');
    document.querySelectorAll('.trajectory-event').forEach((item) => {{
      item.hidden = filter !== 'all' && item.dataset.role !== filter;
    }});
  }});
}});
</script>
</body>
</html>"""

    output = report_dir / "index.html"
    output.write_text(body, encoding="utf-8")
    report_manifest = {
        "status": "generated",
        "output": str(output),
        "inputs": {
            "summary": str(artifacts / "summary.json"),
            "verification": str(artifacts / "verification.json"),
            "benchmark_manifest": str(
                project_dir / "artifacts" / "benchmarks" / "planned_manifest.json"
            ),
            "reward_diagnostic": str(
                project_dir
                / "artifacts"
                / "ablations"
                / "offline_reward_diagnostic.json"
            ),
            "long_horizon_trajectory": str(
                project_dir
                / "artifacts"
                / "long_horizon"
                / "trajectory_qwen3.7_flash.json"
            ),
        },
        "status_counts": counts,
    }
    (report_dir / "report_manifest.json").write_text(
        json.dumps(report_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the evidence-driven AgenticQwen reproduction report"
    )
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(build(args.config.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
