import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const PROJECT = path.resolve(SCRIPT_DIR, "..");
const ROOT = path.resolve(PROJECT, "../..");
const artifactToolDist = process.env.ARTIFACT_TOOL_DIST || "/Users/hongbo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";
const { Presentation, PresentationFile } = await import(pathToFileURL(artifactToolDist).href);

const OUT = process.env.PPTX_OUT ? path.resolve(process.env.PPTX_OUT) : path.join(PROJECT, "slides/AgenticQwen_LongHorizon_Lab.pptx");
const PREVIEW = path.join(ROOT, "work/slides/long-horizon-rendered");
const IMG = path.join(PROJECT, "agenticqwen_report/images");
const ART = path.join(PROJECT, "artifacts/real_qwen3_8b");

const summary = JSON.parse(await fs.readFile(path.join(ART, "summary.json"), "utf8"));
const verification = JSON.parse(await fs.readFile(path.join(ART, "verification.json"), "utf8"));
const completion = JSON.parse(await fs.readFile(path.join(ART, "completion_matrix.json"), "utf8"));
const ablation = JSON.parse(await fs.readFile(path.join(PROJECT, "artifacts/ablations/offline_reward_diagnostic.json"), "utf8"));
const benchmarkPlan = JSON.parse(await fs.readFile(path.join(PROJECT, "artifacts/benchmarks/planned_manifest.json"), "utf8"));
const trajectoryPath = path.join(PROJECT, "artifacts/long_horizon/trajectory_qwen3.7_flash.json");
const trajectory = JSON.parse(await fs.readFile(trajectoryPath, "utf8"));

const C = {
  ink: "#28231F", muted: "#756B62", line: "#E7DED2", paper: "#FFFDF9",
  wash: "#F5EEE5", accent: "#A8472D", accentDark: "#713322",
  green: "#2F6F55", red: "#A12D2D", blue: "#284C75", gold: "#A36B13",
  softGreen: "#EEF7F1", softGold: "#FFF7E8", softRed: "#FCEFED",
};
const W = 1280, H = 720;

async function bytes(file) {
  const b = await fs.readFile(file);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}
async function writeBlob(file, blob) {
  await fs.writeFile(file, new Uint8Array(await blob.arrayBuffer()));
}
function box(slide, name, pos, fill = C.paper, lineFill = C.line, radius = 0) {
  return slide.shapes.add({
    geometry: radius ? "roundRect" : "rect", name, position: pos, fill,
    line: { style: "solid", fill: lineFill, width: 1 },
    ...(radius ? { borderRadius: radius } : {}),
  });
}
function text(slide, name, value, pos, style = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox", name, position: pos, fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = value;
  shape.text.style = {
    fontFamily: style.fontFamily || "PingFang SC",
    fontSize: style.fontSize || 20,
    color: style.color || C.ink,
    bold: style.bold || false,
    alignment: style.alignment || "left",
    verticalAlignment: style.verticalAlignment || "top",
    ...style,
  };
  return shape;
}
function rule(slide, x, y, width, color = C.line, weight = 1) {
  return slide.shapes.add({
    geometry: "line", position: { left: x, top: y, width, height: 0 }, fill: "none",
    line: { style: "solid", fill: color, width: weight },
  });
}
function baseSlide(presentation, number, eyebrow, title, subtitle = "") {
  const slide = presentation.slides.add();
  slide.background.fill = C.wash;
  box(slide, `paper-${number}`, { left: 36, top: 30, width: 1208, height: 660 }, C.paper, C.line);
  text(slide, `eyebrow-${number}`, eyebrow.toUpperCase(), { left: 76, top: 58, width: 760, height: 26 }, { fontSize: 13, color: C.accent, bold: true, letterSpacing: 2 });
  text(slide, `title-${number}`, title, { left: 76, top: 92, width: 1075, height: 58 }, { fontFamily: "Songti SC", fontSize: 39, bold: true });
  if (subtitle) text(slide, `subtitle-${number}`, subtitle, { left: 76, top: 148, width: 1060, height: 42 }, { fontSize: 18, color: C.muted });
  rule(slide, 76, subtitle ? 196 : 164, 1128, C.line, 1);
  text(slide, `page-${number}`, String(number).padStart(2, "0"), { left: 1150, top: 640, width: 52, height: 24 }, { fontSize: 12, color: C.muted, alignment: "right" });
  return slide;
}
function notes(slide, body, sources = []) {
  const sourceLines = sources.map((source) => `- ${source}`).join("\n");
  slide.speakerNotes.textFrame.setText(`${body}\n\n[Sources]\n${sourceLines}\n[/Sources]`);
}
function metric(slide, key, x, y, value, label, color = C.accent) {
  rule(slide, x, y, 220, color, 4);
  text(slide, `metric-value-${key}`, value, { left: x, top: y + 20, width: 220, height: 62 }, { fontFamily: "Georgia", fontSize: 36, bold: true });
  text(slide, `metric-label-${key}`, label, { left: x, top: y + 82, width: 220, height: 52 }, { fontSize: 16, color: C.muted });
}
function pct(value) { return `${(value * 100).toFixed(1)}%`; }
function shortHash(value) { return `${value.slice(0, 12)}…`; }
function statusCount(status) { return completion.items.filter((item) => item.status === status).length; }
function addBar(slide, key, y, label, before, after) {
  text(slide, `bar-label-${key}`, label, { left: 92, top: y - 2, width: 235, height: 30 }, { fontSize: 17, bold: true });
  box(slide, `bar-track-before-${key}`, { left: 345, top: y, width: 520, height: 16 }, "#EEE6DC", "#EEE6DC", 8);
  box(slide, `bar-before-${key}`, { left: 345, top: y, width: Math.max(4, 520 * before), height: 16 }, "#B9ACA0", "#B9ACA0", 8);
  box(slide, `bar-track-after-${key}`, { left: 345, top: y + 29, width: 520, height: 16 }, "#EEE6DC", "#EEE6DC", 8);
  box(slide, `bar-after-${key}`, { left: 345, top: y + 29, width: Math.max(4, 520 * after), height: 16 }, C.accent, C.accent, 8);
  text(slide, `bar-values-${key}`, `${pct(before)}  →  ${pct(after)}`, { left: 900, top: y + 5, width: 215, height: 40 }, { fontFamily: "Georgia", fontSize: 20, bold: true, color: after > before ? C.green : C.muted, alignment: "right" });
}

async function main() {
  await fs.mkdir(PREVIEW, { recursive: true });
  await fs.mkdir(path.dirname(OUT), { recursive: true });
  const fig1 = await bytes(path.join(IMG, "figure1_dual_flywheel.png"));
  const fig2 = await bytes(path.join(IMG, "figure2_flywheel_results.png"));
  const presentation = Presentation.create({ slideSize: { width: W, height: H } });
  const M = summary.metrics;

  // 1 — Title
  {
    const s = presentation.slides.add(); s.background.fill = C.wash;
    box(s, "title-paper", { left: 36, top: 30, width: 1208, height: 660 }, C.paper, C.line);
    rule(s, 80, 78, 170, C.accent, 5);
    text(s, "title-kicker", "STATEFUL TRAJECTORY · REAL UPDATE · VERIFIED", { left: 80, top: 104, width: 630, height: 28 }, { fontSize: 14, bold: true, color: C.accent, letterSpacing: 2 });
    text(s, "deck-title", "AgenticQwen\nLong-Horizon Lab", { left: 80, top: 172, width: 600, height: 160 }, { fontFamily: "Songti SC", fontSize: 61, bold: true });
    text(s, "deck-subtitle", "Qwen3.7-Flash · 8-turn tool trajectory\nQwen3-8B · LoRA-GRPO · verifier 9/9", { left: 82, top: 365, width: 610, height: 86 }, { fontSize: 22, color: C.muted });
    s.images.add({ blob: fig1, contentType: "image/png", alt: "AgenticQwen dual data flywheel overview", fit: "contain", position: { left: 720, top: 126, width: 440, height: 318 } });
    text(s, "title-state", "TRAJECTORY 9/9 PASS · TRAINING 9/9 PASS", { left: 82, top: 592, width: 720, height: 28 }, { fontSize: 14, bold: true, color: C.green });
    notes(s, "开场界定：真实跑通一条 Qwen3.7-Flash 多轮工具轨迹，并保留此前 Qwen3-8B LoRA-GRPO 权重更新与 fresh-process replay。两部分均有独立 verifier，但尚未合并成多轮 response-token RL。", ["https://arxiv.org/abs/2604.21590", trajectoryPath, path.join(ART, "summary.json")]);
  }

  // 2 — What was reproduced
  {
    const s = baseSlide(presentation, 2, "Answer", "完整 Agentic RL 项目，需要四层证据", "环境、轨迹、训练、验证分别回答不同问题。");
    const items = [
      ["01", "状态环境", "identity · order\npayment · policy", C.blue],
      ["02", "真实轨迹", `${trajectory.runtime.agent_turns} agent turns\n${trajectory.runtime.tool_calls} JSON tool calls`, C.accent],
      ["03", "真实训练", "24 rollout groups\n8 optimizer steps", C.green],
      ["04", "独立验证", "trajectory 9/9\ncheckpoint 9/9", C.gold],
    ];
    items.forEach((item, i) => {
      const x = 80 + i * 285;
      box(s, `answer-card-${i}`, { left: x, top: 245, width: 250, height: 250 }, i === 2 ? C.softGreen : C.paper, item[3], 8);
      text(s, `answer-num-${i}`, item[0], { left: x + 22, top: 265, width: 60, height: 32 }, { fontFamily: "Georgia", fontSize: 20, bold: true, color: item[3] });
      text(s, `answer-name-${i}`, item[1], { left: x + 22, top: 320, width: 205, height: 42 }, { fontFamily: "Songti SC", fontSize: 26, bold: true });
      text(s, `answer-detail-${i}`, item[2], { left: x + 22, top: 385, width: 205, height: 80 }, { fontSize: 18, color: C.muted, bold: true });
    });
    text(s, "answer-boundary", "边界：多轮 inference 与单步 GRPO 分别跑通；尚未连接成多轮 response-token RL。", { left: 145, top: 565, width: 990, height: 40 }, { fontSize: 19, bold: true, color: C.accentDark, alignment: "center" });
    notes(s, "四层证据避免把一个漂亮框架误写成完整复现：环境证明状态可变，轨迹证明闭环可运行，训练证明参数真的更新，验证证明结论不是 executor 自报。", [trajectoryPath, path.join(ART, "summary.json"), path.join(ART, "verification.json")]);
  }

  // 3 — Paper method
  {
    const s = baseSlide(presentation, 3, "Paper", "论文靠双数据飞轮持续制造“更难但可验证”的信号", "Reasoning 从错误题扩展；Agentic 从线性 workflow 长成带对抗分支的行为树。");
    s.images.add({ blob: fig1, contentType: "image/png", alt: "AgenticQwen dual data flywheels from paper", fit: "contain", position: { left: 72, top: 210, width: 1136, height: 392 } });
    text(s, "paper-caption", "Paper Figure 1 · 论文训练约 100K 样本，并使用 Qwen3-235B 模拟用户、工具与 reward judge。", { left: 78, top: 606, width: 1060, height: 32 }, { fontSize: 14, color: C.muted });
    notes(s, "说明论文目标：用 GRPO-style 多轮 RL + 双飞轮增强小模型工具使用。此页只展示论文结果，不是本机测量。", ["https://arxiv.org/abs/2604.21590", "https://github.com/haruhi-sudo/data_synth_and_rl"]);
  }

  // 4 — System architecture
  {
    const s = baseSlide(presentation, 4, "System", "Policy 决策，Environment 执行，Verifier 判定", "模型不能直接改状态，也不能用自然语言给自己判成功。");
    const p = [
      { left: 95, top: 270, width: 285, height: 205 },
      { left: 497, top: 270, width: 285, height: 205 },
      { left: 899, top: 270, width: 285, height: 205 },
    ];
    const n = p.map((pos, i) => box(s, `pipeline-node-${i}`, pos, i === 2 ? C.softGreen : C.paper, i === 2 ? C.green : C.line, 8));
    s.shapes.connect(n[0], n[1], { kind: "straight", fromSide: "right", toSide: "left", line: { style: "solid", fill: C.accent, width: 2 }, tail: { type: "arrow", width: "med", length: "med" } });
    s.shapes.connect(n[1], n[2], { kind: "straight", fromSide: "right", toSide: "left", line: { style: "solid", fill: C.accent, width: 2 }, tail: { type: "arrow", width: "med", length: "med" } });
    text(s, "pipeline-arrow-1", "→", { left: 400, top: 342, width: 70, height: 48 }, { fontSize: 30, bold: true, color: C.accent, alignment: "center" });
    text(s, "pipeline-arrow-2", "→", { left: 802, top: 342, width: 70, height: 48 }, { fontSize: 30, bold: true, color: C.accent, alignment: "center" });
    const rows = [
      ["POLICY", "生成追问与函数调用\n给出 JSON 参数\n不拥有 side effect"],
      ["ENV / TOOLS", "身份与权限校验\n先读后写 + 幂等\n执行状态转移"],
      ["VERIFIER", "读取最终 state\n核对 charge / amount\n9 项硬检查"],
    ];
    rows.forEach((row, i) => {
      text(s, `pipeline-k-${i}`, row[0], { left: p[i].left + 24, top: p[i].top + 24, width: 220, height: 30 }, { fontSize: 14, bold: true, color: i === 2 ? C.green : C.accent });
      text(s, `pipeline-b-${i}`, row[1], { left: p[i].left + 24, top: p[i].top + 70, width: 235, height: 120 }, { fontSize: 20, bold: true });
    });
    text(s, "pipeline-foot", "User simulator 提供事实与确认；Trainer 下一步消费多个 trajectory group。", { left: 220, top: 555, width: 840, height: 45 }, { fontSize: 19, color: C.muted, alignment: "center" });
    notes(s, "Policy 只能提出工具调用；环境执行 side effect；verifier 从最终状态判定结果。用户模拟器负责提供事实与明确确认，这就是长程 Agentic RL 的最小安全闭环。", [path.join(PROJECT, "src/agentic_repro/long_horizon_env.py"), path.join(PROJECT, "src/agentic_repro/trajectory_runner.py"), trajectoryPath, "https://help.aliyun.com/en/model-studio/qwen-function-calling"]);
  }

  // 5 — Real trajectory
  {
    const s = baseSlide(presentation, 5, "Trajectory", "一条安全退款轨迹，先读后写", "从模糊诉求到 side effect：每一步都留下 observation、state 与 reward。");
    const steps = [
      ["VERIFY", "索取 email + last4\nlookup_customer"],
      ["READ", "list_orders\npayment → policy"],
      ["CONFIRM", "指出 CHG-9002\n等待明确同意"],
      ["WRITE", "create_refund\nidempotency key"],
    ];
    const positions = steps.map((_, i) => ({ left: 82 + i * 290, top: 265, width: 240, height: 180 }));
    const nodes = positions.map((pos, i) => box(s, `grpo-node-${i}`, pos, i === 3 ? C.softGreen : C.paper, i === 3 ? C.green : C.line, 8));
    for (let i = 0; i < nodes.length - 1; i++) s.shapes.connect(nodes[i], nodes[i + 1], { kind: "straight", fromSide: "right", toSide: "left", line: { style: "solid", fill: C.accent, width: 2 }, tail: { type: "arrow", width: "med", length: "med" } });
    [322, 612, 902].forEach((x, i) => text(s, `grpo-arrow-${i}`, "→", { left: x, top: 330, width: 50, height: 46 }, { fontSize: 25, bold: true, color: C.accent, alignment: "center" }));
    steps.forEach((row, i) => {
      text(s, `grpo-k-${i}`, row[0], { left: positions[i].left + 18, top: 287, width: 190, height: 28 }, { fontSize: 13, bold: true, color: i === 3 ? C.green : C.accent });
      text(s, `grpo-b-${i}`, row[1], { left: positions[i].left + 18, top: 340, width: 200, height: 75 }, { fontSize: 20, bold: true, alignment: "center" });
    });
    metric(s, "layers", 150, 510, String(trajectory.runtime.agent_turns), "agent turns", C.blue);
    metric(s, "rank", 425, 510, String(trajectory.runtime.tool_calls), "tool calls", C.blue);
    metric(s, "groups", 700, 510, String(trajectory.runtime.events), "audited events", C.accent);
    metric(s, "steps", 975, 510, `${trajectory.verification.checks.filter((check) => check.passed).length}/9`, "verifier PASS", C.green);
    notes(s, "真实轨迹没有直接退款：先身份验证，再读取订单、支付记录与策略，明确指出重复扣款 CHG-9002，等待用户确认后才执行 create_refund。", [trajectoryPath, path.join(PROJECT, "artifacts/long_horizon/trajectory_qwen3.7_flash.md"), "https://help.aliyun.com/en/model-studio/multi-round-conversation"]);
  }

  // 6 — Results
  {
    const s = baseSlide(presentation, 6, "Results", "训练前后：提升集中在见过的任务结构", "灰色为 before，红色为 after；完全未见任务没有改善。");
    addBar(s, "train", 260, "Train tasks", M.train.accuracy_before, M.train.accuracy_after);
    addBar(s, "holdout", 370, "Prompt holdout", M.holdout.accuracy_before, M.holdout.accuracy_after);
    addBar(s, "unseen", 480, "Unseen tasks", M.unseen.accuracy_before, M.unseen.accuracy_after);
    text(s, "result-callout", `最有价值的数字不是 ${pct(M.holdout.accuracy_after)}，而是 Unseen 仍为 ${pct(M.unseen.accuracy_after)}。`, { left: 205, top: 590, width: 870, height: 36 }, { fontFamily: "Songti SC", fontSize: 22, bold: true, color: C.accentDark, alignment: "center" });
    notes(s, "解读时避免夸大：prompt holdout 与训练任务共享任务 identity；未见任务不提升说明尚无跨域泛化。", [path.join(ART, "baseline_eval.json"), path.join(ART, "final_eval.json"), path.join(ART, "summary.json")]);
  }

  // 7 — Checkpoint evidence
  {
    const s = baseSlide(presentation, 7, "Evidence", "参数确实变了，而且能从磁盘重放", "三组独立证据：LoRA norm、adapter hash、fresh-process metrics。");
    metric(s, "norm-before", 90, 240, "0.000", "LoRA-B norm before", C.muted);
    metric(s, "norm-after", 365, 240, summary.training.lora_b_norm_after.toFixed(3), "LoRA-B norm after", C.green);
    metric(s, "adapter-size", 640, 240, "417 KiB", "final adapter", C.blue);
    metric(s, "replay", 915, 240, "0.0", "metric replay delta", C.green);
    box(s, "hash-box", { left: 100, top: 425, width: 1080, height: 125 }, C.ink, C.ink, 8);
    text(s, "hash-text", `BASE    ${shortHash(summary.model.weights_sha256)}\nDATA    ${shortHash(summary.dataset.parquet_sha256)}\nADAPTER ${shortHash(summary.training.adapter_final_sha256)}`, { left: 130, top: 448, width: 930, height: 84 }, { fontFamily: "Menlo", fontSize: 19, bold: true, color: C.paper });
    text(s, "hash-badge", "SHA-256", { left: 1030, top: 465, width: 110, height: 34 }, { fontSize: 14, bold: true, color: "#F2C7B8", alignment: "center" });
    notes(s, "解释三种证据互补：norm 证明数值更新，hash 证明文件变化，fresh-process 证明 checkpoint 可加载且指标重现。", [path.join(ART, "adapter/adapters.safetensors"), path.join(ART, "verification.json")]);
  }

  // 8 — Honest failure analysis and ablation diagnosis
  {
    const s = baseSlide(presentation, 8, "Diagnosis", "关键失败：20/24 rollout groups 没有相对学习信号", "离线 PRM-Lite 只重打分，不训练模型；它没有激活任何新 group，是一个有用的负结果。");
    metric(s, "group-total", 90, 230, "24", "rollout groups", C.blue);
    metric(s, "group-active", 365, 230, String(summary.training.updated_groups), "non-zero variance", C.green);
    metric(s, "group-flat", 640, 230, String(summary.training.attempted_groups - summary.training.updated_groups), "saturated groups", C.red);
    metric(s, "prm-new", 915, 230, String(ablation.newly_activated_groups), "PRM newly active", C.gold);
    const variants = [
      ["Turn-Discount", "越早决策越高 credit"],
      ["PRM-Lite", "15 条过程规则"],
      ["LATA", "A/L 与 A/√L"],
      ["Joint", "PRM + LATA"],
    ];
    variants.forEach((row, i) => {
      const x = 90 + i * 275;
      box(s, `ablation-card-${i}`, { left: x, top: 420, width: 245, height: 110 }, i === 1 ? C.softGold : C.paper, i === 1 ? C.gold : C.line, 8);
      text(s, `ablation-name-${i}`, row[0], { left: x + 18, top: 440, width: 205, height: 28 }, { fontSize: 18, bold: true, color: i === 1 ? C.gold : C.ink, alignment: "center" });
      text(s, `ablation-note-${i}`, row[1], { left: x + 18, top: 480, width: 205, height: 32 }, { fontSize: 15, color: C.muted, alignment: "center" });
    });
    text(s, "diagnosis-conclusion", `zero-variance rate: ${pct(ablation.outcome_zero_variance_rate)} → ${pct(ablation.shaped_zero_variance_rate)} · 说明单步 action token 没有足够过程事件`, { left: 145, top: 570, width: 990, height: 45 }, { fontSize: 18, bold: true, color: C.accentDark, alignment: "center" });
    notes(s, "对应 long-horizon 参考项目最重要的 failure mode：group saturation。离线 PRM 诊断保留为负结果，严禁把它写成消融精度；真正验证需要多轮工具轨迹在线训练。", ["https://github.com/qiqihezh/agentic-grpo-longhorizon", path.join(ART, "summary.json"), path.join(PROJECT, "artifacts/ablations/offline_reward_diagnostic.json"), path.join(PROJECT, "configs/ablation_matrix.json")]);
  }

  // 9 — Verification and self-check
  {
    const s = baseSlide(presentation, 9, "Self-check", "四态完成矩阵：跑过、部分跑过、代码就绪、资源阻塞", "每个状态都有证据与补完条件；不再把 CODE_READY 写成 COMPLETE。");
    metric(s, "complete", 90, 230, String(statusCount("COMPLETE")), "COMPLETE", C.green);
    metric(s, "partial", 365, 230, String(statusCount("PARTIAL_RUN")), "PARTIAL_RUN", C.gold);
    metric(s, "ready", 640, 230, String(statusCount("CODE_READY")), "CODE_READY", C.blue);
    metric(s, "blocked", 915, 230, String(statusCount("BLOCKED_RESOURCE")), "BLOCKED_RESOURCE", C.red);
    box(s, "verify-box", { left: 95, top: 420, width: 1090, height: 125 }, C.softGreen, C.green, 8);
    text(s, "verify-title", `独立 checkpoint replay · ${verification.summary.pass} PASS / ${verification.summary.fail} FAIL`, { left: 130, top: 442, width: 500, height: 34 }, { fontFamily: "Songti SC", fontSize: 25, bold: true, color: C.green });
    text(s, "verify-detail", `重新加载 base + adapter · 28 个决策样本 · ${verification.fresh_process_replay_seconds.toFixed(1)}s · peak ${verification.peak_memory_gib.toFixed(2)} GiB`, { left: 130, top: 493, width: 900, height: 34 }, { fontSize: 17, color: C.muted });
    text(s, "verify-rule", "RULE\nNO FAKE SCORE", { left: 1020, top: 444, width: 125, height: 64 }, { fontFamily: "Georgia", fontSize: 17, bold: true, color: C.green, alignment: "center" });
    text(s, "matrix-foot", "完整矩阵逐项列出 evidence 与 completion，不用一个百分比掩盖关键缺口。", { left: 190, top: 580, width: 900, height: 34 }, { fontSize: 18, color: C.muted, alignment: "center" });
    notes(s, "新增真实 API 多轮轨迹后，COMPLETE 与 PARTIAL_RUN 各增加一项，BLOCKED_RESOURCE 减少一项。多轮训练本身仍是 PARTIAL_RUN，不能写成论文指标复现。", [path.join(ART, "verification.json"), path.join(ART, "completion_matrix.json"), trajectoryPath]);
  }

  // 10 — Official benchmark paths
  {
    const s = baseSlide(presentation, 10, "Benchmark", "官方 BFCL-V4 与 TAU-2：代码已接通，本次不长跑", "base / adapter 两个 variant；smoke 与 paper profile；server、命令、结果目录和 manifest 全部落盘。");
    const cols = [
      ["01", "BFCL-V4", `${benchmarkPlan.scope.bfcl_total_tasks} smoke tasks\n4 multi-turn categories`],
      ["02", "TAU-2 v0.2.0", `${benchmarkPlan.scope.tau2_num_tasks_per_domain} smoke/domain\nairline · retail · telecom`],
      ["03", "论文级 profile", "BFCL 800 tasks\nTAU all tasks × 4"],
    ];
    cols.forEach((row, i) => {
      const x = 105 + i * 390;
      text(s, `next-num-${i}`, row[0], { left: x, top: 245, width: 65, height: 38 }, { fontFamily: "Georgia", fontSize: 25, bold: true, color: C.accent });
      text(s, `next-name-${i}`, row[1], { left: x + 75, top: 242, width: 210, height: 38 }, { fontFamily: "Songti SC", fontSize: 25, bold: true });
      text(s, `next-detail-${i}`, row[2], { left: x + 75, top: 302, width: 245, height: 70 }, { fontSize: 18, color: C.muted, bold: true });
    });
    box(s, "command-box", { left: 115, top: 445, width: 1050, height: 105 }, C.ink, C.ink, 8);
    text(s, "command", "PROFILE=smoke BENCHMARK=all ./run_benchmarks.sh", { left: 145, top: 470, width: 990, height: 48 }, { fontFamily: "Menlo", fontSize: 23, bold: true, color: C.paper, alignment: "center" });
    text(s, "close", "现在可以真实声称：核心训练跑过、失败被诊断、官方评测路径可执行；仍不能声称论文指标已复现。", { left: 135, top: 590, width: 1010, height: 48 }, { fontFamily: "Songti SC", fontSize: 22, bold: true, color: C.accentDark, alignment: "center" });
    notes(s, "收束到一键评测入口。BFCL 使用官方 bfcl-eval；TAU 固定论文相关的 v0.2.0。dry-run manifest 已验证命令和规模，但没有执行耗时 benchmark，因此没有 benchmark score。", ["https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard", "https://github.com/sierra-research/tau2-bench/tree/v0.2.0", path.join(PROJECT, "src/agentic_repro/benchmark_runner.py"), path.join(PROJECT, "artifacts/benchmarks/planned_manifest.json")]);
  }

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(path.join(PREVIEW, `${stem}.png`), await presentation.export({ slide, format: "png", scale: 1 }));
    await fs.writeFile(path.join(PREVIEW, `${stem}.layout.json`), await (await slide.export({ format: "layout" })).text());
  }
  await writeBlob(path.join(PREVIEW, "montage.webp"), await presentation.export({ format: "webp", montage: true, scale: 1 }));
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(OUT);
  console.log(OUT);
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
