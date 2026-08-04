import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const PROJECT = path.resolve(SCRIPT_DIR, "..");
const WORKSPACE = path.resolve(PROJECT, "../..");
const TOOL = process.env.ARTIFACT_TOOL_DIST || "/Users/hongbo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";
const { Presentation, PresentationFile } = await import(pathToFileURL(TOOL).href);

const RUN_ROOT = path.resolve(process.argv[2] || process.env.RUN_ROOT || "");
if (!RUN_ROOT || RUN_ROOT === path.parse(RUN_ROOT).root) {
  throw new Error("Usage: node scripts/build_curriculum_deck.mjs <downloaded-run-root>");
}
const OUT = path.resolve(process.env.PPTX_OUT || path.join(PROJECT, "slides/AgenticQwen_Curriculum_Cloud.pptx"));
const PREVIEW = path.resolve(process.env.PPTX_PREVIEW || path.join(WORKSPACE, "work/slides/curriculum-cloud-rendered"));

async function json(file) { return JSON.parse(await fs.readFile(file, "utf8")); }
async function jsonl(file) {
  return (await fs.readFile(file, "utf8")).split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
}
async function bytes(file) {
  const value = await fs.readFile(file);
  return value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength);
}
async function saveBlob(file, blob) { await fs.writeFile(file, new Uint8Array(await blob.arrayBuffer())); }

const run = await json(path.join(RUN_ROOT, "run_summary.json"));
const verify = await json(path.join(RUN_ROOT, "verification.json"));
const stage1 = await json(path.join(RUN_ROOT, "stage1/summary.json"));
const stage2 = await json(path.join(RUN_ROOT, "stage2/summary.json"));
const synth = await json(path.join(RUN_ROOT, "stage2/synthesis_manifest.json"));
const config = await json(path.join(RUN_ROOT, "resolved_config.json"));
let bfcl = {};
try { bfcl = await json(path.join(RUN_ROOT, "benchmarks/bfcl_smoke/manifest.json")); } catch {}
const bfclStatus = bfcl.overall_status || "NOT RUN";
const traces = await jsonl(path.join(RUN_ROOT, "stage2/eval_final_traces.jsonl"));
if (verify.overall_status !== "PASS") {
  throw new Error(`Refusing final deck: curriculum=${verify.overall_status}`);
}
if (process.env.ALLOW_SYNTHETIC_FIXTURE !== "1") {
  if (!["gpu_observed", "cloud_gpu_observed"].includes(run.evidence_class)) throw new Error("Refusing final deck: run is not GPU-observed evidence");
}

const C = {
  navy: "#071A2B", navy2: "#0E2B3D", ink: "#102330", teal: "#18A6A6",
  teal2: "#0B797B", copper: "#C56D4A", gold: "#D6A84B", cream: "#F7F1E8",
  paper: "#FFFDFC", mist: "#E8F0F0", line: "#CAD9D8", muted: "#64747B",
  green: "#228B67", red: "#B44C4C", white: "#FFFFFF", paleGold: "#FAF2DE",
  paleTeal: "#E5F5F3", paleRed: "#F9E9E6",
};
const W = 1280, H = 720;
const FONT = { body: "PingFang SC", serif: "Songti SC", display: "Bodoni 72", script: "Xingkai SC", mono: "Menlo" };

function rect(slide, name, position, fill, lineFill = fill, radius = 0) {
  return slide.shapes.add({ geometry: radius ? "roundRect" : "rect", name, position, fill,
    line: { style: "solid", fill: lineFill, width: lineFill === fill ? 0 : 1 }, ...(radius ? { borderRadius: radius } : {}) });
}
function txt(slide, name, value, position, style = {}) {
  const shape = slide.shapes.add({ geometry: "textbox", name, position, fill: "none", line: { style: "solid", fill: "none", width: 0 } });
  shape.text = String(value);
  shape.text.style = { fontFamily: FONT.body, fontSize: 18, color: C.ink, verticalAlignment: "top", ...style };
  return shape;
}
function rule(slide, x, y, width, color = C.line, weight = 1) {
  return slide.shapes.add({ geometry: "line", position: { left: x, top: y, width, height: 0 }, fill: "none", line: { style: "solid", fill: color, width: weight } });
}
function pill(slide, name, label, x, y, width, fill = C.paleTeal, color = C.teal2) {
  rect(slide, `${name}-bg`, { left: x, top: y, width, height: 30 }, fill, fill, 15);
  txt(slide, `${name}-text`, label, { left: x + 10, top: y + 5, width: width - 20, height: 20 }, { fontSize: 12, bold: true, color, alignment: "center", letterSpacing: 0.7 });
}
function note(slide, body, sources) {
  slide.speakerNotes.textFrame.setText(`${body}\n\n[Sources]\n${sources.map((s) => `- ${s}`).join("\n")}\n[/Sources]`);
}
function frame(p, number, chapter, title, subtitle = "") {
  const s = p.slides.add(); s.background.fill = C.paper;
  rect(s, `rail-${number}`, { left: 0, top: 0, width: 30, height: H }, C.navy);
  txt(s, `flourish-${number}`, "长程多轮智能体强化学习", { left: 70, top: 29, width: 420, height: 32 }, { fontFamily: FONT.script, fontSize: 20, bold: true, color: C.copper, letterSpacing: 1.8 });
  txt(s, `chapter-${number}`, chapter.toUpperCase(), { left: 760, top: 33, width: 420, height: 24 }, { fontFamily: FONT.display, fontSize: 13, bold: true, color: C.teal2, letterSpacing: 2.2, alignment: "right" });
  txt(s, `title-${number}`, title, { left: 70, top: 78, width: 1095, height: 55 }, { fontFamily: FONT.serif, fontSize: 35, bold: true, color: C.navy });
  if (subtitle) txt(s, `subtitle-${number}`, subtitle, { left: 72, top: 137, width: 1080, height: 34 }, { fontSize: 16, color: C.muted });
  rule(s, 70, subtitle ? 184 : 151, 1110, C.line, 1);
  txt(s, `page-${number}`, String(number).padStart(2, "0"), { left: 1125, top: 670, width: 55, height: 20 }, { fontFamily: FONT.display, fontSize: 12, color: C.muted, alignment: "right" });
  return s;
}
function chapterSlide(p, number, roman, zh, en, body) {
  const s = p.slides.add(); s.background.fill = C.navy;
  rect(s, `copper-${number}`, { left: 0, top: 0, width: 18, height: H }, C.copper);
  txt(s, `chapter-script-${number}`, "长程智能体 · 强化学习", { left: 84, top: 68, width: 500, height: 55 }, { fontFamily: FONT.script, fontSize: 29, bold: true, color: C.gold, letterSpacing: 2 });
  txt(s, `roman-${number}`, roman, { left: 840, top: 34, width: 320, height: 240 }, { fontFamily: FONT.display, fontSize: 166, color: C.navy2, bold: true, alignment: "right" });
  rule(s, 86, 171, 170, C.copper, 5);
  txt(s, `zh-${number}`, zh, { left: 82, top: 214, width: 800, height: 86 }, { fontFamily: FONT.serif, fontSize: 57, bold: true, color: C.cream });
  txt(s, `en-${number}`, en.toUpperCase(), { left: 87, top: 318, width: 900, height: 45 }, { fontFamily: FONT.display, fontSize: 24, color: C.teal, bold: true, letterSpacing: 3 });
  txt(s, `body-${number}`, body, { left: 87, top: 410, width: 830, height: 100 }, { fontSize: 21, color: "#C7D6DC", lineSpacing: 1.18 });
  pill(s, `chapter-pill-${number}`, `CHAPTER ${String(number).padStart(2, "0")}`, 87, 592, 150, C.copper, C.white);
  return s;
}
function metric(slide, key, x, y, value, label, color = C.teal) {
  rule(slide, x, y, 210, color, 4);
  txt(slide, `${key}-value`, value, { left: x, top: y + 18, width: 210, height: 55 }, { fontFamily: FONT.display, fontSize: 36, bold: true, color: C.navy });
  txt(slide, `${key}-label`, label, { left: x, top: y + 78, width: 210, height: 44 }, { fontSize: 14, color: C.muted, bold: true });
}
function pct(v) { return `${(100 * Number(v || 0)).toFixed(1)}%`; }
function delta(a, b) { const d = 100 * (Number(b || 0) - Number(a || 0)); return `${d >= 0 ? "+" : ""}${d.toFixed(1)} pt`; }
function short(v) { return v ? `${String(v).slice(0, 13)}…` : "—"; }
function failureNames(a, b) { return [...new Set([...Object.keys(a.failure_counts || {}), ...Object.keys(b.failure_counts || {})])]; }
function scoreRow(variant) {
  const row = (bfcl.variants?.[variant]?.score_rows || [])[0] || {};
  return row["Multi Turn Acc"] ?? row["Overall Acc"] ?? "—";
}

async function main() {
  await fs.mkdir(path.dirname(OUT), { recursive: true });
  await fs.mkdir(PREVIEW, { recursive: true });
  const p = Presentation.create({ slideSize: { width: W, height: H } });
  const paperFig = await bytes(path.join(PROJECT, "agenticqwen_report/images/figure1_dual_flywheel.png"));
  const stageFigure = await bytes(path.join(PROJECT, "agenticqwen_report/images/figure4_curriculum_stage_flow.png"));
  const gpu = stage2.runtime?.gpu?.name || "Cloud GPU";
  const successTrace = traces.find((x) => x.success) || traces[0];

  // 1 — Cover
  {
    const s = p.slides.add(); s.background.fill = C.navy;
    rect(s, "cover-copper", { left: 0, top: 0, width: 20, height: H }, C.copper);
    rect(s, "cover-teal", { left: 1010, top: 0, width: 270, height: H }, C.navy2);
    txt(s, "cover-flourish", "长程多轮智能体强化学习", { left: 84, top: 62, width: 650, height: 60 }, { fontFamily: FONT.script, fontSize: 31, bold: true, color: C.gold, letterSpacing: 3 });
    rule(s, 86, 154, 185, C.copper, 5);
    txt(s, "cover-title", "AgenticQwen\nCurriculum Lab", { left: 80, top: 195, width: 800, height: 160 }, { fontFamily: FONT.display, fontSize: 70, bold: true, color: C.cream, lineSpacing: 0.92 });
    txt(s, "cover-sub", "训练 → 失败诊断 → 困难数据合成 → 再训练", { left: 86, top: 392, width: 790, height: 48 }, { fontFamily: FONT.serif, fontSize: 27, bold: true, color: C.white });
    txt(s, "cover-meta", `Qwen3-8B · response-token QLoRA-GRPO · ${gpu}\n${run.output_root || RUN_ROOT}`, { left: 88, top: 476, width: 790, height: 72 }, { fontSize: 16, color: "#ADC3CC" });
    txt(s, "cover-02", "02", { left: 1040, top: 90, width: 170, height: 170 }, { fontFamily: FONT.display, fontSize: 120, bold: true, color: C.teal, alignment: "center" });
    txt(s, "cover-side", "REAL RUN\nFAILURE LOOP\nAUDITED", { left: 1030, top: 330, width: 195, height: 120 }, { fontFamily: FONT.display, fontSize: 20, bold: true, color: C.cream, alignment: "center", lineSpacing: 1.25, letterSpacing: 1.2 });
    pill(s, "cover-pass", "CURRICULUM PASS", 86, 606, 190, C.green, C.white);
    pill(s, "cover-bfcl", `BFCL ${bfclStatus}`, 294, 606, 190, C.copper, C.white);
    note(s, "开场先讲清证据边界：这是一次真实小规模云端训练闭环，不是论文八卡/100K 数据指标复刻。", [path.join(RUN_ROOT, "run_summary.json"), path.join(RUN_ROOT, "verification.json"), "https://arxiv.org/abs/2604.21590"]);
  }

  // 2 — mental model
  {
    const s = frame(p, 2, "Project map", "拿到这个项目，你应该先建立四层心智模型", "Agentic RL 不是“模型会调工具”这一件事，而是 policy、environment、reward、curriculum 的闭环。");
    const blocks = [
      ["01", "POLICY", "Qwen3-8B 生成完整 assistant response 与工具调用", C.teal],
      ["02", "ENVIRONMENT", "七个 stateful tools 执行读写与真实状态转移", C.copper],
      ["03", "REWARD", "exact outcome + process shaping，按 group 求相对优势", C.gold],
      ["04", "CURRICULUM", "从真实失败类型制造 decoy、timeout 与 replay", C.green],
    ];
    blocks.forEach((b, i) => {
      const x = 72 + i * 282;
      rect(s, `map-${i}`, { left: x, top: 238, width: 244, height: 292 }, i % 2 ? C.cream : C.paper, b[3], 10);
      txt(s, `map-num-${i}`, b[0], { left: x + 20, top: 258, width: 65, height: 38 }, { fontFamily: FONT.display, fontSize: 25, bold: true, color: b[3] });
      txt(s, `map-name-${i}`, b[1], { left: x + 20, top: 322, width: 204, height: 35 }, { fontFamily: FONT.display, fontSize: 18, bold: true, color: C.navy, letterSpacing: 1 });
      txt(s, `map-body-${i}`, b[2], { left: x + 20, top: 390, width: 204, height: 96 }, { fontSize: 18, color: C.muted, bold: true, lineSpacing: 1.2 });
    });
    txt(s, "map-foot", "训练完整性的证据链：trajectory → reward → optimizer step → adapter hash → fresh-process replay", { left: 120, top: 584, width: 1010, height: 38 }, { fontSize: 18, bold: true, color: C.teal2, alignment: "center" });
    note(s, "这是给入门同学的总地图。后面的每一页都落到这四层中的一个证据问题。", [path.join(PROJECT, "docs/curriculum_protocol.md"), path.join(RUN_ROOT, "resolved_config.json")]);
  }

  // 3 — paper to project
  {
    const s = frame(p, 3, "Paper → project", "论文方法如何被拆成一条可执行的工程链路", "保留双飞轮的因果结构，缩小模型、数据和 benchmark 规模，并明确不支持的 claim。");
    s.images.add({ blob: paperFig, contentType: "image/png", alt: "AgenticQwen dual flywheel", fit: "contain", position: { left: 70, top: 214, width: 570, height: 340 } });
    const rows = [
      ["论文", "≈100K 数据 · 235B simulator/judge · 8×H100", C.copper],
      ["本次", `8+12 训练任务 · 24 optimizer steps · 1×${gpu}`, C.teal],
      ["保留", "多轮 response token · stateful tools · GRPO · failure loop", C.green],
      ["不声称", "论文 47.4 平均分 · 全量 TAU-2/BFCL · 统计显著性", C.red],
    ];
    rows.forEach((r, i) => {
      const y = 225 + i * 94;
      rule(s, 700, y + 7, 36, r[2], 5);
      txt(s, `paper-k-${i}`, r[0], { left: 758, top: y - 5, width: 100, height: 30 }, { fontFamily: FONT.serif, fontSize: 20, bold: true, color: r[2] });
      txt(s, `paper-v-${i}`, r[1], { left: 758, top: y + 30, width: 410, height: 48 }, { fontSize: 16, color: C.muted, bold: true });
    });
    note(s, "先比较论文与本项目尺度，避免把工程闭环 PASS 误解成论文分数复现。", ["https://arxiv.org/abs/2604.21590", path.join(RUN_ROOT, "resolved_config.json")]);
  }

  // 4 — architecture
  {
    const s = frame(p, 4, "System architecture", "一次 rollout 究竟发生了什么？", "模型不能直接改业务状态；所有 side effect 都经过工具、先决条件和 verifier。");
    const nodes = [
      ["PROMPT", "模糊退款诉求", 78, C.cream], ["POLICY", "Qwen3-8B\nresponse tokens", 304, C.paleTeal],
      ["TOOLS", "7 个工具\nobservation", 548, C.paleGold], ["STATE", "身份/订单/支付\n确认/退款", 792, C.paleTeal],
      ["REWARD", "outcome +\n0.30 process", 1036, C.paleRed],
    ];
    for (let i = 0; i < nodes.length - 1; i++) {
      s.shapes.add({ geometry: "line", position: { left: nodes[i][2] + 172, top: 361, width: nodes[i + 1][2] - nodes[i][2] - 172, height: 0 }, fill: "none", line: { style: "solid", fill: C.teal2, width: 2 }, tail: { type: "arrow", width: "med", length: "med" } });
    }
    const shapes = nodes.map((n, i) => {
      const sh = rect(s, `arch-${i}`, { left: n[2], top: 285, width: 172, height: 152 }, n[3], i === 4 ? C.copper : C.line, 12);
      txt(s, `arch-k-${i}`, n[0], { left: n[2] + 15, top: 307, width: 142, height: 25 }, { fontFamily: FONT.display, fontSize: 14, bold: true, color: i === 4 ? C.copper : C.teal2, alignment: "center", letterSpacing: 1 });
      txt(s, `arch-v-${i}`, n[1], { left: n[2] + 15, top: 355, width: 142, height: 60 }, { fontSize: 18, bold: true, color: C.navy, alignment: "center" });
      return sh;
    });
    txt(s, "arch-loop", "↺ tool observation 回到同一对话，直到完成或 10 次上限", { left: 310, top: 500, width: 660, height: 40 }, { fontFamily: FONT.serif, fontSize: 20, bold: true, color: C.copper, alignment: "center" });
    metric(s, "arch-tools", 170, 570, "7", "typed tools", C.teal);
    metric(s, "arch-group", 450, 570, "4", "rollouts / group", C.copper);
    metric(s, "arch-max", 730, 570, "10", "max tool turns", C.gold);
    metric(s, "arch-token", 1010, 570, "1,400", "max completion tokens", C.green);
    note(s, "强调模型只生成动作，环境负责执行，reward 从最终状态读取。", [path.join(PROJECT, "src/agentic_repro/curriculum_env.py"), path.join(PROJECT, "src/agentic_repro/curriculum_train.py")]);
  }

  // 5 — environment safety
  {
    const s = frame(p, 5, "State machine", "退款不是一句话：它是一条受约束的状态机", "训练目标既包含任务成功，也包含 read-before-write、精确确认和错误恢复。");
    const steps = ["索取身份", "验证客户", "读取订单", "检查支付", "读取政策", "精确确认", "幂等退款"];
    for (let i = 0; i < steps.length - 1; i++) rule(s, 136 + i * 158, 329, 100, i < 4 ? C.teal : C.copper, 3);
    steps.forEach((label, i) => {
      const x = 78 + i * 158;
      rect(s, `state-dot-${i}`, { left: x, top: 300, width: 58, height: 58 }, i === 6 ? C.green : C.navy, i === 6 ? C.green : C.navy, 29);
      txt(s, `state-n-${i}`, i + 1, { left: x, top: 312, width: 58, height: 30 }, { fontFamily: FONT.display, fontSize: 20, bold: true, color: C.white, alignment: "center" });
      txt(s, `state-label-${i}`, label, { left: x - 26, top: 385, width: 112, height: 48 }, { fontSize: 16, bold: true, color: C.ink, alignment: "center" });
    });
    rect(s, "state-guard", { left: 130, top: 500, width: 1020, height: 92 }, C.navy, C.navy, 10);
    txt(s, "state-guard-text", "WRITE GUARD  ·  verified identity ∧ orders read ∧ payment read ∧ policy read ∧ exact confirmation ∧ duplicate charge ∧ exact amount ∧ idempotency key", { left: 165, top: 524, width: 950, height: 44 }, { fontFamily: FONT.mono, fontSize: 15, bold: true, color: C.cream, alignment: "center" });
    note(s, "这页适合面试时讲 environment design：不可逆动作由代码硬门控制，模型无法通过语言绕过。", [path.join(PROJECT, "src/agentic_repro/curriculum_env.py")]);
  }

  // 6 — GRPO
  {
    const s = frame(p, 6, "Optimization", "为什么用 response-token GRPO，而不是动作分类？", "每个样本是一条完整多轮轨迹；梯度只覆盖 assistant completion token。");
    rect(s, "grpo-formula", { left: 75, top: 230, width: 520, height: 300 }, C.navy, C.navy, 12);
    txt(s, "grpo-formula-title", "GROUP-RELATIVE ADVANTAGE", { left: 110, top: 260, width: 450, height: 30 }, { fontFamily: FONT.display, fontSize: 17, bold: true, color: C.gold, alignment: "center", letterSpacing: 1.5 });
    txt(s, "grpo-formula-main", "Aᵢ = (Rᵢ − μgroup) / (σgroup + ε)", { left: 105, top: 337, width: 460, height: 54 }, { fontFamily: "Georgia", fontSize: 30, bold: true, color: C.white, alignment: "center" });
    txt(s, "grpo-reward", "R = 𝟙[exact outcome] + 0.30 × Rprocess", { left: 110, top: 428, width: 450, height: 38 }, { fontFamily: "Georgia", fontSize: 20, color: "#BFD9DD", alignment: "center" });
    const bullets = [
      ["01", "同 prompt 采样 4 条 trajectory", "不依赖单独 value model"],
      ["02", "DAPO-style clipped objective", "LoRA 更新 Q/K/V/O + MLP"],
      ["03", "NF4 base + BF16 compute", `在单张 ${gpu} 上完成`],
      ["04", "过程分只打破失败平局", "exact outcome 仍是主目标"],
    ];
    bullets.forEach((b, i) => {
      const y = 227 + i * 90;
      txt(s, `grpo-num-${i}`, b[0], { left: 675, top: y, width: 55, height: 30 }, { fontFamily: FONT.display, fontSize: 18, bold: true, color: i % 2 ? C.copper : C.teal });
      txt(s, `grpo-head-${i}`, b[1], { left: 742, top: y - 3, width: 400, height: 28 }, { fontSize: 19, bold: true, color: C.navy });
      txt(s, `grpo-sub-${i}`, b[2], { left: 742, top: y + 34, width: 400, height: 28 }, { fontSize: 15, color: C.muted });
    });
    note(s, "response-token 是本次相对早期单 token demo 的关键升级：整个 assistant completion 参与策略优化。", [path.join(RUN_ROOT, "resolved_config.json"), "https://huggingface.co/docs/trl/en/grpo_trainer"]);
  }

  // 7 — chapter divider
  {
    const s = chapterSlide(p, 7, "II", "课程学习闭环", "DIAGNOSE → SYNTHESIZE → RETRAIN", "真正有价值的不是“又训了一遍”，而是把第一次训练后的残余失败或能力边界变成下一轮可验证的数据分布。");
    note(s, "第二章切入用户最关心的 curriculum：训练—失败—合成更难数据—再训练。", [path.join(RUN_ROOT, "stage2/synthesis_manifest.json")]);
  }

  // 8 — stage1 results
  {
    const s = frame(p, 8, "Stage 1", "第一次训练之后，模型在哪里失败？", "报告保留负结果：成功率下降也不会被隐藏，curriculum 依据 failure composition 而不是漂亮样例。");
    const before = stage1.failure_summary_before, after = stage1.failure_summary_after;
    metric(s, "s1-before", 85, 235, pct(before.success_rate), "probe success · before", C.muted);
    metric(s, "s1-after", 345, 235, pct(after.success_rate), "probe success · after", C.teal);
    metric(s, "s1-delta", 605, 235, delta(before.success_rate, after.success_rate), "success delta", Number(after.success_rate) >= Number(before.success_rate) ? C.green : C.red);
    metric(s, "s1-steps", 865, 235, String(stage1.global_step), "optimizer steps", C.copper);
    const names = failureNames(before, after).slice(0, 5);
    txt(s, "failure-head", "TERMINAL FAILURE TAXONOMY", { left: 88, top: 410, width: 400, height: 28 }, { fontFamily: FONT.display, fontSize: 15, bold: true, color: C.copper, letterSpacing: 1.3 });
    names.forEach((name, i) => {
      const y = 458 + i * 36;
      txt(s, `failure-name-${i}`, name, { left: 92, top: y, width: 380, height: 24 }, { fontFamily: FONT.mono, fontSize: 14, color: C.ink });
      rect(s, `failure-track-${i}`, { left: 490, top: y + 3, width: 500, height: 14 }, C.mist, C.mist, 7);
      const n = Math.max(Number(before.failure_counts?.[name] || 0), Number(after.failure_counts?.[name] || 0), 1);
      rect(s, `failure-before-${i}`, { left: 490, top: y + 3, width: Math.max(5, 175 * Number(before.failure_counts?.[name] || 0) / n), height: 14 }, "#9AA8AD", "#9AA8AD", 7);
      rect(s, `failure-after-${i}`, { left: 690, top: y + 3, width: Math.max(5, 250 * Number(after.failure_counts?.[name] || 0) / n), height: 14 }, C.copper, C.copper, 7);
      txt(s, `failure-count-${i}`, `${before.failure_counts?.[name] || 0} → ${after.failure_counts?.[name] || 0}`, { left: 1020, top: y - 2, width: 90, height: 22 }, { fontFamily: FONT.display, fontSize: 15, bold: true, color: C.navy, alignment: "right" });
    });
    note(s, "Stage 1 后在冻结 probe 上生成轨迹并分类 terminal state；该 trace 的 hash 是合成输入证据。", [path.join(RUN_ROOT, "stage1/summary.json"), path.join(RUN_ROOT, "stage1/eval_probe_traces.jsonl")]);
  }

  // 9 — synthesis
  {
    const s = frame(p, 9, "Hard-data synthesis", "残余失败或能力边界，都会进入下一轮数据生成器", "ground truth 由确定性代码生成；模型自然语言永远不充当 charge、amount 或 verifier target。");
    s.shapes.add({ geometry: "line", position: { left: 375, top: 390, width: 115, height: 0 }, fill: "none", line: { style: "solid", fill: C.copper, width: 3 }, tail: { type: "arrow", width: "med", length: "med" } });
    s.shapes.add({ geometry: "line", position: { left: 790, top: 390, width: 115, height: 0 }, fill: "none", line: { style: "solid", fill: C.teal, width: 3 }, tail: { type: "arrow", width: "med", length: "med" } });
    const left = rect(s, "synth-left", { left: 75, top: 240, width: 300, height: 300 }, C.paleRed, C.copper, 12);
    const mid = rect(s, "synth-mid", { left: 490, top: 240, width: 300, height: 300 }, C.navy, C.navy, 12);
    const right = rect(s, "synth-right", { left: 905, top: 240, width: 300, height: 300 }, C.paleTeal, C.teal, 12);
    txt(s, "synth-left-k", "OBSERVED FAILURES", { left: 105, top: 270, width: 240, height: 30 }, { fontFamily: FONT.display, fontSize: 15, bold: true, color: C.copper, alignment: "center", letterSpacing: 1 });
    txt(s, "synth-left-v", Object.entries(synth.source_summary?.failure_counts || {}).map(([k, v]) => `${k}  ×${v}`).join("\n") || "no observed failure\n→ fallback taxonomy", { left: 105, top: 330, width: 240, height: 150 }, { fontFamily: FONT.mono, fontSize: 14, color: C.ink, alignment: "center" });
    txt(s, "synth-mid-k", "DETERMINISTIC\nTRANSFORMS", { left: 530, top: 272, width: 220, height: 55 }, { fontFamily: FONT.display, fontSize: 16, bold: true, color: C.gold, alignment: "center", letterSpacing: 1.2 });
    txt(s, "synth-mid-v", "+ decoy orders\n+ transient timeout\n+ exact confirmation\n+ old-task replay", { left: 535, top: 360, width: 210, height: 120 }, { fontSize: 18, bold: true, color: C.white, alignment: "center", lineSpacing: 1.2 });
    txt(s, "synth-right-k", "STAGE-2 MIX", { left: 940, top: 270, width: 230, height: 30 }, { fontFamily: FONT.display, fontSize: 15, bold: true, color: C.teal2, alignment: "center", letterSpacing: 1 });
    txt(s, "synth-right-v", `${synth.hard_tasks?.count || 0} hard tasks\n+\n${(synth.replay_tasks || []).length} replay tasks`, { left: 945, top: 348, width: 220, height: 105 }, { fontFamily: FONT.display, fontSize: 27, bold: true, color: C.navy, alignment: "center" });
    txt(s, "synth-hash", `source trace  ${short(synth.source_trace_sha256)}    stage2 data  ${short(synth.stage2_train?.sha256)}`, { left: 185, top: 590, width: 910, height: 30 }, { fontFamily: FONT.mono, fontSize: 14, bold: true, color: C.muted, alignment: "center" });
    note(s, "此页是 curriculum 的核心：failure category 到 task perturbation 的确定性映射，并保留 trace/data hash。", [path.join(RUN_ROOT, "stage2/synthesis_manifest.json"), path.join(RUN_ROOT, "tasks/stage2_hard.jsonl")]);
  }

  // 10 — stage2 results
  {
    const s = frame(p, 10, "Stage 2", "从 Stage-1 adapter 继续训练，效果是否真的改变？", "算法效果和工程完整性分开判断：即使指标不升，只要审计通过，它仍是一个真实负结果。");
    const before = stage2.failure_summary_before, after = stage2.failure_summary_after;
    const bars = [
      ["Final holdout success", before.success_rate, after.success_rate, C.teal],
      ["Mean combined reward", Math.max(0, Number(before.mean_reward || 0) / 1.3), Math.max(0, Number(after.mean_reward || 0) / 1.3), C.copper],
    ];
    bars.forEach((b, i) => {
      const y = 265 + i * 150;
      txt(s, `s2-label-${i}`, b[0], { left: 90, top: y - 8, width: 300, height: 30 }, { fontSize: 20, bold: true, color: C.navy });
      rect(s, `s2-track-a-${i}`, { left: 410, top: y, width: 570, height: 22 }, C.mist, C.mist, 11);
      rect(s, `s2-a-${i}`, { left: 410, top: y, width: Math.max(6, 570 * Number(b[1])), height: 22 }, "#9AA8AD", "#9AA8AD", 11);
      rect(s, `s2-track-b-${i}`, { left: 410, top: y + 48, width: 570, height: 22 }, C.mist, C.mist, 11);
      rect(s, `s2-b-${i}`, { left: 410, top: y + 48, width: Math.max(6, 570 * Number(b[2])), height: 22 }, b[3], b[3], 11);
      txt(s, `s2-value-${i}`, `${i === 0 ? pct(b[1]) : Number(before.mean_reward).toFixed(3)}  →  ${i === 0 ? pct(b[2]) : Number(after.mean_reward).toFixed(3)}`, { left: 1000, top: y + 16, width: 160, height: 38 }, { fontFamily: FONT.display, fontSize: 20, bold: true, color: Number(b[2]) >= Number(b[1]) ? C.green : C.red, alignment: "right" });
    });
    metric(s, "s2-step", 120, 570, String(stage2.global_step), "Stage-2 optimizer steps", C.teal);
    metric(s, "s2-time", 420, 570, `${Math.round(stage2.training_seconds / 60)}m`, "training wall time", C.copper);
    metric(s, "s2-replay", 720, 570, String(verify.fresh_replay?.episodes || 0), "fresh-process episodes", C.green);
    metric(s, "s2-gpu", 1020, 570, "1×", "RTX PRO 6000 · 96GB", C.gold);
    note(s, "结果页必须如实展示正负 delta；不要用完整性 PASS 替代性能提升。", [path.join(RUN_ROOT, "stage2/summary.json"), path.join(RUN_ROOT, "fresh_replay/summary.json")]);
  }

  // 11 — trajectory
  {
    const s = frame(p, 11, "Trajectory evidence", "一条完整轨迹：从模糊诉求到安全 side effect", `task=${successTrace?.task_id || "—"} · reward=${Number(successTrace?.combined_reward || 0).toFixed(3)} · ${successTrace?.failure_type || "—"}`);
    const events = (successTrace?.state?.events || []).slice(0, 7);
    rect(s, "traj-spine", { left: 122, top: 244, width: 3, height: Math.max(36, (events.length - 1) * 56) }, C.line, C.line);
    events.forEach((e, i) => {
      const y = 225 + i * 56;
      rect(s, `traj-dot-${i}`, { left: 105, top: y, width: 36, height: 36 }, e.ok ? C.teal : C.red, e.ok ? C.teal : C.red, 18);
      txt(s, `traj-num-${i}`, e.turn, { left: 105, top: y + 7, width: 36, height: 20 }, { fontFamily: FONT.display, fontSize: 14, bold: true, color: C.white, alignment: "center" });
      txt(s, `traj-tool-${i}`, e.tool, { left: 175, top: y - 1, width: 250, height: 28 }, { fontFamily: FONT.mono, fontSize: 16, bold: true, color: C.navy });
      const payload = JSON.stringify(e.payload || {}).replace(/\s+/g, " ");
      txt(s, `traj-obs-${i}`, payload.length > 90 ? `${payload.slice(0, 87)}…` : payload, { left: 455, top: y - 1, width: 680, height: 34 }, { fontFamily: FONT.mono, fontSize: 12, color: C.muted });
    });
    pill(s, "traj-outcome", successTrace?.success ? "EXACT OUTCOME PASS" : "NEGATIVE TRAJECTORY", 860, 604, 250, successTrace?.success ? C.green : C.red, C.white);
    note(s, "轨迹页用 environment event ledger，而不是模型自述，证明每次读写、observation 和最终状态。", [path.join(RUN_ROOT, "stage2/eval_final_traces.jsonl")]);
  }

  // 12 — benchmark boundary
  {
    const s = frame(p, 12, "Benchmark boundary", "官方 benchmark 没跑，就不应该出现一个分数", "本次验证的是自建 stateful curriculum；BFCL/TAU-2 代码路径保留，但状态必须明确写成 NOT RUN。");
    txt(s, "benchmark-status", bfclStatus, { left: 75, top: 230, width: 470, height: 150 }, { fontFamily: FONT.display, fontSize: 78, bold: true, color: C.copper, alignment: "center" });
    txt(s, "benchmark-label", "BFCL-V4 MULTI-TURN", { left: 95, top: 395, width: 430, height: 38 }, { fontFamily: FONT.display, fontSize: 18, bold: true, color: C.navy, alignment: "center", letterSpacing: 2 });
    rule(s, 640, 230, 470, C.teal, 4);
    txt(s, "benchmark-ready", "代码已经具备", { left: 640, top: 258, width: 470, height: 42 }, { fontFamily: FONT.serif, fontSize: 27, bold: true, color: C.navy });
    txt(s, "benchmark-list", "• 固定 ID 的 base / adapter 对照\n• 官方 multi-turn 结果与评分目录隔离\n• smoke 与 4×200 paper profile 分开\n• 缺 manifest 时报告和 PPT 不生成分数", { left: 640, top: 335, width: 475, height: 180 }, { fontSize: 20, color: C.muted, lineSpacing: 1.25 });
    txt(s, "benchmark-boundary", "能写：curriculum micro-run PASS    不能写：论文 benchmark 已复现", { left: 175, top: 575, width: 930, height: 42 }, { fontFamily: FONT.serif, fontSize: 23, bold: true, color: C.teal2, alignment: "center" });
    note(s, "这页主动声明 BFCL 未运行；代码就绪不等于实验完成。", [path.join(PROJECT, "docs/benchmark_protocol.md"), "https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard"]);
  }

  // 13 — audit
  {
    const s = frame(p, 13, "Integrity audit", "怎样证明不是“看起来训练了”？", "把一次 ML run 拆成可证伪的 gate：任何关键证据缺失，最终报告和 PPT 都拒绝生成。");
    const checks = verify.checks || [];
    checks.slice(0, 8).forEach((c, i) => {
      const col = i % 2, row = Math.floor(i / 2);
      const x = 85 + col * 560, y = 225 + row * 92;
      rect(s, `audit-bg-${i}`, { left: x, top: y, width: 520, height: 72 }, c.status === "PASS" ? C.paleTeal : C.paleRed, c.status === "PASS" ? C.teal : C.red, 9);
      pill(s, `audit-pill-${i}`, c.status, x + 18, y + 20, 75, c.status === "PASS" ? C.green : C.red, C.white);
      txt(s, `audit-name-${i}`, c.name, { left: x + 110, top: y + 17, width: 380, height: 38 }, { fontSize: 16, bold: true, color: C.navy });
    });
    rect(s, "audit-hash", { left: 100, top: 605, width: 1060, height: 50 }, C.navy, C.navy, 8);
    txt(s, "audit-hash-text", `S1 ${short(stage1.adapter_weights_sha256)}   →   S2 ${short(stage2.adapter_weights_sha256)}   ·   child PID ${verify.fresh_replay?.process?.pid || "—"}`, { left: 135, top: 619, width: 990, height: 24 }, { fontFamily: FONT.mono, fontSize: 14, bold: true, color: C.cream, alignment: "center" });
    note(s, "审计门覆盖 adapter、权重变化、Stage-2 继承、数据隔离、轨迹、计划步数和真正的子进程重载。", [path.join(RUN_ROOT, "verification.json"), path.join(RUN_ROOT, "stage1/summary.json"), path.join(RUN_ROOT, "stage2/summary.json")]);
  }

  // 14 — limitations and interview
  {
    const s = frame(p, 14, "Honest self-check", "项目已经完整到哪里，哪些仍然不能写？", "把 COMPLETE、CODE_READY 与 BLOCKED_RESOURCE 分开，是这份复现最重要的研究纪律。");
    const cols = [
      ["COMPLETE", C.green, ["两阶段 response-token GRPO", "frontier hard-task curriculum", "fresh-process adapter replay", "V1 失败 → V2 修复审计"]],
      ["CODE READY / NOT RUN", C.copper, ["BFCL-V4 smoke / paper profile", "TAU-2 Avg@4", "3 seeds × ablation matrix", "更长训练与 replay-ratio sweep"]],
      ["BLOCKED RESOURCE", C.red, ["论文约 100K 数据飞轮", "Qwen3-235B simulator/judge", "8×H100 原配方", "论文 47.4 benchmark claim"]],
    ];
    cols.forEach((c, i) => {
      const x = 75 + i * 385;
      rect(s, `limit-${i}`, { left: x, top: 225, width: 345, height: 345 }, i === 0 ? C.paleTeal : i === 1 ? C.paleGold : C.paleRed, c[1], 12);
      txt(s, `limit-k-${i}`, c[0], { left: x + 24, top: 250, width: 297, height: 34 }, { fontFamily: FONT.display, fontSize: 17, bold: true, color: c[1], alignment: "center", letterSpacing: 1 });
      c[2].forEach((v, j) => txt(s, `limit-v-${i}-${j}`, `${i === 0 ? "✓" : i === 1 ? "◇" : "×"}  ${v}`, { left: x + 28, top: 320 + j * 56, width: 290, height: 36 }, { fontSize: 16, bold: true, color: C.ink }));
    });
    note(s, "面试时最可信的讲法：先讲闭环与证据，再主动说 paper-scale 没做，最后给出扩展实验。", [path.join(RUN_ROOT, "run_summary.json"), path.join(PROJECT, "README.md")]);
  }

  // 15 — mechanism figure
  {
    const s = p.slides.add(); s.background.fill = C.paper;
    rect(s, "mechanism-rail", { left: 0, top: 0, width: 30, height: H }, C.navy);
    txt(s, "mechanism-title", "失败驱动的课程学习闭环", { left: 78, top: 24, width: 720, height: 45 }, { fontFamily: FONT.serif, fontSize: 32, bold: true, color: C.navy });
    txt(s, "mechanism-subtitle", "基础工具链 → 轨迹审计 → 定向合成 → 带分支的持续训练", { left: 785, top: 34, width: 395, height: 28 }, { fontSize: 15, bold: true, color: C.teal2, alignment: "right" });
    rule(s, 76, 76, 1104, C.line, 1);
    s.images.add({ blob: stageFigure, contentType: "image/png", alt: "失败驱动的课程学习闭环机制图", fit: "contain", position: { left: 170, top: 88, width: 940, height: 584 } });
    txt(s, "mechanism-page", "15", { left: 1125, top: 674, width: 55, height: 20 }, { fontFamily: FONT.display, fontSize: 12, color: C.muted, alignment: "right" });
    note(s, "这张机制图解释为什么第二阶段数据不是随机扩充：失败路径被反译成新的环境状态、用户指令、智能体约束与分支工作流。", [path.join(PROJECT, "agenticqwen_report/images/figure4_curriculum_stage_flow.png"), path.join(RUN_ROOT, "stage2/synthesis_manifest.json")]);
  }

  // 16 — closing
  {
    const s = chapterSlide(p, 16, "✓", "从论文到可审计项目", "PLAN · EXECUTE · VERIFY · REPORT", "现在可以真实地讲：我实现并运行了一个 Qwen3-8B 长程 Agentic RL curriculum，模型经历了真实多轮工具环境、失败诊断、难例再训练与独立重载。下一步是运行官方 benchmark、扩大数据与 seeds，而不是补一个更漂亮的故事。");
    txt(s, "close-command", "./run_curriculum_modal.sh  →  ./finalize_curriculum_report.sh  →  build_curriculum_deck.mjs", { left: 86, top: 535, width: 950, height: 35 }, { fontFamily: FONT.mono, fontSize: 16, bold: true, color: C.cream });
    note(s, "结尾把项目浓缩成一句可用于简历和面试、但不越过证据边界的话。", [path.join(RUN_ROOT, "verification.json"), path.join(PROJECT, "docs/benchmark_protocol.md")]);
  }

  const layouts = [];
  for (const [index, slide] of p.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await saveBlob(path.join(PREVIEW, `${stem}.png`), await p.export({ slide, format: "png", scale: 1 }));
    const layoutText = await (await slide.export({ format: "layout" })).text();
    await fs.writeFile(path.join(PREVIEW, `${stem}.layout.json`), layoutText);
    layouts.push(JSON.parse(layoutText));
  }
  await saveBlob(path.join(PREVIEW, "montage.webp"), await p.export({ format: "webp", montage: true, scale: 1 }));
  const pptx = await PresentationFile.exportPptx(p);
  await pptx.save(OUT);
  const manifest = { status: "PASS", slides: p.slides.items.length, run_root: RUN_ROOT, output: OUT, preview: PREVIEW, curriculum: verify.overall_status, bfcl: bfclStatus };
  await fs.writeFile(`${OUT}.manifest.json`, JSON.stringify(manifest, null, 2) + "\n");
  console.log(JSON.stringify(manifest, null, 2));
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
