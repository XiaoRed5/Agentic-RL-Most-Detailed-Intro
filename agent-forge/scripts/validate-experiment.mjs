import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

const target = resolve(process.cwd(), process.argv[2] ?? "data/experiment.example.json");

function fail(message) {
  throw new Error(message);
}

function object(value, path) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    fail(`${path} 必须是对象`);
  }
  return value;
}

function nonEmpty(value, path) {
  if (typeof value !== "string" || value.trim().length === 0) {
    fail(`${path} 必须是非空文本`);
  }
}

function number(value, path) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    fail(`${path} 必须是有限数字`);
  }
}

const source = await readFile(target, "utf8");
const data = JSON.parse(source);
const root = object(data, "根节点");

if (root.version !== 1) fail("version 目前只支持 1");
const meta = object(root.meta, "meta");
const verdict = object(root.verdict, "verdict");

for (const key of ["name", "baselineLabel", "candidateLabel", "defaultMode", "valueNote"]) {
  nonEmpty(meta[key], `meta.${key}`);
}
if (!["qualitative", "exact"].includes(meta.defaultMode)) {
  fail("meta.defaultMode 必须是 qualitative 或 exact");
}
for (const key of ["title", "description", "status", "nextStep"]) {
  nonEmpty(verdict[key], `verdict.${key}`);
}

if (!Array.isArray(root.metrics) || root.metrics.length < 1 || root.metrics.length > 8) {
  fail("metrics 必须包含 1 到 8 项");
}
if (!Array.isArray(root.trends) || root.trends.length < 1 || root.trends.length > 8) {
  fail("trends 必须包含 1 到 8 项");
}
if (!Array.isArray(root.evidence) || root.evidence.length < 1 || root.evidence.length > 12) {
  fail("evidence 必须包含 1 到 12 项");
}

const metricIds = new Set();
for (const [index, rawMetric] of root.metrics.entries()) {
  const metric = object(rawMetric, `metrics[${index}]`);
  for (const key of ["id", "label", "qualitative", "deltaLabel", "note"]) {
    nonEmpty(metric[key], `metrics[${index}].${key}`);
  }
  number(metric.baseline, `metrics[${index}].baseline`);
  number(metric.candidate, `metrics[${index}].candidate`);
  if (typeof metric.higherIsBetter !== "boolean") {
    fail(`metrics[${index}].higherIsBetter 必须是 true 或 false`);
  }
  if (!Number.isInteger(metric.precision) || metric.precision < 0 || metric.precision > 6) {
    fail(`metrics[${index}].precision 必须是 0 到 6 的整数`);
  }
  if (metricIds.has(metric.id)) fail(`metrics 中存在重复 id：${metric.id}`);
  metricIds.add(metric.id);
}

const trendIds = new Set();
for (const [index, rawTrend] of root.trends.entries()) {
  const trend = object(rawTrend, `trends[${index}]`);
  for (const key of ["id", "label", "title", "relation", "note"]) {
    nonEmpty(trend[key], `trends[${index}].${key}`);
  }
  for (const key of ["baseline", "candidate"]) {
    if (!Array.isArray(trend[key]) || trend[key].length < 2 || trend[key].length > 2000) {
      fail(`trends[${index}].${key} 必须包含 2 到 2000 个数字`);
    }
    trend[key].forEach((value, point) => number(value, `trends[${index}].${key}[${point}]`));
  }
  if (trend.baseline.length !== trend.candidate.length) {
    fail(`trends[${index}] 的 baseline 与 candidate 长度必须一致`);
  }
  if (trendIds.has(trend.id)) fail(`trends 中存在重复 id：${trend.id}`);
  trendIds.add(trend.id);
}

for (const [index, rawEvidence] of root.evidence.entries()) {
  const evidence = object(rawEvidence, `evidence[${index}]`);
  for (const key of ["title", "note", "tag", "tone"]) {
    nonEmpty(evidence[key], `evidence[${index}].${key}`);
  }
  if (!["positive", "negative", "neutral"].includes(evidence.tone)) {
    fail(`evidence[${index}].tone 必须是 positive、negative 或 neutral`);
  }
}

console.log(`数据格式正确：${target}`);
console.log(`包含 ${root.metrics.length} 个指标、${root.trends.length} 组趋势和 ${root.evidence.length} 条证据。`);
