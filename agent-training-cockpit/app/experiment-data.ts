import rawDemoData from "../data/experiment.example.json";

export type DisplayMode = "qualitative" | "exact";
export type EvidenceTone = "positive" | "negative" | "neutral";

export type ExperimentMetric = {
  id: string;
  label: string;
  baseline: number;
  candidate: number;
  unit: string;
  precision: number;
  higherIsBetter: boolean;
  qualitative: string;
  deltaLabel: string;
  note: string;
};

export type ExperimentTrend = {
  id: string;
  label: string;
  title: string;
  unit: string;
  precision: number;
  baseline: number[];
  candidate: number[];
  relation: string;
  note: string;
};

export type ExperimentEvidence = {
  title: string;
  note: string;
  tag: string;
  tone: EvidenceTone;
};

export type ExperimentData = {
  version: 1;
  meta: {
    name: string;
    baselineLabel: string;
    candidateLabel: string;
    defaultMode: DisplayMode;
    valueNote: string;
  };
  verdict: {
    title: string;
    description: string;
    status: string;
    nextStep: string;
  };
  metrics: ExperimentMetric[];
  trends: ExperimentTrend[];
  evidence: ExperimentEvidence[];
};

type UnknownRecord = Record<string, unknown>;

function record(value: unknown, path: string): UnknownRecord {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${path} 必须是对象`);
  }
  return value as UnknownRecord;
}

function text(value: unknown, path: string, maxLength = 160): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error(`${path} 必须是非空文本`);
  }
  if (value.length > maxLength) {
    throw new Error(`${path} 不能超过 ${maxLength} 个字符`);
  }
  return value.trim();
}

function finiteNumber(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${path} 必须是有限数字`);
  }
  return value;
}

function precision(value: unknown, path: string): number {
  const parsed = finiteNumber(value, path);
  if (!Number.isInteger(parsed) || parsed < 0 || parsed > 6) {
    throw new Error(`${path} 必须是 0 到 6 的整数`);
  }
  return parsed;
}

function boolean(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") {
    throw new Error(`${path} 必须是 true 或 false`);
  }
  return value;
}

function numberSeries(value: unknown, path: string): number[] {
  if (!Array.isArray(value) || value.length < 2 || value.length > 2000) {
    throw new Error(`${path} 必须包含 2 到 2000 个数字`);
  }
  return value.map((item, index) => finiteNumber(item, `${path}[${index}]`));
}

function array(value: unknown, path: string, min: number, max: number): unknown[] {
  if (!Array.isArray(value) || value.length < min || value.length > max) {
    throw new Error(`${path} 必须包含 ${min} 到 ${max} 项`);
  }
  return value;
}

export function parseExperimentData(value: unknown): ExperimentData {
  const root = record(value, "根节点");
  if (root.version !== 1) {
    throw new Error("version 目前只支持 1");
  }

  const metaInput = record(root.meta, "meta");
  const defaultMode = text(metaInput.defaultMode, "meta.defaultMode");
  if (defaultMode !== "qualitative" && defaultMode !== "exact") {
    throw new Error("meta.defaultMode 必须是 qualitative 或 exact");
  }

  const verdictInput = record(root.verdict, "verdict");
  const metricsInput = array(root.metrics, "metrics", 1, 8);
  const trendsInput = array(root.trends, "trends", 1, 8);
  const evidenceInput = array(root.evidence, "evidence", 1, 12);

  const metrics = metricsInput.map((item, index): ExperimentMetric => {
    const metric = record(item, `metrics[${index}]`);
    return {
      id: text(metric.id, `metrics[${index}].id`, 48),
      label: text(metric.label, `metrics[${index}].label`, 48),
      baseline: finiteNumber(metric.baseline, `metrics[${index}].baseline`),
      candidate: finiteNumber(metric.candidate, `metrics[${index}].candidate`),
      unit: typeof metric.unit === "string" ? metric.unit.slice(0, 16) : "",
      precision: precision(metric.precision, `metrics[${index}].precision`),
      higherIsBetter: boolean(metric.higherIsBetter, `metrics[${index}].higherIsBetter`),
      qualitative: text(metric.qualitative, `metrics[${index}].qualitative`, 48),
      deltaLabel: text(metric.deltaLabel, `metrics[${index}].deltaLabel`, 48),
      note: text(metric.note, `metrics[${index}].note`, 120),
    };
  });

  const trends = trendsInput.map((item, index): ExperimentTrend => {
    const trend = record(item, `trends[${index}]`);
    const baseline = numberSeries(trend.baseline, `trends[${index}].baseline`);
    const candidate = numberSeries(trend.candidate, `trends[${index}].candidate`);
    if (baseline.length !== candidate.length) {
      throw new Error(`trends[${index}] 的 baseline 与 candidate 长度必须一致`);
    }
    return {
      id: text(trend.id, `trends[${index}].id`, 48),
      label: text(trend.label, `trends[${index}].label`, 48),
      title: text(trend.title, `trends[${index}].title`, 80),
      unit: typeof trend.unit === "string" ? trend.unit.slice(0, 16) : "",
      precision: precision(trend.precision, `trends[${index}].precision`),
      baseline,
      candidate,
      relation: text(trend.relation, `trends[${index}].relation`, 80),
      note: text(trend.note, `trends[${index}].note`, 180),
    };
  });

  const uniqueMetricIds = new Set(metrics.map((item) => item.id));
  const uniqueTrendIds = new Set(trends.map((item) => item.id));
  if (uniqueMetricIds.size !== metrics.length || uniqueTrendIds.size !== trends.length) {
    throw new Error("metrics 和 trends 中的 id 不能重复");
  }

  const evidence = evidenceInput.map((item, index): ExperimentEvidence => {
    const entry = record(item, `evidence[${index}]`);
    const tone = text(entry.tone, `evidence[${index}].tone`);
    if (tone !== "positive" && tone !== "negative" && tone !== "neutral") {
      throw new Error(`evidence[${index}].tone 必须是 positive、negative 或 neutral`);
    }
    return {
      title: text(entry.title, `evidence[${index}].title`, 80),
      note: text(entry.note, `evidence[${index}].note`, 120),
      tag: text(entry.tag, `evidence[${index}].tag`, 24),
      tone,
    };
  });

  return {
    version: 1,
    meta: {
      name: text(metaInput.name, "meta.name", 80),
      baselineLabel: text(metaInput.baselineLabel, "meta.baselineLabel", 48),
      candidateLabel: text(metaInput.candidateLabel, "meta.candidateLabel", 48),
      defaultMode,
      valueNote: text(metaInput.valueNote, "meta.valueNote", 160),
    },
    verdict: {
      title: text(verdictInput.title, "verdict.title", 180),
      description: text(verdictInput.description, "verdict.description", 240),
      status: text(verdictInput.status, "verdict.status", 40),
      nextStep: text(verdictInput.nextStep, "verdict.nextStep", 240),
    },
    metrics,
    trends,
    evidence,
  };
}

export const demoExperiment = parseExperimentData(rawDemoData);
export const demoExperimentSource = rawDemoData;
