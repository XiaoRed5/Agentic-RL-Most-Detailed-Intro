import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the importable experiment demo", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Agent Forge · Agent\/RL Experiment Dashboard<\/title>/i);
  assert.match(html, /结论总览/);
  assert.match(html, /趋势对比/);
  assert.match(html, /评测画像/);
  assert.match(html, /定性图集/);
  assert.match(html, /导入数据/);
  assert.match(html, /数值已隐藏/);
  assert.ok(html.indexOf("导入数据") < html.indexOf("结论总览"));
});

test("supports local JSON import without exposing the original experiment", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const sample = JSON.parse(
    await readFile(new URL("../data/experiment.example.json", import.meta.url), "utf8"),
  );

  assert.match(page, /type View = "overview" \| "curves" \| "evaluation" \| "figures" \| "guide"/);
  assert.match(page, /parseExperimentData/);
  assert.match(page, /window\.localStorage/);
  assert.match(page, /选择本地 JSON/);
  assert.equal(sample.meta.defaultMode, "qualitative");
  assert.equal(sample.meta.valueNote, "公开示例使用归一化示意数据，不对应任何真实实验");
  assert.ok(sample.metrics.length >= 1);
  assert.ok(sample.trends.length >= 1);
});
