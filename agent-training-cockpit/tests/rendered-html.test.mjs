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

test("server-renders the Agent Forge cockpit", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Agent Forge · 训练驾驶舱<\/title>/i);
  assert.match(html, /训练总览/);
  assert.match(html, /轨迹回放/);
  assert.match(html, /错误地图/);
  assert.match(html, /版本对比/);
  assert.match(html, /任务成功率/);
});

test("keeps the four cockpit views in the client source", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");

  assert.match(page, /type View = "overview" \| "traces" \| "errors" \| "compare"/);
  assert.match(page, /agent-research-v2\.8/);
  assert.match(page, /最近任务轨迹/);
  assert.match(page, /错误类型分布/);
  assert.match(page, /核心指标对比/);
});
