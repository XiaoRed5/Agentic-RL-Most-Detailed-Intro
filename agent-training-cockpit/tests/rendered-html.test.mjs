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

test("server-renders the qualitative experiment demo", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Agent Forge · 定性实验看板 Demo<\/title>/i);
  assert.match(html, /结论总览/);
  assert.match(html, /趋势对比/);
  assert.match(html, /评测画像/);
  assert.match(html, /定性图集/);
  assert.match(html, /数值已隐藏/);
});

test("keeps only qualitative conclusions in the public source", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");

  assert.match(page, /type View = "overview" \| "curves" \| "evaluation" \| "figures"/);
  assert.match(page, /匿名基线/);
  assert.match(page, /匿名候选/);
  assert.match(page, /只保留方向和权衡/);
  assert.doesNotMatch(page, /from "\.\/data\/|toFixed\(/);
});
