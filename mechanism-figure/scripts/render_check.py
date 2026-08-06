#!/usr/bin/env python3
"""
render_check — 把 SVG/含 SVG 的 HTML 用 Chrome headless 渲成高清 PNG，供 Codex 主 agent 亲眼看。

铁律（见 [[feedback_figure_render_verify_audit]]）：机制图改完必须真渲染、用图像查看工具看 PNG，
不许脑内推坐标。真实图像检查才能抓"字叠/留白/坐标错位/裁切"；judge 文本审给不了这个。

用法：
    python3 render_check.py fig.svg                 # → fig.png（deviceScaleFactor=3）
    python3 render_check.py a.svg b.svg c.svg       # 批量，各出 <name>.png
    python3 render_check.py --contact a.svg b.svg   # 竖排拼一张 contact.png 一次看全

依赖 puppeteer-core + 本机 Chrome。找不到就打印可行的安装/路径提示，不静默退化。
"""
import sys, os, subprocess, tempfile, json, shutil

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
]
PUPPETEER_CANDIDATES = [
    "/opt/homebrew/lib/node_modules/puppeteer-core",
    "/usr/local/lib/node_modules/puppeteer-core",
    "/usr/lib/node_modules/puppeteer-core",
]

def find_one(cands, what):
    for c in cands:
        if c and os.path.exists(c):
            return c
    sys.exit(f"[render_check] 找不到 {what}。候选：{cands}\n"
             f"→ Chrome: 装 Google Chrome；puppeteer: npm i -g puppeteer-core")

def wrap_html(svgs, contact=False):
    if contact:
        blocks = "".join(
            f'<div style="margin:10px;background:#fff;padding:6px">'
            f'<div style="font:700 13px sans-serif;color:#c0392b">{os.path.basename(s)}</div>'
            f'<div style="width:760px">{open(s).read()}</div></div>'
            for s in svgs)
        return f'<html><body style="margin:0;background:#eee">{blocks}</body></html>'
    return f'<html><body style="margin:0;background:#fff"><div style="width:760px">{open(svgs[0]).read()}</div></body></html>'

JS = r'''
const puppeteer=require(process.env.PP);
(async()=>{
  const b=await puppeteer.launch({executablePath:process.env.CHROME,headless:'new',args:['--no-sandbox']});
  const p=await b.newPage();
  await p.setViewport({width:820,height:1000,deviceScaleFactor:3});
  await p.goto('file://'+process.env.HTML,{waitUntil:'networkidle0'});
  await new Promise(r=>setTimeout(r,400));
  const el=await p.$('div');
  await el.screenshot({path:process.env.OUT});
  await b.close(); console.log('ok '+process.env.OUT);
})();
'''

def render(svgs, out, contact=False):
    chrome = find_one(CHROME_CANDIDATES, "Chrome")
    pp     = find_one(PUPPETEER_CANDIDATES, "puppeteer-core")
    html = wrap_html(svgs, contact)
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
        f.write(html); html_path = f.name
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(JS); js_path = f.name
    env = dict(os.environ, PP=pp, CHROME=chrome, HTML=html_path, OUT=out)
    r = subprocess.run(["node", js_path], env=env, capture_output=True, text=True)
    os.unlink(html_path); os.unlink(js_path)
    if r.returncode != 0:
        sys.exit(f"[render_check] 渲染失败：\n{r.stderr}")
    print(r.stdout.strip())

def main():
    args = sys.argv[1:]
    contact = "--contact" in args
    svgs = [a for a in args if not a.startswith("--")]
    if not svgs:
        sys.exit("用法: render_check.py [--contact] fig.svg [fig2.svg ...]")
    if contact:
        render(svgs, os.path.join(os.path.dirname(svgs[0]) or ".", "contact.png"), contact=True)
    else:
        for s in svgs:
            render([s], os.path.splitext(s)[0] + ".png")

if __name__ == "__main__":
    main()
