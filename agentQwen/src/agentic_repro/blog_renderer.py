"""Render the project's canonical Markdown as a yyhdbl-style technical note.

The implementation intentionally uses only the Python standard library so the
report can be rebuilt in the same minimal environment as the reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(text: str) -> str:
    plain = re.sub(r"<[^>]+>", "", text)
    plain = re.sub(r"[^\w\u4e00-\u9fff]+", "-", plain, flags=re.UNICODE)
    return plain.strip("-").lower() or "section"


def _safe_url(url: str) -> str:
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme and parsed.scheme not in {"http", "https", "mailto"}:
        return "#"
    return html.escape(url, quote=True)


def _split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _inline(raw: str) -> str:
    """Render a deliberately small, safe Markdown inline subset."""

    saved: list[str] = []

    def save(fragment: str) -> str:
        saved.append(fragment)
        return f"\x00{len(saved) - 1}\x00"

    tick = chr(96)

    def code_sub(match: re.Match[str]) -> str:
        return save(f"<code>{html.escape(match.group(1))}</code>")

    raw = re.sub(tick + r"([^" + tick + r"\n]+)" + tick, code_sub, raw)
    escaped = html.escape(raw, quote=False)

    image_pattern = r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+[&quot;'\"]([^&quot;'\"]*)[&quot;'\"])?\)"
    escaped = re.sub(
        image_pattern,
        lambda match: save(
            '<img src="'
            + _safe_url(html.unescape(match.group(2)))
            + '" alt="'
            + html.escape(html.unescape(match.group(1)), quote=True)
            + '" loading="lazy">'
        ),
        escaped,
    )

    link_pattern = r"\[([^\]]+)\]\(([^)\s]+)\)"
    escaped = re.sub(
        link_pattern,
        lambda match: save(
            '<a href="'
            + _safe_url(html.unescape(match.group(2)))
            + '">'
            + match.group(1)
            + "</a>"
        ),
        escaped,
    )
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", escaped)
    escaped = re.sub(
        r"\$\$([^$\n]+)\$\$",
        lambda match: save("$$" + match.group(1) + "$$"),
        escaped,
    )
    escaped = re.sub(
        r"(?<!\$)\$([^$\n]+)\$(?!\$)",
        lambda match: save("$" + match.group(1) + "$"),
        escaped,
    )

    for index, fragment in enumerate(saved):
        escaped = escaped.replace(f"\x00{index}\x00", fragment)
    return escaped


def render_markdown(markdown: str) -> tuple[str, list[tuple[int, str, str]]]:
    lines = markdown.splitlines()
    output: list[str] = []
    headings: list[tuple[int, str, str]] = []
    paragraph: list[str] = []
    list_kind: str | None = None
    quote: list[str] = []
    used_ids: dict[str, int] = {}
    index = 0

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            output.append("<p>" + _inline(" ".join(part.strip() for part in paragraph)) + "</p>")
            paragraph = []

    def flush_list() -> None:
        nonlocal list_kind
        if list_kind:
            output.append(f"</{list_kind}>")
            list_kind = None

    def flush_quote() -> None:
        nonlocal quote
        if quote:
            output.append("<blockquote>" + _inline(" ".join(quote)) + "</blockquote>")
            quote = []

    while index < len(lines):
        line = lines[index]

        if line.startswith(chr(96) * 3):
            flush_paragraph()
            flush_list()
            flush_quote()
            language = line[len(chr(96) * 3) :].strip()
            code_lines: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].startswith(chr(96) * 3):
                code_lines.append(lines[index])
                index += 1
            language_class = f' class="language-{html.escape(language)}"' if language else ""
            output.append(
                f"<pre><code{language_class}>{html.escape(chr(10).join(code_lines))}</code></pre>"
            )
            index += 1
            continue

        if not line.strip():
            flush_paragraph()
            flush_list()
            flush_quote()
            index += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading:
            flush_paragraph()
            flush_list()
            flush_quote()
            level = len(heading.group(1))
            title = heading.group(2)
            base = _slug(title)
            count = used_ids.get(base, 0)
            used_ids[base] = count + 1
            anchor = base if count == 0 else f"{base}-{count + 1}"
            headings.append((level, re.sub(r"[*_]", "", title), anchor))
            output.append(f'<h{level} id="{anchor}">{_inline(title)}</h{level}>')
            index += 1
            continue

        if line.startswith(">"):
            flush_paragraph()
            flush_list()
            quote.append(line[1:].strip())
            index += 1
            continue
        flush_quote()

        if "|" in line and index + 1 < len(lines):
            separator = lines[index + 1].strip()
            if re.match(r"^\|?\s*:?-{3,}", separator):
                flush_paragraph()
                flush_list()
                headers = _split_row(line)
                output.append('<div class="table-wrap"><table><thead><tr>')
                output.extend(f"<th>{_inline(cell)}</th>" for cell in headers)
                output.append("</tr></thead><tbody>")
                index += 2
                while index < len(lines) and "|" in lines[index] and lines[index].strip():
                    cells = _split_row(lines[index])
                    output.append("<tr>")
                    output.extend(f"<td>{_inline(cell)}</td>" for cell in cells)
                    output.append("</tr>")
                    index += 1
                output.append("</tbody></table></div>")
                continue

        bullet = re.match(r"^\s*[-*+]\s+(.+)$", line)
        ordered = re.match(r"^\s*\d+\.\s+(.+)$", line)
        if bullet or ordered:
            flush_paragraph()
            kind = "ul" if bullet else "ol"
            if list_kind != kind:
                flush_list()
                output.append(f"<{kind}>")
                list_kind = kind
            match = bullet or ordered
            output.append(f"<li>{_inline(match.group(1))}</li>")
            index += 1
            continue
        flush_list()

        image_only = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", line)
        if image_only:
            flush_paragraph()
            alt = image_only.group(1)
            src = image_only.group(2).strip()
            output.append(
                '<figure><img src="'
                + _safe_url(src)
                + '" alt="'
                + html.escape(alt, quote=True)
                + '" loading="lazy"><figcaption>'
                + _inline(alt)
                + "</figcaption></figure>"
            )
            index += 1
            continue

        if line.strip() == "---":
            flush_paragraph()
            output.append("<hr>")
            index += 1
            continue

        paragraph.append(line)
        index += 1

    flush_paragraph()
    flush_list()
    flush_quote()
    return "\n".join(output), headings


def _toc_html(headings: list[tuple[int, str, str]]) -> str:
    items = [
        f'<li class="toc-level-{level}"><a href="#{anchor}">{html.escape(title)}</a></li>'
        for level, title, anchor in headings
        if level in {2, 3}
    ]
    if not items:
        return ""
    return (
        '<details class="article-toc" open><summary>这篇长文会讲什么</summary><ol>'
        + "".join(items)
        + "</ol></details>"
    )


def render_file(
    source: Path,
    output: Path,
    *,
    title: str,
    brand: str = "AgenticQwen/notes",
) -> dict[str, str]:
    source = source.resolve()
    output = output.resolve()
    body, headings = render_markdown(source.read_text(encoding="utf-8"))
    source_hash = _sha256(source)
    generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    toc = _toc_html(headings)
    article_body = body
    first_paragraph_end = article_body.find("</p>")
    if toc and first_paragraph_end >= 0:
        insertion = first_paragraph_end + len("</p>")
        article_body = article_body[:insertion] + toc + article_body[insertion:]
    css = r"""
:root {
  --ink: #28231f;
  --muted: #756b62;
  --line: #e7ded2;
  --paper: #fffdf9;
  --wash: #f5eee5;
  --accent: #a8472d;
  --accent-dark: #713322;
  --code: #241f1b;
  --shadow: 0 18px 55px rgba(80, 47, 27, .09);
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  color: var(--ink);
  background:
    radial-gradient(circle at 8% 5%, rgba(168,71,45,.055), transparent 28rem),
    var(--wash);
  font: 16px/1.9 -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC",
    "PingFang SC", "Microsoft YaHei", sans-serif;
  text-rendering: optimizeLegibility;
}
a { color: var(--accent-dark); text-decoration-color: rgba(113,51,34,.38); text-underline-offset: .18em; }
a:hover, a:focus-visible { color: var(--accent); text-decoration-color: currentColor; }
.site-header {
  position: sticky;
  top: 0;
  z-index: 10;
  border-bottom: 1px solid rgba(231,222,210,.92);
  background: rgba(245,238,229,.9);
  backdrop-filter: blur(16px);
}
.nav-shell, .page-shell { width: min(1080px, calc(100% - 36px)); margin: 0 auto; }
.nav-shell {
  min-height: 64px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
}
.brand {
  color: var(--ink);
  font: 700 20px/1 Georgia, "Songti SC", serif;
  text-decoration: none;
  letter-spacing: -.02em;
}
.brand span { color: var(--accent); }
.site-nav { display: flex; gap: 24px; }
.site-nav a { color: var(--muted); font-size: 14px; text-decoration: none; }
.page-shell { padding: 70px 0 100px; }
.article {
  max-width: 820px;
  margin: 0 auto;
  padding: 58px clamp(24px, 6vw, 76px) 76px;
  background: var(--paper);
  box-shadow: var(--shadow);
  border: 1px solid rgba(231,222,210,.7);
}
.source-note {
  margin: 0 0 18px;
  color: var(--accent);
  font-size: 12px;
  font-weight: 750;
  letter-spacing: .12em;
  text-transform: uppercase;
}
h1, h2, h3, h4 {
  font-family: Georgia, "Songti SC", "STSong", serif;
  line-height: 1.28;
  letter-spacing: -.025em;
  text-wrap: balance;
  scroll-margin-top: 84px;
}
h1 { margin: 0 0 24px; font-size: clamp(34px, 5vw, 55px); font-weight: 700; }
h2 {
  margin: 64px 0 18px;
  padding-top: 14px;
  border-top: 1px solid var(--line);
  font-size: clamp(25px, 3vw, 33px);
}
h3 { margin: 38px 0 12px; font-size: 22px; }
h4 { margin: 28px 0 10px; font-size: 18px; }
p { margin: 0 0 20px; }
h1 + p {
  margin-bottom: 28px;
  color: #5e554e;
  font: 20px/1.8 Georgia, "Songti SC", serif;
}
strong { color: #181411; }
hr { height: 1px; margin: 48px 0; border: 0; background: var(--line); }
blockquote {
  margin: 28px 0;
  padding: 18px 22px;
  border-left: 4px solid var(--accent);
  background: #f8f0e7;
  color: #554b43;
}
blockquote p:last-child { margin-bottom: 0; }
ul, ol { margin: 8px 0 22px; padding-left: 1.55em; }
li { margin: 5px 0; }
code {
  padding: .12em .36em;
  border-radius: 4px;
  background: #f1e8de;
  color: var(--accent-dark);
  font: .9em/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  overflow-wrap: anywhere;
}
pre {
  overflow: auto;
  margin: 26px 0;
  padding: 22px;
  border-radius: 8px;
  background: var(--code);
  color: #eee4d8;
  line-height: 1.65;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,.05);
}
pre code { padding: 0; background: transparent; color: inherit; white-space: pre; }
.table-wrap {
  overflow-x: auto;
  margin: 24px 0 30px;
  border: 1px solid var(--line);
  border-radius: 7px;
}
table { width: 100%; border-collapse: collapse; font-size: 14px; line-height: 1.62; }
th, td { min-width: 110px; padding: 12px 14px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
th { color: #4e443d; background: #f7f0e7; font-weight: 700; }
tr:last-child td { border-bottom: 0; }
figure { margin: 34px -18px; }
figure img { display: block; width: 100%; max-width: 100%; height: auto; border: 1px solid var(--line); background: #fff; }
figcaption { margin-top: 10px; color: var(--muted); font-size: 13px; line-height: 1.6; text-align: center; }
.article-toc {
  margin: 34px 0 40px;
  padding: 16px 20px;
  border: 1px solid var(--line);
  background: #fbf7f1;
}
.article-toc summary { cursor: pointer; min-height: 30px; color: var(--accent-dark); font-weight: 750; }
.article-toc ol { columns: 2; column-gap: 32px; margin: 12px 0 2px; padding-left: 1.25em; }
.article-toc li { break-inside: avoid; margin: 4px 0; color: var(--muted); font-size: 13px; }
.article-toc .toc-level-3 { margin-left: 12px; }
.article-toc a { color: inherit; text-decoration: none; }
.article-meta {
  margin-top: 66px;
  padding-top: 20px;
  border-top: 1px solid var(--line);
  color: var(--muted);
  font-size: 12px;
  overflow-wrap: anywhere;
}
.site-footer { padding: 0 18px 52px; color: var(--muted); font-size: 13px; text-align: center; }
@media (max-width: 680px) {
  .nav-shell, .page-shell { width: min(100% - 24px, 1080px); }
  .nav-shell { min-height: 58px; }
  .site-nav { gap: 14px; }
  .site-nav a { font-size: 13px; }
  .page-shell { padding: 18px 0 52px; }
  .article { padding: 34px 20px 52px; }
  h1 { font-size: 36px; }
  h2 { margin-top: 52px; }
  figure { margin-left: 0; margin-right: 0; }
  .article-toc ol { columns: 1; }
}
@media (max-width: 360px) {
  .brand { font-size: 18px; }
  .site-nav { gap: 9px; }
  .site-nav a { font-size: 12px; }
}
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
}
"""
    brand_html = html.escape(brand).replace("/", '<span>/</span>')
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="generator" content="agentic_repro.blog_renderer">
  <meta name="source-path" content="{html.escape(str(source), quote=True)}">
  <meta name="source-sha256" content="{source_hash}">
  <meta name="generated-at" content="{generated_at}">
  <title>{html.escape(title)}</title>
  <style>{css}</style>
  <script>
    window.MathJax = {{ tex: {{ inlineMath: [['$', '$']], displayMath: [['$$', '$$']] }} }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>
<body>
  <header class="site-header">
    <div class="nav-shell">
      <a class="brand" href="#article">{brand_html}</a>
      <nav class="site-nav" aria-label="主要导航">
        <a href="#article">长文</a>
        <a href="#复现账本">账本</a>
        <a href="#面试怎么讲">面试</a>
      </nav>
    </div>
  </header>
  <main class="page-shell">
    <article class="article" id="article">
      <p class="source-note">A long-horizon Agentic RL reproduction note</p>
      {article_body}
      <div class="article-meta">
        规范源：{html.escape(source.name)} · SHA-256 {source_hash[:16]}… · 生成于 {generated_at}
      </div>
    </article>
  </main>
  <footer class="site-footer">写给想真正理解 Agentic RL 的人，也写给要把项目讲进面试的人。</footer>
  <script>
    if (window.matchMedia('(max-width: 680px)').matches) {{
      document.querySelector('.article-toc')?.removeAttribute('open');
    }}
  </script>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    output_hash = _sha256(output)
    return {
        "source": str(source),
        "output": str(output),
        "source_sha256": source_hash,
        "output_sha256": output_hash,
        "generated_at": generated_at,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="AgenticQwen：从失败轨迹到下一轮训练")
    parser.add_argument("--brand", default="AgenticQwen/notes")
    args = parser.parse_args()
    render_file(args.source, args.output, title=args.title, brand=args.brand)


if __name__ == "__main__":
    main()
