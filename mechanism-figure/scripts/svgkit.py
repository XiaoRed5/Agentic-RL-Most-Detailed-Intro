#!/usr/bin/env python3
"""
svgkit — 画「纯变量机制图」的确定性 SVG 工具箱。

设计原则（见 SKILL.md）：本文件把"细节多/一眼看懂/字大/不叠/留白少"这几个
互相打架的诉求，靠"换编码 + Python 算坐标"消解成几个可复用函数。
writer 不要手写 SVG 坐标 —— 手摆一定会字叠或留白；一律用这里的常量和 helper。

用法：
    from svgkit import *
    doc = Svg(760, 470, title="token 先压到低维再路由 → ≈1.08×")
    doc.formula_row(380, 78, "d_model → d_latent → Experts(top-k) → d_model")
    x0 = doc.vvecbox(50, 110, [0.9,0.35,0.72,0.5,0.85,0.22,0.6,0.42], BLUE, label="x", sub="d_model=8")
    doc.arrow(70,174, 120,174, label="W_d")
    ...
    print(doc.render())          # 打到 stdout，主 agent 收走

核心手法：
  * vvecbox: 向量→一列方格，fill-opacity=数值，深浅即大小（替掉说明文字）
  * 三档字号常量 FS_*：数值/标签/底注，下限 14，宁删内容不缩字
  * 坐标用 col_x()/row_y() 等距推：天然不叠不留白
  * arrow: 把算子"动作"画出来（不是虚线暗示）
"""

# ---- 调色板（固定 6 色，跨图一致）----------------------------------------
BLUE   = "#1a3a8a"   # 主/输入
GOLD   = "#e0a040"   # 强调/选中/收益
RED    = "#c0392b"   # 警示/坏例/净倍率
GREEN  = "#1e6b1e"   # 输出/好例
PURPLE = "#8a5aa8"   # 中间态/latent
GREY   = "#888"      # 底注/次要
INK    = "#222"      # 正文黑
LINE   = "#ccc"      # 分隔线

# ---- 三档字号（用户逐条提过"字太小"，下限 14；宁可删内容不缩字）----------
FS_TITLE = 21   # 图标题
FS_VAL   = 20   # 数值/主变量（范围 18–23，按密度取）
FS_LABEL = 15   # 变量名/列标签（范围 14–16）
FS_NOTE  = 14   # 底注/机制注（范围 14–15）
FS_TINY  = 12   # 格子内 E1/α₁ 这类挤在小方块里的角标（唯一可 <14 处）

# ---- 箭头（stroke ~2.0，marker ~8px；提过"箭头很小"也提过"箭头稍微小一点"，取中）
ARROW_W  = 2.0
MARKER   = 8

CELL = 16       # 方格边长（vvecbox 单元）

SERIF = "STIX Two Text, Times New Roman, serif"   # 变量/公式用衬线斜体
SANS  = "PingFang SC, sans-serif"                  # 中文标签/底注用无衬线


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


class Svg:
    def __init__(self, w=760, h=470, title=None, bg="#ffffff"):
        self.w, self.h = w, h
        self.p = []
        self.p.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'font-family="{SANS}">')
        self.p.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="{bg}"/>')
        self.p.append(
            '<defs><marker id="ar" markerWidth="%d" markerHeight="%d" refX="6" refY="3" '
            'orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#555"/></marker></defs>'
            % (MARKER, MARKER))
        if title:
            self.title(title)

    # -- 原语 ---------------------------------------------------------------
    def raw(self, s):
        self.p.append(s); return self

    def text(self, x, y, s, size=FS_LABEL, fill=INK, anchor="start",
             weight=None, italic=False, family=None, opacity=None):
        a = [f'x="{x}"', f'y="{y}"', f'font-size="{size}"', f'fill="{fill}"',
             f'text-anchor="{anchor}"', f'font-family="{family or SANS}"']
        if weight:  a.append(f'font-weight="{weight}"')
        if italic:  a.append('font-style="italic"')
        if opacity is not None: a.append(f'opacity="{opacity}"')
        self.p.append(f'<text {" ".join(a)}>{esc(s)}</text>'); return self

    def line(self, x1, y1, x2, y2, stroke=LINE, w=1, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.p.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                      f'stroke="{stroke}" stroke-width="{w}"{d}/>'); return self

    def rect(self, x, y, w, h, fill="none", stroke=None, sw=1, op=None, rx=None):
        a = [f'x="{x}"', f'y="{y}"', f'width="{w}"', f'height="{h}"', f'fill="{fill}"']
        if op is not None:  a.append(f'fill-opacity="{op}"')
        if stroke: a += [f'stroke="{stroke}"', f'stroke-width="{sw}"']
        if rx: a.append(f'rx="{rx}"')
        self.p.append(f'<rect {" ".join(a)}/>'); return self

    # -- 组件 ---------------------------------------------------------------
    def title(self, s, fill=BLUE, y=32):
        self.text(self.w/2, y, s, size=FS_TITLE, fill=fill, anchor="middle", weight="700")
        self.line(30, y+20, self.w-30, y+20, stroke=LINE, w=1); return self

    def note(self, s, y=None, fill=GREY, size=FS_NOTE, x=None, anchor="middle"):
        """底部机制注：占满主图后剩下的空地，别留大白。"""
        self.text(x if x is not None else self.w/2, y if y is not None else self.h-12,
                  s, size=size, fill=fill, anchor=anchor); return self

    def formula_row(self, x, y, steps, colors=None, size=16, anchor="middle"):
        """
        公式流水线行：把 ['d_model','d_latent','Experts','d_model'] 画成
        彩色算子 + 灰色箭头分隔，顶在图上方对应技法②。steps 可传 str（用 → 拆）或 list。
        """
        if isinstance(steps, str):
            steps = [s.strip() for s in steps.replace("->", "→").split("→")]
        colors = colors or [BLUE, PURPLE, GOLD, GREEN, RED]
        tsp = []
        for i, s in enumerate(steps):
            col = colors[i % len(colors)]
            tsp.append(f'<tspan fill="{col}" font-style="italic">{esc(s)}</tspan>')
            if i < len(steps)-1:
                tsp.append(f'<tspan fill="{GREY}">  →  </tspan>')
        self.p.append(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
            f'font-family="{SERIF}">' + "".join(tsp) + '</text>')
        return self

    def arrow(self, x1, y1, x2, y2, label=None, sub=None, color="#555",
              lab_color=None, lab_family=SERIF):
        """把算子动作画成箭头；label 在上、sub 在下。不要用虚线暗示 —— 画实。"""
        self.line(x1, y1, x2, y2, stroke=color, w=ARROW_W)
        self.p[-1] = self.p[-1][:-2] + ' marker-end="url(#ar)"/>'
        mx = (x1+x2)/2
        if label: self.text(mx, min(y1,y2)-8, label, size=13, fill=lab_color or PURPLE,
                            anchor="middle", italic=True, family=lab_family)
        if sub:   self.text(mx, max(y1,y2)+16, sub, size=FS_TINY, fill=GREY, anchor="middle")
        return self

    def vvecbox(self, x, y, vec, color, label=None, sub=None, idx=None,
                lab_dy=None):
        """
        向量 → 一列方格，fill-opacity = 该维数值(0–1)，深浅编码大小。
        这是"细节多却不堆字"的根本手法：读者看深浅就知道数值。
        返回该列右边缘 x，方便接箭头。
        """
        n = len(vec)
        for j, val in enumerate(vec):
            self.rect(x, y+j*CELL, CELL, CELL, fill=color,
                      op=round(max(0.0, min(1.0, val))*0.85, 2), stroke="#fff", sw=0.8)
        self.rect(x, y, CELL, CELL*n, stroke=color, sw=1.1)              # 外框
        cx = x + CELL/2
        if idx is not None:
            self.text(cx, y-6, idx, size=FS_LABEL, fill=color, anchor="middle")
        base = y + CELL*n + (lab_dy or 18)
        if label: self.text(cx, base, label, size=FS_VAL, fill=color, anchor="middle",
                            italic=True, family=SERIF)
        if sub:   self.text(cx, base+16, sub, size=13, fill=GREY, anchor="middle")
        return x + CELL

    def dotprod(self, x, y, wvec, kvec, colw=BLUE, colk=GOLD, out=None):
        """
        画点积"动作"本身：w 横向量（上）叠 k 横向量（下），中间大「·」→ 分数。
        AttnRes 教训：点积只用连线暗示会"看不见"，必须把动作画出来。
        返回分数标注锚点 x。
        """
        n = len(wvec)
        for j in range(n):
            self.rect(x+j*CELL, y,      CELL, CELL, fill=colw, op=round(wvec[j]*0.85,2),
                      stroke="#fff", sw=0.8)
            self.rect(x+j*CELL, y+CELL,  CELL, CELL, fill=colk, op=round(kvec[j]*0.85,2),
                      stroke="#fff", sw=0.8)
        midx = x+n*CELL+14
        self.text(midx, y+CELL+4, "·", size=26, fill=INK, anchor="middle", weight="700")
        self.arrow(midx+12, y+CELL, midx+40, y+CELL)
        if out is not None:
            self.text(midx+58, y+CELL+5, out, size=FS_VAL, fill=RED, anchor="start", weight="700")
        return midx+58

    def col_x(self, i, start, step):     # 等距列坐标：天然不叠不留白
        return start + i*step

    def row_y(self, i, start, step):
        return start + i*step

    def render(self):
        return "\n".join(self.p + ['</svg>'])


# 自测：直接跑本文件输出一张 demo（也是 render_check 的冒烟对象）
if __name__ == "__main__":
    d = Svg(760, 300, title="svgkit demo · 变量逐步计算流")
    d.vvecbox(60, 90, [0.9,0.4,0.7,0.5,0.85,0.3], BLUE, label="x", sub="d=6", idx="①")
    d.arrow(78, 150, 150, 150, label="W", sub="proj")
    d.vvecbox(158, 120, [0.8,0.45,0.6], PURPLE, label="z", sub="ℓ=3", idx="②")
    d.note("深浅 = 数值大小；坐标等距推，字号≥14，动作画实不虚线")
    print(d.render())
