#!/usr/bin/env python3
"""Generate a Chinese, mechanism-only Agentic RL curriculum figure.

The diagram deliberately omits run-specific scores, step counts and rollout
counts.  It explains the causal workflow: basic tool-use training, trajectory
auditing, failure-conditioned task synthesis, and branched Stage-2 training.
"""

from __future__ import annotations

import sys
from pathlib import Path


SKILL_DIR = Path("/Users/hongbo/.codex/skills/mechanism-figure")
sys.path.insert(0, str(SKILL_DIR / "scripts"))
from svgkit import Svg  # noqa: E402


W, H = 1900, 1180
BG = "#FFFDFC"
INK = "#102A43"
MUTED = "#5E6F7D"
BLUE = "#17628A"
BLUE_LINE = "#1D6590"
BLUE_BAND = "#D9E8F3"
BLUE_PANEL = "#EAF3F9"
PEACH = "#F8DCC6"
PEACH_PANEL = "#FFF1E4"
PINK = "#EFBFBE"
PINK_PANEL = "#FBE5E4"
GREEN = "#2F7D32"
GREEN_FILL = "#E4F1D8"
BEIGE = "#E8CBAA"
BEIGE_FILL = "#F8ECDC"
RED = "#C62828"
RED_FILL = "#FCE0DC"
PURPLE = "#7B4FA3"
WHITE = "#FFFFFF"
GREY = "#A7ADB2"


def box(doc: Svg, x: float, y: float, w: float, h: float, fill: str, stroke: str,
        radius: int = 14, sw: float = 2.0):
    doc.rect(x, y, w, h, fill=fill, stroke=stroke, sw=sw, rx=radius)


def band(doc: Svg, y: float, title: str, fill: str, color: str):
    box(doc, 42, y, W - 84, 54, fill, fill, 26, 0)
    doc.text(W / 2, y + 36, title, size=25, fill=color, anchor="middle",
             weight="700", family="Songti SC")


def multiline(doc: Svg, x: float, y: float, text: str, *, size: int = 16,
              fill: str = INK, anchor: str = "middle", weight: str | None = None,
              gap: int = 23, family: str = "PingFang SC"):
    for i, line in enumerate(text.split("\n")):
        doc.text(x, y + i * gap, line, size=size, fill=fill, anchor=anchor,
                 weight=weight, family=family)


def card(doc: Svg, x: float, y: float, w: float, h: float, title: str, body: str,
         fill: str = WHITE, stroke: str = BLUE_LINE, title_color: str = INK):
    box(doc, x, y, w, h, fill, stroke, 12, 2)
    doc.text(x + w / 2, y + 30, title, size=18, fill=title_color,
             anchor="middle", weight="700", family="Songti SC")
    multiline(doc, x + w / 2, y + 59, body, size=14, fill=MUTED,
              anchor="middle", gap=20, weight="700")


def node(doc: Svg, x: float, y: float, w: float, text: str,
         fill: str = GREEN_FILL, stroke: str = GREEN, color: str = INK,
         h: float = 52):
    box(doc, x, y, w, h, fill, stroke, 18, 2)
    doc.text(x + w / 2, y + h / 2 + 7, text, size=16, fill=color,
             anchor="middle", weight="700", family="Songti SC")


def arrow(doc: Svg, x1: float, y1: float, x2: float, y2: float,
          color: str = GREEN, label: str | None = None):
    doc.arrow(x1, y1, x2, y2, color=color, label=label,
              lab_color=color, lab_family="PingFang SC")


def main() -> None:
    doc = Svg(W, H, title=None, bg=BG)

    # ------------------------------------------------------------------
    # 第一阶段：基础任务和线性工具链
    # ------------------------------------------------------------------
    band(doc, 28, "一、第一阶段：基础策略学习与线性工具链", BLUE_BAND, INK)

    card(doc, 115, 135, 270, 150, "基础任务与环境", "重复扣款诉求\n客户、订单、支付、政策状态", BLUE_PANEL, BLUE_LINE)
    arrow(doc, 385, 210, 435, 210, BLUE)
    card(doc, 435, 120, 300, 180, "策略模型训练", "生成完整回复与工具调用\n环境执行动作并返回观察\n结果奖励 + 过程奖励", BLUE_PANEL, BLUE_LINE)

    # Complete linear workflow.
    box(doc, 790, 108, 1010, 202, BLUE_PANEL, BLUE_LINE, 15, 2)
    doc.text(1295, 140, "期望的完整行为工作流", size=21, fill=INK,
             anchor="middle", weight="700", family="Songti SC")
    workflow = ["身份确认", "查询客户", "查询订单", "读取支付记录", "读取退款政策", "请求用户确认", "创建正确退款"]
    x0, y0, nw, gap = 815, 190, 120, 20
    for i, label in enumerate(workflow):
        x = x0 + i * (nw + gap)
        node(doc, x, y0, nw, label, GREEN_FILL, GREEN, h=58)
        if i < len(workflow) - 1:
            arrow(doc, x + nw, y0 + 29, x + nw + gap - 4, y0 + 29, GREEN)
    doc.text(1295, 286, "先读后写 · 精确确认 · 正确重复扣款 · 幂等写入", size=14,
             fill=GREEN, anchor="middle", weight="700")

    arrow(doc, 735, 210, 785, 210, BLUE)
    doc.text(525, 330, "每条轨迹都保存工具事件、环境状态、终止原因与奖励分解", size=15,
             fill=BLUE, anchor="middle", weight="700")
    arrow(doc, 740, 327, 920, 327, BLUE, "轨迹进入审计")

    # ------------------------------------------------------------------
    # Failure audit and directed synthesis.
    # ------------------------------------------------------------------
    band(doc, 360, "二、失败驱动的数据飞轮：轨迹审计 → 失败标签 → 定向合成", PEACH, INK)

    box(doc, 48, 430, 850, 310, PEACH_PANEL, PEACH, 16, 2)
    doc.text(473, 462, "失败轨迹分析", size=22, fill=INK, anchor="middle",
             weight="700", family="Songti SC")
    card(doc, 78, 490, 235, 112, "环境状态", "哪些信息已经读取？\n写操作是否满足前置条件？", WHITE, BLUE_LINE)
    card(doc, 333, 490, 235, 112, "智能体行为", "工具顺序是否正确？\n临时错误是否得到恢复？", WHITE, BLUE_LINE)
    card(doc, 588, 490, 235, 112, "用户确认状态", "是否确认精确目标？\n是否存在模糊或越权写入？", WHITE, BLUE_LINE)
    box(doc, 78, 628, 745, 80, WHITE, RED, 12, 2)
    doc.text(450, 655, "轨迹审计器：读取状态／事件账本，不使用模型自报作为真值", size=16,
             fill=RED, anchor="middle", weight="700")
    doc.text(450, 683, "输出可操作的失败分类与合成条件", size=14,
             fill=MUTED, anchor="middle", weight="700")

    box(doc, 930, 430, 922, 310, PEACH_PANEL, PEACH, 16, 2)
    doc.text(1391, 462, "失败标签如何变成困难样本", size=22, fill=INK,
             anchor="middle", weight="700", family="Songti SC")

    failure_to_stress = [
        ("没有读取支付历史", "加入相似订单与干扰扣款"),
        ("没有正确核对身份", "身份查询首次返回临时超时"),
        ("没有处理临时错误", "要求使用相同参数进行一次重试"),
        ("没有请求用户确认", "保留精确确认的不可绕过硬门"),
        ("过早执行退款", "加强先读后写与状态前置检查"),
        ("没有绑定幂等键", "增加重复写风险与幂等键校验"),
    ]
    for i, (failure, stress) in enumerate(failure_to_stress):
        row = i // 2
        col = i % 2
        x = 965 + col * 430
        y = 500 + row * 72
        box(doc, x, y, 170, 48, RED_FILL, RED, 10, 1.6)
        doc.text(x + 85, y + 30, failure, size=14, fill=RED,
                 anchor="middle", weight="700")
        arrow(doc, x + 174, y + 24, x + 206, y + 24, BLUE)
        box(doc, x + 212, y, 190, 48, GREEN_FILL, GREEN, 10, 1.6)
        multiline(doc, x + 307, y + 20, stress, size=12, fill=INK,
                  anchor="middle", gap=15, weight="700")

    # Back-translation arrows, mirroring the reference figure.
    doc.text(950, 772, "从失败路径反译新的环境状态、用户指令与智能体约束", size=20,
             fill=RED, anchor="middle", weight="700", family="Songti SC")
    arrow(doc, 1110, 785, 880, 785, GREY)
    arrow(doc, 790, 785, 560, 785, GREY)

    # ------------------------------------------------------------------
    # Stage 2: harder contexts + branched behavior tree.
    # ------------------------------------------------------------------
    band(doc, 805, "三、第二阶段：带分支的困难工作流与持续训练", PINK, INK)

    box(doc, 48, 875, 820, 250, PINK_PANEL, PINK, 16, 2)
    doc.text(458, 907, "困难任务上下文", size=22, fill=INK, anchor="middle",
             weight="700", family="Songti SC")
    card(doc, 78, 935, 230, 142, "环境状态增强", "多个相似订单\n真实与干扰扣款并存\n工具出现一次性超时", WHITE, BLUE_LINE)
    card(doc, 328, 935, 230, 142, "智能体指令增强", "相同参数重试\n严格先读后写\n写操作绑定幂等键", WHITE, BLUE_LINE)
    card(doc, 578, 935, 230, 142, "用户指令增强", "不允许猜测目标\n必须说明精确扣款\n明确确认后才能写入", WHITE, BLUE_LINE)
    doc.text(443, 1108, "旧任务回放样本与困难任务混合，避免只适配新增分支", size=14,
             fill=PURPLE, anchor="middle", weight="700")

    box(doc, 900, 875, 952, 250, PINK_PANEL, PINK, 16, 2)
    doc.text(1376, 907, "带分支的受约束行为树", size=22, fill=INK,
             anchor="middle", weight="700", family="Songti SC")

    # Main path and recovery branch.
    node(doc, 930, 960, 118, "身份确认")
    node(doc, 1100, 960, 118, "查询客户")
    arrow(doc, 1048, 986, 1096, 986, GREEN)
    node(doc, 1270, 930, 120, "查询成功")
    node(doc, 1270, 1003, 120, "临时超时", BEIGE_FILL, BEIGE)
    arrow(doc, 1218, 986, 1265, 956, GREEN)
    arrow(doc, 1218, 986, 1265, 1029, BEIGE)
    node(doc, 1438, 1003, 115, "同参重试", GREEN_FILL, GREEN)
    arrow(doc, 1390, 1029, 1433, 1029, GREEN)
    arrow(doc, 1495, 1003, 1495, 950, GREEN)

    node(doc, 1438, 930, 118, "查询订单")
    arrow(doc, 1390, 956, 1433, 956, GREEN)
    node(doc, 1605, 904, 120, "目标订单")
    node(doc, 1605, 968, 120, "干扰订单", BEIGE_FILL, BEIGE)
    arrow(doc, 1556, 956, 1600, 930, GREEN)
    arrow(doc, 1556, 956, 1600, 994, BEIGE)

    # Condensed read-confirm-write branch at the bottom of the tree.
    node(doc, 970, 1070, 130, "读取支付")
    node(doc, 1150, 1070, 130, "读取政策")
    node(doc, 1330, 1070, 130, "精确确认")
    node(doc, 1510, 1070, 130, "幂等退款")
    for x in [1100, 1280, 1460]:
        arrow(doc, x, 1096, x + 46, 1096, GREEN)
    node(doc, 1690, 1070, 125, "写入被阻断", RED_FILL, RED, RED)
    arrow(doc, 1725, 1020, 1752, 1065, RED, "跳过读取 / 模糊确认")

    # One bounded feedback arrow replaces page-edge loops.  It sits entirely
    # below the Stage-2 panels and points back toward the audit stage.
    arrow(doc, 1745, 1140, 1035, 1140, BLUE, "新轨迹回流，进入下一轮失败审计")
    doc.text(W / 2, 1170, "训练 → 轨迹审计 → 失败标签 → 定向合成 → 再训练；数据难度沿真实能力边界逐轮扩展",
             size=16, fill=MUTED, anchor="middle", weight="700")

    print(doc.render())


if __name__ == "__main__":
    main()
