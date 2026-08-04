from pathlib import Path

from agentic_repro.blog_renderer import render_markdown
from agentic_repro.blog_report import build_report


PROJECT = Path(__file__).resolve().parents[1]


def test_blog_report_keeps_run_and_plan_boundaries_separate():
    report, evidence = build_report(PROJECT)

    assert "20/24 个 rollout group 饱和" in report
    assert "SFT LoRA | NOT IMPLEMENTED" in report
    assert "PLANNED_NOT_RUN" in report
    assert "论文 47.4 | NOT REPRODUCED" in report
    assert "0.375" not in report
    assert "0.625" not in report
    assert evidence["claim_boundary"]["paper_scale_claimed"] is False
    assert evidence["claim_boundary"]["cloud_curriculum_completed"] is False
    assert evidence["not_run"]["cloud_run_file_count"] == 0


def test_blog_renderer_escapes_html_and_builds_tables():
    rendered, headings = render_markdown(
        "# 标题\n\n<script>alert(1)</script>\n\n"
        "| 状态 | 说明 |\n|---|---|\n| COMPLETE | 真跑过 |\n"
    )

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "<table>" in rendered
    assert headings[0][1] == "标题"
