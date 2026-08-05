from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_official_agentic_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_official_agentic_data", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(row_id: str) -> dict:
    return {
        "id": row_id,
        "prompt": [
            {"role": "system", "content": "<policy>p</policy><tools>[]</tools>"},
            {"role": "user", "content": "do it"},
        ],
        "task_background": "background",
        "rubrics": "rubric",
        "test_policy": "policy",
        "user_escape_strategy": "escape",
        "tool_return_expected": {"normal_path": [], "hack_path": []},
    }


def row_with_paths(row_id: str, normal_steps: int, hack_steps: int) -> dict:
    value = row(row_id)
    value["tool_return_expected"] = {
        "normal_path": [{"tool_name": f"normal-{i}"} for i in range(normal_steps)],
        "hack_path": [{"tool_name": f"hack-{i}"} for i in range(hack_steps)],
    }
    return value


def write(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_official_rows_are_not_capped_and_synthetic_is_train_only(tmp_path: Path) -> None:
    official = tmp_path / "official.json"
    synthetic = tmp_path / "synthetic.json"
    write(official, [row(f"official-{i}") for i in range(12)])
    write(synthetic, [row("synthetic-0"), row("synthetic-1")])

    manifest = MODULE.build_dataset(
        official_path=official,
        output_dir=tmp_path / "out",
        validation_count=2,
        official_train_count=None,
        seed="fixed",
        synthetic_path=synthetic,
        synthetic_cap=2,
        allow_missing_synthetic=False,
    )

    assert manifest["sources"]["official"]["rows"] == 12
    assert manifest["splits"]["train"]["rows"] == 12
    assert manifest["splits"]["train"]["official_rows"] == 10
    assert manifest["splits"]["train"]["synthetic_rows"] == 2
    assert manifest["splits"]["validation"]["rows"] == 2
    assert manifest["splits"]["validation"]["synthetic_rows"] == 0
    assert manifest["checks"]["train_validation_disjoint"] is True


def test_split_is_deterministic(tmp_path: Path) -> None:
    official = tmp_path / "official.json"
    rows = [row(f"official-{i}") for i in range(12)]
    write(official, rows)

    manifests = []
    for name in ("a", "b"):
        manifests.append(
            MODULE.build_dataset(
                official_path=official,
                output_dir=tmp_path / name,
                validation_count=3,
                official_train_count=None,
                seed="fixed",
                synthetic_path=None,
                synthetic_cap=10,
                allow_missing_synthetic=False,
            )
        )
    assert manifests[0]["splits"]["train"]["ids_sha256"] == manifests[1]["splits"]["train"]["ids_sha256"]
    assert manifests[0]["splits"]["validation"]["ids_sha256"] == manifests[1]["splits"]["validation"]["ids_sha256"]


def test_rejects_synthetic_over_cap(tmp_path: Path) -> None:
    official = tmp_path / "official.json"
    synthetic = tmp_path / "synthetic.json"
    write(official, [row(f"official-{i}") for i in range(4)])
    write(synthetic, [row("synth-0"), row("synth-1")])

    with pytest.raises(ValueError, match="exceeding cap"):
        MODULE.build_dataset(
            official_path=official,
            output_dir=tmp_path / "out",
            validation_count=1,
            official_train_count=None,
            seed="fixed",
            synthetic_path=synthetic,
            synthetic_cap=1,
            allow_missing_synthetic=False,
        )


def test_missing_synthetic_can_be_explicitly_allowed(tmp_path: Path) -> None:
    official = tmp_path / "official.json"
    write(official, [row(f"official-{i}") for i in range(4)])
    manifest = MODULE.build_dataset(
        official_path=official,
        output_dir=tmp_path / "out",
        validation_count=1,
        official_train_count=None,
        seed="fixed",
        synthetic_path=tmp_path / "missing.json",
        synthetic_cap=10,
        allow_missing_synthetic=True,
    )
    assert manifest["sources"]["synthetic"]["state"] == "missing_allowed"
    assert manifest["splits"]["train"]["official_rows"] == 3


def test_official_train_subset_does_not_change_holdout(tmp_path: Path) -> None:
    official = tmp_path / "official.json"
    write(official, [row(f"official-{i}") for i in range(20)])

    manifest = MODULE.build_dataset(
        official_path=official,
        output_dir=tmp_path / "out",
        validation_count=4,
        official_train_count=6,
        seed="fixed",
        synthetic_path=None,
        synthetic_cap=10,
        allow_missing_synthetic=False,
    )
    assert manifest["sources"]["official"]["rows"] == 20
    assert manifest["sources"]["official"]["selected_train_rows"] == 6
    assert manifest["sources"]["official"]["selected_validation_rows"] == 4
    assert manifest["sources"]["official"]["unselected_rows"] == 10
    assert manifest["splits"]["train"]["official_rows"] == 6
    assert manifest["splits"]["validation"]["official_rows"] == 4


def test_repeated_official_base_ids_are_split_as_groups(tmp_path: Path) -> None:
    official = tmp_path / "official.json"
    rows = []
    for base_id in ("base-a", "base-b", "base-c", "base-d"):
        for variant in range(2):
            value = row(base_id)
            value["prompt"][1]["content"] = f"variant-{variant}"
            rows.append(value)
    write(official, rows)

    manifest = MODULE.build_dataset(
        official_path=official,
        output_dir=tmp_path / "out",
        validation_count=2,
        official_train_count=None,
        seed="fixed",
        synthetic_path=None,
        synthetic_cap=10,
        allow_missing_synthetic=False,
    )
    train_rows = json.loads((tmp_path / "out" / "tool_use_data_train.json").read_text())
    validation_rows = json.loads((tmp_path / "out" / "tool_use_data_val.json").read_text())
    assert {item["id"] for item in train_rows}.isdisjoint(
        {item["id"] for item in validation_rows}
    )
    assert manifest["sources"]["official"]["unique_base_ids"] == 4
    assert manifest["sources"]["official"]["repeated_id_rows"] == 4
    assert manifest["policy"]["holdout_method"] == "deterministic_group_subset_sum_exact"


def test_rejects_exact_duplicate_official_rows(tmp_path: Path) -> None:
    official = tmp_path / "official.json"
    duplicate = row("same")
    write(official, [duplicate, duplicate, row("other")])

    with pytest.raises(ValueError, match="exact duplicate row"):
        MODULE.build_dataset(
            official_path=official,
            output_dir=tmp_path / "out",
            validation_count=1,
            official_train_count=None,
            seed="fixed",
            synthetic_path=None,
            synthetic_cap=10,
            allow_missing_synthetic=False,
        )


def test_official_curriculum_filter_selects_short_workflows_before_split(tmp_path: Path) -> None:
    official = tmp_path / "official.json"
    write(
        official,
        [
            row_with_paths("easy-a", 0, 0),
            row_with_paths("easy-b", 1, 1),
            row_with_paths("easy-c", 1, 0),
            row_with_paths("hard-normal", 2, 1),
            row_with_paths("hard-hack", 1, 2),
        ],
    )

    manifest = MODULE.build_dataset(
        official_path=official,
        output_dir=tmp_path / "out",
        validation_count=1,
        official_train_count=None,
        seed="fixed",
        synthetic_path=None,
        synthetic_cap=10,
        allow_missing_synthetic=False,
        max_normal_path_steps=1,
        max_hack_path_steps=1,
    )

    assert manifest["sources"]["official"]["rows"] == 5
    assert manifest["sources"]["official"]["eligible_rows"] == 3
    assert manifest["sources"]["official"]["excluded_by_curriculum_filter"] == 2
    assert manifest["splits"]["train"]["official_rows"] == 2
    assert manifest["splits"]["validation"]["official_rows"] == 1
    assert manifest["policy"]["official_curriculum_filter"] == {
        "max_normal_path_steps": 1,
        "max_hack_path_steps": 1,
    }


def test_exact_variant_hash_filter_selects_only_requested_published_rows(tmp_path: Path) -> None:
    official = tmp_path / "official.json"
    rows = [row_with_paths(f"official-{i}", 1, 0) for i in range(5)]
    write(official, rows)
    selected_hashes = {MODULE.sha256_json(rows[1]), MODULE.sha256_json(rows[3]), MODULE.sha256_json(rows[4])}

    manifest = MODULE.build_dataset(
        official_path=official,
        output_dir=tmp_path / "out",
        validation_count=1,
        official_train_count=None,
        seed="fixed",
        synthetic_path=None,
        synthetic_cap=10,
        allow_missing_synthetic=False,
        include_row_sha256=selected_hashes,
    )

    output_rows = json.loads((tmp_path / "out" / "tool_use_data_train.json").read_text())
    output_rows += json.loads((tmp_path / "out" / "tool_use_data_val.json").read_text())
    assert {MODULE.sha256_json(item) for item in output_rows} == selected_hashes
    assert manifest["sources"]["official"]["selected_by_exact_variant_filter"] == 3
    assert manifest["sources"]["official"]["excluded_by_exact_variant_filter"] == 2
    assert manifest["policy"]["official_exact_variant_filter"]["row_sha256"] == sorted(selected_hashes)


def test_exact_variant_hash_filter_rejects_missing_hash(tmp_path: Path) -> None:
    official = tmp_path / "official.json"
    write(official, [row("official-0"), row("official-1")])

    with pytest.raises(ValueError, match="were not found"):
        MODULE.build_dataset(
            official_path=official,
            output_dir=tmp_path / "out",
            validation_count=1,
            official_train_count=None,
            seed="fixed",
            synthetic_path=None,
            synthetic_cap=10,
            allow_missing_synthetic=False,
            include_row_sha256={"0" * 64},
        )
