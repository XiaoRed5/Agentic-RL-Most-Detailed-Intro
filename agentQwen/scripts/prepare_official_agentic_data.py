#!/usr/bin/env python3
"""Build an auditable AgenticQwen train/validation split.

The upstream repository ships a 100-row ``tool_use_demo.json`` dataset but
does not ship the derived train/validation JSON files expected by its parquet
converter.  This wrapper keeps a deterministic official-data holdout and may
append a separately generated curriculum batch to *training only*.

The synthetic cap applies only to newly generated rows.  Official rows are
controlled separately by ``--official-train-count``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "id",
    "prompt",
    "task_background",
    "rubrics",
    "test_policy",
    "user_escape_strategy",
    "tool_return_expected",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_official_parquet_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert the published AgenticQwen-Data schema to upstream raw JSON."""
    prompt = json.loads(row["messages_json"])
    tool_return_expected = json.loads(row["tool_return_expected_json"])
    return {
        "id": row["id"],
        "prompt": prompt,
        "task_background": row["task_background"],
        "rubrics": row["rubrics"],
        "test_policy": row["test_policy"],
        "user_escape_strategy": row["user_escape_strategy"],
        "tool_return_expected": tool_return_expected,
    }


def load_rows(path: Path, *, label: str) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - exercised in the remote runtime
            raise RuntimeError("pyarrow is required to read the published AgenticQwen parquet") from exc
        required = [
            "id",
            "task_background",
            "rubrics",
            "test_policy",
            "user_escape_strategy",
            "messages_json",
            "tool_return_expected_json",
        ]
        table = pq.read_table(path, columns=required)
        return [_normalize_official_parquet_row(row) for row in table.to_pylist()]

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{label} must be a JSON list of objects: {path}")
    return value


def validate_rows(
    rows: list[dict[str, Any]],
    *,
    label: str,
    allow_repeated_ids: bool,
) -> dict[str, int]:
    seen_ids: set[str] = set()
    seen_row_hashes: set[str] = set()
    repeated_id_rows = 0
    for index, row in enumerate(rows):
        missing = sorted(REQUIRED_FIELDS - row.keys())
        if missing:
            raise ValueError(f"{label}[{index}] is missing fields: {missing}")
        row_id = str(row["id"])
        if not row_id:
            raise ValueError(f"{label}[{index}] has an empty id")
        if row_id in seen_ids:
            repeated_id_rows += 1
            if not allow_repeated_ids:
                raise ValueError(f"{label} contains duplicate id: {row_id}")
        seen_ids.add(row_id)
        row_hash = sha256_json(row)
        if row_hash in seen_row_hashes:
            raise ValueError(f"{label} contains an exact duplicate row at index {index}")
        seen_row_hashes.add(row_hash)
        prompt = row["prompt"]
        if not isinstance(prompt, list) or len(prompt) < 2:
            raise ValueError(f"{label}[{index}] prompt must contain system and user messages")
        roles = [message.get("role") for message in prompt if isinstance(message, dict)]
        if roles[:2] != ["system", "user"]:
            raise ValueError(f"{label}[{index}] first prompt roles must be system,user; got {roles[:2]}")
        system = str(prompt[0].get("content", ""))
        if "<policy>" not in system or "</policy>" not in system:
            raise ValueError(f"{label}[{index}] system prompt lacks <policy> block")
        if "<tools>" not in system or "</tools>" not in system:
            raise ValueError(f"{label}[{index}] system prompt lacks <tools> block")
        expected = row["tool_return_expected"]
        if not isinstance(expected, (dict, str)):
            raise ValueError(f"{label}[{index}] tool_return_expected must be dict or string")
        if isinstance(expected, dict) and not {"normal_path", "hack_path"}.issubset(expected):
            raise ValueError(f"{label}[{index}] lacks normal_path/hack_path")
    return {
        "rows": len(rows),
        "unique_base_ids": len(seen_ids),
        "repeated_id_rows": repeated_id_rows,
        "exact_duplicate_rows": 0,
    }


def stable_group_key(row_id: str, *, seed: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}\0group\0{row_id}".encode("utf-8")).hexdigest()
    return digest, row_id


def stable_row_key(row: dict[str, Any], *, seed: str) -> tuple[str, str, str]:
    row_id = str(row["id"])
    content_hash = sha256_json(row)
    digest = hashlib.sha256(
        f"{seed}\0row\0{row_id}\0{content_hash}".encode("utf-8")
    ).hexdigest()
    return digest, row_id, content_hash


def select_group_disjoint_holdout(
    rows: list[dict[str, Any]],
    *,
    validation_count: int,
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Select a deterministic holdout without leaking variants of the same base id."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["id"]), []).append(row)
    ordered_ids = sorted(groups, key=lambda row_id: stable_group_key(row_id, seed=seed))

    # Prefer an exact row count while keeping each base-id group intact.
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for row_id in ordered_ids:
        group_size = len(groups[row_id])
        for subtotal, chosen in list(reachable.items())[::-1]:
            candidate = subtotal + group_size
            if candidate <= validation_count and candidate not in reachable:
                reachable[candidate] = chosen + (row_id,)
        if validation_count in reachable:
            break

    if validation_count in reachable:
        validation_ids = set(reachable[validation_count])
        method = "deterministic_group_subset_sum_exact"
    else:
        validation_ids = set()
        selected_rows = 0
        for row_id in ordered_ids:
            validation_ids.add(row_id)
            selected_rows += len(groups[row_id])
            if selected_rows >= validation_count:
                break
        method = "deterministic_group_prefix_at_least_target"

    validation_rows = [row for row in rows if str(row["id"]) in validation_ids]
    train_candidates = [row for row in rows if str(row["id"]) not in validation_ids]
    validation_rows.sort(key=lambda row: stable_row_key(row, seed=seed))
    train_candidates.sort(key=lambda row: stable_row_key(row, seed=seed))
    return train_candidates, validation_rows, method


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def workflow_lengths(row: dict[str, Any]) -> tuple[int, int]:
    expected = row["tool_return_expected"]
    if isinstance(expected, str):
        expected = json.loads(expected)
    if not isinstance(expected, dict):
        raise ValueError(f"invalid tool_return_expected for row {row.get('id')}")
    return len(expected.get("normal_path") or []), len(expected.get("hack_path") or [])


def filter_by_workflow_complexity(
    rows: list[dict[str, Any]],
    *,
    max_normal_path_steps: int | None,
    max_hack_path_steps: int | None,
) -> list[dict[str, Any]]:
    """Select an official curriculum tier without synthesizing new examples."""
    selected: list[dict[str, Any]] = []
    for row in rows:
        normal_steps, hack_steps = workflow_lengths(row)
        if max_normal_path_steps is not None and normal_steps > max_normal_path_steps:
            continue
        if max_hack_path_steps is not None and hack_steps > max_hack_path_steps:
            continue
        selected.append(row)
    return selected


def filter_by_row_hashes(
    rows: list[dict[str, Any]],
    *,
    include_row_sha256: set[str] | None,
) -> list[dict[str, Any]]:
    """Select exact published variants while keeping their source IDs intact."""
    if not include_row_sha256:
        return rows
    invalid = sorted(
        value
        for value in include_row_sha256
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
    )
    if invalid:
        raise ValueError(f"invalid include_row_sha256 values: {invalid[:3]}")
    selected = [row for row in rows if sha256_json(row) in include_row_sha256]
    matched = {sha256_json(row) for row in selected}
    missing = sorted(include_row_sha256 - matched)
    if missing:
        raise ValueError(f"requested official row hashes were not found: {missing[:3]}")
    return selected


def build_dataset(
    *,
    official_path: Path,
    output_dir: Path,
    validation_count: int,
    official_train_count: int | None,
    seed: str,
    synthetic_path: Path | None,
    synthetic_cap: int,
    allow_missing_synthetic: bool,
    max_normal_path_steps: int | None = None,
    max_hack_path_steps: int | None = None,
    include_row_sha256: set[str] | None = None,
) -> dict[str, Any]:
    if validation_count <= 0:
        raise ValueError("validation_count must be positive")
    if official_train_count is not None and official_train_count <= 0:
        raise ValueError("official_train_count must be positive when provided")
    if synthetic_cap < 0:
        raise ValueError("synthetic_cap must be non-negative")
    if max_normal_path_steps is not None and max_normal_path_steps < 0:
        raise ValueError("max_normal_path_steps must be non-negative")
    if max_hack_path_steps is not None and max_hack_path_steps < 0:
        raise ValueError("max_hack_path_steps must be non-negative")

    official_rows = load_rows(official_path, label="official")
    official_stats = validate_rows(
        official_rows,
        label="official",
        allow_repeated_ids=True,
    )
    selected_official_rows = filter_by_row_hashes(
        official_rows,
        include_row_sha256=include_row_sha256,
    )
    eligible_official_rows = filter_by_workflow_complexity(
        selected_official_rows,
        max_normal_path_steps=max_normal_path_steps,
        max_hack_path_steps=max_hack_path_steps,
    )
    if validation_count >= len(eligible_official_rows):
        raise ValueError("validation_count must be smaller than the eligible official dataset")

    official_train_rows, validation_rows, holdout_method = select_group_disjoint_holdout(
        eligible_official_rows,
        validation_count=validation_count,
        seed=seed,
    )
    if official_train_count is not None:
        official_train_rows = official_train_rows[:official_train_count]

    synthetic_rows: list[dict[str, Any]] = []
    synthetic_state = "not_configured"
    if synthetic_path is not None:
        if synthetic_path.is_file():
            synthetic_rows = load_rows(synthetic_path, label="synthetic")
            validate_rows(
                synthetic_rows,
                label="synthetic",
                allow_repeated_ids=False,
            )
            synthetic_state = "loaded"
        elif allow_missing_synthetic:
            synthetic_state = "missing_allowed"
        else:
            raise FileNotFoundError(f"synthetic dataset does not exist: {synthetic_path}")
    if len(synthetic_rows) > synthetic_cap:
        raise ValueError(
            f"synthetic dataset contains {len(synthetic_rows)} rows, exceeding cap {synthetic_cap}"
        )

    official_ids = {str(row["id"]) for row in official_rows}
    synthetic_ids = {str(row["id"]) for row in synthetic_rows}
    overlap = sorted(official_ids & synthetic_ids)
    if overlap:
        raise ValueError(f"official/synthetic id overlap: {overlap[:10]}")

    train_rows = official_train_rows + sorted(synthetic_rows, key=lambda row: str(row["id"]))
    train_ids = [str(row["id"]) for row in train_rows]
    validation_ids = [str(row["id"]) for row in validation_rows]
    if set(train_ids) & set(validation_ids):
        raise AssertionError("train/validation leakage detected")

    train_path = output_dir / "tool_use_data_train.json"
    validation_path = output_dir / "tool_use_data_val.json"
    manifest_path = output_dir / "dataset_manifest.json"
    write_json(train_path, train_rows)
    write_json(validation_path, validation_rows)

    manifest = {
        "schema_version": 1,
        "policy": {
            "official_data_usage": (
                "published official pool with deterministic train subset and holdout"
            ),
            "synthetic_data_usage": "new curriculum rows are appended to training only",
            "synthetic_cap": synthetic_cap,
            "official_curriculum_filter": {
                "max_normal_path_steps": max_normal_path_steps,
                "max_hack_path_steps": max_hack_path_steps,
            },
            "official_exact_variant_filter": {
                "row_sha256": sorted(include_row_sha256 or []),
            },
            "holdout_seed": seed,
            "holdout_method": holdout_method,
            "leakage_policy": (
                "all variants sharing an official base id stay on one side of the split; "
                "synthetic rows never enter validation"
            ),
        },
        "sources": {
            "official": {
                "path": str(official_path.resolve()),
                "sha256": sha256_file(official_path),
                "rows": len(official_rows),
                "selected_by_exact_variant_filter": len(selected_official_rows),
                "eligible_rows": len(eligible_official_rows),
                "excluded_by_exact_variant_filter": len(official_rows) - len(selected_official_rows),
                "excluded_by_curriculum_filter": len(selected_official_rows) - len(eligible_official_rows),
                "unique_base_ids": official_stats["unique_base_ids"],
                "repeated_id_rows": official_stats["repeated_id_rows"],
                "exact_duplicate_rows": official_stats["exact_duplicate_rows"],
                "selected_train_rows": len(official_train_rows),
                "selected_validation_rows": len(validation_rows),
                "validation_target_rows": validation_count,
                "unselected_rows": len(official_rows) - len(official_train_rows) - len(validation_rows),
            },
            "synthetic": {
                "path": str(synthetic_path.resolve()) if synthetic_path else None,
                "state": synthetic_state,
                "sha256": sha256_file(synthetic_path) if synthetic_path and synthetic_path.is_file() else None,
                "rows": len(synthetic_rows),
            },
        },
        "splits": {
            "train": {
                "path": str(train_path.resolve()),
                "rows": len(train_rows),
                "official_rows": len(official_train_rows),
                "synthetic_rows": len(synthetic_rows),
                "ids_sha256": sha256_json(train_ids),
            },
            "validation": {
                "path": str(validation_path.resolve()),
                "rows": len(validation_rows),
                "official_rows": len(validation_rows),
                "synthetic_rows": 0,
                "ids_sha256": sha256_json(validation_ids),
            },
        },
        "checks": {
            "required_fields": sorted(REQUIRED_FIELDS),
            "official_base_ids_may_repeat": True,
            "no_exact_duplicate_official_rows": official_stats["exact_duplicate_rows"] == 0,
            "train_validation_base_id_disjoint": not bool(set(train_ids) & set(validation_ids)),
            "train_validation_disjoint": not bool(set(train_ids) & set(validation_ids)),
            "synthetic_within_cap": len(synthetic_rows) <= synthetic_cap,
        },
    }
    write_json(manifest_path, manifest)
    manifest["outputs"] = {
        "train_sha256": sha256_file(train_path),
        "validation_sha256": sha256_file(validation_path),
        "manifest_path": str(manifest_path.resolve()),
    }
    write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-count", type=int, default=10)
    parser.add_argument(
        "--official-train-count",
        type=int,
        help="Deterministic official training subset size; omit to use every non-holdout row",
    )
    parser.add_argument("--seed", default="agenticqwen-official-holdout-v1")
    parser.add_argument("--synthetic-data", type=Path)
    parser.add_argument("--synthetic-cap", type=int, default=10)
    parser.add_argument("--allow-missing-synthetic", action="store_true")
    parser.add_argument(
        "--max-normal-path-steps",
        type=int,
        help="Use only official rows whose expected compliant workflow is at most this long",
    )
    parser.add_argument(
        "--max-hack-path-steps",
        type=int,
        help="Use only official rows whose adversarial workflow is at most this long",
    )
    parser.add_argument(
        "--include-row-sha256",
        action="append",
        default=[],
        help="Exact normalized official row SHA-256 to include; repeat for multiple variants",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_dataset(
        official_path=args.official_data.resolve(),
        output_dir=args.output_dir.resolve(),
        validation_count=args.validation_count,
        official_train_count=args.official_train_count,
        seed=args.seed,
        synthetic_path=args.synthetic_data.resolve() if args.synthetic_data else None,
        synthetic_cap=args.synthetic_cap,
        allow_missing_synthetic=args.allow_missing_synthetic,
        max_normal_path_steps=args.max_normal_path_steps,
        max_hack_path_steps=args.max_hack_path_steps,
        include_row_sha256=set(args.include_row_sha256),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
