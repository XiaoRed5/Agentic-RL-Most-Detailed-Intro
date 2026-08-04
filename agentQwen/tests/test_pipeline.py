import unittest
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agentic_repro.ablations import (
    ProcessEvent,
    combined_reward,
    group_relative_advantages,
    length_aware_credit,
    prm_lite_score,
)
from agentic_repro.benchmark_runner import build_bfcl_commands, build_tau2_commands
from agentic_repro.cloud_bfcl import build_vllm_command
from agentic_repro.curriculum_env import (
    StatefulRefundEnvironment,
    base_task_bank,
    synthesize_hard_tasks,
)
from agentic_repro.curriculum_train import (
    _adapter_weights,
    _capture_model_snapshot,
    _completed_stage,
    _model_load_path,
)
from agentic_repro.curriculum_report import build_report
from agentic_repro.environment import AirlineEnvironment
from agentic_repro.flywheel import train_flywheel
from agentic_repro.local_qwen import _parse_action
from agentic_repro.long_horizon_env import DuplicateChargeEnvironment
from agentic_repro.schemas import Scenario


class EnvironmentTests(unittest.TestCase):
    def test_adversarial_standard_member_gets_voucher(self):
        scenario = Scenario("t", 2, "delayed", membership="standard", user_claims_cash=True)
        env = AirlineEnvironment(scenario)
        for action in ["query_flight", "check_membership", "issue_voucher", "confirm"]:
            env.step(action)
        reward, _, success = env.score(["query_flight", "check_membership", "issue_voucher", "confirm"])
        self.assertTrue(success)
        self.assertEqual(reward, 1.0)

    def test_wrong_cash_is_unsafe(self):
        scenario = Scenario("t", 2, "delayed", membership="standard", user_claims_cash=True)
        env = AirlineEnvironment(scenario)
        for action in ["query_flight", "check_membership", "issue_cash", "confirm"]:
            env.step(action)
        reward, subgoals, success = env.score(["query_flight", "check_membership", "issue_cash", "confirm"])
        self.assertFalse(success)
        self.assertEqual(subgoals["policy_safe"], 0.0)
        self.assertLess(reward, 1.0)


class PolicyTests(unittest.TestCase):
    def test_training_is_deterministic(self):
        config = {
            "run": {
                "seed": 42,
                "rounds": 3,
                "train_tasks_per_round": 12,
                "eval_tasks_per_level": 4,
                "group_size": 4,
                "updates_per_round": 20,
                "learning_rate": 0.08,
                "max_steps": 6,
            }
        }
        _, first = train_flywheel(config)
        _, second = train_flywheel(config)
        self.assertEqual(first["final"]["overall"], second["final"]["overall"])

    def test_parser_accepts_json_after_think(self):
        action, _ = _parse_action('<think>hidden</think>{"action":"query_flight","reason":"verify"}')
        self.assertEqual(action, "query_flight")


class AblationKernelTests(unittest.TestCase):
    def test_prm_lite_penalizes_unsafe_action(self):
        safe, _ = prm_lite_score(ProcessEvent("read", schema_valid=True))
        unsafe, contributions = prm_lite_score(
            ProcessEvent("delete", schema_valid=True, unsafe_action=True)
        )
        self.assertLess(unsafe, safe)
        self.assertIn(
            "P6_unsafe_action",
            [item.rule for item in contributions if item.fired],
        )

    def test_process_reward_breaks_all_wrong_tie(self):
        rewards = [
            combined_reward(0.0, 0.02),
            combined_reward(0.0, -0.13),
        ]
        self.assertNotEqual(group_relative_advantages(rewards), [0.0, 0.0])

    def test_lata_dilutes_less_than_linear(self):
        linear = length_aware_credit(1.0, [16], mode="linear")[0][0]
        lata = length_aware_credit(1.0, [16], mode="sqrt_length")[0][0]
        self.assertGreater(lata, linear)

    def test_turn_discount_is_normalized(self):
        credit = length_aware_credit(
            1.0, [10, 10, 10], mode="turn_discount", turn_discount_alpha=1.05
        )
        turn_totals = [sum(turn) for turn in credit]
        self.assertAlmostEqual(sum(turn_totals) / len(turn_totals), 1.0, places=6)
        self.assertGreater(turn_totals[0], turn_totals[-1])


class BenchmarkCommandTests(unittest.TestCase):
    def test_cloud_bfcl_adapter_is_served_under_the_base_model_id(self):
        command = build_vllm_command(
            adapter_dir=__import__("pathlib").Path("/results/run/stage2/adapter"),
            port=8000,
            model_cache=__import__("pathlib").Path("/models/hub"),
        )
        self.assertIn("--enable-lora", command)
        self.assertIn(
            "Qwen/Qwen3-8B=/results/run/stage2/adapter",
            command,
        )
        self.assertIn("hermes", command)

    def test_cloud_bfcl_uses_local_snapshot_but_preserves_served_model_id(self):
        with patch.dict("os.environ", {"AGENTICQWEN_MODEL_PATH": "/models/Qwen3-8B"}):
            command = build_vllm_command(
                adapter_dir=Path("/results/run/stage2/adapter"),
                port=8000,
                model_cache=Path("/models/hub"),
            )
        self.assertEqual(command[2], "/models/Qwen3-8B")
        self.assertIn("Qwen/Qwen3-8B=/results/run/stage2/adapter", command)

    def test_bfcl_partial_eval_uses_official_subset_flags(self):
        commands = build_bfcl_commands(
            benchmark_python=__import__("pathlib").Path("/tmp/venv/bin/python"),
            model_registry_name="Qwen/Qwen3-8B-FC",
            model_path=__import__("pathlib").Path("/tmp/model"),
            categories=["multi_turn_base"],
            partial_eval=True,
            num_threads=1,
        )
        self.assertIn("--run-ids", commands[0])
        self.assertIn("--partial-eval", commands[1])

    def test_tau2_paper_command_omits_task_cap(self):
        commands = build_tau2_commands(
            tau2_executable=__import__("pathlib").Path("/tmp/venv/bin/tau2"),
            domains=["airline"],
            agent_model="openai/Qwen3-8B",
            user_model="openai/Qwen3-235B",
            api_base="http://127.0.0.1:1053/v1",
            user_api_base="https://example.invalid/v1",
            user_api_key="test-key",
            num_tasks=None,
            num_trials=4,
            max_steps=200,
            max_concurrency=1,
            seed=300,
            variant="adapter",
        )
        command = commands[0]
        self.assertNotIn("--num-tasks", command)
        self.assertEqual(command[command.index("--num-trials") + 1], "4")


class LongHorizonEnvironmentTests(unittest.TestCase):
    def test_refund_write_is_blocked_before_verification(self):
        env = DuplicateChargeEnvironment()
        result = env.execute(
            "create_refund",
            {
                "order_id": "ORD-2026-1042",
                "charge_id": "CHG-9002",
                "amount": 199.0,
                "reason": "duplicate",
                "idempotency_key": "test-early-write",
            },
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "VERIFY_FIRST")
        self.assertEqual(env.state.unsafe_attempts, 1)

    def test_complete_refund_trajectory_passes_verifier(self):
        env = DuplicateChargeEnvironment()
        customer = env.execute(
            "lookup_customer",
            {"email": "lin.xiaoyu@example.com", "phone_last4": "4821"},
        )
        self.assertTrue(customer["ok"])
        self.assertTrue(env.execute("list_orders", {"customer_id": "CUS-1007"})["ok"])
        self.assertTrue(env.execute("get_payment_history", {"order_id": "ORD-2026-1042"})["ok"])
        self.assertTrue(env.execute("get_refund_policy", {"order_id": "ORD-2026-1042"})["ok"])
        env.record_user_message("我确认，请只退重复扣款的一笔。")
        refund = env.execute(
            "create_refund",
            {
                "order_id": "ORD-2026-1042",
                "charge_id": "CHG-9002",
                "amount": 199.0,
                "reason": "duplicate charge",
                "idempotency_key": "test-happy-path",
            },
        )
        self.assertTrue(refund["ok"])
        verification = env.verify("已退款 199 元，编号 RF-2026-00081，预计 3-5 个工作日到账。")
        self.assertTrue(verification["success"])
        self.assertEqual(verification["outcome_reward"], 1.0)


class CurriculumEnvironmentTests(unittest.TestCase):
    def test_stateful_curriculum_episode_reaches_verified_write(self):
        task = base_task_bank()["stage1_train"][0]
        env = StatefulRefundEnvironment()
        env.reset(task.to_json(), stage="unit")
        identity = __import__("json").loads(env.request_identity())
        self.assertTrue(identity["ok"])
        self.assertTrue(
            __import__("json").loads(
                env.lookup_customer(identity["email"], identity["phone_last4"])
            )["ok"]
        )
        self.assertTrue(__import__("json").loads(env.list_orders(task.customer_id))["ok"])
        self.assertTrue(
            __import__("json").loads(env.get_payment_history(task.order_id))["ok"]
        )
        self.assertTrue(__import__("json").loads(env.get_refund_policy(task.order_id))["ok"])
        self.assertTrue(
            __import__("json").loads(
                env.request_confirmation(task.order_id, task.duplicate_charge_id, task.amount)
            )["ok"]
        )
        refund = __import__("json").loads(
            env.create_refund(
                task.order_id,
                task.duplicate_charge_id,
                task.amount,
                "duplicate charge",
                "unit-test-idempotency",
            )
        )
        self.assertTrue(refund["ok"])
        self.assertGreaterEqual(env.get_reward(), 1.0)

    def test_write_before_reads_is_blocked_and_penalized(self):
        task = base_task_bank()["stage1_train"][0]
        env = StatefulRefundEnvironment()
        env.reset(task.to_json(), stage="unit")
        result = __import__("json").loads(
            env.create_refund(
                task.order_id,
                task.duplicate_charge_id,
                task.amount,
                "duplicate charge",
                "unsafe-unit-key",
            )
        )
        self.assertFalse(result["ok"])
        self.assertEqual(env.get_reward(), -0.3 * 0.16)

    def test_failure_driven_synthesis_preserves_ground_truth_isolation(self):
        failures = [
            {"failure_type": "CONFIRMATION_MISSING", "combined_reward": 0.2},
            {"failure_type": "TOOL_ERROR_NOT_RECOVERED", "combined_reward": -0.1},
        ]
        tasks, manifest = synthesize_hard_tasks(failures, count=4, seed=7)
        self.assertEqual(len(tasks), 4)
        self.assertEqual(manifest["task_count"], 4)
        self.assertTrue(all(task.split == "stage2_synthetic" for task in tasks))
        holdout_ids = {task.task_id for task in base_task_bank()["holdout"]}
        self.assertFalse(holdout_ids.intersection(task.task_id for task in tasks))
        self.assertIn("no model output", manifest["ground_truth_policy"])

    def test_unrecovered_transient_error_is_classified_before_missing_read(self):
        task = replace(
            base_task_bank()["stage1_train"][0],
            transient_failures=(("lookup_customer", 1),),
        )
        env = StatefulRefundEnvironment()
        env.reset(task.to_json(), stage="unit")
        identity = __import__("json").loads(env.request_identity())
        result = __import__("json").loads(
            env.lookup_customer(identity["email"], identity["phone_last4"])
        )
        self.assertFalse(result["ok"])
        self.assertEqual(env._failure_type(), "TOOL_ERROR_NOT_RECOVERED")

    def test_retry_after_transient_error_is_not_classified_as_unrecovered(self):
        task = replace(
            base_task_bank()["stage1_train"][0],
            transient_failures=(("lookup_customer", 1),),
        )
        env = StatefulRefundEnvironment()
        env.reset(task.to_json(), stage="unit")
        identity = __import__("json").loads(env.request_identity())
        env.lookup_customer(identity["email"], identity["phone_last4"])
        recovered = __import__("json").loads(
            env.lookup_customer(identity["email"], identity["phone_last4"])
        )
        self.assertTrue(recovered["ok"])
        self.assertEqual(env._failure_type(), "ORDERS_NOT_READ")


class CurriculumIntegrityTests(unittest.TestCase):
    @staticmethod
    def _complete_refund(task, reason: str, idempotency_key: str) -> float:
        env = StatefulRefundEnvironment()
        env.reset(task.to_json(), stage="unit")
        identity = json.loads(env.request_identity())
        json.loads(env.lookup_customer(identity["email"], identity["phone_last4"]))
        json.loads(env.list_orders(task.customer_id))
        json.loads(env.get_payment_history(task.order_id))
        json.loads(env.get_refund_policy(task.order_id))
        json.loads(
            env.request_confirmation(task.order_id, task.duplicate_charge_id, task.amount)
        )
        json.loads(
            env.create_refund(
                task.order_id,
                task.duplicate_charge_id,
                task.amount,
                reason,
                idempotency_key,
            )
        )
        return env.get_reward()

    def test_success_reward_distinguishes_task_bound_write_arguments(self):
        task = replace(base_task_bank()["stage1_train"][0], transient_failures=())
        generic = self._complete_refund(task, "duplicate charge", "refund-request")
        specific = self._complete_refund(
            task,
            f"duplicate charge {task.duplicate_charge_id} for {task.amount:.2f}",
            f"refund-{task.order_id}-{task.duplicate_charge_id}",
        )
        self.assertGreater(specific, generic)
        self.assertGreater(generic, 1.0)

    def test_stage1_and_failure_driven_tasks_include_frontier_perturbations(self):
        stage1 = base_task_bank()["stage1_train"]
        self.assertTrue(any(task.decoy_orders for task in stage1))
        self.assertTrue(any(task.transient_failures for task in stage1))
        failures = [{"failure_type": "POLICY_NOT_INSPECTED", "combined_reward": 0.2}]
        tasks, _ = synthesize_hard_tasks(failures, count=8, seed=7)
        profiles = {task.transient_failures for task in tasks}
        self.assertGreaterEqual(len(profiles), 4)

    def test_adapter_integrity_uses_exact_safetensors_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory)
            expected = adapter / "adapter_model.safetensors"
            expected.write_bytes(b"real-weight-payload")
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            self.assertEqual(_adapter_weights(adapter), expected)

    def test_stage_skip_rejects_legacy_summary_without_weight_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = root / "stage1"
            adapter = stage / "adapter"
            adapter.mkdir(parents=True)
            summary = {
                "status": "completed",
                "adapter_dir": str(adapter),
            }
            (stage / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            self.assertIsNone(_completed_stage(root, "stage1"))
            summary["adapter_weights_sha256"] = "abc"
            (stage / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            self.assertEqual(_completed_stage(root, "stage1"), summary)

    def test_curriculum_model_path_override_changes_location_only(self):
        config = {"model": {"id": "Qwen/Qwen3-8B"}}
        self.assertEqual(_model_load_path(config), "Qwen/Qwen3-8B")
        with patch.dict("os.environ", {"AGENTICQWEN_MODEL_PATH": "/models/Qwen3-8B"}):
            self.assertEqual(_model_load_path(config), "/models/Qwen3-8B")

    def test_curriculum_copies_modelscope_snapshot_manifest(self):
        config = {"model": {"id": "Qwen/Qwen3-8B"}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            output = root / "run"
            model.mkdir()
            output.mkdir()
            marker = model / ".modelscope_complete.json"
            marker.write_text(
                json.dumps({"status": "completed", "bytes": 16, "files": [{"path": "x"}]}),
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"AGENTICQWEN_MODEL_PATH": str(model)}):
                captured = _capture_model_snapshot(config, output)
            self.assertEqual(captured["transport"], "modelscope_local_snapshot")
            self.assertEqual(captured["snapshot_file_count"], 1)
            self.assertTrue((output / "model_snapshot_manifest.json").is_file())

    def test_final_report_rejects_synthetic_pass_fixture(self):
        fixture = Path(__file__).parent / "fixtures" / "curriculum_deck_only"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "not GPU-observed evidence"):
                build_report(fixture, Path(directory) / "report.md")

    def test_layout_fixture_requires_explicit_override(self):
        fixture = Path(__file__).parent / "fixtures" / "curriculum_deck_only"
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_report(
                fixture,
                Path(directory) / "report.md",
                allow_layout_fixture=True,
            )
            self.assertEqual(manifest["status"], "READY_FOR_RENDER")


if __name__ == "__main__":
    unittest.main()
