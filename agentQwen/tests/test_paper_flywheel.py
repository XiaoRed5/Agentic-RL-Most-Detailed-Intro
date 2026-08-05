import json
import inspect
import tempfile
import unittest
from pathlib import Path

from agentic_repro.paper_flywheel import (
    AgenticQwenDataFlywheel,
    DeterministicStrongModel,
    _extract_json_object,
    linear_seed_tree,
    validate_candidate,
    write_flywheel_artifacts,
)
from agentic_repro.paper_flywheel_env import execute_path
from agentic_repro.paper_flywheel_env import PaperFlightEnvironment
from agentic_repro.paper_grpo_train import (
    build_training_plan,
    read_policy_rollouts,
    run_pipeline,
    seed_task_variants,
    task_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class WrongBranchTeacher(DeterministicStrongModel):
    model_name = "wrong-branch-test-teacher"

    def solve_task(self, task):
        return task.hack_path


class PaperFlywheelTests(unittest.TestCase):
    def setUp(self):
        self.teacher = DeterministicStrongModel()
        self.rounds = AgenticQwenDataFlywheel(self.teacher).evolve(rounds=3)

    def test_three_rounds_grow_linear_workflow_into_behavior_tree(self):
        self.assertEqual([item.tree.revision for item in self.rounds], [1, 2, 3])
        self.assertEqual([len(item.tree.branches()) for item in self.rounds], [2, 4, 5])
        self.assertTrue(all(item.rollout_count > 0 for item in self.rounds))
        self.assertTrue(all("rollout_sha256=" in item.tree.provenance for item in self.rounds))

    def test_synthesis_budget_caps_new_tasks_per_round(self):
        bounded = AgenticQwenDataFlywheel(self.teacher).evolve(
            rounds=3,
            max_new_tasks_per_round=2,
        )
        self.assertTrue(all(len(item.tasks) <= 2 for item in bounded))
        self.assertLessEqual(sum(len(item.tasks) for item in bounded), 6)

    def test_branch_to_task_inversion_has_three_inputs_and_auditable_paths(self):
        task = next(
            task
            for task in self.rounds[1].tasks
            if "issue_voucher" in task.normal_actions
        )
        streams = task.training_streams()
        self.assertEqual(
            set(streams),
            {"environment_input", "user_input", "agent_input"},
        )
        self.assertEqual(task.provenance["algorithm"], "branch_to_task_inversion")
        self.assertIn("issue_voucher", task.normal_actions)
        self.assertIn("issue_cash", task.hack_actions)
        self.assertIn("User claims never override", task.agent_instruction)
        compat = task.official_compatibility_fields()
        self.assertEqual(compat["test_policy"], task.agent_instruction)
        self.assertTrue(compat["user_escape_strategy"])
        self.assertEqual(
            compat["tool_return_expected"]["normal_path"][-1]["tool_name"],
            "confirm",
        )
        self.assertIn("intended-branch-hit", compat["rubrics"])

    def test_standard_member_adversary_cannot_turn_voucher_into_cash(self):
        task = next(
            task
            for task in self.rounds[1].tasks
            if "issue_voucher" in task.normal_actions
        )
        normal = execute_path(task, task.normal_path)
        hacked = execute_path(task, task.hack_path)
        self.assertEqual(normal.reward, 1.0)
        self.assertTrue(normal.intended_branch_hit)
        self.assertTrue(normal.success)
        self.assertGreater(hacked.final_state["unsafe_action_count"], 0)
        self.assertLess(hacked.reward, 1.0)
        self.assertFalse(hacked.success)
        self.assertGreaterEqual(hacked.reward, 0.0)

    def test_every_adversarial_candidate_exercises_mock_user(self):
        adversarial_tasks = [
            task
            for item in self.rounds
            for task in item.tasks
            if task.user_input.adversarial_strategy
        ]
        self.assertTrue(adversarial_tasks)
        for task in adversarial_tasks:
            trajectory = execute_path(task, task.normal_path)
            self.assertGreaterEqual(trajectory.final_state["user_turns"], 1)
            self.assertEqual(trajectory.rubric_scores["mock-user-turn"], 1.0)
            self.assertTrue(trajectory.success)

    def test_teacher_gate_rejects_candidate_that_misses_selected_branch(self):
        task = next(task for task in self.rounds[1].tasks if task.hack_path)
        result = validate_candidate(task, WrongBranchTeacher())
        self.assertFalse(result.solved)
        self.assertFalse(result.intended_branch_hit)
        self.assertFalse(result.retained)

    def test_task_rows_do_not_leak_ground_truth_path_into_policy_prompt(self):
        task = self.rounds[-1].tasks[-1]
        row = task_rows((task,), "round3")[0]
        rendered = json.dumps(row["prompt"], ensure_ascii=False)
        self.assertIn(task.user_input.request, rendered)
        self.assertIn(task.agent_instruction, rendered)
        self.assertNotIn("normal_path", rendered)
        self.assertNotIn(task.source_branch_id, rendered)
        self.assertIn("training_streams", json.loads(row["task_json"]))
        self.assertIn("agenticqwen_official_compat", json.loads(row["task_json"]))
        self.assertEqual(row["extra_info"]["test_policy"], task.agent_instruction)
        self.assertEqual(
            row["extra_info"]["tool_return_expected"]["hack_path"],
            task.official_compatibility_fields()["tool_return_expected"]["hack_path"],
        )

    def test_environment_exposes_no_audit_backdoor_as_a_tool(self):
        reserved = {"reset", "get_reward"}
        expected_tools = {
            "ask_user",
            "query_flight",
            "book_flight",
            "search_hsr",
            "book_hsr",
            "search_nearby_airport",
            "book_nearby_flight",
            "check_membership",
            "issue_cash",
            "issue_voucher",
            "confirm",
        }
        public = {
            name
            for name, value in inspect.getmembers(PaperFlightEnvironment, predicate=callable)
            if not name.startswith("_")
        }
        self.assertEqual(public - reserved, expected_tools)

    def test_contract_artifacts_have_hash_verifiable_round_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_flywheel_artifacts(root, self.rounds)
            self.assertEqual(manifest["round_count"], 3)
            for item in manifest["rounds"]:
                self.assertTrue((root / item["paths"]["tree"]).is_file())
                self.assertTrue((root / item["paths"]["tasks"]).is_file())
                self.assertTrue((root / item["paths"]["validation"]).is_file())
                rows = [
                    json.loads(line)
                    for line in (root / item["paths"]["validation"]).read_text(encoding="utf-8").splitlines()
                ]
                self.assertTrue(all(row["retained"] for row in rows))

    def test_policy_trace_reader_enforces_normalized_reward(self):
        task = self.rounds[0].tasks[0]
        trajectory = execute_path(task, task.normal_path)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text(json.dumps(trajectory.to_dict()) + "\n", encoding="utf-8")
            values = read_policy_rollouts(path)
            self.assertEqual(len(values), 1)
            bad = trajectory.to_dict()
            bad["reward"] = 1.1
            path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside"):
                read_policy_rollouts(path)

    def test_micro_config_has_round_zero_through_three(self):
        config = json.loads(
            (PROJECT_ROOT / "configs" / "agenticqwen_paper_micro.json").read_text(encoding="utf-8")
        )
        plan = build_training_plan(config)
        self.assertEqual(
            [item["name"] for item in plan["stages"]],
            ["round0", "round1", "round2", "round3"],
        )
        self.assertEqual(plan["reward_range"], [0.0, 1.0])
        self.assertFalse(plan["paper_scale_claimed"])
        self.assertEqual(plan["synthesis_budget"]["max_total_tasks"], 10)

    def test_full_train_refuses_contract_teacher_by_default(self):
        config_path = PROJECT_ROOT / "configs" / "agenticqwen_paper_micro.json"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "real strong-model endpoint"):
                run_pipeline(config_path, Path(directory))

    def test_json_extractor_accepts_fenced_teacher_output(self):
        self.assertEqual(_extract_json_object("```json\n{\"ok\": true}\n```"), {"ok": True})

    def test_seed_tree_is_exactly_one_synthagent_happy_path(self):
        tree = linear_seed_tree()
        branches = tree.branches()
        self.assertEqual(len(branches), 1)
        self.assertEqual(
            branches[0].actions,
            ("query_flight", "book_flight", "confirm"),
        )

    def test_round_zero_instantiates_linear_seed_in_multiple_contexts(self):
        seed = AgenticQwenDataFlywheel(self.teacher).seed_task()
        variants = seed_task_variants(seed, ("Beijing", "Shanghai", "Shenzhen", "Chengdu"))
        self.assertEqual(len(variants), 4)
        self.assertEqual(len({task.task_id for task in variants}), 4)
        for task in variants:
            destination = task.environment_state["destination"]
            self.assertIn(destination, task.user_input.request)
            self.assertTrue(
                all(
                    call.arguments.get("destination", destination) == destination
                    for call in task.normal_path
                )
            )


if __name__ == "__main__":
    unittest.main()
