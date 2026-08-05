import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentic_repro.industrial_pipeline import IndustrialRunner, load_config, preflight, tree_inventory


class IndustrialPipelineTests(unittest.TestCase):
    def setUp(self):
        self.project = Path(__file__).resolve().parents[1]
        self.config_path = self.project / "configs" / "industrial_agenticqwen.json"

    @staticmethod
    def _fake_upstream(root: Path) -> Path:
        upstream = root / "data_synth_and_rl"
        for relative in [
            "tool_use_data_synthesis/run_data_gen.py",
            "tool_use_data_synthesis/run_solve_task.py",
            "tool_use_data_synthesis/run_rubrics.py",
            "tool_use_data_synthesis/make_filtered_verl_data.py",
            "RL/my_script/data_process/virtual_tool_use_convert_parquet.py",
        ]:
            path = upstream / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# portable test fixture\n")
        return upstream

    def test_preflight_accepts_pinned_archive_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            upstream = self._fake_upstream(Path(tmp))
            with patch.dict(os.environ, {"AGENTICQWEN_UPSTREAM_REPO": str(upstream)}):
                config = load_config(self.config_path)
                result = preflight(config, Path(tmp) / "run")
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["upstream"]["git"]["archive_lock"])
        self.assertTrue(result["upstream"]["remote_ok"])
        self.assertIsInstance(result["credentials"]["configured"], bool)

    def test_dry_run_is_resumable_and_redacts_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            upstream = self._fake_upstream(tmp_path)
            with patch.dict(os.environ, {"AGENTICQWEN_UPSTREAM_REPO": str(upstream)}):
                config = load_config(self.config_path)
                runner = IndustrialRunner(config, tmp_path / "run", dry_run=True)
                first = runner.run(["preflight", "data_gen"])
                second = runner.run(["preflight", "data_gen"])
            self.assertEqual(first["status"], "completed")
            self.assertEqual(second["status"], "completed")
            state = json.loads((tmp_path / "run" / "stages" / "data_gen.json").read_text())
            self.assertEqual(state["status"], "dry_run")
            self.assertNotIn("sensitive-test-token", json.dumps(state))
            self.assertIn("AIGC_APP_ID", state["command"]["env_names"])

    def test_inventory_changes_when_output_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out"
            path.mkdir()
            (path / "a.txt").write_text("a")
            first = tree_inventory(path)
            (path / "a.txt").write_text("b")
            second = tree_inventory(path)
            self.assertNotEqual(first["sha256"], second["sha256"])


if __name__ == "__main__":
    unittest.main()
