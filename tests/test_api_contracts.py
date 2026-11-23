"""Contract tests verifying repository structure, API files, and secret hygiene."""

import json
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestRepoAndApiContracts(unittest.TestCase):
    def test_required_standard_directories_exist(self) -> None:
        for folder in ("src", "lib", "bin", "docs", "tests", "research"):
            self.assertTrue(
                (REPO_ROOT / folder).is_dir(), f"Missing required folder: {folder}"
            )

    def test_vercel_config_is_valid_json(self) -> None:
        vercel_cfg = REPO_ROOT / "src" / "vercel.json"
        self.assertTrue(vercel_cfg.is_file())
        parsed = json.loads(vercel_cfg.read_text(encoding="utf-8"))
        self.assertIsInstance(parsed, dict)

    def test_required_api_endpoints_exist(self) -> None:
        for endpoint in ("health.js", "gemini-status.js", "gemini-canonicalize.js"):
            self.assertTrue((REPO_ROOT / "src" / "api" / endpoint).is_file())

    def test_no_env_local_secret_file_in_repo(self) -> None:
        self.assertFalse((REPO_ROOT / ".env.local").exists())
        self.assertFalse((REPO_ROOT / "src" / ".env.local").exists())


if __name__ == "__main__":
    unittest.main()
