#!/usr/bin/env python3
"""Regression test for tracked secret/path scanning."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_git_safety.sh"


def run(script: Path, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script)],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="atec_git_safety_") as directory:
        repo = Path(directory)
        scripts = repo / "scripts"
        scripts.mkdir()
        copied = scripts / SCRIPT.name
        shutil.copy2(SCRIPT, copied)
        (repo / "README.md").write_text("portable repository\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "add", "README.md", "scripts/check_git_safety.sh"], cwd=repo, check=True)

        safe = run(copied, repo)
        assert safe.returncode == 0, safe.stdout

        leak = repo / "leak.txt"
        leak.write_text(str(Path.home() / "private/data") + "\n", encoding="utf-8")
        subprocess.run(["git", "add", "leak.txt"], cwd=repo, check=True)
        home_result = run(copied, repo)
        assert home_result.returncode == 1, home_result.stdout
        assert "敏感检查失败" in home_result.stdout

        leak.write_text("gh" + "o_example_token\n", encoding="utf-8")
        subprocess.run(["git", "add", "leak.txt"], cwd=repo, check=True)
        token_result = run(copied, repo)
        assert token_result.returncode == 1, token_result.stdout
        assert "敏感检查失败" in token_result.stdout

    print("GIT_SAFETY_ASSERTIONS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
