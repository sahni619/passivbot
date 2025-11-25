from __future__ import annotations

import subprocess
from pathlib import Path


MERGE_CONFLICT_MARKERS = ("<<<<<<<", "=======", ">>>>>>>")


def _iter_tracked_files(repo_root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    return [repo_root / line for line in result.stdout.splitlines() if line]


def _contains_merge_markers(path: Path) -> bool:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return False
    return any(line.strip().startswith(MERGE_CONFLICT_MARKERS) for line in lines)


def test_repository_has_no_merge_conflict_markers() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    flagged = [path for path in _iter_tracked_files(repo_root) if _contains_merge_markers(path)]
    assert not flagged, f"Merge conflict markers present in tracked files: {', '.join(str(p) for p in flagged)}"
