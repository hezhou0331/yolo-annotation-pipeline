"""Portable path serialization and legacy-workspace relocation helpers.

Repository-owned files should store paths relative to the file that contains
them. Older snapshots stored producer-machine absolute paths instead. The
helpers in this module keep new output portable while allowing those legacy
paths to be interpreted inside the current checkout.
"""
from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT_ROOT = REPOSITORY_ROOT / "projects" / "atec_real"

_PROJECT_MARKERS = frozenset({"data", "datasets", "manifests", "assets", "reports"})
_REPOSITORY_MARKERS = frozenset({"models", "third_party"})
_PROJECT_DIRECTORIES = frozenset({"data", "datasets", "manifests", "assets", "reports"})


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _find_legacy_marker(path: Path, repo: Path, project: Path) -> tuple[int, str] | None:
    """Find a marker directly owned by a recognizable old checkout root."""

    parts = path.parts
    markers = _PROJECT_MARKERS | _REPOSITORY_MARKERS
    project_names = {name for name in (project.name, DEFAULT_PROJECT_ROOT.name) if name}
    candidates: set[tuple[int, str]] = set()

    def add(marker_index: int) -> None:
        if marker_index < len(parts) and parts[marker_index] in markers:
            candidates.add((marker_index, parts[marker_index]))

    for index, part in enumerate(parts):
        if part == repo.name:
            add(index + 1)
            if (
                index + 3 < len(parts)
                and parts[index + 1] == "projects"
                and parts[index + 2] in project_names
            ):
                add(index + 3)
        if part in project_names:
            add(index + 1)

    return max(candidates, key=lambda item: item[0]) if candidates else None


def infer_project_root(
    path: str | Path,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
    default: str | Path | None = None,
) -> Path:
    """Infer the project root that owns a manifest, dataset, or scene path."""

    candidate = _resolved(path)
    repo = _resolved(repository_root)
    fallback = _resolved(default) if default is not None else repo / "projects" / "atec_real"

    if _is_relative_to(candidate, repo):
        relative = candidate.relative_to(repo)
        parts = relative.parts
        if len(parts) >= 3 and parts[0] == "projects" and parts[2] in _PROJECT_DIRECTORIES:
            return repo / parts[0] / parts[1]
        return fallback

    # Do not derive ownership from a purely lexical marker in a missing external
    # path. Callers can still pass an explicit/default project root for outputs.
    if not candidate.exists():
        return fallback

    for current in (candidate, *candidate.parents):
        if current.name in _PROJECT_DIRECTORIES:
            return current.parent
    return fallback


def resolve_compatible_path(
    value: str | Path,
    *,
    base: str | Path | None = None,
    repository_root: str | Path = REPOSITORY_ROOT,
    project_root: str | Path | None = None,
) -> Path:
    """Resolve a relative path or relocate a legacy absolute repository path.

    A missing absolute path is rebased only when a supported marker is a
    direct child of a repository/project-name anchor (or follows the canonical
    "projects/<project-name>" layout). Common external directory names alone
    are deliberately not treated as migration evidence.
    """

    raw = Path(value).expanduser()
    repo = _resolved(repository_root)
    project = _resolved(project_root) if project_root is not None else repo / "projects" / "atec_real"
    if not raw.is_absolute():
        anchor = _resolved(base) if base is not None else Path.cwd().resolve()
        return (anchor / raw).resolve(strict=False)

    absolute = raw.resolve(strict=False)
    if _is_relative_to(absolute, repo) or _is_relative_to(absolute, project):
        return absolute
    if absolute.exists():
        return absolute

    match = _find_legacy_marker(absolute, repo, project)
    if match is None:
        return absolute

    index, marker = match
    suffix = Path(*absolute.parts[index + 1 :])
    anchor = project / marker if marker in _PROJECT_MARKERS else repo / marker
    return (anchor / suffix).resolve(strict=False)


def portable_path(
    value: str | Path,
    *,
    relative_to: str | Path,
    repository_root: str | Path = REPOSITORY_ROOT,
    project_root: str | Path | None = None,
) -> str:
    """Serialize repository/project paths relatively and leave externals absolute."""

    path = _resolved(value)
    base = _resolved(relative_to)
    repo = _resolved(repository_root)
    project = _resolved(project_root) if project_root is not None else repo / "projects" / "atec_real"
    if not (_is_relative_to(path, repo) or _is_relative_to(path, project)):
        return str(path)
    return Path(os.path.relpath(path, start=base)).as_posix()
