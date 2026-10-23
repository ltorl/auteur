"""On-disk project layout.

<home>/<slug>/
  project.json        everything the director decided + render state
  cast/<id>.png       character reference sheets
  board/<shot>.png    storyboard keyframes
  clips/<shot>.mp4    rendered shots
  cut/                normalised segments + title cards
  <slug>.mp4          the film
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .schemas import Project, slugify


class Store:
    def __init__(self, home: Path):
        self.home = Path(home).expanduser()

    def dir(self, p: Project | str) -> Path:
        return self.home / (p.slug if isinstance(p, Project) else p)

    def file(self, p: Project | str) -> Path:
        return self.dir(p) / "project.json"

    def unique_slug(self, text: str) -> str:
        base = slugify(text)
        slug, n = base, 2
        while self.dir(slug).exists():
            slug, n = f"{base}-{n}", n + 1
        return slug

    def save(self, p: Project) -> Path:
        p.updated = datetime.now()
        d = self.dir(p)
        for sub in ("cast", "board", "clips", "cut"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        tmp = d / "project.json.tmp"
        tmp.write_text(p.model_dump_json(indent=2))
        tmp.replace(d / "project.json")
        return d

    def load(self, ref: str | Path) -> Project:
        path = Path(ref).expanduser()
        if path.is_dir() and (path / "project.json").exists():
            f = path / "project.json"
        elif path.suffix == ".json" and path.exists():
            f = path
        else:
            f = self.file(str(ref))
            if not f.exists():
                matches = [p for p in self.all() if p.slug.startswith(str(ref))]
                if len(matches) == 1:
                    return matches[0]
                raise FileNotFoundError(f"no project '{ref}' in {self.home}")
        return Project.model_validate_json(f.read_text())

    def all(self) -> list[Project]:
        if not self.home.exists():
            return []
        out = []
        for f in self.home.glob("*/project.json"):
            try:
                out.append(Project.model_validate_json(f.read_text()))
            except Exception:
                continue
        return sorted(out, key=lambda p: p.updated, reverse=True)

    def latest(self) -> Project | None:
        cwd = Path.cwd() / "project.json"
        if cwd.exists():
            return Project.model_validate_json(cwd.read_text())
        projects = self.all()
        return projects[0] if projects else None

    def rel(self, p: Project, path: str | Path | None) -> Path | None:
        """Paths are stored relative to the project dir so folders stay portable."""
        if path is None:
            return None
        path = Path(path)
        return path if path.is_absolute() else self.dir(p) / path
