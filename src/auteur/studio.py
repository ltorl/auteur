"""The production pipeline: pitch → plan → cast → board → shoot → cut."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import imagery, prompts
from .config import Settings
from .llm import LLM, Progress
from .render import base, media
from .schemas import Message, Project, Revision, Shot, ShotList, ShotPlan, Treatment
from .store import Store

ItemDone = Callable[[str, Path | None, Exception | None], None]


def shot_seed(project: Project, shot_id: str) -> int:
    return int(hashlib.sha256(f"{project.slug}:{shot_id}".encode()).hexdigest()[:8], 16)


def treatment_text(t: Treatment) -> str:
    return t.model_dump_json(indent=1)


class Studio:
    def __init__(self, settings: Settings, store: Store | None = None, llm: LLM | None = None):
        self.settings = settings
        self.store = store or Store(settings.home_path)
        self.llm = llm or LLM(settings)

    # ---- writing ------------------------------------------------------------ #
    def create(self, idea: str, template: str, seconds: int, aspect: str, style: str = "") -> Project:
        p = Project(slug=self.store.unique_slug(idea), idea=idea, template=template,
                    target_seconds=seconds, aspect=aspect, style_hint=style)
        self.store.save(p)
        return p

    def pitch(self, p: Project, progress: Progress | None = None) -> Treatment:
        tpl = prompts.TEMPLATES.get(p.template, prompts.TEMPLATES["short-film"])
        user = prompts.PITCH.format(
            idea=p.idea, template_label=tpl["label"], template_brief=tpl["brief"],
            seconds=p.target_seconds, aspect=p.aspect, style_line=prompts.style_line(p.style_hint),
        )
        t = self.llm.structured(prompts.DIRECTOR, user, Treatment, progress)
        p.treatment = t
        self.store.save(p)
        return t

    def plan(self, p: Project, progress: Progress | None = None) -> list[Shot]:
        if p.treatment is None:
            self.pitch(p, progress)
        user = prompts.SHOTLIST.format(treatment=treatment_text(p.treatment), seconds=p.target_seconds,
                                       aspect=p.aspect)
        plan = self.llm.structured(prompts.DIRECTOR, user, ShotList, progress)
        p.shots = [Shot(**s.model_dump(), seed=shot_seed(p, s.id)) for s in plan.shots]
        p.final = None
        self.store.save(p)
        return p.shots

    def revise(self, p: Project, note: str, progress: Progress | None = None) -> Revision:
        state = {"treatment": p.treatment.model_dump() if p.treatment else None,
                 "shots": [ShotPlan(**s.model_dump(include=set(ShotPlan.model_fields))).model_dump()
                           for s in p.shots]}
        history = []
        for m in p.chat[-8:]:
            history.append({"role": "user" if m.role == "user" else "assistant", "content": m.text})
        user = prompts.REVISE.format(state=json.dumps(state, indent=1), note=note)
        rev = self.llm.structured(prompts.DIRECTOR, user, Revision, progress, history=history)
        p.chat += [Message(role="user", text=note), Message(role="director", text=rev.reply)]
        if rev.treatment is not None:
            self._apply_treatment(p, rev.treatment)
        if rev.shots is not None:
            self._apply_shots(p, rev.shots)
        self.store.save(p)
        return rev

    def _apply_treatment(self, p: Project, t: Treatment) -> None:
        old = {c.id: c for c in (p.treatment.characters if p.treatment else [])}
        changed = {c.id for c in t.characters if c.id not in old or old[c.id].look != c.look}
        for cid in changed:
            p.cast_sheets.pop(cid, None)
        style_changed = p.treatment is None or p.treatment.style != t.style
        p.treatment = t
        for s in p.shots:
            if style_changed or changed & set(s.characters):
                self._invalidate(s, keyframe=True)

    def _apply_shots(self, p: Project, plans: list[ShotPlan]) -> None:
        old = {s.id: s for s in p.shots}
        visual = {"keyframe_prompt", "characters", "location", "shot_size", "lens"}
        motion = {"motion_prompt", "duration", "camera", "dialogue", "sound", "action"}
        shots = []
        for plan in plans:
            prev = old.get(plan.id)
            if prev is None:
                shots.append(Shot(**plan.model_dump(), seed=shot_seed(p, plan.id)))
                continue
            data = plan.model_dump()
            prev_data = prev.model_dump()
            s = Shot(**{**prev_data, **data})  # validated, unlike model_copy(update=)
            if any(data[k] != prev_data[k] for k in visual):
                self._invalidate(s, keyframe=True)
            elif any(data[k] != prev_data[k] for k in motion):
                self._invalidate(s, keyframe=False)
            shots.append(s)
        p.shots = shots
        p.final = None

    @staticmethod
    def _invalidate(s: Shot, keyframe: bool) -> None:
        s.clip = None
        s.error = None
        if keyframe:
            s.keyframe = None
            s.status = "planned"
        elif s.status == "shot":
            s.status = "boarded" if s.keyframe else "planned"

    # ---- art department ----------------------------------------------------- #
    def _style(self, p: Project) -> str:
        st = p.treatment.style
        return f"{st.visual_style}; {st.texture}".strip("; ")

    def cast(self, p: Project, ids: Iterable[str] | None = None, force: bool = False,
             on_item: ItemDone | None = None, workers: int = 3) -> None:
        assert p.treatment, "pitch first"
        chars = [c for c in p.treatment.characters if ids is None or c.id in set(ids)]
        todo = [c for c in chars if force or not self._exists(p, p.cast_sheets.get(c.id))]
        d = self.store.dir(p)
        model = self.llm.image_model

        def job(c):
            prompt = prompts.CAST_SHEET.format(name=c.name, look=c.look, style=self._style(p))
            return imagery.generate(self.llm.client, model, prompt, "1536x1024", d / "cast" / f"{c.id}.png")

        for c, path, err in self._parallel(todo, job, workers):
            if path:
                p.cast_sheets[c.id] = str(path.relative_to(d))
                self.store.save(p)
            if on_item:
                on_item(c.name, path, err)

    def keyframe_prompt(self, p: Project, s: Shot) -> str:
        st = p.treatment.style
        names = {c.id: c for c in p.treatment.characters}
        cast = "".join(f"{names[c].name}: {names[c].look}. " for c in s.characters if c in names)
        refs = [c for c in s.characters if c in p.cast_sheets]
        if refs:
            order = ", ".join(f"reference image {i + 1} is {names[c].name}" for i, c in enumerate(refs))
            cast = f"Keep characters identical to the references ({order}). {cast}"
        return prompts.KEYFRAME.format(
            aspect=p.aspect, prompt=s.keyframe_prompt, style=st.visual_style, palette=st.palette,
            lighting=st.lighting, texture=st.texture, cast=cast + "\n" if cast else "",
        )

    def board(self, p: Project, ids: Iterable[str] | None = None, force: bool = False,
              on_item: ItemDone | None = None, workers: int = 3) -> None:
        assert p.treatment, "pitch first"
        wanted = {i.upper() for i in ids} if ids else None
        todo = [s for s in p.shots if (wanted is None or s.id.upper() in wanted)
                and (force or not self._exists(p, s.keyframe))]
        d = self.store.dir(p)
        model = self.llm.image_model
        size = imagery.api_size(p.aspect)

        def job(s: Shot):
            out = d / "board" / f"{s.id}.png"
            prompt = self.keyframe_prompt(p, s)
            refs = [self.store.rel(p, p.cast_sheets[c]) for c in s.characters if c in p.cast_sheets][:4]
            refs = [r for r in refs if r and r.exists()]
            if refs:
                try:
                    return imagery.edit(self.llm.client, model, prompt, refs, size, out)
                except Exception:
                    pass  # some accounts/models reject edits — fall back to plain generation
            return imagery.generate(self.llm.client, model, prompt, size, out)

        for s, path, err in self._parallel(todo, job, workers):
            if path:
                s.keyframe = str(path.relative_to(d))
                s.status = "boarded"
                s.clip = None
            else:
                s.error = str(err)
            self.store.save(p)
            if on_item:
                on_item(s.id, path, err)

    def _exists(self, p: Project, rel: str | None) -> bool:
        path = self.store.rel(p, rel)
        return bool(path and path.exists())

    @staticmethod
    def _parallel(items, fn, workers):
        if not items:
            return
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futs = {pool.submit(fn, it): it for it in items}
            for fut in as_completed(futs):
                it = futs[fut]
                try:
                    yield it, fut.result(), None
                except Exception as e:  # noqa: BLE001 — reported per item
                    yield it, None, e

    # ---- camera department -------------------------------------------------- #
    def backend(self, name: str | None = None, quality: str | None = None):
        name = name or self.settings.backend
        quality = quality or self.settings.quality
        if name == "animatic":
            from .render.animatic import AnimaticBackend

            return AnimaticBackend()
        if name == "ltx-pipelines":
            from .render.ltx_official import LTXOfficialBackend

            return LTXOfficialBackend(self.settings.ltx2_repo, self.settings.ltx_checkpoints)
        if name == "airstream":
            from .render.ltx import LTXStreamEngine

            return LTXStreamEngine(
                self.settings.weights, device=self.settings.device,
                cache_root=self.settings.layer_cache, compress=self.settings.compress,
                two_stage=quality == "hq", text_batch=self.settings.text_batch,
            )
        raise ValueError(f"unknown backend '{name}' (airstream | ltx-pipelines | animatic)")

    def jobs(self, p: Project, ids: Iterable[str] | None, quality: str, force: bool) -> list[base.ShotJob]:
        wanted = {i.upper() for i in ids} if ids else None
        w, h = base.resolution(p.aspect, quality)
        d = self.store.dir(p)
        out = []
        for s in p.shots:
            if wanted is not None and s.id.upper() not in wanted:
                continue
            if not force and self._exists(p, s.clip):
                continue
            out.append(base.ShotJob(
                shot_id=s.id, prompt=s.motion_prompt, out=d / "clips" / f"{s.id}.mp4",
                seconds=s.duration, width=w, height=h, seed=s.seed,
                image=self.store.rel(p, s.keyframe), camera=s.camera,
            ))
        return out

    def shoot(self, p: Project, emit: base.Emit, ids: Iterable[str] | None = None,
              backend: str | None = None, quality: str | None = None, force: bool = False,
              engine=None) -> list[Path]:
        quality = quality or self.settings.quality
        jobs = self.jobs(p, ids, quality, force)
        if not jobs:
            return []
        engine = engine or self.backend(backend, quality)
        d = self.store.dir(p)
        by_id = {s.id: s for s in p.shots}

        def track(e: base.Event):
            if e.kind == "done" and e.shot:
                s = by_id[e.shot]
                s.clip = str(Path(e.text).relative_to(d))
                s.status, s.backend, s.error = "shot", engine.name, None
                p.final = None
                self.store.save(p)
            emit(e)

        try:
            return engine.render(jobs, track)
        finally:
            close = getattr(engine, "close", None)
            if close:
                close()

    # ---- editorial ---------------------------------------------------------- #
    def cut(self, p: Project, titles: bool = True, music: Path | None = None,
            on_step: Callable[[str], None] | None = None) -> Path:
        d = self.store.dir(p)
        width, height = base.DELIVERY.get(p.aspect, base.DELIVERY["16:9"])
        fps = base.FPS
        shots = [s for s in p.shots if self._exists(p, s.clip)]
        if not shots:
            raise RuntimeError("no shots rendered yet — run `auteur shoot` first")
        missing = [s.id for s in p.shots if s not in shots]
        segs: list[media.Segment] = []
        say = on_step or (lambda _m: None)
        if titles and p.treatment:
            say("Title card")
            card = d / "cut" / "title.png"
            imagery.title_card(p.treatment.title, p.treatment.logline, width, height).save(card)
            segs.append(media.Segment(media.still_clip(card, d / "cut" / "title.mp4", 3.0, width, height, fps)))
        for s in shots:
            say(f"Conform {s.id}")
            norm = media.normalize_clip(self.store.rel(p, s.clip), d / "cut" / f"{s.id}.mp4", width, height, fps)
            segs.append(media.Segment(norm, "dissolve" if titles and len(segs) == 1 else s.transition))
        if titles and p.treatment:
            card = d / "cut" / "end.png"
            imagery.title_card(p.treatment.title, "Directed with Auteur", width, height).save(card)
            segs.append(media.Segment(media.still_clip(card, d / "cut" / "end.mp4", 2.5, width, height, fps),
                                      "fade"))
        say("Assembling" + (f" (skipping unshot {', '.join(missing)})" if missing else ""))
        out = media.assemble(segs, d / f"{p.slug}.mp4", music=music)
        p.final = str(out.relative_to(d))
        self.store.save(p)
        return out
