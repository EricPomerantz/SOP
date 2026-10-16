"""The SOP builder's API (/builder/...): drafts, detector import, generated steps, publishing (sopvision.builder)."""

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from sopvision.builder import detector as det
from sopvision.builder import drafts
from sopvision.builder.drafts import DraftError
from sopvision.builder.status import checklist, recordings
from sopvision.config import ConfigError
from sopvision.server.runs import RunConflictError, RunManager
from sopvision.settings import Settings

MAX_UPLOAD = 1 << 30  # 1 GB: detection weights are tens to hundreds of MB


class NewSop(BaseModel):
    id: str
    name: str = ""
    description: str = ""


class SaveSop(BaseModel):
    data: dict


class ExistingModel(BaseModel):
    path: str


class Activate(BaseModel):
    checkpoint: str | None  # a folder under data/training/models; None: back to the shared checkpoint


class Timeline(BaseModel):
    rows: list[tuple[str, float, float]]


def builder_router(settings: Settings, manager: RunManager) -> APIRouter:
    r = APIRouter(prefix="/builder")

    def view(sop_id: str, warnings: list[str] | None = None) -> dict:
        draft, published = drafts.find(settings, sop_id)
        sf = draft or published
        if sf is None:
            raise HTTPException(404, f"No SOP {sop_id}")
        return {"id": sf.id, "file": str(sf.path), "draft": sf.draft, "data": sf.data, "warnings": warnings or [],
                "status": checklist(settings, sf, published, sop_id in manager.sops) | {"runs": recordings(settings, sf.data)}}

    def draft_of(sop_id: str) -> drafts.SopFile:
        draft, published = drafts.find(settings, sop_id)
        if draft is None:
            raise HTTPException(409 if published else 404,
                                f"{sop_id} is published: start editing it first" if published else f"No SOP {sop_id}")
        return draft

    @r.get("/sops")
    def list_sops():
        out = {}
        for sf in drafts.scan(settings):
            entry = out.setdefault(sf.id, {"id": sf.id, "name": sf.data.get("name") or sf.id, "draft": False,
                                           "published": False, "loaded": sf.id in manager.sops})
            entry["draft" if sf.draft else "published"] = True
            if sf.draft:
                entry["name"] = sf.data.get("name") or sf.id
        return sorted(out.values(), key=lambda e: e["name"].lower())

    @r.post("/sops", status_code=201)
    def create(body: NewSop):
        try:
            drafts.new_draft(settings, body.id, body.name, body.description)
        except DraftError as e:
            raise HTTPException(400, str(e)) from None
        return view(body.id)

    @r.get("/sops/{sop_id}")
    def get(sop_id: str):
        return view(sop_id)

    @r.put("/sops/{sop_id}")
    def save(sop_id: str, body: SaveSop):
        try:
            drafts.write(draft_of(sop_id), body.data)
        except DraftError as e:
            raise HTTPException(400, str(e)) from None
        return view(sop_id)

    @r.post("/sops/{sop_id}/edit")
    def edit(sop_id: str):
        try:
            drafts.start_edit(settings, sop_id)
        except DraftError as e:
            raise HTTPException(404, str(e)) from None
        return view(sop_id)

    @r.post("/sops/{sop_id}/discard")
    def discard(sop_id: str):
        try:
            drafts.discard(settings, sop_id)
        except DraftError as e:
            raise HTTPException(404, str(e)) from None
        draft, published = drafts.find(settings, sop_id)
        return view(sop_id) if published else {"id": sop_id, "deleted": True}

    @r.post("/sops/{sop_id}/publish")
    def publish(sop_id: str):
        draft_of(sop_id)
        if manager.current() is not None or manager.recording is not None:
            raise HTTPException(409, "Stop the current run or recording first: publishing reloads the SOPs.")
        try:
            path = drafts.publish(settings, sop_id)
        except DraftError as e:
            raise HTTPException(400, str(e)) from None
        try:
            manager.reload_sops()
        except (ConfigError, RunConflictError) as e:
            raise HTTPException(500, f"Published {path}, but the app couldn't reload its SOPs: {e}") from None
        return view(sop_id)

    @r.post("/sops/{sop_id}/generate-steps")
    def generate_steps(sop_id: str):
        """Steps, order and groups from the draft's slots (not saved: the page shows them for editing)."""
        return drafts.generate_steps(drafts.current(settings, sop_id).data)

    @r.get("/sops/{sop_id}/autofit")
    def autofit_result(sop_id: str):
        """The last slot auto-fit (data/builder/<id>/autofit.json), or null before the first."""
        import json

        from sopvision.builder.autofit import output_dir

        drafts.current(settings, sop_id)
        path = output_dir(sop_id) / "autofit.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    @r.get("/sops/{sop_id}/frames/{run}.jpg")
    def sample_frame(sop_id: str, run: str):
        from fastapi.responses import FileResponse

        from sopvision.builder.autofit import output_dir

        path = output_dir(sop_id) / "frames" / f"{Path(run).name}.jpg"
        if not drafts.SOP_ID.match(sop_id) or not path.exists():
            raise HTTPException(404, "No sample frame")
        return FileResponse(path, media_type="image/jpeg")

    def run_folder(sop_id: str, run: str) -> tuple[drafts.SopFile, Path]:
        import re

        sf = drafts.current(settings, sop_id)
        dataset = sf.data.get("dataset") or {}
        folder = settings.datasets_dir / (dataset.get("dir") or sop_id) / "runs" / run
        if not re.match(r"^[A-Za-z0-9_.-]{1,64}$", run) or run.startswith(".") or not (folder / "video.mp4").exists():
            raise HTTPException(404, f"No recording {run}")
        return sf, folder

    @r.get("/sops/{sop_id}/runs/{run}/timeline")
    def get_timeline(sop_id: str, run: str):
        """A run's step timeline (saved and drafted), torque presses and video length, for the timeline page."""
        import json

        from sopvision.builder.timelines import read_rows, video_info
        from sopvision.sop.torque import read_torque_log

        sf, folder = run_folder(sop_id, run)
        fps, duration = video_info(folder / "video.mp4")
        info = folder / "steps.draft.json"
        steps = sf.data.get("steps") or []
        return {
            "run": run, "sop_file": str(sf.path), "fps": fps, "duration": duration, "preview": (folder / "preview.webm").exists(),
            "steps": read_rows(folder / "steps.csv") if (folder / "steps.csv").exists() else None,
            "draft": read_rows(folder / "steps.draft.csv") if (folder / "steps.draft.csv").exists() else None,
            "draft_info": json.loads(info.read_text(encoding="utf-8")) if info.exists() else None,
            "torque": [{"time_s": t.t, "ok": t.ok} for t in read_torque_log(folder / "torque.csv")]
            if (folder / "torque.csv").exists() else [],
            "step_ids": [s.get("id") for s in steps],
            "labels": {s.get("id"): s.get("label") or s.get("id") for s in steps},
        }

    @r.put("/sops/{sop_id}/runs/{run}/timeline")
    def save_timeline(sop_id: str, run: str, body: Timeline):
        from sopvision.builder.timelines import check_rows, video_info, write_rows

        sf, folder = run_folder(sop_id, run)
        _, duration = video_info(folder / "video.mp4")
        try:
            check_rows(body.rows, {s.get("id") for s in sf.data.get("steps") or []}, duration or None)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        write_rows(folder / "steps.csv", body.rows)
        return get_timeline(sop_id, run)

    @r.get("/sops/{sop_id}/runs/{run}/preview.webm")
    def preview(sop_id: str, run: str):
        from fastapi.responses import FileResponse

        _, folder = run_folder(sop_id, run)
        if not (folder / "preview.webm").exists():
            raise HTTPException(404, "No preview yet: draft the timelines first")
        return FileResponse(folder / "preview.webm", media_type="video/webm")

    @r.get("/sops/{sop_id}/checkpoints")
    def checkpoints(sop_id: str):
        """The SOP's trained checkpoints (with their reports), which one answers for it, and the shared ones."""
        from sopvision.builder import training
        from sopvision.decide import laya_registry

        drafts.current(settings, sop_id)
        own = training.versions(sop_id)
        mine = {v["checkpoint"] for v in own}
        shared = sorted(p.name for p in training.MODELS.glob("*") if (p / "model.safetensors").exists()
                        and p.name not in mine and not p.name.endswith(".eval"))
        return {"active": laya_registry.load(settings.laya_registry).get(sop_id), "versions": own, "shared": shared}

    @r.post("/sops/{sop_id}/activate")
    def activate(sop_id: str, body: Activate):
        """Make a checkpoint answer for this SOP's runs (or none: the shared one), and have Laya load it."""
        import httpx

        from sopvision.builder import training
        from sopvision.decide import laya_registry

        drafts.current(settings, sop_id)
        name = Path(body.checkpoint).name if body.checkpoint else None
        if name and not (training.MODELS / name / "model.safetensors").exists():
            raise HTTPException(404, f"No checkpoint {name}")
        laya_registry.set_checkpoint(settings.laya_registry, sop_id, name)
        warning = None
        try:
            httpx.post(f"{settings.laya_url}/admin/reload", timeout=120).raise_for_status()
        except httpx.HTTPError as e:
            warning = f"Saved, but the Laya service didn't reload ({e}); it picks the change up when it next starts."
        return checkpoints(sop_id) | {"warning": warning}

    @r.get("/models")
    def models():
        return det.existing_models()

    @r.put("/sops/{sop_id}/detector")
    async def upload_detector(sop_id: str, request: Request):
        """The request body is the .pt file itself (Content-Type: application/octet-stream)."""
        sf = draft_of(sop_id)
        det.MODELS.mkdir(exist_ok=True)
        size = 0
        with tempfile.NamedTemporaryFile(dir=det.MODELS, suffix=".pt", delete=False) as f:
            tmp = Path(f.name)
            try:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_UPLOAD:
                        raise HTTPException(413, "That file is over 1 GB; is it a detection model?")
                    f.write(chunk)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
        try:
            _, warnings = det.import_upload(sf, tmp)
        except DraftError as e:
            raise HTTPException(400, str(e)) from None
        finally:
            tmp.unlink(missing_ok=True)
        return view(sop_id, warnings)

    @r.post("/sops/{sop_id}/detector")
    def use_existing_detector(sop_id: str, body: ExistingModel):
        try:
            _, warnings = det.import_existing(draft_of(sop_id), body.path)
        except DraftError as e:
            raise HTTPException(400, str(e)) from None
        return view(sop_id, warnings)

    return r
