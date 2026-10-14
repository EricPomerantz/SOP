"""FastAPI app: SOPs and sources, run control, MJPEG video, WebSocket status, built dashboard (spec §9.1)."""

import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sopvision.config import ConfigError
from sopvision.server.runs import NoTorqueToolError, NotFoundError, NotPausableError, RunConflictError, RunManager
from sopvision.settings import Settings
from sopvision.sources.base import SourceUnavailableError
from sopvision.sources.registry import list_sources
from sopvision.storage.report import report_csv

WEB_DIST = Path("web/dist")


class StartRun(BaseModel):
    sop: str
    source_id: str


class LatencyMeasure(BaseModel):
    source_id: str
    seconds: float = 10.0
    save: bool = False  # store the result as this camera's delay


class StartRecording(BaseModel):
    sop_file: str
    source_id: str
    split: Literal["train", "val", "test"] = "train"
    name: str | None = None  # the run folder name; default r<date>-<time>
    fps: float = 25.0


class Torque(BaseModel):
    ok: bool  # the tightening reached the target torque
    torque_nm: float | None = None


class Control(BaseModel):
    action: Literal["pause", "resume"] | None = None
    speed: float | None = None


def create_app(settings: Settings, manager: RunManager) -> FastAPI:
    app = FastAPI(title="SOP Vision")

    @app.get("/sops")
    def sops():
        return [{"id": s.sop, "name": s.display_name, "version": s.version, "description": s.description,
                 "torque": bool(s.layout.tools),  # the dashboard shows torque buttons for these
                 "steps": [{"id": st.id, "label": s.step_label(st.id), "text": st.text} for st in s.steps]}
                for s in manager.sops.values()]

    @app.get("/config")
    def config():
        """What the dashboard needs before any run: which model makes the decisions."""
        from sopvision.decide.providers import authoritative_provider

        return {"decision_model": authoritative_provider(settings), "decision_mode": settings.decision_mode}

    @app.get("/sources")
    def sources():
        return [s.model_dump(mode="json") for s in list_sources(settings)]

    @app.post("/runs", status_code=201)
    async def start_run(body: StartRun):
        try:
            return (await manager.start(body.sop, body.source_id)).model_dump()
        except RunConflictError as e:
            raise HTTPException(409, str(e)) from None
        except SourceUnavailableError as e:
            raise HTTPException(400, str(e)) from None
        except NotFoundError as e:
            raise HTTPException(404, str(e)) from None

    @app.post("/runs/{run_id}/stop")
    async def stop_run(run_id: str):
        try:
            await manager.stop(run_id)
        except NotFoundError as e:
            raise HTTPException(404, str(e)) from None
        return {"stopped": run_id}

    @app.post("/runs/{run_id}/torque")
    def torque(run_id: str, body: Torque):
        """One tightening result from the torque wrench system (or a bridge in front of it), sent as it happens."""
        try:
            t = manager.torque(run_id, body.ok, body.torque_nm)
        except NotFoundError as e:
            raise HTTPException(404, str(e)) from None
        except NoTorqueToolError as e:
            raise HTTPException(409, str(e)) from None
        return {"t": t}

    @app.get("/latency/clock")
    def latency_clock():
        """The app's clock, so the latency page can draw its QR code on the same timeline the app measures with."""
        import time

        return {"now_ms": time.time() * 1000}

    @app.get("/latency")
    def latency_results():
        from sopvision.sources.latency import load_delays

        return load_delays(settings)

    @app.post("/latency/measure")
    async def latency_measure(body: LatencyMeasure):
        """Read the camera for a few seconds while it films the latency page's QR clock; report the delay."""
        from sopvision.sources.latency import measure, save_delay
        from sopvision.sources.registry import open_source

        if manager.current() is not None or manager.recording is not None:
            raise HTTPException(409, "Stop the current run or recording first: the measurement needs the camera to itself.")
        spec = next((s for s in list_sources(settings) if s.id == body.source_id), None)
        if spec is None:
            raise HTTPException(404, f"Unknown source {body.source_id}")
        if spec.kind == "file":
            raise HTTPException(400, "Recorded videos have no camera delay; pick a live camera.")
        seconds = min(max(body.seconds, 2.0), 60.0)
        try:
            source = open_source(spec, settings)
        except SourceUnavailableError as e:
            raise HTTPException(400, str(e)) from None
        try:
            result = await asyncio.wait_for(asyncio.to_thread(measure, source.frames(), seconds), seconds + 15)
        except TimeoutError:
            raise HTTPException(504, f"No frames from {spec.label} within {seconds + 15:.0f} s") from None
        finally:
            source.close()
        if not result.get("decoded"):
            raise HTTPException(422, f"Got {result['frames']} frames but no readable QR clock: point the camera at the "
                                     "latency page so the code fills a good part of the picture.")
        if body.save:
            save_delay(settings, spec.id, result)
        return result | {"source_id": spec.id, "saved": body.save}

    @app.get("/recording/sops")
    def recording_sops():
        return manager.recording_sops()

    @app.get("/recording/sop-errors")
    def recording_sop_errors():
        return manager.recording_sop_errors()

    @app.get("/recording")
    def recording_status():
        return manager.recording.status() if manager.recording is not None else None

    @app.post("/recording/start")
    def recording_start(body: StartRecording):
        try:
            return manager.start_recording(body.sop_file, body.source_id, body.split, body.name, body.fps)
        except RunConflictError as e:
            raise HTTPException(409, str(e)) from None
        except NotFoundError as e:
            raise HTTPException(404, str(e)) from None
        except (ConfigError, SourceUnavailableError) as e:
            raise HTTPException(400, str(e)) from None

    @app.post("/recording/torque")
    def recording_torque(body: Torque):
        """A "torque reached" (or not reached) press while recording; stored on the video's timeline."""
        try:
            return {"time_s": manager.recording_torque(body.ok, body.torque_nm)}
        except NotFoundError as e:
            raise HTTPException(404, str(e)) from None
        except NotPausableError as e:
            raise HTTPException(409, str(e)) from None

    @app.post("/recording/discard")
    def recording_discard():
        """Stop recording and delete the run (nothing is saved, splits.yaml is unchanged)."""
        try:
            return manager.discard_recording()
        except NotFoundError as e:
            raise HTTPException(404, str(e)) from None

    @app.post("/recording/stop")
    def recording_stop():
        try:
            return manager.stop_recording()
        except NotFoundError as e:
            raise HTTPException(404, str(e)) from None
        except ConfigError as e:
            raise HTTPException(400, str(e)) from None

    @app.post("/runs/{run_id}/control")
    def control(run_id: str, body: Control):
        try:
            manager.control(run_id, body.action, body.speed)
        except NotFoundError as e:
            raise HTTPException(404, str(e)) from None
        except NotPausableError as e:
            raise HTTPException(400, str(e)) from None
        return {"ok": True}

    @app.get("/runs/current")
    def current():
        snap = manager.current()
        return Response(status_code=204) if snap is None else snap

    @app.get("/runs/{run_id}/report")
    def report(run_id: str, format: Literal["json", "csv"] = "json"):
        try:
            r = manager.report(run_id)
        except NotFoundError as e:
            raise HTTPException(404, str(e)) from None
        if format == "csv":
            return Response(report_csv(r), media_type="text/csv",
                            headers={"Content-Disposition": f'attachment; filename="{run_id}.csv"'})
        return asdict(r)

    @app.get("/video.mjpg")
    async def video():
        async def frames():
            seq = 0
            while True:  # every new annotated frame, as it arrives (the pipeline sets the pace, ~30 fps)
                seq, jpeg = await manager.frames.next(seq)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode()
                       + b"\r\n\r\n" + jpeg + b"\r\n")

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        queue: asyncio.Queue = asyncio.Queue()
        manager.subscribers.add(queue)
        try:
            while True:
                await socket.send_json(await queue.get())
        except WebSocketDisconnect:
            pass
        finally:
            manager.subscribers.discard(queue)

    if WEB_DIST.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
    return app
