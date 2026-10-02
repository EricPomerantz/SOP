"""FastAPI app: SOPs and sources, run control, MJPEG video, WebSocket status, built dashboard (spec §9.1)."""

import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sopvision.server.runs import NotFoundError, NotPausableError, RunConflictError, RunManager
from sopvision.settings import Settings
from sopvision.sources.base import SourceUnavailableError
from sopvision.sources.registry import list_sources
from sopvision.storage.report import report_csv

WEB_DIST = Path("web/dist")


class StartRun(BaseModel):
    sop: str
    source_id: str


class Control(BaseModel):
    action: Literal["pause", "resume"] | None = None
    speed: float | None = None


def create_app(settings: Settings, manager: RunManager) -> FastAPI:
    app = FastAPI(title="SOP Vision")

    @app.get("/sops")
    def sops():
        return [{"id": s.sop, "version": s.version, "description": s.description,
                 "steps": [{"id": st.id, "label": s.step_label(st.id), "text": st.text} for st in s.steps]}
                for s in manager.sops.values()]

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
