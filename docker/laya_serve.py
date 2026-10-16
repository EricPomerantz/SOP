"""laya-serve, answering with fine-tuned checkpoints: a shared one in place of the shipped English model, and one per SOP.

laya-serve only serves published checkpoints, and a request cannot name a local one.

- LAYA_CHECKPOINT: a fine-tune's directory (model.safetensors + rl_agent_config.json, as laya-train writes it),
  attached under the name "english", where the router sends English states: requests that name no model reach it.
- The registry (LAYA_REGISTRY, default /data/models/registry.json, written when a checkpoint is activated in the
  dashboard) maps SOP ids to checkpoint folders under /data/models:  {"sops": {"lug_bolts_torque": "laya-sop-v2-e12"}}
  Each checkpoint is registered once, as `ckpt-<folder>` (sopvision.decide.laya_registry.model_name), and the app
  names it in the request's `model` field for that SOP's runs. POST /admin/reload re-reads the registry.

laya 0.3.28's request handler maps `model` through the published names only and auto-routes anything else, so a
registered name would silently get the default checkpoint: `_resolve_model` is wrapped to accept registered names.
Without LAYA_CHECKPOINT or a registry this is plain laya-serve.
"""

import json
import os
from pathlib import Path

import uvicorn

MODELS = Path("/data/models")


def _registry() -> dict[str, str]:
    """model name -> checkpoint directory, for every checkpoint the registry names."""
    path = Path(os.environ.get("LAYA_REGISTRY", str(MODELS / "registry.json")))
    try:
        sops = json.loads(path.read_text(encoding="utf-8")).get("sops", {})
    except FileNotFoundError:
        return {}
    out = {}
    for folder in sorted(set(sops.values())):
        directory = MODELS / Path(folder).name
        if (directory / "model.safetensors").exists():
            out[f"ckpt-{Path(folder).name}".lower()] = str(directory)
        else:
            print(f"[laya] registry names {folder}, which has no model.safetensors under {MODELS}; skipped", flush=True)
    return out


def main() -> None:
    checkpoint = os.environ.get("LAYA_CHECKPOINT", "").strip()
    if checkpoint:
        os.environ["LAYA_PRELOAD"] = "0"  # don't build the shipped English checkpoint just to replace it
    import laya.serve as serve
    from laya.serve import build_router, create_app

    router = build_router()
    agent = None
    if checkpoint:
        import laya

        agent = laya.load(checkpoint, device=os.environ.get("LAYA_DEVICE") or None)
        router.attach("english", agent)
        print(f"[laya] serving fine-tuned checkpoint {checkpoint} as 'english'", flush=True)

    sources: dict[str, str] = {}
    candidates: set[str] = set()  # registered for evaluation (POST /admin/register), kept across reloads

    def reload() -> dict[str, str]:
        wanted = _registry()
        for name, directory in wanted.items():
            if sources.get(name) != directory:
                if agent is not None and Path(directory).resolve() == Path(checkpoint).resolve():
                    router.attach(name, agent)  # the shared checkpoint, already loaded: not a second copy
                else:
                    router.register(name, directory, description="per-SOP fine-tune (registry.json)")
                sources[name] = directory
                print(f"[laya] registered {directory} as '{name}'", flush=True)
        for name in [n for n in sources if n not in wanted and n not in candidates]:
            del sources[name]  # no SOP uses it now: no longer accepted by name (the router evicts it when full)
        if wanted:
            router.max_loaded = max(router.max_loaded, len(wanted) + 1)
            router.preload([n for n in wanted if n not in getattr(router, "_agents", {})])  # load now: a first request shouldn't wait for a checkpoint build
        return dict(sources)

    original = serve._resolve_model

    def resolve_model(model):
        key = str(model or "").strip().lower()
        return key if key in sources else original(model)

    serve._resolve_model = resolve_model
    reload()

    app = create_app(router)

    @app.post("/admin/reload")
    def admin_reload():
        """Re-read the registry after a checkpoint was activated."""
        return {"models": reload()}

    @app.post("/admin/register")
    def admin_register(body: dict):
        """Serve one checkpoint folder under /data/models by name, without the registry: a candidate under
        evaluation (the builder's training job benchmarks it before anyone activates it)."""
        from fastapi import HTTPException

        folder = Path(str(body.get("folder", ""))).name
        directory = MODELS / folder
        if not folder or not (directory / "model.safetensors").exists():
            raise HTTPException(404, f"No checkpoint {folder} under {MODELS}")
        name = f"ckpt-{folder}".lower()
        router.register(name, str(directory), description="candidate under evaluation")
        sources[name] = str(directory)
        candidates.add(name)
        router.max_loaded = max(router.max_loaded, len(sources) + 1)
        return {"model": name}

    @app.get("/admin/models")
    def admin_models():
        return {"models": dict(sources), "loaded": list(getattr(router, "_order", []))}

    uvicorn.run(app, host=os.environ.get("LAYA_HOST", "0.0.0.0"), port=int(os.environ.get("LAYA_PORT", "8000")),
                log_level=os.environ.get("LAYA_LOG_LEVEL", "info"))


if __name__ == "__main__":
    main()
