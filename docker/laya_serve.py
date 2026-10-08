"""laya-serve, optionally answering with a fine-tuned checkpoint instead of the shipped English one.

laya-serve only serves published checkpoints, and a request cannot name a local one. With LAYA_CHECKPOINT set to a
fine-tune's directory (model.safetensors + rl_agent_config.json, as laya-train writes it), that checkpoint is attached
under the name "english", which is where the router sends English states, so every client request reaches it unchanged.
Without LAYA_CHECKPOINT this is plain laya-serve.
"""

import os

import uvicorn


def main() -> None:
    checkpoint = os.environ.get("LAYA_CHECKPOINT", "").strip()
    if checkpoint:
        os.environ["LAYA_PRELOAD"] = "0"  # don't build the shipped English checkpoint just to replace it
    from laya.serve import build_router, create_app

    router = build_router()
    if checkpoint:
        import laya

        agent = laya.load(checkpoint, device=os.environ.get("LAYA_DEVICE") or None)
        router.attach("english", agent)
        print(f"[laya] serving fine-tuned checkpoint {checkpoint} as 'english'", flush=True)
    uvicorn.run(create_app(router), host=os.environ.get("LAYA_HOST", "0.0.0.0"), port=int(os.environ.get("LAYA_PORT", "8000")),
                log_level=os.environ.get("LAYA_LOG_LEVEL", "info"))


if __name__ == "__main__":
    main()
