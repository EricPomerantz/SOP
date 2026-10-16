"""Environment settings, read from the process environment and `.env`."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from sopvision.config import ConfigError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    typesafe_api_key: SecretStr = SecretStr("")
    # Optional Vercel AI Gateway routing; the names match what typesafe-sdk itself reads.
    typesafe_base_url: str | None = None
    typesafe_model: str | None = Field(default=None, validation_alias="TYPESAFE_DEFAULT_MODEL")
    # Who answers SOP decisions: jev (default), decider or laya (local models), or compare (Jev decides, the
    # compare_shadow model gets the same requests and both answers are logged).
    decision_mode: Literal["jev", "decider", "laya", "compare"] = "jev"
    compare_shadow: Literal["decider", "laya"] = "decider"
    decider_url: str = "http://decider:8001"
    decider_timeout: float = 5.0
    # Laya (convaiinnovations/laya), a 421M decision model behind the same wire format; evaluation only so far.
    laya_url: str = "http://laya:8000"
    laya_model: str = "english"  # english | multilingual | typed-decisions: SOPs without a checkpoint of their own
    laya_registry: Path = Path("data/training/models/registry.json")  # per-SOP checkpoints (decide/laya_registry.py)
    laya_max_len: int | None = None  # token budgets; None = the checkpoint's own (english 512 / 192)
    laya_head_max_len: int | None = None
    sop_data_root: Path = Path("/data/server_fan")  # the server_fan (clips) dataset
    datasets_dir: Path = Path("/datasets")  # timeline-format datasets (new SOPs, recordings) live in <datasets_dir>/<dir>
    sopvision_rtsp_urls: Annotated[list[str], NoDecode] = []
    rtsp_low_latency: bool = True  # open RTSP streams with FFmpeg's low-buffering options
    # How far a live camera's frames trail reality, for timing torque results. A camera measured with the latency
    # check (dashboard /#/latency) uses its measured value instead (runs/camera_latency.json).
    camera_delay_ms: float = 0.0
    sops_dir: Path = Path("sops")
    runs_dir: Path = Path("runs")
    scenarios_dir: Path = Path("data/scenarios")
    jobs_dir: Path = Path("data/jobs")  # background jobs, run by the worker service (sopvision.jobs)

    @field_validator("sopvision_rtsp_urls", mode="before")
    @classmethod
    def _split_urls(cls, value: object) -> object:
        if isinstance(value, str):
            return [u.strip() for u in value.split(",") if u.strip()]
        return value

    def check_data_root(self) -> None:
        """Fail with the variable to fix when the dataset folder is missing or isn't a recordings dataset."""
        # In the container SOP_DATA_ROOT is fixed by docker-compose; the user-facing knob is SOP_DATASET_DIR.
        fix = "Set SOP_DATASET_DIR in .env (docker-compose mounts it at /data)."
        root = self.sop_data_root
        if not root.is_dir():
            raise ConfigError(f"SOP_DATA_ROOT points to {root}, which doesn't exist. {fix}")
        if not ((root / "raw").is_dir() and (root / "test").is_dir()):
            raise ConfigError(f"SOP_DATA_ROOT points to {root}, which isn't a recordings dataset (needs raw/ and test/ "
                              f"folders). {fix}")
