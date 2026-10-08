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
    decider_url: str = "http://localhost:8001"
    decider_timeout: float = 5.0
    # Laya (convaiinnovations/laya), a 421M decision model behind the same wire format; evaluation only so far.
    laya_url: str = "http://localhost:8002"
    laya_model: str = "english"  # english | multilingual | typed-decisions
    laya_max_len: int | None = None  # token budgets; None = the checkpoint's own (english 512 / 192)
    laya_head_max_len: int | None = None
    sop_data_root: Path = Path("sop-sample-training-data/server_fan")
    sopvision_in_container: bool = False
    sopvision_rtsp_urls: Annotated[list[str], NoDecode] = []
    sops_dir: Path = Path("sops")
    runs_dir: Path = Path("runs")
    scenarios_dir: Path = Path("data/scenarios")

    @field_validator("sopvision_rtsp_urls", mode="before")
    @classmethod
    def _split_urls(cls, value: object) -> object:
        if isinstance(value, str):
            return [u.strip() for u in value.split(",") if u.strip()]
        return value

    def check_data_root(self) -> None:
        """Fail with the variable to fix when the dataset folder is missing or isn't a recordings dataset."""
        # In the container SOP_DATA_ROOT is fixed by docker-compose; the user-facing knob is SOP_DATASET_DIR.
        fix = ("Set SOP_DATASET_DIR in .env (docker-compose mounts it at /data)." if self.sopvision_in_container
               else None)
        root = self.sop_data_root
        if not root.is_dir():
            raise ConfigError(f"SOP_DATA_ROOT points to {root}, which doesn't exist. {fix or 'Set it in .env.'}")
        if not ((root / "raw").is_dir() and (root / "test").is_dir()):
            raise ConfigError(f"SOP_DATA_ROOT points to {root}, which isn't a recordings dataset (needs raw/ and test/ "
                              f"folders). {fix or 'Set SOP_DATA_ROOT in .env.'}")
