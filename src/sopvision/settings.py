"""Environment settings, read from the process environment and `.env`."""

from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from sopvision.config import ConfigError, SopConfig


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    typesafe_api_key: SecretStr = SecretStr("")
    # Optional Vercel AI Gateway routing; the names match what typesafe-sdk itself reads.
    typesafe_base_url: str | None = None
    typesafe_model: str | None = Field(default=None, validation_alias="TYPESAFE_DEFAULT_MODEL")
    sop_data_root: Path = Path("sop-sample-training-data/server_fan")
    sopvision_in_container: bool = False
    sopvision_rtsp_urls: Annotated[list[str], NoDecode] = []
    sops_dir: Path = Path("sops")
    runs_dir: Path = Path("runs")
    scenarios_dir: Path = Path("data/scenarios")
    # The decision model's name as people see it (dashboard, alerts); "Kev" when docker-compose.kev.yml runs Kev.
    sopvision_decider_name: str = "Jev"
    # A decision model's own thresholds (e.g. Kev's, set in docker-compose.kev.yml); unset = the SOP file's values.
    sopvision_step_confidence: float | None = Field(default=None, ge=0, le=1)
    sopvision_done_confidence: float | None = Field(default=None, ge=0, le=1)
    sopvision_anomaly_confidence: float | None = Field(default=None, ge=0, le=1)

    @field_validator("sopvision_rtsp_urls", mode="before")
    @classmethod
    def _split_urls(cls, value: object) -> object:
        if isinstance(value, str):
            return [u.strip() for u in value.split(",") if u.strip()]
        return value

    def apply_decision_overrides(self, sop: SopConfig) -> SopConfig:
        updates = {"step_confidence": self.sopvision_step_confidence, "done_confidence": self.sopvision_done_confidence,
                   "anomaly_confidence": self.sopvision_anomaly_confidence}
        updates = {k: v for k, v in updates.items() if v is not None}
        if not updates:
            return sop
        return sop.model_copy(update={"decisions": sop.decisions.model_copy(update=updates)})

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
