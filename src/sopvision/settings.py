"""Environment settings, read from the process environment and `.env`."""

from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from sopvision.config import ConfigError


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

    @field_validator("sopvision_rtsp_urls", mode="before")
    @classmethod
    def _split_urls(cls, value: object) -> object:
        if isinstance(value, str):
            return [u.strip() for u in value.split(",") if u.strip()]
        return value

    def check_data_root(self) -> None:
        if not self.sop_data_root.is_dir():
            raise ConfigError(f"SOP_DATA_ROOT points to {self.sop_data_root}, which doesn't exist. Set it in .env.")
