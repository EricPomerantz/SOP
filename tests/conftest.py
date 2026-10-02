from pathlib import Path

import pytest

from sopvision.config import SopConfig, load_sop

SERVER_FAN = Path("sops/server_fan.yaml")


@pytest.fixture
def sop() -> SopConfig:
    return load_sop(SERVER_FAN)
