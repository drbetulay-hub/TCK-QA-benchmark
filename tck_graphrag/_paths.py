"""Package paths — load .env only from this repository root."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PACKAGE_ROOT / ".env"


def load_project_dotenv() -> None:
    from dotenv import load_dotenv

    if ENV_FILE.is_file():
        load_dotenv(ENV_FILE, override=True)
