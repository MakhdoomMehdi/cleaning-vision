"""Environment-based application settings.

All secrets come from the environment / .env file and are never hardcoded.
"""

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Cleaning Vision AI"
    app_env: str = "local"
    log_level: str = "INFO"

    # NoDecode: parse the comma-separated string ourselves instead of as JSON.
    cors_origins: Annotated[list[str], NoDecode] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    database_url: str = "sqlite:///./data/cleaning_vision.db"
    upload_dir: Path = Path("./uploads")

    max_upload_mb: int = Field(default=10, ge=1)
    min_image_dimension: int = Field(default=320, ge=1)
    max_image_dimension: int = Field(default=8000, ge=1)
    max_images_per_request: int = Field(default=6, ge=1)
    # Before photos sent with one inspection. More than this is rejected, never silently dropped.
    max_reference_images_per_inspection: int = Field(default=12, ge=1)

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5.4"
    openai_timeout_seconds: float = Field(default=90.0, gt=0)
    openai_max_retries: int = Field(default=2, ge=0)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def upload_root(self) -> Path:
        """Absolute upload directory (relative values resolve against backend/, not the CWD)."""
        path = self.upload_dir
        return path if path.is_absolute() else (BACKEND_DIR / path).resolve()

    @property
    def resolved_database_url(self) -> str:
        """Make relative SQLite file paths absolute (against backend/) so CWD doesn't matter."""
        prefix = "sqlite:///"
        url = self.database_url
        if not url.startswith(prefix):
            return url
        raw = url[len(prefix) :]
        if raw in ("", ":memory:") or raw.startswith("file:"):
            return url
        path = Path(raw)
        if not path.is_absolute():
            path = (BACKEND_DIR / path).resolve()
        return f"{prefix}{path.as_posix()}"

    @property
    def openai_configured(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.get_secret_value().strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
