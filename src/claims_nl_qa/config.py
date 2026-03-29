from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_data_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "docs" / "synthetic_claims.csv"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_MODEL")
    data_path: Path = Field(default_factory=_default_data_path, validation_alias="DATA_PATH")


@lru_cache
def get_settings() -> Settings:
    return Settings()
