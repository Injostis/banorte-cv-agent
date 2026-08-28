from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str
    agent_bearer_token: str
    claude_model: str = "claude-sonnet-5"
    profile_path: Path = Path(__file__).resolve().parent.parent / "data" / "profile.yaml"
    ps_trophies_path: Path = Path(__file__).resolve().parent.parent / "data" / "ps_trophies.json"
    log_level: str = "INFO"

    # Sin llaves, Langfuse queda deshabilitado.
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"


settings = Settings()
