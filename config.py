"""Application configuration loaded from environment / .env file."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Database ---
    DB_USER: str
    DB_PASSWORD: SecretStr
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str

    # --- Telegram ---
    BOT_TOKEN: SecretStr

    # --- OpenAI ---
    OPENAI_API_KEY: SecretStr
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_TIMEOUT: float = Field(default=30.0, gt=0)

    # --- RAG ---
    # Maximum allowed cosine distance (0 = identical, 2 = opposite).
    # If the nearest chunk is farther than this value, the LLM is NOT called.
    RAG_THRESHOLD: float = Field(default=0.5, ge=0.0, le=2.0)
    RAG_TOP_K: int = Field(default=3, ge=1)

    LOG_LEVEL: str = "INFO"

    @property
    def database_url(self) -> str:
        from sqlalchemy.engine import URL

        return URL.create(
            drivername="postgresql+asyncpg",
            username=self.DB_USER,
            password=self.DB_PASSWORD.get_secret_value(),
            host=self.DB_HOST,
            port=self.DB_PORT,
            database=self.DB_NAME,
        ).render_as_string(hide_password=False)


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
