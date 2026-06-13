from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql://postgres:postgres@localhost:5432/hs_classification"
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    llm_model: str = "gpt-4o"
    pipeline_version: str = "0.1.0"
    confidence_threshold: float = 0.80
    retrieval_top_k: int = 10
    # Path to the fitted M6 calibrator pickle.  Created by:
    #   python -m app.pipeline.calibration
    calibrator_path: Path = Path("data/calibrator.pkl")


settings = Settings()
