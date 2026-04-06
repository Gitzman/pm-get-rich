from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # Data paths
    data_dir: Path = Path("data")
    duckdb_path: Path = Path("data/pmgetrich.duckdb")

    # Model settings
    chronos_model: str = "amazon/chronos-2"
    device: str = "cuda"

    # Divergence thresholds
    divergence_threshold: float = 0.05
    divergence_lookback_days: int = 30

    # Market selection criteria
    min_volume: int = 1000
    min_liquidity: int = 500
    max_markets: int = 100

    # HuggingFace
    hf_dataset: str = "SII-WANGZJ/Polymarket_data"


settings = Settings()
