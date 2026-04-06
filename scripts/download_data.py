"""Download Polymarket dataset from HuggingFace and load into DuckDB."""

from src.config import settings


def main() -> None:
    print(f"Downloading dataset: {settings.hf_dataset}")
    print(f"Target DuckDB: {settings.duckdb_path}")
    raise NotImplementedError("download_data not yet implemented")


if __name__ == "__main__":
    main()
