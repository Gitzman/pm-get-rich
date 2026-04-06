"""Download Polymarket parquet files from HuggingFace."""

from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

from src.config import settings

RAW_DIR = settings.data_dir / "raw"
EXPECTED_FILES = ["quant.parquet", "markets.parquet"]


def list_parquet_files(repo_id: str | None = None) -> list[str]:
    """List all parquet files in the HuggingFace dataset repo."""
    repo_id = repo_id or settings.hf_dataset
    files = list_repo_files(repo_id, repo_type="dataset")
    return [f for f in files if f.endswith(".parquet")]


def download_dataset(
    repo_id: str | None = None,
    output_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Download parquet files from HuggingFace to data/raw/.

    Args:
        repo_id: HuggingFace dataset ID. Defaults to config setting.
        output_dir: Where to save files. Defaults to data/raw/.
        force: Re-download even if files exist locally.

    Returns:
        Mapping of filename to local path for each downloaded file.
    """
    repo_id = repo_id or settings.hf_dataset
    output_dir = output_dir or RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = list_parquet_files(repo_id)
    if not parquet_files:
        raise RuntimeError(f"No parquet files found in {repo_id}")

    downloaded: dict[str, Path] = {}
    for filename in parquet_files:
        local_path = output_dir / Path(filename).name
        if local_path.exists() and not force:
            print(f"  ✓ {filename} already exists, skipping")
            downloaded[filename] = local_path
            continue

        print(f"  ↓ Downloading {filename}...")
        cached = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(output_dir),
        )
        # hf_hub_download with local_dir puts the file at output_dir/filename
        local_path = Path(cached)
        print(f"  ✓ Saved to {local_path}")
        downloaded[filename] = local_path

    return downloaded
