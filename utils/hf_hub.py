"""
Hugging Face Hub integration utilities for SPCAC.

Supports:
- Downloading model checkpoints from HF Model Hub (private repos supported)
- Downloading dataset files from HF Dataset Hub (private repos supported)
- Uploading checkpoints/datasets to HF Hub
- Local-first strategy: local files always take priority over remote downloads

Configuration via environment variables:
    SPCAC_HF_CHECKPOINT_REPO : HF repo ID for model checkpoints (e.g. "my-org/SPCAC-checkpoints")
    SPCAC_HF_DATASET_REPO    : HF repo ID for datasets (e.g. "my-org/SPCAC-datasets")
    HF_TOKEN                 : HF access token (for private repos; also read by huggingface_hub)
    HF_HUB_CACHE             : Override default HF cache directory
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — read from environment variables
# ---------------------------------------------------------------------------

CHECKPOINT_REPO_ID = os.environ.get("SPCAC_HF_CHECKPOINT_REPO", "")
DATASET_REPO_ID = os.environ.get("SPCAC_HF_DATASET_REPO", "")

# HF token — `huggingface_hub` natively reads HF_TOKEN, but we expose it for
# explicit upload / download calls where needed.
HF_TOKEN = os.environ.get("HF_TOKEN", None)


def _is_configured(repo_id: str) -> bool:
    """Return True if a HF repo ID has been configured."""
    return bool(repo_id and repo_id.strip())


def _ensure_hf_available():
    """Lazy-import huggingface_hub and verify it is installed."""
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        raise ImportError(
            "huggingface_hub is not installed. "
            "Install it with: pip install huggingface_hub"
        )


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def download_checkpoint(
    repo_filename: str,
    repo_id: Optional[str] = None,
    local_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> str:
    """
    Download a single checkpoint file from HF Hub.

    Parameters
    ----------
    repo_filename : str
        Relative path inside the HF repo, e.g.
        ``coco3d/elpcac/quality_3/eb_las.pth``.
    repo_id : str, optional
        HF repo ID.  Defaults to ``SPCAC_HF_CHECKPOINT_REPO`` env var.
    local_dir : str, optional
        Directory to save the downloaded file.  Defaults to a mirror of the
        repo structure under ``./checkpoints/``.
    token : str, optional
        HF access token.  Defaults to ``HF_TOKEN`` env var.

    Returns
    -------
    str
        Absolute path to the downloaded (or cached) file.
    """
    _ensure_hf_available()
    from huggingface_hub import hf_hub_download

    repo = repo_id or CHECKPOINT_REPO_ID
    if not _is_configured(repo):
        raise RuntimeError(
            "HF checkpoint repo not configured. "
            "Set SPCAC_HF_CHECKPOINT_REPO environment variable."
        )

    tok = token or HF_TOKEN

    if local_dir is None:
        local_dir = os.path.join("checkpoints")

    logger.info("Downloading checkpoint from HF: %s / %s", repo, repo_filename)

    return hf_hub_download(
        repo_id=repo,
        filename=repo_filename,
        repo_type="model",
        local_dir=local_dir,
        token=tok,
    )


def download_dataset_file(
    repo_filename: str,
    repo_id: Optional[str] = None,
    local_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> str:
    """
    Download a single dataset file (hdf5 / ply archive) from HF Hub.

    Parameters
    ----------
    repo_filename : str
        Relative path inside the HF repo, e.g. ``coco3d/coco3d.hdf5``.
    repo_id : str, optional
        HF repo ID.  Defaults to ``SPCAC_HF_DATASET_REPO`` env var.
    local_dir : str, optional
        Directory to save the downloaded file.  Defaults to ``./datasets/``.
    token : str, optional
        HF access token.  Defaults to ``HF_TOKEN`` env var.

    Returns
    -------
    str
        Absolute path to the downloaded (or cached) file.
    """
    _ensure_hf_available()
    from huggingface_hub import hf_hub_download

    repo = repo_id or DATASET_REPO_ID
    if not _is_configured(repo):
        raise RuntimeError(
            "HF dataset repo not configured. "
            "Set SPCAC_HF_DATASET_REPO environment variable."
        )

    tok = token or HF_TOKEN

    if local_dir is None:
        local_dir = os.path.join("datasets")

    logger.info("Downloading dataset from HF: %s / %s", repo, repo_filename)

    return hf_hub_download(
        repo_id=repo,
        filename=repo_filename,
        repo_type="dataset",
        local_dir=local_dir,
        token=tok,
    )


def snapshot_download_dataset(
    repo_id: Optional[str] = None,
    local_dir: Optional[str] = None,
    allow_patterns: Optional[list] = None,
    token: Optional[str] = None,
) -> str:
    """
    Download an entire dataset snapshot from HF Hub.

    Parameters
    ----------
    repo_id : str, optional
        HF repo ID.  Defaults to ``SPCAC_HF_DATASET_REPO`` env var.
    local_dir : str, optional
        Local directory to download into.  Defaults to ``./datasets/``.
    allow_patterns : list, optional
        Glob patterns to filter which files to download.
    token : str, optional
        HF access token.  Defaults to ``HF_TOKEN`` env var.

    Returns
    -------
    str
        Path to the local directory containing the downloaded files.
    """
    _ensure_hf_available()
    from huggingface_hub import snapshot_download

    repo = repo_id or DATASET_REPO_ID
    if not _is_configured(repo):
        raise RuntimeError(
            "HF dataset repo not configured. "
            "Set SPCAC_HF_DATASET_REPO environment variable."
        )

    tok = token or HF_TOKEN

    if local_dir is None:
        local_dir = os.path.join("datasets")

    logger.info("Downloading dataset snapshot from HF: %s", repo)

    return snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=allow_patterns,
        token=tok,
    )


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------


def upload_checkpoint(
    local_path: str,
    path_in_repo: str,
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    commit_message: str = "Upload checkpoint",
):
    """
    Upload a single checkpoint file to HF Hub.

    Parameters
    ----------
    local_path : str
        Path to the local checkpoint file.
    path_in_repo : str
        Destination path inside the HF repo.
    repo_id : str, optional
        HF repo ID.  Defaults to ``SPCAC_HF_CHECKPOINT_REPO`` env var.
    token : str, optional
        HF access token with write permission.
    commit_message : str
        Commit message for the upload.
    """
    _ensure_hf_available()
    from huggingface_hub import HfApi

    repo = repo_id or CHECKPOINT_REPO_ID
    if not _is_configured(repo):
        raise RuntimeError(
            "HF checkpoint repo not configured. "
            "Set SPCAC_HF_CHECKPOINT_REPO environment variable."
        )

    tok = token or HF_TOKEN

    api = HfApi(token=tok)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo,
        repo_type="model",
        commit_message=commit_message,
    )
    logger.info("Uploaded checkpoint to HF: %s / %s", repo, path_in_repo)


def upload_dataset_file(
    local_path: str,
    path_in_repo: str,
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    commit_message: str = "Upload dataset file",
):
    """
    Upload a single dataset file to HF Hub.

    Parameters
    ----------
    local_path : str
        Path to the local dataset file (hdf5 / ply).
    path_in_repo : str
        Destination path inside the HF repo.
    repo_id : str, optional
        HF repo ID.  Defaults to ``SPCAC_HF_DATASET_REPO`` env var.
    token : str, optional
        HF access token with write permission.
    commit_message : str
        Commit message for the upload.
    """
    _ensure_hf_available()
    from huggingface_hub import HfApi

    repo = repo_id or DATASET_REPO_ID
    if not _is_configured(repo):
        raise RuntimeError(
            "HF dataset repo not configured. "
            "Set SPCAC_HF_DATASET_REPO environment variable."
        )

    tok = token or HF_TOKEN

    api = HfApi(token=tok)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo,
        repo_type="dataset",
        commit_message=commit_message,
    )
    logger.info("Uploaded dataset file to HF: %s / %s", repo, path_in_repo)


def upload_folder(
    local_folder: str,
    path_in_repo: str = "",
    repo_id: Optional[str] = None,
    repo_type: str = "model",
    token: Optional[str] = None,
    commit_message: str = "Upload folder",
):
    """
    Upload an entire folder to HF Hub.

    Parameters
    ----------
    local_folder : str
        Local folder path to upload.
    path_in_repo : str
        Destination subdirectory inside the HF repo.
    repo_id : str, optional
        HF repo ID.
    repo_type : str
        "model" or "dataset".
    token : str, optional
        HF access token with write permission.
    commit_message : str
        Commit message for the upload.
    """
    _ensure_hf_available()
    from huggingface_hub import HfApi

    repo = repo_id or CHECKPOINT_REPO_ID
    if not _is_configured(repo):
        raise RuntimeError(
            "HF repo not configured. Set SPCAC_HF_CHECKPOINT_REPO or "
            "SPCAC_HF_DATASET_REPO environment variable."
        )

    tok = token or HF_TOKEN

    api = HfApi(token=tok)
    api.upload_folder(
        folder_path=local_folder,
        path_in_repo=path_in_repo,
        repo_id=repo,
        repo_type=repo_type,
        commit_message=commit_message,
    )
    logger.info("Uploaded folder to HF: %s / %s", repo, path_in_repo)
