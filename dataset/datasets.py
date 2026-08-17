import os
import yaml
import logging

from .sensat_urban import(
    sensat_urban,
)

from .coco3d import(
    coco3d,
)

from .scannet import(
    scannet,
)

from .j8ivfbv2 import(
    j8ivfbv2,
)

from .owlii import(
    owlii,
)

datasets = {
    'sensat_urban': sensat_urban,
    'coco3d': coco3d,
    'scannet': scannet,
    'j8ivfbv2': j8ivfbv2,
    'owlii': owlii,
}

BASE_DIR = 'configs/dataset'

logger = logging.getLogger(__name__)


def _ensure_dataset_available(root_path, dataset_name):
    """
    Ensure the dataset files exist locally.

    If the local path does not exist, attempt to download from
    Hugging Face Hub as a fallback.

    Parameters
    ----------
    root_path : str
        Local directory path for the dataset.
    dataset_name : str
        Dataset name (e.g. 'coco3d', 'j8ivfbv2').

    Returns
    -------
    bool
        True if the dataset is now available locally.
    """
    if os.path.exists(root_path):
        return True

    # Try Hugging Face Hub fallback
    try:
        from utils.hf_hub import snapshot_download_dataset, DATASET_REPO_ID
    except ImportError:
        logger.warning(
            "huggingface_hub not installed; cannot auto-download dataset."
        )
        return False

    if not DATASET_REPO_ID:
        logger.warning(
            "SPCAC_HF_DATASET_REPO not set; skipping HF dataset download."
        )
        return False

    # Determine the relative path inside the HF repo
    # The dataset root_path is typically './datasets/<dataset_name>' or
    # './datasets/<dataset_name>/<subset>'
    rel_path = os.path.relpath(root_path, './datasets')

    logger.info(
        "Dataset '%s' not found locally at '%s'. "
        "Downloading from Hugging Face Hub...",
        dataset_name, root_path
    )

    try:
        snapshot_download_dataset(
            local_dir=os.path.join('datasets'),
            allow_patterns=[f"{rel_path}/**"],
        )
        if os.path.exists(root_path):
            logger.info("Dataset '%s' downloaded successfully.", dataset_name)
            return True
        else:
            logger.warning(
                "Dataset '%s' download completed but path still not found.",
                dataset_name
            )
            return False
    except Exception as e:
        logger.warning(
            "Failed to download dataset '%s' from HF Hub: %s",
            dataset_name, e
        )
        return False


def _ensure_hdf5_available(hdf5_path, dataset_name):
    """
    Ensure an hdf5 dataset file exists locally, downloading from HF if needed.

    Parameters
    ----------
    hdf5_path : str
        Local path to the hdf5 file (e.g. './datasets/coco3d.hdf5').
    dataset_name : str
        Dataset name for logging.

    Returns
    -------
    bool
        True if the hdf5 file is now available locally.
    """
    if os.path.exists(hdf5_path):
        return True

    # Try Hugging Face Hub fallback
    try:
        from utils.hf_hub import download_dataset_file, DATASET_REPO_ID
    except ImportError:
        return False

    if not DATASET_REPO_ID:
        return False

    repo_filename = os.path.relpath(hdf5_path, './datasets')

    logger.info(
        "HDF5 '%s' not found locally. Downloading from Hugging Face Hub...",
        hdf5_path
    )

    try:
        download_dataset_file(repo_filename)
        if os.path.exists(hdf5_path):
            logger.info("HDF5 '%s' downloaded successfully.", hdf5_path)
            return True
        return False
    except Exception as e:
        logger.warning(
            "Failed to download hdf5 '%s' from HF Hub: %s",
            hdf5_path, e
        )
        return False


def get_dataset(profile, mode):
    profile_path = os.path.join(BASE_DIR, '{}.yaml'.format(profile))
    # load yaml config
    with open(profile_path, 'r') as f:
        dataset_profile = yaml.load(f, Loader=yaml.FullLoader)

    dataset_name = dataset_profile['dataset']
    root_path = dataset_profile['root_path']
    depth = dataset_profile['depth']

    # Try to ensure the dataset is available locally (HF Hub fallback)
    _ensure_dataset_available(root_path, dataset_name)

    dataset = datasets[dataset_name](root_path, depth, mode)
    category = dataset_profile['category']
    attributes = dataset_profile['attributes']

    # draw a table
    print('-------------------------')
    print('Dataset : {}'.format(dataset_name))
    print('Root Path: {}'.format(root_path))
    print('Mode: {}'.format(mode))
    print('Category: {}'.format(category))
    print('Attributes: {}'.format(attributes))
    print('Depth: {}'.format(depth))
    print('-------------------------')

    dataset.name = dataset_name
    dataset.category = category
    dataset.attributes = attributes

    return dataset