from . import datasets

def load_dataset(profile, mode):
    return datasets.get_dataset(profile, mode)
