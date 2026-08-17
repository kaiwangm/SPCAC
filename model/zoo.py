"""Model zoo: registry of all supported models and factory helpers.

A model profile YAML under ``configs/model`` selects one of the models
registered in ``models`` and provides per-quality latent widths and
rate-distortion lambda.
"""

import os
import yaml

# Baselines
from model.basic.factorized_prior import factorized_prior
from model.basic.mean_scale_hyperprior import mean_scale_hyperprior
from model.basic.grouping import grouping

# ELPCAC family
from model.elpcac.elpcac import elpcac
from model.elpcac.elpcac_l import elpcac_l


# Registry mapping model profile names to model classes
models = {
    'factorized_prior': factorized_prior,
    'mean_scale_hyperprior': mean_scale_hyperprior,
    'grouping': grouping,
    'elpcac_l': elpcac_l,
    'elpcac': elpcac,
}

BASE_DIR = 'configs/model'


def get_model(profile, quality, category):
    """Build a model instance from a model profile YAML.

    Args:
        profile: profile name, selects ``configs/model/{profile}.yaml``.
        quality: quality level index (q0-q6), selects the ``levels`` entry.
        category: point cloud category tag.

    Returns:
        (model, lam): the built model and the rate-distortion lambda.
    """

    profile_path = os.path.join(BASE_DIR, '{}.yaml'.format(profile))
    # Load the model profile YAML
    with open(profile_path, 'r') as f:
        method_profile = yaml.load(f, Loader=yaml.FullLoader)

    # Resolve model name, category, and per-quality hyper-parameters
    model_name = method_profile['model']
    category = method_profile['category']
    cfg = method_profile['levels']['q{}'.format(quality)]

    # Optional overrides: input channels and number of downsample layers
    channels = 3
    if 'channels' in method_profile:
        channels = method_profile['channels']
    paramemts = cfg['paraments']
    num_layers = 1
    if 'num_layers' in method_profile:
        num_layers = method_profile['num_layers']

    # Instantiate the model with latent widths from the profile
    model = models[model_name](
        *paramemts, channels=channels, num_layers=num_layers)

    lam = cfg['lambda']

    # Print a summary of the resolved configuration
    print('-------------------------')
    print('Profile Path: {}'.format(profile_path))
    print('Model: {}'.format(model_name))
    print('Category: {}'.format(category))
    print('Quality: {}'.format(quality))
    print('Lambda: {}'.format(lam))
    print('Paraments: {}'.format(paramemts))
    print('Num Layers: {}'.format(num_layers))
    print('-------------------------')

    return model, lam


def load_model(profile, quality, category='dense'):
    """Convenience alias of :func:`get_model` with a default category."""
    return get_model(profile, quality, category)
