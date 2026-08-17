"""SPCAC model package.

Set OpenMP threads before any submodule imports MinkowskiEngine,
so the ME startup warning is avoided unless the user already configured it.
"""

import os

os.environ.setdefault('OMP_NUM_THREADS', '8')
