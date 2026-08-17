"""3-pass progressive coding (parity-based 3-way grouping)."""
from model.basic.group_base import group_base
from model.core.utils import group_sp_3


class grouping(group_base):
    _num_passes = 3
    _group_fn = staticmethod(group_sp_3)
