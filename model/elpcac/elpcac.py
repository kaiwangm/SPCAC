"""ELPCAC: 3-pass progressive coding (3-group parity partitioning)."""
from model.elpcac.base import elpcac_base, GROUP_FN


class elpcac(elpcac_base):
    _num_passes = 3
    _return_weight_attention = True
    _group_fn = staticmethod(GROUP_FN[3])
