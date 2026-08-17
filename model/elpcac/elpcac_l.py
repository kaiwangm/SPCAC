"""ELPCAC-L: lightweight encoder/decoder for 3-pass progressive coding."""
from model.elpcac.elpcac import elpcac
from model.core.layers import make_encoder, make_decoder


class elpcac_l(elpcac):
    def __init__(self, N, M, HyM, channels, num_layers) -> None:
        super().__init__(N, M, HyM, channels, num_layers)
        self.g_a = make_encoder(channels, N, M, num_layers, kernel_size=3)
        self.g_s = make_decoder(channels, N, M, num_layers, kernel_size=3)
