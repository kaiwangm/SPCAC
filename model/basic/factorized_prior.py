import torch
import MinkowskiEngine as ME
from compressai.entropy_models import EntropyBottleneck
from model.core.base import compression_model
from model.core.utils import apply_ent, apply_cmp, apply_dcmp, group_sp
from model.core.layers import make_encoder, make_decoder


class factorized_prior(compression_model):
    def __init__(self, N, M, channels, num_layers, kernel_size=3, use_attention=False) -> None:
        super(factorized_prior, self).__init__()

        self.g_a = make_encoder(channels, N, M, num_layers, kernel_size=kernel_size, use_attention=use_attention)
        self.entropy_bottleneck = EntropyBottleneck(M)
        self.g_s = make_decoder(channels, N, M, num_layers, kernel_size=kernel_size, use_attention=use_attention)

    def forward(self, points, colors):
        x = self.preprocess(points, colors)
        x, _, _ = group_sp(x)

        y = self.g_a(x)
        y_hat, y_likelihoods = apply_ent(self.entropy_bottleneck, y)
        x_hat = self.g_s(y_hat)

        return {
            'likelihoods': {'y': y_likelihoods},
            'x_hat': x_hat,
            'x': x,
        }

    def compress(self, points, colors):
        x = self.preprocess(points, colors)
        x, _, _ = group_sp(x)

        y = self.g_a(x)
        y_strings, y_shape = apply_cmp(self.entropy_bottleneck, y)

        return {
            'strings': {'y': y_strings},
            'x': x,
            'y': y,
            'y_shape': y_shape,
        }

    def decompress(self, out_enc):
        y_hat = apply_dcmp(self.entropy_bottleneck,
                           out_enc['strings']['y'], out_enc['y'], out_enc['y_shape'])
        x_hat = self.g_s(y_hat)

        return {'x_hat': x_hat}
