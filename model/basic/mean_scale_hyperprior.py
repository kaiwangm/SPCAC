import torch
import MinkowskiEngine as ME
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from model.core.base import compression_model
from model.core.utils import apply_ent, apply_cmp, apply_dcmp, group_sp
from model.core.layers import make_encoder, make_decoder, make_hyper_encoder, make_hyper_decoder


class mean_scale_hyperprior(compression_model):
    def __init__(self, N, M, HyM, channels, num_layers) -> None:
        super(mean_scale_hyperprior, self).__init__()

        self.g_a = make_encoder(channels, N, M, num_layers)
        self.h_a = make_hyper_encoder(M, N, HyM, activation='leaky_relu')
        self.h_s = make_hyper_decoder(HyM, N, M, activation='leaky_relu')
        self.gaussian_conditional = GaussianConditional(None)
        self.entropy_bottleneck = EntropyBottleneck(HyM)
        self.g_s = make_decoder(channels, N, M, num_layers)

    def _hyper_bottleneck(self, y):
        """Encode/decode hyper latent and compute params."""
        z = self.h_a(y)
        z_hat, z_likelihoods = apply_ent(self.entropy_bottleneck, z)
        hy_hat = self.h_s(z_hat)
        scales_hat, means_hat = self.chunk_gaussian_params(hy_hat)
        return z, z_hat, z_likelihoods, hy_hat, scales_hat, means_hat

    def _hyper_compress(self, y):
        """Compress/decompress hyper latent."""
        z = self.h_a(y)
        z_strings, z_shape = apply_cmp(self.entropy_bottleneck, z)
        z_hat = apply_dcmp(self.entropy_bottleneck, z_strings, z, z_shape)
        hy_hat = self.h_s(z_hat)
        scales_hat, means_hat = self.chunk_gaussian_params(hy_hat)
        return z_strings, z_shape, z_hat, hy_hat, scales_hat, means_hat

    def forward(self, points, colors):
        x = self.preprocess(points, colors)
        x, _, _ = group_sp(x)

        y = self.g_a(x)
        _, _, z_likelihoods, _, scales_hat, means_hat = self._hyper_bottleneck(y)
        y_hat, y_likelihoods = apply_ent(self.gaussian_conditional, y, scales_hat, means_hat)

        x_hat = self.g_s(y_hat)

        return {
            'likelihoods': {'y': y_likelihoods, 'zl': z_likelihoods},
            'x_hat': x_hat,
            'x': x,
        }

    def compress(self, points, colors):
        x = self.preprocess(points, colors)
        x, _, _ = group_sp(x)

        y = self.g_a(x)
        z_strings, z_shape, z_hat, _, scales_hat, means_hat = self._hyper_compress(y)

        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings, _ = apply_cmp(self.gaussian_conditional, y, indexes, means_hat)

        return {
            'strings': {'zl': z_strings, 'y': y_strings},
            'z_hat': z_hat,
            'z_shape': z_shape,
            'x': x,
            'y': y,
        }

    def decompress(self, out_enc):
        z_hat = apply_dcmp(self.entropy_bottleneck,
                           out_enc['strings']['zl'], out_enc['z_hat'], out_enc['z_shape'])
        hy_hat = self.h_s(z_hat)
        scales_hat, means_hat = self.chunk_gaussian_params(hy_hat)

        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = apply_dcmp(self.gaussian_conditional,
                           out_enc['strings']['y'], out_enc['y'], indexes, means=means_hat)

        x_hat = self.g_s(y_hat)
        return {'x_hat': x_hat}
