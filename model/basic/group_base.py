"""Unified base class for N-pass group-based progressive coding models.

Subclass attributes:
    _num_passes: int – number of progressive passes.
    _group_fn: callable – grouping function (e.g. group_sp_3).
"""

import torch
import MinkowskiEngine as ME
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

from model.core.base import compression_model
from model.core.utils import (
    make_sparse_tensor, make_spzeros_channel,
    apply_ent, apply_cmp, apply_dcmp, apply_nos,
    mask_spzeros_numpt_n,
)
from model.core.layers import (
    sconv, make_encoder, make_decoder,
    make_hyper_encoder, make_hyper_decoder, make_fusion_network,
)


class group_base(compression_model):
    """Generic N-pass group-based progressive coding model."""

    _num_passes: int = 3
    _group_fn = None

    def __init__(self, N, M, HyM, channels, num_layers) -> None:
        super().__init__()
        self.N = N
        self.M = M

        self.g_a = make_encoder(channels, N, M, num_layers)
        self.g_s = make_decoder(channels, N, M, num_layers)
        self.h_a = make_hyper_encoder(M, N, HyM, activation='relu')
        self.h_s = make_hyper_decoder(HyM, N, M, activation='relu')

        self.gaussian_conditional = GaussianConditional(None)
        self.entropy_bottleneck = EntropyBottleneck(HyM)

        self.context_prediction = torch.nn.Sequential(
            sconv(M, M * 2, kernel_size=5, stride=1, bias=True),
        )
        self.entropy_parameters = make_fusion_network(M)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def forward(self, points, colors):
        x = self.preprocess(points, colors)
        grouped = self._group_fn(x)
        x, num_gps = grouped[0], list(grouped[1:])

        y = self.g_a(x)
        y_hat = apply_nos(self.gaussian_conditional, y, "noise")

        # Hyper
        z = self.h_a(y)
        z_hat, z_likelihoods = apply_ent(self.entropy_bottleneck, z)
        hy_hat = self.h_s(z_hat)

        # Progressive passes
        y_likelihoods_list = []
        y_hat_accum = None

        for k in range(self._num_passes):
            if k == 0:
                ctx_params = make_spzeros_channel(y_hat, hy_hat.F.shape[1])
            else:
                ctx_params = self.context_prediction(y_hat_accum)

            ctx_fusion = ME.cat([hy_hat, ctx_params])
            scales_hat, means_hat = self.chunk_gaussian_params(
                self.entropy_parameters(ctx_fusion))

            _, y_likelihoods = apply_ent(
                self.gaussian_conditional, y, scales_hat, means_hat)

            y_hat_k = mask_spzeros_numpt_n(y_hat, num_gps, k)
            start = sum(num_gps[:k])
            end = sum(num_gps[:k + 1])
            y_likelihoods_list.append(y_likelihoods[start:end, :])

            y_hat_accum = y_hat_k if y_hat_accum is None else y_hat_accum + y_hat_k

        x_hat = self.g_s(y_hat)

        likelihoods = {'zl': z_likelihoods}
        for i, y_lk in enumerate(y_likelihoods_list):
            likelihoods[f'y{i + 1}'] = y_lk

        return {'likelihoods': likelihoods, 'x_hat': x_hat, 'x': x}

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------
    def compress(self, points, colors):
        x = self.preprocess(points, colors)
        grouped = self._group_fn(x)
        x, num_gps = grouped[0], list(grouped[1:])
        total_pts = sum(num_gps)
        N = self._num_passes

        y = self.g_a(x)
        y_hat = apply_nos(self.gaussian_conditional, y, "symbols")

        # Hyper
        z = self.h_a(y)
        z_strings, z_shape = apply_cmp(self.entropy_bottleneck, z)
        z_hat = apply_dcmp(self.entropy_bottleneck, z_strings, z, z_shape)
        hy_hat = self.h_s(z_hat)

        # Progressive passes
        y_strings_dict = {}
        y_sparse_dict = {}
        padded_feats_list = []
        y_hat_accum = None

        for k in range(N):
            if k == 0:
                ctx_params = make_spzeros_channel(y_hat, hy_hat.F.shape[1])
            else:
                ctx_params = self.context_prediction(y_hat_accum)

            ctx_fusion = ME.cat([hy_hat, ctx_params])
            scales_hat, means_hat = self.chunk_gaussian_params(
                self.entropy_parameters(ctx_fusion))

            indexes = self.gaussian_conditional.build_indexes(scales_hat)
            start = sum(num_gps[:k])
            end = sum(num_gps[:k + 1])

            # Sub-tensor for this pass
            y_k = make_sparse_tensor(
                coordinates=y.C[start:end, :],
                features=y.F[start:end, :],
                tensor_stride=8, dimension=3, device=points.device,
            )
            indexes_k = indexes[:, :, start:end]
            means_hat_k = means_hat[:, :, start:end]

            y_strings_k, _ = apply_cmp(
                self.gaussian_conditional, y_k, indexes_k, means_hat_k)
            y_strings_dict[f'y_{k + 1}'] = y_strings_k
            y_sparse_dict[f'y_{k + 1}'] = y_k

            # Local decode for context (skip last pass)
            if k < N - 1:
                y_hat_k = make_spzeros_channel(y_hat, y_hat.F.shape[1])
                y_hat_k_f = apply_dcmp(
                    self.gaussian_conditional, y_strings_k, y_k, indexes_k, means=means_hat_k)

                zeros_before = torch.zeros(
                    start, y_hat.F.shape[1], device=y_hat.device)
                zeros_after = torch.zeros(
                    total_pts - end, y_hat.F.shape[1], device=y_hat.device)
                padded = torch.cat(
                    [zeros_before, y_hat_k_f.F, zeros_after], dim=0)
                padded_feats_list.append(padded)

                accum_feats = sum(padded_feats_list)
                y_hat_accum = make_sparse_tensor(
                    features=accum_feats,
                    tensor_stride=8, dimension=3,
                    coordinate_manager=y_hat_k.coordinate_manager,
                    coordinate_map_key=y_hat_k.coordinate_map_key,
                    device=y_hat_k.device,
                )

        out = {
            'strings': {'z': z_strings, **y_strings_dict},
            'z_hat': z_hat,
            'y_hat': y_hat,
            'x': x,
            'z_shape': z_shape,
            **{f'num_gp_{i + 1}': n for i, n in enumerate(num_gps)},
            **y_sparse_dict,
        }
        return out

    # ------------------------------------------------------------------
    # Decompression
    # ------------------------------------------------------------------
    def decompress(self, out_enc):
        N = self._num_passes
        num_gps = [out_enc[f'num_gp_{i + 1}'] for i in range(N)]
        total_pts = sum(num_gps)

        z_hat = apply_dcmp(self.entropy_bottleneck,
                           out_enc['strings']['z'], out_enc['z_hat'], out_enc['z_shape'])
        hy_hat = self.h_s(z_hat)

        y_hat_k_f_list = []
        padded_feats_list = []
        y_hat_accum = None

        for k in range(N):
            if k == 0:
                ctx_params = make_spzeros_channel(hy_hat, self.M * 2)
            else:
                ctx_params = self.context_prediction(y_hat_accum)

            ctx_fusion = ME.cat([hy_hat, ctx_params])
            scales_hat, means_hat = self.chunk_gaussian_params(
                self.entropy_parameters(ctx_fusion))

            indexes = self.gaussian_conditional.build_indexes(scales_hat)
            start = sum(num_gps[:k])
            end = sum(num_gps[:k + 1])

            indexes_k = indexes[:, :, start:end]
            means_hat_k = means_hat[:, :, start:end]

            y_hat_k = make_spzeros_channel(
                out_enc['y_hat'], out_enc['y_hat'].F.shape[1])
            y_hat_k_f = apply_dcmp(
                self.gaussian_conditional,
                out_enc['strings'][f'y_{k + 1}'],
                out_enc[f'y_{k + 1}'],
                indexes_k,
                means=means_hat_k,
            )
            y_hat_k_f_list.append(y_hat_k_f)

            zeros_before = torch.zeros(
                start, y_hat_k.F.shape[1], device=y_hat_k_f.F.device)
            zeros_after = torch.zeros(
                total_pts - end, y_hat_k.F.shape[1], device=y_hat_k_f.F.device)
            padded = torch.cat(
                [zeros_before, y_hat_k_f.F, zeros_after], dim=0)
            padded_feats_list.append(padded)

            accum_feats = sum(padded_feats_list)
            y_hat_accum = make_sparse_tensor(
                features=accum_feats,
                tensor_stride=8, dimension=3,
                coordinate_manager=y_hat_k.coordinate_manager,
                coordinate_map_key=y_hat_k.coordinate_map_key,
                device=y_hat_k.device,
            )

        y_hat = make_sparse_tensor(
            features=torch.cat([f.F for f in y_hat_k_f_list], dim=0),
            tensor_stride=8, dimension=3,
            coordinate_manager=y_hat_accum.coordinate_manager,
            coordinate_map_key=y_hat_accum.coordinate_map_key,
            device=y_hat_accum.device,
        )

        x_hat = self.g_s(y_hat)
        return {'x_hat': x_hat}
