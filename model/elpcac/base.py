"""Unified base class for ELPCAC progressive coding models.

Supports N-pass progressive coding with a single parameterized implementation.
Subclasses only need to specify num_passes and group_fn.
"""

import torch
import MinkowskiEngine as ME
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

from model.core.base import compression_model
from model.core.utils import (
    make_sparse_tensor,
    make_spzeros_channel,
    make_spones_channel,
    apply_ent,
    apply_cmp,
    apply_dcmp,
    apply_nos,
    mask_spzeros_numpt_n,
)
from model.core.layers import (
    minkowski_local_self_attention_block,
    minkowski_mutil_residual_blocks_stack,
    global_hyper_encoder,
    global_parameter_model,
    make_encoder,
    make_decoder,
    make_hyper_encoder,
    make_hyper_decoder,
    make_fusion_network,
)

from model.core.geo import geometry_context_net


# ---------------------------------------------------------------------------
# Group → num_passes mapping – imported from utils so subclasses pick the right one
# ---------------------------------------------------------------------------
from model.core.utils import group_sp_3  # 3-pass: (all-even, even-pos, odd-pos)

GROUP_FN = {
    3: group_sp_3,
}


# ---------------------------------------------------------------------------
# Unified progressive-pass base class
# ---------------------------------------------------------------------------
class elpcac_base(compression_model):
    """Generic ELPCAC model with configurable N-pass progressive coding.

    Subclass attributes to set:
        _num_passes: int – number of progressive passes.
        _return_weight_attention: bool – compute final weight_attention in compress.
        _group_fn: callable – grouping function (e.g. GROUP_FN[3]).
    """

    _num_passes: int = 3
    _return_weight_attention: bool = False
    _group_fn = None

    # ------------------------------------------------------------------
    # Architecture builder
    # ------------------------------------------------------------------
    def __init__(self, N, M, HyM, channels, num_layers) -> None:
        super().__init__()

        self.N = N
        self.M = M

        # Encoder / Decoder (heavy by default; _l variants override)
        self.g_a = make_encoder(channels, N, M, num_layers, kernel_size=5, use_attention=True)
        self.g_s = make_decoder(channels, N, M, num_layers, kernel_size=5, use_attention=True)

        # ------------------ Hyper ------------------
        self.h_a = make_hyper_encoder(M, N, HyM, activation='relu', kernel_size=5)
        self.h_s = make_hyper_decoder(HyM, N, M, activation='relu', kernel_size=5)

        # Entropy models
        self.gaussian_conditional = GaussianConditional(None)
        self.entropy_bottleneck = EntropyBottleneck(HyM)

        # Fusion
        self.entropy_parameters = make_fusion_network(M)

        # Geometry attention
        self.geo = geometry_context_net(N, M)

        # Global hyper
        self.global_hyper_encoder = global_hyper_encoder(M * 2)
        self.entropy_bottleneck_zg = EntropyBottleneck(M * 2)
        self.global_hyper_decoder = torch.nn.Sequential(
            torch.nn.Linear(M * 2, M * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(M * 2, M * 2),
        )
        self.global_parameter_model = global_parameter_model(M * 2)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def preprocess(self, points, colors):
        """Collate input → sparse tensor → group → geometry features."""
        x_c, x_f = ME.utils.sparse_collate(
            coords=[points[i] for i in range(points.shape[0])],
            feats=[colors[i] for i in range(points.shape[0])],
        )
        x = make_sparse_tensor(
            coordinates=x_c, features=x_f,
            tensor_stride=1, dimension=3, device=points.device,
        )
        grouped = self._group_fn(x)
        x = grouped[0]
        num_gps = list(grouped[1:])  # [num_gp_1, num_gp_2, ...]

        x_geo = make_spones_channel(x, 1)
        hy_geo = self.geo(x_geo)
        return x, num_gps, x_geo, hy_geo

    def encode_global_hyper(self, y, hy_geo, mode):
        """Global hyper-encoder. mode ∈ {'train', 'compress'}."""
        spt_yg = ME.cat([y, hy_geo])
        zg, _ = self.global_hyper_encoder(spt_yg)
        zg = zg.permute(0, 2, 1)

        if mode == 'train':
            zg_hat, zg_likelihoods = self.entropy_bottleneck_zg(zg)
            zg_hat = zg_hat.permute(0, 2, 1)
            zg_likelihoods = zg_likelihoods.permute(0, 2, 1)
            hyg_hat = self.global_hyper_decoder(zg_hat)
            return hyg_hat, zg_likelihoods
        else:  # compress
            zg_strings = [self.entropy_bottleneck_zg.compress(zg)]
            zg_shape = zg.size()[-1:]
            zg_hat = self.entropy_bottleneck_zg.decompress(zg_strings[0], zg_shape)
            zg_hat = zg_hat.permute(0, 2, 1)
            hyg_hat = self.global_hyper_decoder(zg_hat)
            return hyg_hat, zg_strings, zg_shape

    def decode_global_hyper(self, out_enc):
        """Global hyper-decoder for decompression."""
        zg_hat = self.entropy_bottleneck_zg.decompress(
            out_enc['strings']['zg'][0], out_enc['zg_shape'])
        zg_hat = zg_hat.permute(0, 2, 1)
        return self.global_hyper_decoder(zg_hat)

    def encode_local_hyper(self, y, mode):
        """Local hyper-encoder. mode ∈ {'train', 'compress'}."""
        zl = self.h_a(y)
        if mode == 'train':
            zl_hat, zl_likelihoods = apply_ent(self.entropy_bottleneck, zl)
            hyl_hat = self.h_s(zl_hat)
            return hyl_hat, zl_likelihoods
        else:  # compress
            zl_strings, zl_shape = apply_cmp(self.entropy_bottleneck, zl)
            zl_hat = apply_dcmp(self.entropy_bottleneck, zl_strings, zl, zl_shape)
            hyl_hat = self.h_s(zl_hat)
            return hyl_hat, zl_strings, zl_shape, zl_hat

    def decode_local_hyper(self, out_enc):
        """Local hyper-decoder for decompression."""
        zl_hat = apply_dcmp(
            self.entropy_bottleneck,
            out_enc['strings']['zl'], out_enc['zl_hat'], out_enc['zl_shape'])
        return self.h_s(zl_hat)

    def compute_gaussian_params(self, ctx_params, hyg_hat, hyl_hat):
        """Compute scales/means for one progressive pass."""
        hym_hat, _ = self.global_parameter_model(ctx_params, hyg_hat)
        ctx_fusion = ME.cat([hym_hat, hyl_hat])
        gaussian_params_sp = self.entropy_parameters(ctx_fusion)
        gaussian_params = gaussian_params_sp.F.unsqueeze(0).permute(0, 2, 1)
        return gaussian_params.chunk(2, 1)

    # ------------------------------------------------------------------
    # Top-level API
    # ------------------------------------------------------------------
    def forward(self, points, colors):
        x, num_gps, x_geo, hy_geo = self.preprocess(points, colors)
        N = self._num_passes

        # Encoder
        y = self.g_a(x)

        # Global hyper (train)
        hyg_hat, zg_likelihoods = self.encode_global_hyper(y, hy_geo, 'train')

        # Quantize
        y_hat = apply_nos(self.gaussian_conditional, y, "noise")

        # Local hyper (train)
        hyl_hat, zl_likelihoods = self.encode_local_hyper(y, 'train')

        # Progressive passes
        y_likelihoods_list = []
        y_hat_accum = None  # sum of previous y_hat_k

        for k in range(N):
            # Context for this pass
            if k == 0:
                ctx = ME.cat([make_spzeros_channel(hy_geo, self.M), hy_geo])
            else:
                ctx = ME.cat([y_hat_accum, hy_geo])

            scales_hat, means_hat = self.compute_gaussian_params(ctx, hyg_hat, hyl_hat)
            _, y_likelihoods = apply_ent(self.gaussian_conditional, y, scales_hat, means_hat)

            # Mask & slice likelihoods
            y_hat_k = mask_spzeros_numpt_n(y_hat, num_gps, k)
            start = sum(num_gps[:k])
            end = sum(num_gps[:k + 1])
            y_likelihoods_list.append(y_likelihoods[start:end, :])

            # Accumulate for next pass
            y_hat_accum = y_hat_k if y_hat_accum is None else y_hat_accum + y_hat_k

        # Decoder
        x_hat = self.g_s(y_hat)

        likelihoods = {'zg': zg_likelihoods, 'zl': zl_likelihoods}
        for i, y_lk in enumerate(y_likelihoods_list):
            likelihoods[f'y{i + 1}'] = y_lk

        return {'likelihoods': likelihoods, 'x_hat': x_hat, 'x': x}

    def compress(self, points, colors):
        x, num_gps, x_geo, hy_geo = self.preprocess(points, colors)
        total_pts = sum(num_gps)
        N = self._num_passes

        # Encoder
        y = self.g_a(x)

        # Global hyper (compress)
        hyg_hat, zg_strings, zg_shape = self.encode_global_hyper(y, hy_geo, 'compress')

        # Quantize
        y_hat = apply_nos(self.gaussian_conditional, y, "symbols")

        # Local hyper (compress)
        hyl_hat, zl_strings, zl_shape, zl_hat = self.encode_local_hyper(y, 'compress')

        # Progressive passes
        y_strings_dict = {}
        y_sparse_dict = {}
        padded_feats_list = []  # each entry: [total_pts, M] tensor
        y_hat_accum = None

        for k in range(N):
            # Context
            if k == 0:
                ctx = ME.cat([make_spzeros_channel(hy_geo, self.M), hy_geo])
            else:
                ctx = ME.cat([y_hat_accum, hy_geo])

            scales_hat, means_hat = self.compute_gaussian_params(ctx, hyg_hat, hyl_hat)
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

            # Local decode for context building (skip last pass – no next pass)
            if k < N - 1:
                y_hat_k = make_spzeros_channel(y_hat, y_hat.F.shape[1])
                y_hat_k_f = apply_dcmp(
                    self.gaussian_conditional, y_strings_k, y_k, indexes_k, means=means_hat_k)

                # Pad features to full size
                zeros_before = torch.zeros(start, y_hat.F.shape[1], device=y_hat.device)
                zeros_after = torch.zeros(total_pts - end, y_hat.F.shape[1], device=y_hat.device)
                padded = torch.cat([zeros_before, y_hat_k_f.F, zeros_after], dim=0)
                padded_feats_list.append(padded)

                # Build accumulated context
                accum_feats = sum(padded_feats_list)  # y_hat_k.F is zeros
                y_hat_accum = make_sparse_tensor(
                    features=accum_feats,
                    tensor_stride=8, dimension=3,
                    coordinate_manager=y_hat_k.coordinate_manager,
                    coordinate_map_key=y_hat_k.coordinate_map_key,
                    device=y_hat_k.device,
                )

        # Optional weight_attention
        weight_attention = None
        if self._return_weight_attention:
            ctx = ME.cat([y_hat, hy_geo])
            _, weight_attention = self.global_parameter_model(ctx, hyg_hat)

        out = {
            'strings': {
                'zg': zg_strings,
                'zl': zl_strings,
                **y_strings_dict,
            },
            'zl_hat': zl_hat,
            'y_hat': y_hat,
            'x': x,
            'x_geo': x_geo,
            'zg_shape': zg_shape,
            'zl_shape': zl_shape,
            **{f'num_gp_{i + 1}': n for i, n in enumerate(num_gps)},
            **y_sparse_dict,
        }
        if weight_attention is not None:
            out['weight_attention'] = weight_attention
        return out

    def decompress(self, out_enc):
        N = self._num_passes
        num_gps = [out_enc[f'num_gp_{i + 1}'] for i in range(N)]
        total_pts = sum(num_gps)

        # Global hyper
        hyg_hat = self.decode_global_hyper(out_enc)

        # Local hyper
        hyl_hat = self.decode_local_hyper(out_enc)

        # Geometry
        hy_geo = self.geo(out_enc['x_geo'])

        # Progressive passes
        y_hat_k_f_list = []   # raw decoded features for final cat
        padded_feats_list = []
        y_hat_accum = None

        for k in range(N):
            # Context
            if k == 0:
                ctx = ME.cat([make_spzeros_channel(hy_geo, self.M), hy_geo])
            else:
                ctx = ME.cat([y_hat_accum, hy_geo])

            scales_hat, means_hat = self.compute_gaussian_params(ctx, hyg_hat, hyl_hat)
            indexes = self.gaussian_conditional.build_indexes(scales_hat)

            start = sum(num_gps[:k])
            end = sum(num_gps[:k + 1])

            indexes_k = indexes[:, :, start:end]
            means_hat_k = means_hat[:, :, start:end]

            # Decode this pass
            y_hat_k = make_spzeros_channel(out_enc['y_hat'], out_enc['y_hat'].F.shape[1])
            y_hat_k_f = apply_dcmp(
                self.gaussian_conditional,
                out_enc['strings'][f'y_{k + 1}'],
                out_enc[f'y_{k + 1}'],
                indexes_k,
                means=means_hat_k,
            )
            y_hat_k_f_list.append(y_hat_k_f)

            # Pad features to full size
            zeros_before = torch.zeros(start, y_hat_k.F.shape[1], device=y_hat_k_f.F.device)
            zeros_after = torch.zeros(total_pts - end, y_hat_k.F.shape[1], device=y_hat_k_f.F.device)
            padded = torch.cat([zeros_before, y_hat_k_f.F, zeros_after], dim=0)
            padded_feats_list.append(padded)

            # Build accumulated context
            accum_feats = sum(padded_feats_list)
            y_hat_accum = make_sparse_tensor(
                features=accum_feats,
                tensor_stride=8, dimension=3,
                coordinate_manager=y_hat_k.coordinate_manager,
                coordinate_map_key=y_hat_k.coordinate_map_key,
                device=y_hat_k.device,
            )

        # Final reconstruction
        y_hat = make_sparse_tensor(
            features=torch.cat([f.F for f in y_hat_k_f_list], dim=0),
            tensor_stride=8, dimension=3,
            coordinate_manager=y_hat_accum.coordinate_manager,
            coordinate_map_key=y_hat_accum.coordinate_map_key,
            device=y_hat_accum.device,
        )

        x_hat = self.g_s(y_hat)
        return {'x_hat': x_hat}
