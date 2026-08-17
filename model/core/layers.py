from typing import Any
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function
import MinkowskiEngine as ME
from torch import nn, einsum
from pytorch3d.ops import knn_points, knn_gather
from model.core.utils import make_sparse_tensor


def sconv(in_channels, out_channels, kernel_size=3, stride=1, bias=False):
    return ME.MinkowskiConvolution(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        bias=bias,
        dimension=3,
    )


def sconvt(in_channels, out_channels, kernel_size=3, stride=1, bias=False):
    return ME.MinkowskiConvolutionTranspose(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        bias=bias,
        dimension=3,
    )


# ---------------------------------------------------------------------------
# Model component factories (shared by basic/ and elpcac/ sub-packages)
# ---------------------------------------------------------------------------
def make_encoder(channels, N, M, num_layers, kernel_size=3, use_attention=False):
    """Encoder factory.

    kernel_size  – convolution kernel size (e.g. 3 for lightweight, 5 for full).
    use_attention – when True, adds residual blocks and self-attention.
    """
    k = kernel_size
    if not use_attention:
        return nn.Sequential(
            sconv(channels, 64, kernel_size=k, stride=1, bias=True),
            sconv(64, N, kernel_size=k, stride=2, bias=True),
            *([ME.MinkowskiReLU(inplace=True),
               sconv(N, N, kernel_size=k, stride=1, bias=True),
               sconv(N, N, kernel_size=k, stride=2, bias=True)] * num_layers),
            ME.MinkowskiReLU(inplace=True),
            sconv(N, N, kernel_size=k, stride=1, bias=True),
            sconv(N, M, kernel_size=k, stride=2, bias=True),
        )
    return nn.Sequential(
        sconv(channels, 64, kernel_size=3, stride=1, bias=True),
        sconv(64, N, kernel_size=k, stride=2, bias=True),
        minkowski_mutil_residual_blocks_stack(N),
        *([sconv(N, N, kernel_size=k, stride=2, bias=True),
           minkowski_mutil_residual_blocks_stack(N),
           minkowski_local_self_attention_block(N)] * num_layers),
        sconv(N, M, kernel_size=k, stride=2, bias=True),
        minkowski_mutil_residual_blocks_stack(M),
        minkowski_local_self_attention_block(M),
    )


def make_decoder(channels, N, M, num_layers, kernel_size=3, use_attention=False):
    """Decoder factory.

    kernel_size  – convolution kernel size (e.g. 3 for lightweight, 5 for full).
    use_attention – when True, adds residual blocks and self-attention.
    """
    k = kernel_size
    if not use_attention:
        return nn.Sequential(
            sconvt(M, N, kernel_size=k, stride=2, bias=True),
            sconv(N, N, kernel_size=k, stride=1, bias=True),
            *([ME.MinkowskiReLU(inplace=True),
               sconvt(N, N, kernel_size=k, stride=2, bias=True),
               sconv(N, N, kernel_size=k, stride=1, bias=True)] * num_layers),
            ME.MinkowskiReLU(inplace=True),
            sconvt(N, 64, kernel_size=k, stride=2, bias=True),
            sconv(64, channels, kernel_size=k, stride=1, bias=True),
        )
    return nn.Sequential(
        minkowski_local_self_attention_block(M),
        minkowski_mutil_residual_blocks_stack(M),
        sconvt(M, N, kernel_size=k, stride=2, bias=True),
        *([minkowski_local_self_attention_block(N),
           minkowski_mutil_residual_blocks_stack(N),
           sconvt(N, N, kernel_size=k, stride=2, bias=True)] * num_layers),
        minkowski_mutil_residual_blocks_stack(N),
        sconvt(N, 64, kernel_size=k, stride=2, bias=True),
        sconv(64, channels, kernel_size=3, stride=1, bias=True),
    )


def make_hyper_encoder(M, N, HyM, activation='relu', kernel_size=3):
    """Hyper encoder for latent features."""
    k = kernel_size
    act = ME.MinkowskiLeakyReLU if activation == 'leaky_relu' else ME.MinkowskiReLU
    return nn.Sequential(
        sconv(M, N, kernel_size=3, stride=1, bias=True),
        act(inplace=True),
        sconv(N, N, kernel_size=3, stride=1, bias=True),
        sconv(N, N, kernel_size=k, stride=2, bias=True),
        act(inplace=True),
        sconv(N, N, kernel_size=3, stride=1, bias=True),
        sconv(N, HyM, kernel_size=k, stride=2, bias=True),
    )


def make_hyper_decoder(HyM, N, M, activation='relu', kernel_size=3):
    """Hyper decoder for latent features."""
    k = kernel_size
    act = ME.MinkowskiLeakyReLU if activation == 'leaky_relu' else ME.MinkowskiReLU
    return nn.Sequential(
        sconvt(HyM, N, kernel_size=k, stride=2, bias=True),
        sconv(N, N, kernel_size=3, stride=1, bias=True),
        act(inplace=True),
        sconvt(N, N * 3 // 2, kernel_size=k, stride=2, bias=True),
        sconv(N * 3 // 2, N * 3 // 2, kernel_size=3, stride=1, bias=True),
        act(inplace=True),
        sconv(N * 3 // 2, M * 2, kernel_size=3, stride=1, bias=True),
    )


def make_fusion_network(M):
    """Entropy parameter fusion: M*4 → M*2."""
    return nn.Sequential(
        sconv(M * 4, M * 3, kernel_size=1, stride=1, bias=True),
        ME.MinkowskiReLU(inplace=True),
        sconv(M * 3, M * 2, kernel_size=1, stride=1, bias=True),
        ME.MinkowskiReLU(inplace=True),
        sconv(M * 2, M * 2, kernel_size=1, stride=1, bias=True),
    )


class minkowski_masked_convolution(ME.MinkowskiConvolution):
    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.kernel))
        hwd, _, _ = self.mask.size()
        self.mask[hwd // 2:, :, :] = 0

    def forward(self, x):
        # TODO(begaintj): weight assigment is not supported by torchscript
        with torch.no_grad():
            self.kernel *= self.mask
        return super().forward(x)


class minkowski_point_transformer(nn.Module):
    def __init__(self, in_channel, dim=32, n_knn=3, pos_hidden_dim=32, attn_hidden_multiplier=3):
        super(minkowski_point_transformer, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(dim, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )

        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, sp_tensor):
        """feed forward of transformer
        Args:
            x: Tensor of features, (B, in_channel, n)
            pos: Tensor of positions, (B, 3, n)
        Returns:
            y: Tensor of features with attention, (B, in_channel, n)
        """

        x = sp_tensor.F.unsqueeze(0).permute(0, 2, 1)

        identity = x

        x = self.linear_start(x)
        b, dim, n = x.shape

        x_flipped = x.permute(0, 2, 1).contiguous()

        _, idx_knn, _ = knn_points(x_flipped, x_flipped, K=self.n_knn)
        idx_knn = idx_knn.contiguous()

        key = self.conv_key(x)
        value = self.conv_value(x)
        query = self.conv_query(x)

        key = knn_gather(key.permute(0, 2, 1), idx_knn).permute(0, 3, 1, 2)

        qk_rel = query.reshape((b, -1, n, 1)) - key

        attention = self.attn_mlp(qk_rel)
        attention = torch.softmax(attention, -1)

        value = value.reshape((b, -1, n, 1))

        agg = einsum('b c i j, b c i j -> b c i',
                     attention, value)  # b, dim, n
        y = self.linear_end(agg)

        ans_feats = y + identity

        # ----------------------------
        ans_sp_tensor = make_sparse_tensor(
            features=ans_feats.permute(0, 2, 1).squeeze(0),
            tensor_stride=sp_tensor.tensor_stride,
            dimension=3,
            device=sp_tensor.device,
            coordinate_manager=sp_tensor.coordinate_manager,
            coordinate_map_key=sp_tensor.coordinate_map_key,
        )

        return ans_sp_tensor


class minkowski_local_self_attention_block(nn.Module):
    def __init__(self, N):
        super(minkowski_local_self_attention_block, self).__init__()

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self, N):
                super().__init__()
                self.conv = nn.Sequential(
                    sconv(N, N // 2, kernel_size=1, stride=1, bias=True),
                    ME.MinkowskiReLU(inplace=True),
                    sconv(N // 2, N // 2, kernel_size=3, stride=1, bias=True),
                    ME.MinkowskiReLU(inplace=True),
                    sconv(N // 2, N, kernel_size=1, stride=1, bias=True),
                )
                self.relu = ME.MinkowskiReLU(inplace=True)

            def forward(self, x):
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(
            ResidualUnit(N),
            ResidualUnit(N),
            ResidualUnit(N),
        )

        self.conv_b = nn.Sequential(
            ResidualUnit(N),
            ResidualUnit(N),
            ResidualUnit(N),
            sconv(N, N, kernel_size=1, stride=1, bias=True),
        )

        self.sigmoid = ME.MinkowskiSigmoid()

    def forward(self, x):
        identity = x
        a = self.conv_a(x)
        b = self.sigmoid(self.conv_b(x))
        # b = ME.SparseTensor(
        #     coordinates=b.C,
        #     features=b.F,
        #     coordinate_manager=a.coordinate_manager,
        # )

        out = a * b
        out += identity
        return out


class minkowski_residual_block(nn.Module):
    def __init__(self, N):
        super(minkowski_residual_block, self).__init__()
        self.conv = nn.Sequential(
            sconv(N, N, kernel_size=1, stride=1, bias=True),
            ME.MinkowskiReLU(inplace=True),
            sconv(N, N, kernel_size=3, stride=1, bias=True),
            ME.MinkowskiReLU(inplace=True),
            sconv(N, N, kernel_size=1, stride=1, bias=True),
        )

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out += identity
        return out


class minkowski_mutil_residual_blocks_stack(nn.Module):
    def __init__(self, N):
        super(minkowski_mutil_residual_blocks_stack, self).__init__()
        self.conv = nn.Sequential(
            minkowski_residual_block(N),
            minkowski_residual_block(N),
            minkowski_residual_block(N),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class minkowski_self_mutilhead_attention(nn.Module):
    def __init__(self, M):
        super(minkowski_self_mutilhead_attention, self).__init__()

        self.M = M
        self.mutilattention = torch.nn.MultiheadAttention(
            embed_dim=M, num_heads=8, batch_first=True)
        self.norm1 = torch.nn.LayerNorm(M)
        self.norm2 = torch.nn.LayerNorm(M)
        self.mlp = nn.Sequential(
            torch.nn.Linear(M, 2 * M),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * M, M),
        )

    def forward(self, x):
        feats = self.norm1(x.F.unsqueeze(0))

        query = feats
        key = feats
        value = feats

        hym, weight = self.mutilattention(query, key, value)
        hym = self.norm1(hym + feats)

        out = self.mlp(hym)
        out = self.norm2(out + hym)

        # ----------------------------
        ans_sp_tensor = make_sparse_tensor(
            features=out.squeeze(0),
            tensor_stride=x.tensor_stride,
            dimension=3,
            device=x.device,
            coordinate_manager=x.coordinate_manager,
            coordinate_map_key=x.coordinate_map_key,
        )

        return ans_sp_tensor, weight


class global_hyper_encoder(nn.Module):
    def __init__(self, M):
        super(global_hyper_encoder, self).__init__()
        self.M = M
        self.mutilattention = torch.nn.MultiheadAttention(
            embed_dim=M, num_heads=8, batch_first=True)
        self.u = torch.nn.Parameter(torch.randn(
            size=(1, 8, M)), requires_grad=True)
        self.norm1 = torch.nn.LayerNorm(M)
        self.norm2 = torch.nn.LayerNorm(M)
        self.norm3 = torch.nn.LayerNorm(M)
        self.mlp1 = nn.Sequential(
            torch.nn.Linear(M, M),
            torch.nn.ReLU(),
            torch.nn.Linear(M, M),
        )
        self.mlp2 = torch.nn.Linear(M, M)

    def forward(self, x):
        query = self.norm1(self.u)

        kv = self.norm2(x.F.unsqueeze(0))
        key = kv
        value = kv

        up, weight = self.mutilattention(query, key, value)
        up = up + self.u
        up = self.mlp1(self.norm3(up)) + up

        up = self.mlp2(up)

        return up, weight


class global_parameter_model(nn.Module):
    def __init__(self, M):
        super(global_parameter_model, self).__init__()
        self.M = M
        self.mutilattention = torch.nn.MultiheadAttention(
            embed_dim=M, num_heads=8, batch_first=True)
        self.norm1 = torch.nn.LayerNorm(M)
        self.norm2 = torch.nn.LayerNorm(M)
        self.norm3 = torch.nn.LayerNorm(M)

        self.mlp1 = nn.Sequential(
            torch.nn.Linear(M, M),
            torch.nn.ReLU(),
            torch.nn.Linear(M, M),
        )

    def forward(self, ctx, hyg):
        feats_hyl = self.norm1(ctx.F.unsqueeze(0))
        query = self.norm1(feats_hyl)

        kv = self.norm2(hyg)
        key = kv
        value = kv

        hym, weight = self.mutilattention(query, key, value)
        hym = hym + feats_hyl
        hym = self.mlp1(self.norm3(hym)) + hym

        # ----------------------------
        ans_sp_tensor = make_sparse_tensor(
            features=hym.squeeze(0),
            tensor_stride=ctx.tensor_stride,
            dimension=3,
            device=ctx.device,
            coordinate_manager=ctx.coordinate_manager,
            coordinate_map_key=ctx.coordinate_map_key,
        )

        return ans_sp_tensor, weight
