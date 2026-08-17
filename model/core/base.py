import torch
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from typing import cast
from torch import Tensor
import math

from model.core.utils import make_sparse_tensor, group_sp
import MinkowskiEngine as ME

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    """Returns table of logarithmically scales."""
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class compression_model(torch.nn.Module):
    def update(self, scale_table=None, force=False):
        """Updates EntropyBottleneck and GaussianConditional CDFs.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (torch.Tensor): table of scales (i.e. stdev)
                for initializing the Gaussian distributions
                (default: 64 logarithmically spaced scales from 0.11 to 256)
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if at least one of the modules was updated.
        """
        if scale_table is None:
            scale_table = get_scale_table()
        updated = False
        for _, module in self.named_modules():
            if isinstance(module, EntropyBottleneck):
                updated |= module.update(force=force)
            if isinstance(module, GaussianConditional):
                updated |= module.update_scale_table(scale_table, force=force)
        return updated

    def aux_loss(self) -> Tensor:
        loss = sum(m.loss()
                   for m in self.modules() if isinstance(m, EntropyBottleneck))
        return cast(Tensor, loss)
    
    def preprocess(self, points, colors):
        """Collate points/colors into a sparse tensor.

        Shared by all basic/ models to eliminate repeated collate + make_sparse_tensor.
        """
        x_c, x_f = ME.utils.sparse_collate(
            coords=[points[i] for i in range(points.shape[0])],
            feats=[colors[i] for i in range(points.shape[0])],
        )
        return make_sparse_tensor(
            coordinates=x_c, features=x_f,
            tensor_stride=1, dimension=3, device=points.device,
        )

    @staticmethod
    def chunk_gaussian_params(sp_tensor):
        """Extract (scales, means) from a sparse tensor of Gaussian params."""
        gaussian_params = sp_tensor.F.unsqueeze(0).permute(0, 2, 1)
        return gaussian_params.chunk(2, 1)

    def transonly(self, points, colors):
        x_c, x_f = ME.utils.sparse_collate(
            coords=[points[i] for i in range(points.shape[0])],
            feats=[colors[i] for i in range(points.shape[0])]
        )
        x = make_sparse_tensor(
            coordinates=x_c,
            features=x_f,
            tensor_stride=1,
            dimension=3,
            device=points.device
        )

        x, num_gp_1, _ = group_sp(x)

        y = self.g_a(x)
        x_hat = self.g_s(y)

        return {
            'x_hat': x_hat,
            'x': x,
        }
        
