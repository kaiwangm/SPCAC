import torch
import MinkowskiEngine as ME
from model.core.layers import sconv


class geometry_context_net(torch.nn.Module):
    def __init__(self, N, M) -> None:
        super(geometry_context_net, self).__init__()

        self.g_a = torch.nn.Sequential(
            sconv(1, 64, kernel_size=3, stride=1, bias=True),
            sconv(64, N, kernel_size=3, stride=2, bias=True),
            ME.MinkowskiReLU(inplace=True),
            sconv(N, N, kernel_size=3, stride=1, bias=True),
            sconv(N, N, kernel_size=3, stride=2, bias=True),
            ME.MinkowskiReLU(inplace=True),
            sconv(N, N, kernel_size=3, stride=1, bias=True),
            sconv(N, M, kernel_size=3, stride=2, bias=True),
        )

    def forward(self, x):
        y = self.g_a(x)
        return y