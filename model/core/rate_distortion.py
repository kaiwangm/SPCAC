import math
import torch
import torch.nn as nn

from model.core.utils import sort_sparse_tensor, make_sparse_tensor
import numpy as np
from kornia.color import rgb_to_ycbcr


def rgb2yuv(rgb):
    # input is n*3 -> (1, 3, n, 1)
    rgb_ = torch.einsum('nc->cn', rgb)
    rgb_ = rgb_.unsqueeze(0)
    rgb_ = rgb_.unsqueeze(3)

    yuv_ = rgb_to_ycbcr(rgb_)

    yuv = yuv_.squeeze(3)
    yuv = yuv.squeeze(0)
    yuv = torch.einsum('cn->nc', yuv)

    return yuv


class rate_distortion_loss(torch.nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lam, attributes='RGB', metric='mse'):
        super().__init__()
        self.mse = nn.MSELoss()

        self.metric = metric
        self.lam = lam
        self.attributes = attributes

    def forward(self, output):
        N, C = output['x'].F.size()
        out = {}

        y_likelihoods = torch.cat(
            [likelihood for k, likelihood in output['likelihoods'].items() if k.startswith('y')], dim=0)

        loss_R_y = torch.log(y_likelihoods).sum() / (-math.log(2) * N)
        out['loss_R_y'] = loss_R_y
        if 'zl' in output['likelihoods'].keys():
            zl_likelihoods = output['likelihoods']['zl']
            loss_R_zl = torch.log2(zl_likelihoods).sum() / (-math.log(2) * N)
            out['loss_R_z'] = loss_R_zl
            loss_R = loss_R_y + loss_R_zl

            if 'zg' in output['likelihoods'].keys():
                zg_likelihoods = output['likelihoods']['zg']
                loss_R_zg = torch.log2(
                    zg_likelihoods).sum() / (-math.log(2) * N)
                out['loss_R_zg'] = loss_R_zg
                loss_R = loss_R + loss_R_zg
        else:
            loss_R = loss_R_y

        x = sort_sparse_tensor(output['x'])
        x_hat = sort_sparse_tensor(output['x_hat'])

        if self.attributes == 'RGB':
            rgb = x.F
            rgb_hat = x_hat.F

            yuv = rgb2yuv(rgb)
            yuv_hat = rgb2yuv(rgb_hat)

            loss_D = self.mse(yuv_hat[:, 0], yuv[:, 0]) + self.mse(
                yuv_hat[:, 1], yuv[:, 1]) + self.mse(yuv_hat[:, 2], yuv[:, 2])
            loss = self.lam * 255**2 * loss_D + loss_R
        elif self.attributes == 'intensity':
            intensity = x.F
            intensity_hat = x_hat.F

            loss_D = self.mse(intensity_hat, intensity)
            loss = self.lam * 255**2 * loss_D + loss_R

        out['loss_R'] = loss_R
        out['loss_D'] = loss_D
        out['loss'] = loss

        psnr = 10.0 * torch.log10((1.0 * 1.0) / loss_D)
        out['psnr'] = psnr

        return out

    def evaluate(self, output):
        N, C = output['x'].F.size()
        out = {}

        x = sort_sparse_tensor(output['x'])
        x_hat = sort_sparse_tensor(output['x_hat'])

        x_hat = make_sparse_tensor(
            coordinates=x_hat.C,
            features=torch.clip(x_hat.F, 0.0, 1.0),
            tensor_stride=1,
            dimension=3,
            device=x_hat.device,
        )

        if self.attributes == 'RGB':
            rgb = x.F
            rgb_hat = x_hat.F

            yuv = rgb2yuv(rgb)
            yuv_hat = rgb2yuv(rgb_hat)

            loss_D_yuv = (4.0 * self.mse(yuv_hat[:, 0], yuv[:, 0]) + self.mse(yuv_hat[:, 1], yuv[:, 1]) + self.mse(yuv_hat[:, 2], yuv[:, 2])) / 6.0
            psnr_yuv = 10.0 * torch.log10((1.0 * 1.0) / loss_D_yuv)
            out['psnr_yuv'] = psnr_yuv

            loss_D_y = self.mse(yuv_hat[:, 0], yuv[:, 0])
            psnr_y = 10.0 * torch.log10((1.0 * 1.0) / loss_D_y)
            out['psnr_y'] = psnr_y
        elif self.attributes == 'intensity':
            intensity = x.F
            intensity_hat = x_hat.F

            loss_D = self.mse(intensity_hat, intensity)
            psnr = 10.0 * torch.log10((1.0 * 1.0) / loss_D)
            out['psnr'] = psnr

        bpp = 0.0
        if 'likelihoods' in output.keys():
            bpp = sum(torch.log(likelihood).sum() / (-math.log(2) * N)
                      for likelihood in output['likelihoods'].values())
        elif 'strings' in output.keys():
            bpp = sum([len(string[0][0])
                       for string in output['strings'].values()]) * 8.0 / N

        out['bpp'] = bpp

        return out


class distortion_loss(nn.Module):
    """Distortion loss."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output):
        out = {}

        x = sort_sparse_tensor(output['x'])
        x_hat = sort_sparse_tensor(output['x_hat'])

        loss_D = self.mse(x_hat.F, x.F)
        loss = 255**2 * loss_D

        out['loss_D'] = loss_D
        out['loss'] = loss

        psnr = 10.0 * torch.log10((1.0 * 1.0) / loss_D)
        out['psnr'] = psnr

        return out
