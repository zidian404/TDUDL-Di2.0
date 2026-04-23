import torch
import torch.nn as nn
from typing import Any
from torch import Tensor
from Net.restormer_arch import Restormer11


class denoise_Net_admm_restormer(nn.Module):
    def __init__(self, opt):
        super(denoise_Net_admm_restormer, self).__init__()

        self.n_channels = opt["n_channels"]
        self.d_size = opt["d_size"]
        self.stage = opt["stage"]

        self.headnet = HeadNet(self.n_channels, self.n_channels, 3)

        self.m_channels = 16
        self.stride = 1

        self.D_0 = default_conv(self.n_channels, self.m_channels, self.d_size, self.stride)
        self.D_0T = default_conv(self.m_channels, self.n_channels, self.d_size, self.stride)
        self.A = nn.ModuleList([
            default_conv(self.n_channels, self.m_channels, self.d_size, self.stride)
            for _ in range(self.stage)
        ])
        self.B = nn.ModuleList([
            default_conv(self.m_channels, self.n_channels, self.d_size, self.stride)
            for _ in range(self.stage)
        ])
        self.D = default_conv(self.m_channels, self.n_channels, self.d_size, self.stride)

        self.unet = Restormer11(
            inp_channels=self.m_channels + 1,
            out_channels=self.m_channels,
            dim=self.m_channels
        )

        self.body = BodyNet(self.n_channels, self.d_size, self.A, self.B, self.unet)

        self.hypa_list_ = nn.ModuleList()
        for _ in range(self.stage):
            self.hypa_list_.append(HyPaNet(in_nc=1, out_nc=5))

    def forward(self, input, sigma=None):
        device = input.device
        b, _, _, _ = input.shape

        if sigma is None:
            sigma = torch.full((b, 1), 0.01, device=device)

        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(1)
        sigma = sigma.to(device=device, dtype=input.dtype)
        sigma = sigma.unsqueeze(-1).unsqueeze(-1)

        X = self.headnet(input, sigma)

        preds = []
        for k in range(self.stage):
            hypas = self.hypa_list_[k](sigma)
            alpha_ = hypas[:, 0].unsqueeze(-1)
            rho_ = hypas[:, 1].unsqueeze(-1)
            gamma1 = hypas[:, 2].unsqueeze(-1)
            gamma2 = hypas[:, 3].unsqueeze(-1)
            gamma3 = hypas[:, 4].unsqueeze(-1)
            gamma = [gamma1, gamma2, gamma3]
            alpha = alpha_
            rho = rho_

            if k == 0:
                X_in = X
                X1 = self.D_0(X_in)
                temp = torch.sub(self.D_0T(X1), input)
                X2 = self.D_0(temp)
                X_ = X2 + torch.mul(rho, X1)
                X = X1 - torch.mul(alpha, X_)

                rho_map = (1 / rho.sqrt()).repeat(1, 1, X.size(2), X.size(3))
                Z, samfeats, enc, dec = self.unet(
                    torch.cat([X, rho_map], dim=1),
                    stage_inter=True
                )

                beta = gamma[1] * X - gamma[2] * Z

                output = self.D(X)
                preds.append(output)

            else:
                X, Z, beta, samfeats, enc, dec = self.body(
                    X, input, Z, beta, alpha, rho, gamma, k, samfeats, enc, dec
                )
                output = self.D(X)
                preds.append(output)

        temp = torch.sub(self.B[-1](X), input)
        X_1 = self.A[-1](temp)
        X_2 = torch.mul(rho, X - Z - beta)
        X_out = X - torch.mul(alpha, X_1 + X_2)

        output = self.D(X_out)
        preds.append(output)
        return output, preds


class HeadNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, d_size: int):
        super(HeadNet, self).__init__()
        self.head_x = nn.Sequential(
            nn.Conv2d(in_channels + 1, 64, d_size, padding=(d_size - 1) // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1, bias=False)
        )

    def forward(self, y: Any, sigma):
        sigma = sigma.repeat(1, 1, y.size(2), y.size(3))
        x = self.head_x(torch.cat([y, sigma], dim=1))
        return x


class BodyNet(nn.Module):
    def __init__(self, n_channels, d_size, A, B, unet):
        super(BodyNet, self).__init__()
        self.unet = unet
        self.D = B
        self.D_T = A

    def forward(self, X_in, Y, Z, beta, alpha, rho, gamma, k, samfeats, enc, dec):
        X = X_in
        temp = torch.sub(self.D[k](X), Y)
        X_1 = self.D_T[k](temp)
        X_2 = torch.mul(rho, X - Z + beta)
        X_out = X - torch.mul(alpha, X_1 + X_2)

        rho_map = (1 / rho.sqrt()).repeat(1, 1, X.size(2), X.size(3))
        Z, samfeats, enc_, dec_ = self.unet(
            torch.cat([X_out, rho_map], dim=1),
            samfeats, enc, dec,
            stage_inter=True
        )

        beta = gamma[0] * beta + gamma[1] * X_out - gamma[2] * Z
        return X_out, Z, beta, samfeats, enc_, dec_


class HyPaNet(nn.Module):
    def __init__(self, in_nc: int = 1, nc: int = 64, out_nc: int = 8):
        super(HyPaNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_nc, nc, 1, padding=0, bias=True),
            nn.Sigmoid(),
            nn.Conv2d(nc, out_nc, 1, padding=0, bias=True),
            nn.Softplus()
        )

    def forward(self, x: Tensor):
        x = (x - 0.098) / 0.0566
        x = self.mlp(x) + 1e-6
        return x


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=(stride, stride)
    )


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=(stride, stride), bias=bias
    )


def conv_down(in_chn, out_chn, kernel_size, stride=2, bias=False):
    return nn.Conv2d(
        in_chn, out_chn, kernel_size,
        stride=(stride, stride),
        padding=(kernel_size - 1) // 2,
        bias=bias
    )


def conv_up(in_chn, out_chn, kernel_size, stride=2, bias=False):
    return nn.ConvTranspose2d(
        in_chn, out_chn, kernel_size,
        stride=(stride, stride),
        padding=(kernel_size - 1) // 2,
        output_padding=stride - 1,
        bias=bias
    )