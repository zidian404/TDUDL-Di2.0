import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, List

from Net.restormer_arch import Restormer11


##########################################################################
# Basic modules
##########################################################################

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


##########################################################################
# HyPaNet
##########################################################################

class HyPaNet(nn.Module):
    def __init__(self, in_nc: int = 1, nc: int = 64, out_nc: int = 5):
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


##########################################################################
# HeadNet
##########################################################################

class HeadNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, d_size: int):
        super(HeadNet, self).__init__()
        self.head_x = nn.Sequential(
            nn.Conv2d(in_channels + 1, 64, d_size, padding=(d_size - 1) // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1, bias=False)
        )

    def forward(self, y: Any, sigma: Tensor):
        sigma = sigma.repeat(1, 1, y.size(2), y.size(3))
        x = self.head_x(torch.cat([y, sigma], dim=1))
        return x


##########################################################################
# Orientation-aware DictAdapterV2
##########################################################################

class OrientationBlock(nn.Module):
    """
    轻量方向感知分支：
    用多尺度 depthwise conv 抽取局部方向/纹理响应。
    """
    def __init__(self, channels: int):
        super(OrientationBlock, self).__init__()
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=True)
        self.dw5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=True)
        self.pw = nn.Conv2d(channels * 2, channels, 1, bias=True)
        self.act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: Tensor):
        f3 = self.dw3(x)
        f5 = self.dw5(x)
        f = self.pw(torch.cat([f3, f5], dim=1))
        f = self.act(f)
        g = self.pool(f)
        return g


class DictAdapterV2(nn.Module):
    """
    M2 核心：
    全局通道统计 + 方向感知特征 -> 生成样本特异性通道权重 w
    """
    def __init__(self, Cx: int, reduction: int = 4):
        super(DictAdapterV2, self).__init__()
        hidden = max(Cx // reduction, 4)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.ori = OrientationBlock(Cx)

        self.fc = nn.Sequential(
            nn.Conv2d(Cx * 2, hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, Cx, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, X: Tensor) -> Tensor:
        g_global = self.pool(X)
        g_orient = self.ori(X)
        w = self.fc(torch.cat([g_global, g_orient], dim=1))
        return w


##########################################################################
# BodyNet for M2
##########################################################################

class BodyNetM2(nn.Module):
    def __init__(self, unet, S, S_T, dict_adapter: nn.Module):
        super(BodyNetM2, self).__init__()
        self.unet = unet
        self.S = S
        self.S_T = S_T
        self.dict_adapter = dict_adapter

    def forward(
        self,
        X_in: Tensor,
        Y: Tensor,
        Z: Tensor,
        beta: Tensor,
        alpha: Tensor,
        rho: Tensor,
        gamma: List[Tensor],
        samfeats,
        enc,
        dec
    ):
        X = X_in

        # w^{k-1} = DictAdapterV2(X^{k-1})
        w = self.dict_adapter(X)

        # X-step
        # grad_data = w ⊙ S^T( S(w ⊙ X) - Y )
        Y_hat = self.S(X * w)
        res = Y_hat - Y
        grad_data = self.S_T(res) * w

        grad_aug = rho * (X - Z + beta)
        X_out = X - alpha * (grad_data + grad_aug)

        # Z-step: learned proximal / Restormer
        rho_map = (1.0 / torch.sqrt(rho)).repeat(1, 1, X_out.size(2), X_out.size(3))
        Z_out, samfeats, enc_, dec_ = self.unet(
            torch.cat([X_out, rho_map], dim=1),
            samfeats, enc, dec, stage_inter=True
        )

        # beta-step
        beta_out = gamma[0] * beta + gamma[1] * X_out - gamma[2] * Z_out

        return X_out, Z_out, beta_out, samfeats, enc_, dec_


##########################################################################
# Main network: M2 = Baseline + DictAdapterV2
##########################################################################

class denoise_Net_admm_restormer_M2(nn.Module):
    def __init__(self, opt):
        super(denoise_Net_admm_restormer_M2, self).__init__()

        self.n_channels = opt["n_channels"]
        self.d_size = opt["d_size"]
        self.stage = opt["stage"]

        self.headnet = HeadNet(self.n_channels, self.n_channels, 3)

        self.m_channels = 16

        self.unet = Restormer11(
            inp_channels=self.m_channels + 1,
            out_channels=self.m_channels,
            dim=self.m_channels
        )

        k = self.d_size
        Cx = self.m_channels
        Cy = self.n_channels

        self.S = nn.Conv2d(Cx, Cy, k, padding=k // 2, bias=True)
        self.S_T = nn.Conv2d(Cy, Cx, k, padding=k // 2, bias=True)

        self.dict_adapter = DictAdapterV2(Cx=Cx)
        self.body = BodyNetM2(self.unet, self.S, self.S_T, self.dict_adapter)

        self.hypa_list_ = nn.ModuleList([
            HyPaNet(in_nc=1, out_nc=5) for _ in range(self.stage)
        ])

    def _format_sigma(self, sigma: Tensor, device):
        sigma = sigma.to(device)
        if sigma.dim() == 1:
            sigma = sigma.view(-1, 1, 1, 1)
        elif sigma.dim() == 2:
            sigma = sigma.view(sigma.size(0), 1, 1, 1)
        elif sigma.dim() == 3:
            sigma = sigma.view(sigma.size(0), 1, 1, 1)
        return sigma

    def reconstruct(self, X: Tensor) -> Tensor:
        w = self.dict_adapter(X)
        return self.S(X * w)

    def forward(self, input: Tensor, sigma: Tensor):
        device = input.device
        sigma = self._format_sigma(sigma, device)

        # initialization
        X_img0 = self.headnet(input, sigma)
        X = self.S_T(X_img0)

        Z = torch.zeros_like(X)
        beta = torch.zeros_like(X)

        preds = []
        samfeats = enc = dec = None

        for k in range(self.stage):
            hypas = self.hypa_list_[k](sigma)
            alpha = hypas[:, 0:1, :, :]
            rho   = hypas[:, 1:2, :, :]
            gamma1 = hypas[:, 2:3, :, :]
            gamma2 = hypas[:, 3:4, :, :]
            gamma3 = hypas[:, 4:5, :, :]
            gamma = [gamma1, gamma2, gamma3]

            X, Z, beta, samfeats, enc, dec = self.body(
                X, input, Z, beta, alpha, rho, gamma,
                samfeats, enc, dec
            )

            output_k = self.reconstruct(X)
            preds.append(output_k)

        output = self.reconstruct(X)
        preds.append(output)

        return output, preds


##########################################################################
# Optional ST
##########################################################################

class ST(nn.Module):
    def __init__(self):
        super(ST, self).__init__()

    def forward(self, x, t, samfeats=None, enc_in=None, dec_in=None):
        return x.sign() * F.relu(x.abs() - t), samfeats, enc_in, dec_in