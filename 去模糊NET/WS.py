import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any

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


##########################################################################
# HyPaNet
##########################################################################

class HyPaNet(nn.Module):
    def __init__(
        self,
        in_nc: int = 1,
        nc: int = 64,
        out_nc: int = 5,
    ):
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
            nn.Conv2d(
                in_channels + 1, 64, d_size,
                padding=(d_size - 1) // 2, bias=False
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1, bias=False)
        )

    def forward(self, y: Any, blur_level: Tensor):
        blur_level = blur_level.repeat(1, 1, y.size(2), y.size(3))
        x = self.head_x(torch.cat([y, blur_level], dim=1))
        return x


##########################################################################
# DictAdapter
##########################################################################

class DictAdapter(nn.Module):
    def __init__(self, Cx: int):
        super(DictAdapter, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(Cx, Cx, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(Cx, Cx, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, X: Tensor) -> Tensor:
        g = self.pool(X)
        w = self.fc(g)
        return w


##########################################################################
# BodyNet
##########################################################################

class BodyNet(nn.Module):
    def __init__(self, unet, S, S_T, dict_adapter: DictAdapter):
        super(BodyNet, self).__init__()
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
        gamma: list,
        samfeats,
        enc,
        dec
    ):
        X = X_in

        w = self.dict_adapter(X)
        X_mod = X * w

        Y_hat = self.S(X_mod)
        res = Y_hat - Y

        grad_S = self.S_T(res)
        grad_D = grad_S * w
        grad = grad_S + grad_D

        X_term = X - Z + beta
        X_out = X - alpha * (grad + rho * X_term)

        rho_ = (1 / rho.sqrt()).repeat(1, 1, X_out.size(2), X_out.size(3))
        Z, samfeats, enc_, dec_ = self.unet(
            torch.cat([X_out, rho_], dim=1),
            samfeats, enc, dec, stage_inter=True
        )

        beta = gamma[0] * beta + gamma[1] * X_out - gamma[2] * Z

        return X_out, Z, beta, samfeats, enc_, dec_


##########################################################################
# Main network
##########################################################################

class denoise_Net_admm_restormer(nn.Module):
    def __init__(self, opt):
        super(denoise_Net_admm_restormer, self).__init__()

        self.n_channels = opt["n_channels"]
        self.d_size = opt["d_size"]
        self.stage = opt["stage"]

        self.headnet = HeadNet(self.n_channels, self.n_channels, 3)

        self.m_channels = 16
        self.stride = 1

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

        self.dict_adapter = DictAdapter(Cx=Cx)
        self.body = BodyNet(self.unet, self.S, self.S_T, self.dict_adapter)

        self.hypa_list_ = nn.ModuleList()
        for _ in range(self.stage):
            self.hypa_list_.append(HyPaNet(in_nc=1, out_nc=5))

    def forward(self, input: Tensor, blur_level: Tensor = None):
        device = input.device
        b = input.size(0)

        if blur_level is None:
            blur_level = torch.full((b, 1, 1, 1), 0.01, device=device, dtype=input.dtype)
        else:
            blur_level = blur_level.to(device=device, dtype=input.dtype)
            if blur_level.dim() == 1:
                blur_level = blur_level.view(-1, 1, 1, 1)
            elif blur_level.dim() == 2:
                blur_level = blur_level.view(blur_level.size(0), 1, 1, 1)
            elif blur_level.dim() == 3:
                blur_level = blur_level.view(blur_level.size(0), 1, 1, 1)

        X_img0 = self.headnet(input, blur_level)
        X = self.S_T(X_img0)

        preds = []
        Z = torch.zeros_like(X)
        beta = torch.zeros_like(X)

        samfeats = enc = dec = None

        for k in range(self.stage):
            hypas = self.hypa_list_[k](blur_level)
            alpha = hypas[:, 0:1, :, :]
            rho = hypas[:, 1:2, :, :]
            gamma1 = hypas[:, 2:3, :, :]
            gamma2 = hypas[:, 3:4, :, :]
            gamma3 = hypas[:, 4:5, :, :]
            gamma = [gamma1, gamma2, gamma3]

            if k == 0:
                w = self.dict_adapter(X)
                X_mod = X * w

                X1_img = self.S(X_mod)
                temp_back = self.S_T(X1_img) - self.S_T(input)

                temp_mod = temp_back * w
                X2_img = self.S(temp_mod)

                X1_coef = self.S_T(X1_img)
                X2_coef = self.S_T(X2_img)

                X_ = X2_coef + rho * X1_coef
                X = X1_coef - alpha * X_

                rho_map = (1 / rho.sqrt()).repeat(1, 1, X.size(2), X.size(3))
                Z, samfeats, enc, dec = self.unet(
                    torch.cat([X, rho_map], dim=1),
                    stage_inter=True
                )

                beta = gamma[1] * X - gamma[2] * Z

                w_out = self.dict_adapter(X)
                Y_S_out = self.S(X)
                Y_D_out = self.S(X * w_out) - Y_S_out
                output = Y_S_out + Y_D_out
                preds.append(output)

            else:
                X, Z, beta, samfeats, enc, dec = self.body(
                    X, input, Z, beta, alpha, rho, gamma,
                    samfeats, enc, dec
                )

                w_out = self.dict_adapter(X)
                Y_S_out = self.S(X)
                Y_D_out = self.S(X * w_out) - Y_S_out
                output = Y_S_out + Y_D_out
                preds.append(output)

        w_final = self.dict_adapter(X)
        Y_S_final = self.S(X * w_final)
        temp = Y_S_final - input

        X_1 = self.S_T(temp) * (1.0 + w_final)
        X_2 = rho * (X - Z - beta)
        X_out = X - alpha * (X_1 + X_2)

        w_out = self.dict_adapter(X_out)
        Y_S_out = self.S(X_out)
        Y_D_out = self.S(X_out * w_out) - Y_S_out
        output = Y_S_out + Y_D_out
        preds.append(output)

        return output, preds


##########################################################################
# ST
##########################################################################

class ST(nn.Module):
    def __init__(self):
        super(ST, self).__init__()

    def forward(self, x, t, samfeats=None, enc_in=None, dec_in=None):
        return x.sign() * F.relu(x.abs() - t), samfeats, enc_in, dec_in