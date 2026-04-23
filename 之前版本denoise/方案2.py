import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any
from Net.restormer_arch import Restormer11


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,
                     padding=(kernel_size // 2), stride=(stride, stride), bias=bias)


class HyPaNet(nn.Module):
    def __init__(self, in_nc=1, nc=64, out_nc=5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_nc, nc, 1, bias=True),
            nn.Sigmoid(),
            nn.Conv2d(nc, out_nc, 1, bias=True),
            nn.Softplus()
        )

    def forward(self, x):
        x = (x - 0.098) / 0.0566
        return self.mlp(x) + 1e-6


class HeadNet(nn.Module):
    def __init__(self, in_channels, out_channels, d_size):
        super().__init__()
        self.head_x = nn.Sequential(
            nn.Conv2d(in_channels + 1, 64, d_size, padding=(d_size - 1)//2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1, bias=False)
        )

    def forward(self, y, sigma):
        sigma = sigma.repeat(1, 1, y.size(2), y.size(3))
        return self.head_x(torch.cat([y, sigma], dim=1))


class DictAdapter(nn.Module):
    def __init__(self, Cx):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(Cx, Cx, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(Cx, Cx, 1),
            nn.Sigmoid()
        )

    def forward(self, X):
        return self.fc(self.pool(X))


class HighFreqExtractor(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        kernel = torch.tensor([[1, 2, 1],
                               [2, 4, 2],
                               [1, 2, 1]], dtype=torch.float32) / 16.0
        weight = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.blur = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.blur.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        low = self.blur(x)
        return x - low


class HFGuidance(nn.Module):
    def __init__(self, x_channels=16, out_channels=16):
        super().__init__()
        self.hf = HighFreqExtractor(channels=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(x_channels + 1, out_channels, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True)
        )

    def forward(self, x):
        g = x.mean(dim=1, keepdim=True)
        hf = self.hf(g)
        return self.fuse(torch.cat([x, hf], dim=1))


class BodyNet(nn.Module):
    def __init__(self, unet, S, S_T, dict_adapter, hf_guidance):
        super().__init__()
        self.unet = unet
        self.S = S
        self.S_T = S_T
        self.dict_adapter = dict_adapter
        self.hf_guidance = hf_guidance

    def forward(self, X_in, Y, Z, beta, alpha, rho, gamma, samfeats, enc, dec):
        X = X_in
        w = self.dict_adapter(X)
        X_mod = X * w

        Y_hat = self.S(X_mod)
        res = Y_hat - Y
        grad = self.S_T(res) * w

        X_out = X - alpha * (grad + rho * (X - Z + beta))

        rho_ = (1 / rho.sqrt()).repeat(1, 1, X_out.size(2), X_out.size(3))
        hf_feat = self.hf_guidance(X_out)
        Z, samfeats, enc_, dec_ = self.unet(torch.cat([hf_feat, rho_], dim=1), samfeats, enc, dec, stage_inter=True)

        beta = gamma[0] * beta + gamma[1] * X_out - gamma[2] * Z
        return X_out, Z, beta, samfeats, enc_, dec_


class denoise_Net_admm_restormer_hf(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.n_channels = opt["n_channels"]
        self.d_size = opt["d_size"]
        self.stage = opt["stage"]

        self.headnet = HeadNet(self.n_channels, self.n_channels, 3)
        self.m_channels = 16

        self.unet = Restormer11(inp_channels=self.m_channels + 1, out_channels=self.m_channels, dim=self.m_channels)

        k = self.d_size
        Cx = self.m_channels
        Cy = self.n_channels
        self.S = nn.Conv2d(Cx, Cy, k, padding=k // 2, bias=True)
        self.S_T = nn.Conv2d(Cy, Cx, k, padding=k // 2, bias=True)

        self.dict_adapter = DictAdapter(Cx)
        self.hf_guidance = HFGuidance(Cx, Cx)
        self.body = BodyNet(self.unet, self.S, self.S_T, self.dict_adapter, self.hf_guidance)

        self.hypa_list_ = nn.ModuleList([HyPaNet(in_nc=1, out_nc=5) for _ in range(self.stage)])

    def forward(self, input, sigma):
        device = input.device
        sigma = sigma.to(device)
        if sigma.dim() == 1:
            sigma = sigma.view(-1, 1, 1, 1)
        elif sigma.dim() != 4:
            sigma = sigma.view(sigma.size(0), 1, 1, 1)

        X_img0 = self.headnet(input, sigma)
        X = self.S_T(X_img0)

        Z = torch.zeros_like(X)
        beta = torch.zeros_like(X)
        preds = []
        samfeats = enc = dec = None

        for k in range(self.stage):
            hypas = self.hypa_list_[k](sigma)
            alpha = hypas[:, 0:1]
            rho = hypas[:, 1:2]
            gamma = [hypas[:, 2:3], hypas[:, 3:4], hypas[:, 4:5]]

            if k == 0:
                w = self.dict_adapter(X)
                X_mod = X * w
                X1_img = self.S(X_mod)
                temp_back = self.S_T(X1_img) - self.S_T(input)
                X2_img = self.S(temp_back * w)
                X1_coef = self.S_T(X1_img)
                X2_coef = self.S_T(X2_img)
                X_ = X2_coef + rho * X1_coef
                X = X1_coef - alpha * X_

                rho_map = (1 / rho.sqrt()).repeat(1, 1, X.size(2), X.size(3))
                hf_feat = self.hf_guidance(X)
                Z, samfeats, enc, dec = self.unet(torch.cat([hf_feat, rho_map], dim=1), samfeats, enc, dec, stage_inter=True)
                beta = gamma[1] * X - gamma[2] * Z

                output = self.S(X) + (self.S(X * self.dict_adapter(X)) - self.S(X))
                preds.append(output)
            else:
                X, Z, beta, samfeats, enc, dec = self.body(X, input, Z, beta, alpha, rho, gamma, samfeats, enc, dec)
                output = self.S(X) + (self.S(X * self.dict_adapter(X)) - self.S(X))
                preds.append(output)

        w_final = self.dict_adapter(X)
        Y_S_final = self.S(X * w_final)
        temp = Y_S_final - input
        X_1 = self.S_T(temp) * (1.0 + w_final)
        X_2 = rho * (X - Z - beta)
        X_out = X - alpha * (X_1 + X_2)

        output = self.S(X_out) + (self.S(X_out * self.dict_adapter(X_out)) - self.S(X_out))
        preds.append(output)
        return output, preds