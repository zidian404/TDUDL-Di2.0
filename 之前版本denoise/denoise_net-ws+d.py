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
        out_nc: int = 5,  # alpha, rho, gamma1, gamma2, gamma3
    ):
        super(HyPaNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_nc, nc, 1, padding=0, bias=True),
            nn.Sigmoid(),
            nn.Conv2d(nc, out_nc, 1, padding=0, bias=True),
            nn.Softplus()
        )

    def forward(self, x: Tensor):
        # x: [B,1,1,1]
        x = (x - 0.098) / 0.0566
        x = self.mlp(x) + 1e-6  # [B,5,1,1]
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

    def forward(self, y: Any, sigma: Tensor):
        # sigma: [B,1,1,1] -> [B,1,H,W]
        sigma = sigma.repeat(1, 1, y.size(2), y.size(3))
        x = self.head_x(torch.cat([y, sigma], dim=1))
        return x

##########################################################################
# DictAdapter：S ⊙ w(i) 这一支（保持不变）
##########################################################################

class DictAdapter(nn.Module):
    """
    输入：X [B, Cx, H, W]
    输出：w [B, Cx, 1, 1]，用来对 X 做逐通道缩放，相当于 S⊙w(i) 这一支。
    """
    def __init__(self, Cx: int):
        super(DictAdapter, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(Cx, Cx, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(Cx, Cx, 1, bias=True),
            nn.Sigmoid()  # (0,1)
        )

    def forward(self, X: Tensor) -> Tensor:
        g = self.pool(X)   # [B,Cx,1,1]
        w = self.fc(g)     # [B,Cx,1,1]
        return w

##########################################################################
# DeltaDGenerator：ΔD(i) 这一支（新增）
##########################################################################

class DeltaDGenerator(nn.Module):
    """
    根据当前系数 X 生成 per-image 卷积核增量 ΔD(i)
    X: [B, Cx, H, W]
    输出 ΔD: [B, Cy_delta, Cx, k, k]
    一般可以设 Cy_delta <= Cy，减少过拟合风险。
    """
    def __init__(self, Cx: int, Cy_delta: int, k: int, hidden: int = 64):
        super(DeltaDGenerator, self).__init__()
        self.Cx = Cx
        self.Cy_delta = Cy_delta
        self.k = k
        out_dim = Cy_delta * Cx * k * k

        self.mlp = nn.Sequential(
            nn.Conv2d(Cx, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_dim, 1, bias=True)
        )

    def forward(self, X: Tensor) -> Tensor:
        B, Cx, H, W = X.shape
        g = F.adaptive_avg_pool2d(X, 1)          # [B,Cx,1,1]
        z = self.mlp(g)                          # [B, Cy_delta*Cx*k*k,1,1]
        z = z.view(B, self.Cy_delta, self.Cx, self.k, self.k)
        return z

##########################################################################
# ΔD 的卷积与转置卷积
##########################################################################

def apply_DeltaD(X: Tensor, DeltaD: Tensor) -> Tensor:
    """
    X: [B, Cx, H, W]
    DeltaD: [B, Cy_delta, Cx, k, k]
    返回 Y_delta: [B, Cy_delta, H, W]
    """
    B, Cx, H, W = X.shape
    B2, Cy_delta, Cx2, k, k2 = DeltaD.shape
    assert B2 == B and Cx2 == Cx and k == k2

    patches = F.unfold(X, kernel_size=k, padding=k // 2)  # [B, Cx*k*k, N]
    patches = patches.transpose(1, 2)                     # [B, N, Cx*k*k]
    K_flat = DeltaD.view(B, Cy_delta, Cx * k * k)         # [B, Cy_delta, Cx*k*k]
    Y_flat = torch.bmm(K_flat, patches.transpose(1, 2))   # [B, Cy_delta, N]
    Y = F.fold(Y_flat, output_size=(H, W), kernel_size=1)
    return Y

def apply_DeltaD_T(Y: Tensor, DeltaD: Tensor) -> Tensor:
    """
    Y: [B, Cy_delta, H, W]
    DeltaD: [B, Cy_delta, Cx, k, k]
    返回 X_grad: [B, Cx, H, W]
    """
    B, Cy_delta, H, W = Y.shape
    B2, Cy_delta2, Cx, k, k2 = DeltaD.shape
    assert B2 == B and Cy_delta2 == Cy_delta and k == k2

    patches = F.unfold(Y, kernel_size=k, padding=k // 2)  # [B, Cy_delta*k*k, N]
    patches = patches.transpose(1, 2)                     # [B, N, Cy_delta*k*k]

    K_T = DeltaD.permute(0, 2, 1, 3, 4).contiguous()      # [B, Cx, Cy_delta, k, k]
    K_T_flat = K_T.view(B, Cx, Cy_delta * k * k)          # [B, Cx, Cy_delta*k*k]
    X_flat = torch.bmm(K_T_flat, patches.transpose(1, 2)) # [B, Cx, N]
    X_grad = F.fold(X_flat, output_size=(H, W), kernel_size=1)
    return X_grad

##########################################################################
# BodyNet：S⊙w + ΔD，Z / beta 更新保持 ADMM 形式
##########################################################################

class BodyNet(nn.Module):
    def __init__(
        self,
        unet,
        S,
        S_T,
        dict_adapter: DictAdapter,
        delta_gen: DeltaDGenerator,
        delta_scale: float = 0.1
    ):
        super(BodyNet, self).__init__()
        self.unet = unet
        self.S = S
        self.S_T = S_T
        self.dict_adapter = dict_adapter
        self.delta_gen = delta_gen
        self.delta_scale = delta_scale  # 控制 ΔD 的相对权重，避免压过 S

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

        # --- S ⊙ w 部分 ---
        w = self.dict_adapter(X)         # [B,Cx,1,1]
        X_mod = X * w                    # [B,Cx,H,W]
        Y_S = self.S(X_mod)              # [B,Cy,H,W]

        # --- ΔD 部分 ---
        DeltaD = self.delta_gen(X)       # [B, Cy_delta, Cx, k, k]
        Y_delta = apply_DeltaD(X, DeltaD)  # [B, Cy_delta, H, W]

        # 为了和 Y 的通道对齐，这里简单用 1x1 conv 做映射
        # 为简化，这里把 Cy_delta 设为 Cy，直接相加即可；否则需要一个映射层
        Y_hat = Y_S + self.delta_scale * Y_delta  # [B,Cy,H,W]

        # 残差
        res = Y_hat - Y

        # 梯度：共享字典分支
        grad_S = self.S_T(res)          # [B,Cx,H,W]
        # ΔD 分支的梯度
        grad_D = apply_DeltaD_T(self.delta_scale * res, DeltaD)  # [B,Cx,H,W]

        grad = grad_S + grad_D

        # X-step
        X_term = X - Z + beta
        X_out = X - alpha * (grad + rho * X_term)

        # Z-step（Restormer）
        rho_ = (1.0 / rho.sqrt()).repeat(1, 1, X_out.size(2), X_out.size(3))
        Z, samfeats, enc_, dec_ = self.unet(
            torch.cat([X_out, rho_], dim=1),
            samfeats, enc, dec, stage_inter=True
        )

        # beta-step
        beta = gamma[0] * beta + gamma[1] * X_out - gamma[2] * Z

        return X_out, Z, beta, samfeats, enc_, dec_

##########################################################################
# 主网络：S⊙w + ΔD
##########################################################################

class denoise_Net_admm_restormer(nn.Module):
    def __init__(self, opt):
        super(denoise_Net_admm_restormer, self).__init__()

        self.n_channels = opt["n_channels"]  # Cy
        self.d_size = opt["d_size"]
        self.stage = opt["stage"]

        self.headnet = HeadNet(self.n_channels, self.n_channels, 3)

        self.m_channels = 16  # Cx
        self.stride = 1

        self.unet = Restormer11(
            inp_channels=self.m_channels + 1,
            out_channels=self.m_channels,
            dim=self.m_channels
        )

        k = self.d_size
        Cx = self.m_channels
        Cy = self.n_channels

        # 共享字典 S / S_T
        self.S = nn.Conv2d(Cx, Cy, k, padding=k // 2, bias=True)
        self.S_T = nn.Conv2d(Cy, Cx, k, padding=k // 2, bias=True)

        # S⊙w 部分
        self.dict_adapter = DictAdapter(Cx=Cx)

        # ΔD 部分：这里设 Cy_delta = Cy，方便直接和 Y 相加
        self.delta_gen = DeltaDGenerator(Cx=Cx, Cy_delta=Cy, k=k, hidden=64)

        self.body = BodyNet(
            self.unet, self.S, self.S_T,
            self.dict_adapter, self.delta_gen,
            delta_scale=0.1  # 可以以后调参
        )

        self.hypa_list_: nn.ModuleList = nn.ModuleList()
        for _ in range(self.stage):
            self.hypa_list_.append(HyPaNet(in_nc=1, out_nc=5))

    def forward(self, input: Tensor, sigma: Tensor):
        device = input.device
        sigma = sigma.to(device)
        if sigma.dim() == 1:
            sigma = sigma.view(-1, 1, 1, 1)
        elif sigma.dim() == 2:
            sigma = sigma.view(sigma.size(0), 1, 1, 1)
        elif sigma.dim() == 3:
            sigma = sigma.view(sigma.size(0), 1, 1, 1)

        # 初始化 X^0
        X_img0 = self.headnet(input, sigma)  # [B,Cy,H,W]
        X = self.S_T(X_img0)                 # [B,Cx,H,W]

        preds = []
        Z = torch.zeros_like(X)
        beta = torch.zeros_like(X)
        samfeats = enc = dec = None

        for k in range(self.stage):
            hypas = self.hypa_list_[k](sigma)    # [B,5,1,1]
            alpha = hypas[:, 0:1, :, :]
            rho   = hypas[:, 1:2, :, :]
            gamma1 = hypas[:, 2:3, :, :]
            gamma2 = hypas[:, 3:4, :, :]
            gamma3 = hypas[:, 4:5, :, :]
            gamma = [gamma1, gamma2, gamma3]

            if k == 0:
                # 初始步可以沿用原来 S⊙w 逻辑，再加 ΔD
                w0 = self.dict_adapter(X)
                X_mod0 = X * w0
                Y_S0 = self.S(X_mod0)
                DeltaD0 = self.delta_gen(X)
                Y_delta0 = apply_DeltaD(X, DeltaD0)
                Y_hat0 = Y_S0 + 0.1 * Y_delta0

                temp_back = self.S_T(Y_hat0) - self.S_T(input)
                X2_img = self.S(temp_back * w0)
                X1_coef = self.S_T(Y_hat0)
                X2_coef = self.S_T(X2_img)

                X_ = X2_coef + rho * X1_coef
                X = X1_coef - alpha * X_

                rho_map = (1.0 / rho.sqrt()).repeat(1, 1, X.size(2), X.size(3))
                Z, samfeats, enc, dec = self.unet(
                    torch.cat([X, rho_map], dim=1),
                    stage_inter=True
                )
                beta = gamma[1] * X - gamma[2] * Z

                # 中间重构
                w_out = self.dict_adapter(X)
                Y_S_out = self.S(X * w_out)
                DeltaD_out = self.delta_gen(X)
                Y_delta_out = apply_DeltaD(X, DeltaD_out)
                output = Y_S_out + 0.1 * Y_delta_out
                preds.append(output)
            else:
                X, Z, beta, samfeats, enc, dec = self.body(
                    X, input, Z, beta, alpha, rho, gamma,
                    samfeats, enc, dec
                )
                w_out = self.dict_adapter(X)
                Y_S_out = self.S(X * w_out)
                DeltaD_out = self.delta_gen(X)
                Y_delta_out = apply_DeltaD(X, DeltaD_out)
                output = Y_S_out + 0.1 * Y_delta_out
                preds.append(output)

        # Final step
        DeltaD_final = self.delta_gen(X)
        w_final = self.dict_adapter(X)

        Y_S_final = self.S(X * w_final)
        Y_delta_final = apply_DeltaD(X, DeltaD_final)
        Y_hat_final = Y_S_final + 0.1 * Y_delta_final

        temp = Y_hat_final - input
        X_1 = self.S_T(temp) + apply_DeltaD_T(0.1 * temp, DeltaD_final)
        X_2 = rho * (X - Z - beta)
        X_out = X - alpha * (X_1 + X_2)

        w_out = self.dict_adapter(X_out)
        Y_S_out = self.S(X_out * w_out)
        DeltaD_out = self.delta_gen(X_out)
        Y_delta_out = apply_DeltaD(X_out, DeltaD_out)
        output = Y_S_out + 0.1 * Y_delta_out
        preds.append(output)

        return output, preds