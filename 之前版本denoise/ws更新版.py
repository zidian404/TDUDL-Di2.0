import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, List, Tuple

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
        out_nc: int = 5,  # 输出 5 个超参数：alpha, rho, gamma1, gamma2, gamma3
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
# DictAdapter：动态生成通道权重 w
##########################################################################

class DictAdapter(nn.Module):
    """
    给共享字典 S / S_T 提供 per-image 的通道调制权重，相当于隐式的特异性字典。
    输入：X (系数域) [B, Cx, H, W]
    输出：w [B, Cx, 1, 1]，用于对 X 做逐通道缩放。
    """
    def __init__(self, Cx: int):
        super(DictAdapter, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(Cx, Cx, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(Cx, Cx, 1, bias=True),
            nn.Sigmoid()        # 限制在 (0,1) 区间
        )

    def forward(self, X: Tensor) -> Tensor:
        g = self.pool(X)    # [B,Cx,1,1]
        w = self.fc(g)      # [B,Cx,1,1]
        return w


##########################################################################
# BodyNet：与基础ADMM版本结构一一对应，但添加了w调制
##########################################################################

class BodyNet(nn.Module):
    def __init__(self, unet, S, S_T, dict_adapter: DictAdapter):
        super(BodyNet, self).__init__()
        self.unet = unet
        self.S = S          # 图像域字典
        self.S_T = S_T      # 系数域字典
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
        """
        与基础 ADMM 版本结构一一对应：
        X_step: X_out = X - alpha * (grad + rho * (X - Z + beta))
        Z_step: Z = unet(X_out, rho)
        beta_step: beta = gamma[0] * beta + gamma[1] * X_out - gamma[2] * Z
        
        区别：添加了 w 调制，使得 S 和 S_T 具有特异性
        """
        # === 动态适配器：w(X) ===
        w = self.dict_adapter(X_in)        # [B,Cx,1,1]
        X_mod = X_in * w                   # [B,Cx,H,W] 每图特异性调整

        # ---- X-step: 对应基础版的 res = S(X) - Y ----
        # 基础版: res = self.S(X_in) - Y
        # 调制版: res = self.S(X_mod) - Y
        res = self.S(X_mod) - Y            # [B,Cy,H,W]

        # 基础版: grad = self.S_T(res)
        # 调制版: grad = self.S_T(res) + 调制项（模拟 Di^T 作用）
        grad_S = self.S_T(res)             # [B,Cx,H,W]
        grad_D = grad_S * w                # 用同一个 w 调制，模拟特异性反向投影
        grad = grad_S + grad_D

        # X_term = X - Z + beta (与基础版完全一致)
        X_term = X_in - Z + beta

        # X_out = X - alpha * (grad + rho * X_term)
        X_out = X_in - alpha * (grad + rho * X_term)

        # ---- Z-step（与基础版完全一致）----
        Z_input = X_out + beta
        rho_map = (1 / rho.sqrt()).repeat(1, 1, X_out.size(2), X_out.size(3))
        Z, samfeats, enc_, dec_ = self.unet(
            torch.cat([Z_input, rho_map], dim=1),
            samfeats, enc, dec, stage_inter=True
        )

        # ---- beta-step（与基础版完全一致）----
        beta = gamma[0] * beta + gamma[1] * X_out - gamma[2] * Z

        return X_out, Z, beta, samfeats, enc_, dec_


##########################################################################
# 主网络：共享 S / S_T + 动态 DictAdapter
##########################################################################

class denoise_Net_admm_restormer(nn.Module):
    def __init__(self, opt):
        super(denoise_Net_admm_restormer, self).__init__()

        self.n_channels = opt["n_channels"]   # 图像域通道 Cy
        self.d_size = opt["d_size"]
        self.stage = opt["stage"]

        # HeadNet：从 (Y, sigma) 初始化系数域 X
        self.headnet = HeadNet(self.n_channels, self.n_channels, 3)

        # 系数域通道 Cx
        self.m_channels = 16
        self.stride = 1

        # Restormer (Z-step): 输入 m_channels+1 -> 输出 m_channels
        self.unet = Restormer11(
            inp_channels=self.m_channels + 1,  # 16 + 1 (rho_map)
            out_channels=self.m_channels,      # 16
            dim=self.m_channels
        )

        # ---- 共享字典 S / S_T ----
        k = self.d_size
        Cx = self.m_channels   # 系数域
        Cy = self.n_channels   # 图像域

        # S: 系数域 X[Cx] -> 图像域 Y[Cy]
        self.S = nn.Conv2d(Cx, Cy, k, padding=k // 2, bias=True)
        # S_T: 图像域 Y[Cy] -> 系数域 X[Cx]
        self.S_T = nn.Conv2d(Cy, Cx, k, padding=k // 2, bias=True)

        # 动态特异性适配器（per-image）
        self.dict_adapter = DictAdapter(Cx=Cx)

        self.body = BodyNet(self.unet, self.S, self.S_T, self.dict_adapter)

        # 每个 stage 的 HyPaNet
        self.hypa_list_: nn.ModuleList = nn.ModuleList()
        for _ in range(self.stage):
            self.hypa_list_.append(HyPaNet(in_nc=1, out_nc=5))

    def forward(self, input: Tensor, sigma: Tensor):
        """
        input: [B,Cy,H,W] 噪声图像 Y
        sigma: [B] / [B,1] / [B,1,1] / [B,1,1,1] 噪声标准差
        """
        device = input.device

        # sigma -> [B,1,1,1]
        sigma = sigma.to(device)
        if sigma.dim() == 1:
            sigma = sigma.view(-1, 1, 1, 1)
        elif sigma.dim() == 2:
            sigma = sigma.view(sigma.size(0), 1, 1, 1)
        elif sigma.dim() == 3:
            sigma = sigma.view(sigma.size(0), 1, 1, 1)

        # 初始化 X^0：
        # 先用 HeadNet 得到一个图像域特征，再用 S_T 映射到系数域
        X_img0 = self.headnet(input, sigma)   # [B,Cy,H,W]
        X = self.S_T(X_img0)                  # [B,Cx,H,W] 作为 X^0

        preds = []
        Z = torch.zeros_like(X)
        beta = torch.zeros_like(X)

        samfeats = enc = dec = None

        for k in range(self.stage):
            # HyPaNet: [B,5,1,1]
            hypas = self.hypa_list_[k](sigma)
            alpha = hypas[:, 0:1, :, :]   # [B,1,1,1]
            rho   = hypas[:, 1:2, :, :]
            gamma1 = hypas[:, 2:3, :, :]
            gamma2 = hypas[:, 3:4, :, :]
            gamma3 = hypas[:, 4:5, :, :]
            gamma = [gamma1, gamma2, gamma3]

            if k == 0:
                # ========== k==0：与基础ADMM版本结构一一对应，但添加w调制 ==========
                # 基础版对应关系：
                #   X1 = D_0(X)      -> X1_img = S(X_mod)
                #   temp = D_0T(X1) - input -> temp = S_T(X1_img) - input
                #   X2 = D_0(temp)   -> X2_img = S(temp_mod)
                #   X_ = X2 + rho * X1 -> X_ = X2_coef + rho * X1_coef
                #   X = X1 - alpha * X_ -> X = X1_coef - alpha * X_
                
                # 动态适配器
                w = self.dict_adapter(X)       # [B,Cx,1,1]
                X_mod = X * w                  # [B,Cx,H,W]
                
                # X1 = S(X_mod)  (图像域)
                X1_img = self.S(X_mod)         # [B,Cy,H,W]
                
                # temp = S_T(X1_img) - input  (注意：基础版是 D_0T(X1) - input)
                # 这里 input 是 Y，所以 temp 是在图像域的残差
                temp = X1_img - input          # [B,Cy,H,W]
                
                # X2_img = S(temp_mod)，其中 temp_mod = S_T(temp) * w
                # 先投影回系数域，再用 w 调制
                temp_coef = self.S_T(temp)     # [B,Cx,H,W]
                temp_mod = temp_coef * w       # 调制
                X2_img = self.S(temp_mod)      # [B,Cy,H,W]
                
                # 转回系数域
                X1_coef = self.S_T(X1_img)     # [B,Cx,H,W]
                X2_coef = self.S_T(X2_img)     # [B,Cx,H,W]
                
                # X_ = X2_coef + rho * X1_coef
                X_ = X2_coef + rho * X1_img
                
                # X = X1_coef - alpha * X_
                X = X1_coef - alpha * X_
                
                # Z-step (与基础版完全一致)
                Z_input=X + beta
                rho_map = (1 / rho.sqrt()).repeat(1, 1, X.size(2), X.size(3))
                Z, samfeats, enc, dec = self.unet(
                    torch.cat([Z_input, rho_map], dim=1),
                    stage_inter=True
                )
                
                # beta-step (与基础版完全一致)
                beta = gamma[1] * X - gamma[2] * Z
                
                # 中间输出重构
                w_out = self.dict_adapter(X)
                output = self.S(X * w_out)
                preds.append(output)
                
            else:
                # 其余阶段用 BodyNet（内部已用 S + 动态适配器）
                X, Z, beta, samfeats, enc, dec = self.body(
                    X, input, Z, beta, alpha, rho,
                    gamma, samfeats, enc, dec
                )
                
                # 中间输出重构
                w_out = self.dict_adapter(X)
                output = self.S(X * w_out)
                preds.append(output)

        # ========== FINAL STEP：与基础ADMM版本结构一一对应 ==========
        # 基础版:
        #   temp = B[-1](X) - input
        #   X_1 = A[-1](temp)
        #   X_2 = rho * (X - Z - beta)
        #   X_out = X - alpha * (X_1 + X_2)
        #
        # 调制版对应:
        #   temp = S(X * w) - input
        #   X_1 = S_T(temp) * (1 + w)
        #   X_2 = rho * (X - Z - beta)
        #   X_out = X - alpha * (X_1 + X_2)
        
        w_final = self.dict_adapter(X)
        
        # temp = S(X * w) - input
        temp = self.S(X * w_final) - input     # [B,Cy,H,W]
        
        # X_1 = S_T(temp) * (1 + w)  模拟 (S_T + Di_T) 的作用
        X_1 = self.S_T(temp) * (1.0 + w_final)  # [B,Cx,H,W]
        
        # X_2 = rho * (X - Z - beta)  注意这里是减 beta (与基础版一致)
        X_2 = rho * (X - Z  + beta)
        
        # X_out = X - alpha * (X_1 + X_2)
        X_out = X - alpha * (X_1 + X_2)
        
        # 最终重构
        w_out = self.dict_adapter(X_out)
        output = self.S(X_out * w_out)
        preds.append(output)

        return output, preds


##########################################################################
# ST（保留接口，不在当前结构中使用）
##########################################################################

class ST(nn.Module):
    def __init__(self):
        super(ST, self).__init__()

    def forward(self, x, t, samfeats=None, enc_in=None, dec_in=None):
        return x.sign() * F.relu(x.abs() - t), samfeats, enc_in, dec_in