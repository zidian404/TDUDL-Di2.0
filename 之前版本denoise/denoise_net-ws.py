import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, List, Tuple

# 注意：请确保你的项目目录下有 Net/restormer_arch.py 且包含 Restormer11 类
from Net.restormer_arch import Restormer11 

##########################################################################
# 1. 基础卷积工具函数 (Basic Modules)
##########################################################################

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=(stride, stride))

def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=(stride, stride), bias=bias)

def conv_down(in_chn, out_chn, kernel_size, stride=2, bias=False):
    return nn.Conv2d(in_chn, out_chn, kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=bias)

def conv_up(in_chn, out_chn, kernel_size, stride=2, bias=False):
    return nn.ConvTranspose2d(in_chn, out_chn, kernel_size, stride=stride, 
                              padding=(kernel_size - 1) // 2, output_padding=stride-1, bias=bias)

class ST(nn.Module):
    """ 软阈值算子 (Shrinkage-Thresholding) """
    def __init__(self):
        super(ST, self).__init__()
    def forward(self, x, t, samfeats=None, enc_in=None, dec_in=None):
        return x.sign() * F.relu(x.abs() - t), samfeats, enc_in, dec_in

##########################################################################
# 2. 矩阵运算算子 (针对 Batch 维度的 Di 卷积)
##########################################################################

def apply_Di(X, D_i):
    """ X: [B, Cx, H, W], D_i: [B, Cy, Cx, k, k] -> [B, Cy, H, W] """
    B, Cx, H, W = X.shape
    _, Cy, _, k, _ = D_i.shape
    patches = F.unfold(X, kernel_size=k, padding=k // 2)
    patches = patches.transpose(1, 2)
    K_flat = D_i.view(B, Cy, Cx * k * k)
    Y_flat = torch.bmm(K_flat, patches.transpose(1, 2))
    return F.fold(Y_flat, output_size=(H, W), kernel_size=1)

def apply_Di_T(Y, D_i):
    """ Y: [B, Cy, H, W], D_i: [B, Cy, Cx, k, k] -> [B, Cx, H, W] """
    B, Cy, H, W = Y.shape
    _, _, Cx, k, _ = D_i.shape
    patches = F.unfold(Y, kernel_size=k, padding=k // 2)
    patches = patches.transpose(1, 2)
    K_T = D_i.permute(0, 2, 1, 3, 4).contiguous()
    K_T_flat = K_T.view(B, Cx, Cy * k * k)
    X_flat = torch.bmm(K_T_flat, patches.transpose(1, 2))
    return F.fold(X_flat, output_size=(H, W), kernel_size=1)

##########################################################################
# 3. 子网络定义 (Head, HyPa, Generator)
##########################################################################

class HeadNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, d_size: int):
        super(HeadNet, self).__init__()
        self.head_x = nn.Sequential(
            nn.Conv2d(in_channels + 1, 64, d_size, padding=(d_size - 1) // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1, bias=False))

    def forward(self, y, sigma):
        sigma = sigma.repeat(1, 1, y.size(2), y.size(3))
        return self.head_x(torch.cat([y, sigma], dim=1))

class HyPaNet(nn.Module):
    def __init__(self, in_nc: int = 1, nc: int = 64, out_nc: int = 5):
        super(HyPaNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_nc, nc, 1, padding=0, bias=True), nn.Sigmoid(),
            nn.Conv2d(nc, out_nc, 1, padding=0, bias=True), nn.Softplus())

    def forward(self, x: Tensor):
        x = (x - 0.098) / 0.0566 # 使用你提供的标准化公式
        return self.mlp(x) + 1e-6

class DiGenerator(nn.Module):
    """ 针对织物周期性图像设计的特异性字典生成器 """
    def __init__(self, in_channels, out_channels, k_size, m_channels):
        super(DiGenerator, self).__init__()
        self.out_c, self.in_c, self.k = out_channels, m_channels, k_size
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        # 空间注意力：帮助定位织物纹理特征
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid())
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(inplace=True),
            nn.Linear(64, out_channels * m_channels * k_size * k_size))

    def forward(self, y):
        b = y.size(0)
        feat = self.extractor(y)
        avg_out = torch.mean(feat, dim=1, keepdim=True)
        max_out, _ = torch.max(feat, dim=1, keepdim=True)
        attn = self.sa(torch.cat([avg_out, max_out], dim=1))
        pooled = self.global_pool(feat * attn).view(b, -1)
        return self.mlp(pooled).view(b, self.out_c, self.in_c, self.k, self.k)

##########################################################################
# 4. BodyNet 与主模型
##########################################################################

class BodyNet(nn.Module):
    def __init__(self, unet, S, S_T):
        super(BodyNet, self).__init__()
        self.unet = unet
        self.S, self.S_T = S, S_T

    def forward(self, X_in, Y, Z, beta, alpha, rho, gamma, Di_batch, samfeats, enc, dec):
        # X-step
        res = (self.S(X_in) + apply_Di(X_in, Di_batch)) - Y
        grad = self.S_T(res) + apply_Di_T(res, Di_batch)
        X_out = X_in - alpha * (grad + rho * (X_in - Z + beta))
        # Z-step
        rho_map = (1 / rho.sqrt()).repeat(1, 1, X_out.size(2), X_out.size(3))
        Z, samfeats, enc_, dec_ = self.unet(torch.cat([X_out, rho_map], dim=1), samfeats, enc, dec, stage_inter=True)
        # beta-step
        beta = gamma[0] * beta + gamma[1] * X_out - gamma[2] * Z
        return X_out, Z, beta, samfeats, enc_, dec_

class denoise_Net_admm_restormer(nn.Module):
    def __init__(self, opt):
        super(denoise_Net_admm_restormer, self).__init__()
        self.n_channels = opt["n_channels"]
        self.d_size = opt["d_size"]
        self.stage = opt["stage"]
        self.m_channels = 16

        self.headnet = HeadNet(self.n_channels, self.n_channels, 3)
        self.unet = Restormer11(inp_channels=self.m_channels+1, out_channels=self.m_channels, dim=self.m_channels)
        
        # 共享字典 S 及其转置 S_T
        self.S = default_conv(self.m_channels, self.n_channels, self.d_size)
        self.S_T = default_conv(self.n_channels, self.m_channels, self.d_size)
        
        # 动态字典生成网络
        self.di_gen = DiGenerator(self.n_channels, self.n_channels, self.d_size, self.m_channels)
        
        self.body = BodyNet(self.unet, self.S, self.S_T)
        self.hypa_list = nn.ModuleList([HyPaNet(in_nc=1, out_nc=5) for _ in range(self.stage)])

    def forward(self, input, sigma):
        # 1. 动态生成特异性字典
        Di_batch = self.di_gen(input) 

        # 2. 初始化
        sigma = sigma.view(sigma.size(0), 1, 1, 1).to(input.device)
        X = self.S_T(self.headnet(input, sigma)) # 系数域初始化

        preds = []
        Z = torch.zeros_like(X)
        beta = torch.zeros_like(X)
        samfeats = enc = dec = None

        # 3. ADMM 迭代阶段
        for k in range(self.stage):
            hypas = self.hypa_list[k](sigma)
            alpha, rho = hypas[:, 0:1], hypas[:, 1:2]
            gamma = [hypas[:, 2:3], hypas[:, 3:4], hypas[:, 4:5]]

            if k == 0:
                # 初始步：(S+Di) 组合逻辑
                X1_img = self.S(X) + apply_Di(X, Di_batch)
                temp = (self.S_T(X1_img) + apply_Di_T(input, Di_batch)) - self.S_T(input)
                X_ = self.S_T(self.S(temp) + apply_Di(temp, Di_batch)) + rho * self.S_T(X1_img)
                X = self.S_T(X1_img) - alpha * X_

                rho_m = (1 / rho.sqrt()).repeat(1, 1, X.size(2), X.size(3))
                Z, samfeats, enc, dec = self.unet(torch.cat([X, rho_m], dim=1), stage_inter=True)
                beta = gamma[1] * X - gamma[2] * Z
            else:
                X, Z, beta, samfeats, enc, dec = self.body(
                    X, input, Z, beta, alpha, rho, gamma, Di_batch, samfeats, enc, dec)
            
            # 记录中间重构结果
            preds.append(self.S(X) + apply_Di(X, Di_batch))

        # 4. Final Step (最后一层 X 更新)
        res_f = (self.S(X) + apply_Di(X, Di_batch)) - input
        grad_f = self.S_T(res_f) + apply_Di_T(res_f, Di_batch)
        X_out = X - alpha * (grad_f + rho * (X - Z - beta))
        
        final_out = self.S(X_out) + apply_Di(X_out, Di_batch)
        preds.append(final_out)

        return final_out, preds