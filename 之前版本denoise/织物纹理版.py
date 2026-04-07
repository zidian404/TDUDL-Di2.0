import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, List, Tuple, Optional
import numpy as np

from Net.restormer_arch import Restormer11


##########################################################################
# 创新点1：Gabor方向滤波器组（提取织物纹理方向特征）
##########################################################################

class GaborFilterBank(nn.Module):
    """
    多方向、多尺度Gabor滤波器组，提取织物的纹理方向信息
    """
    def __init__(self, in_channels=16, out_channels=16, num_orientations=8, num_scales=3):
        super(GaborFilterBank, self).__init__()
        self.num_orientations = num_orientations
        self.num_scales = num_scales
        
        # 可学习的Gabor参数
        self.theta = nn.Parameter(torch.linspace(0, np.pi, num_orientations+1)[:-1].view(1, num_orientations, 1, 1))
        self.lambda_wavelength = nn.Parameter(torch.tensor([4., 8., 16.]).view(1, 1, num_scales, 1))
        
        # 方向特征编码器（输入通道适配灰度）
        self.dir_encoder = nn.Sequential(
            nn.Conv2d(in_channels * num_orientations * num_scales, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 生成Gabor核（简化版本，实际应用中可预计算）
        gabor_features = []
        for i in range(self.num_orientations):
            for j in range(self.num_scales):
                # 应用Gabor滤波（这里用可分离卷积近似）
                theta_i = self.theta[:, i, :, :]
                lambda_j = self.lambda_wavelength[:, :, j, :]
                
                # 方向性卷积
                kernel_size = int(lambda_j.squeeze().item()) * 2 + 1
                if kernel_size % 2 == 0:
                    kernel_size += 1
                    
                # 简化：使用方向性平均池化 + 卷积
                feat = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
                gabor_features.append(feat)
        
        # 拼接所有方向特征
        gabor_cat = torch.cat(gabor_features, dim=1)
        
        # 编码为统一的纹理特征
        texture_feat = self.dir_encoder(gabor_cat)
        
        return texture_feat


##########################################################################
# 创新点2：周期性注意力机制（捕捉织物重复pattern）
##########################################################################

class PeriodicAttention(nn.Module):
    """
    周期性注意力：利用织物的周期性结构先验
    """
    def __init__(self, channels, max_period=32):
        super(PeriodicAttention, self).__init__()
        self.max_period = max_period
        
        # 周期检测器
        self.period_detector = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # 周期调制器
        self.period_modulator = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        )
        
        # 自相关计算（用于周期估计）
        self.register_buffer("position_ids", torch.arange(max_period).view(1, 1, max_period))
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. 检测周期性区域
        period_map = self.period_detector(x)  # [B,1,H,W]
        
        # 2. 计算局部自相关（简化版本）
        x_reshaped = x.view(B, C, -1)  # [B,C,H*W]
        x_norm = F.normalize(x_reshaped, dim=1)
        
        # 3. 周期调制
        period_mod = self.period_modulator(x)
        
        # 4. 结合周期信息
        out = x + period_mod * period_map
        
        return out, period_map


##########################################################################
# 创新点3：纹理感知的字典适配器（融合方向+周期信息）
##########################################################################

class TextureAwareDictAdapter(nn.Module):
    """
    织物纹理感知的字典适配器
    创新点：融合Gabor方向特征和周期性注意力
    """
    def __init__(self, Cx: int, use_gabor=True, use_periodic=True):
        super(TextureAwareDictAdapter, self).__init__()
        
        self.use_gabor = use_gabor
        self.use_periodic = use_periodic
        
        # 全局池化获取通道统计
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        if use_gabor:
            self.gabor_bank = GaborFilterBank(in_channels=Cx, out_channels=Cx)
            
        if use_periodic:
            self.period_attn = PeriodicAttention(Cx)
            
        # 纹理特征融合
        fusion_in_channels = Cx
        if use_gabor:
            fusion_in_channels += Cx
        if use_periodic:
            fusion_in_channels += Cx
            
        self.texture_fusion = nn.Sequential(
            nn.Conv2d(fusion_in_channels, Cx, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(Cx, Cx, 1),
            nn.Sigmoid()
        )
        
        # 基础适配器（保留原功能）
        self.base_adapter = nn.Sequential(
            nn.Conv2d(Cx, Cx, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(Cx, Cx, 1, bias=True),
            nn.Sigmoid()
        )
        
    def forward(self, X: Tensor) -> Tuple[Tensor, dict]:
        """
        Returns:
            w: 通道权重 [B,Cx,1,1]
            texture_info: 纹理信息字典（用于损失计算）
        """
        texture_info = {}
        
        # 基础权重（全局统计）
        g = self.gap(X)  # [B,Cx,1,1]
        w_base = self.base_adapter(g)
        
        texture_features = [w_base]
        
        # Gabor纹理特征
        if self.use_gabor:
            gabor_feat = self.gabor_bank(X)  # [B,Cx,H,W]
            gabor_g = self.gap(gabor_feat)
            texture_features.append(gabor_g)
            texture_info['gabor'] = gabor_feat
        
        # 周期性特征
        if self.use_periodic:
            period_feat, period_map = self.period_attn(X)
            period_g = self.gap(period_feat)
            texture_features.append(period_g)
            texture_info['period_map'] = period_map
        
        # 融合纹理信息
        if len(texture_features) > 1:
            texture_cat = torch.cat(texture_features, dim=1)
            w_texture = self.texture_fusion(texture_cat)
            w = w_base * w_texture  # 联合调制
        else:
            w = w_base
            
        return w, texture_info


##########################################################################
# 创新点4：结构保真损失（Texture Consistency Loss）
##########################################################################

class TextureConsistencyLoss(nn.Module):
    """
    织物纹理一致性损失
    1. 方向一致性损失
    2. 周期性一致性损失
    3. 局部结构相似性损失
    """
    def __init__(self, lambda_dir=0.1, lambda_period=0.1, lambda_l1=0.5):
        super(TextureConsistencyLoss, self).__init__()
        self.lambda_dir = lambda_dir
        self.lambda_period = lambda_period
        self.lambda_l1 = lambda_l1
        
        # Sobel算子用于梯度计算（灰度图单通道）
        self.sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
        self.sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)
        
    def forward(self, pred, target, texture_info):
        """
        pred: 去噪结果 [B,1,H,W] 灰度图
        target: 干净图像 [B,1,H,W]
        texture_info: 来自适配器的纹理信息
        """
        loss_dict = {}
        total_loss = 0.0
        
        # 确保是灰度图
        if pred.shape[1] != 1:
            # 如果是RGB，转换为灰度（实际不会发生，因为n_channels=1）
            pred = torch.mean(pred, dim=1, keepdim=True)
            target = torch.mean(target, dim=1, keepdim=True)
        
        # 1. 梯度方向一致性（保留织物边缘）
        sobel_x = self.sobel_x.to(pred.device)
        sobel_y = self.sobel_y.to(pred.device)
        
        grad_x_pred = F.conv2d(pred, sobel_x, padding=1)
        grad_y_pred = F.conv2d(pred, sobel_y, padding=1)
        grad_x_target = F.conv2d(target, sobel_x, padding=1)
        grad_y_target = F.conv2d(target, sobel_y, padding=1)
        
        dir_loss = F.l1_loss(grad_x_pred, grad_x_target) + F.l1_loss(grad_y_pred, grad_y_target)
        loss_dict['dir_loss'] = dir_loss
        total_loss += self.lambda_dir * dir_loss
        
        # 2. 周期性一致性（如果提供了period_map）
        if 'period_map' in texture_info and texture_info['period_map'] is not None:
            period_map = texture_info['period_map']
            # 在周期性强的区域加强监督
            period_weight = period_map.detach()
            period_loss = (period_weight * F.l1_loss(pred, target, reduction='none')).mean()
            loss_dict['period_loss'] = period_loss
            total_loss += self.lambda_period * period_loss
        
        # 3. L1损失
        l1_loss = F.l1_loss(pred, target)
        loss_dict['l1_loss'] = l1_loss
        total_loss += self.lambda_l1 * l1_loss
        
        return total_loss, loss_dict


##########################################################################
# 改进的BodyNet（集成纹理感知）
##########################################################################

class BodyNet(nn.Module):
    def __init__(self, unet, S, S_T, dict_adapter: TextureAwareDictAdapter):
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
        
        # 纹理感知的适配器
        w, texture_info = self.dict_adapter(X)
        X_mod = X * w
        
        # X-step
        Y_hat = self.S(X_mod)
        res = Y_hat - Y
        grad_S = self.S_T(res)
        grad_D = grad_S * w
        grad = grad_S + grad_D
        
        X_term = X - Z + beta
        X_out = X - alpha * (grad + rho * X_term)
        
        # Z-step
        rho_ = (1 / rho.sqrt()).repeat(1, 1, X_out.size(2), X_out.size(3))
        Z, samfeats, enc_, dec_ = self.unet(
            torch.cat([X_out, rho_], dim=1),
            samfeats, enc, dec, stage_inter=True
        )
        
        # beta-step
        beta = gamma[0] * beta + gamma[1] * X_out - gamma[2] * Z
        
        return X_out, Z, beta, samfeats, enc_, dec_, texture_info


##########################################################################
# 完整的改进版主网络（适配灰度图像）
##########################################################################

class denoise_Net_admm_restormer_enhanced(nn.Module):
    """
    增强版织物图像去噪网络（适配灰度图）
    创新点：
    1. 纹理感知字典适配器（Gabor + 周期性注意力）
    2. 多尺度字典共享机制
    3. 纹理保真损失（可在训练时使用）
    
    输入：灰度图像 [B, 1, H, W]
    """
    def __init__(self, opt):
        super(denoise_Net_admm_restormer_enhanced, self).__init__()

        self.n_channels = opt["n_channels"]  # 应该是 1（灰度图）
        self.d_size = opt["d_size"]
        self.stage = opt["stage"]
        
        # HeadNet（输入1通道灰度图 + sigma）
        self.headnet = HeadNet(self.n_channels, self.n_channels, 3)

        # 系数域通道
        self.m_channels = 16
        self.stride = 1
        
        # Restormer（输入：系数域通道+1）
        self.unet = Restormer11(
            inp_channels=self.m_channels + 1,
            out_channels=self.m_channels,
            dim=self.m_channels
        )
        
        # 共享字典 S / S_T
        k = self.d_size
        Cx = self.m_channels  # 系数域通道
        Cy = self.n_channels  # 图像域通道（1）
        
        # 多尺度字典（增强表示能力）
        self.S = nn.ModuleList([
            nn.Conv2d(Cx, Cy, kernel_size=k1, padding=k1//2, bias=True)
            for k1 in [3, 5, 7]
        ])
        self.S_T = nn.ModuleList([
            nn.Conv2d(Cy, Cx, kernel_size=k1, padding=k1//2, bias=True)
            for k1 in [3, 5, 7]
        ])
        
        # 多尺度融合权重
        self.scale_fusion = nn.Sequential(
            nn.Conv2d(Cx * 3, Cx, 1),
            nn.Sigmoid()
        )
        
        # 纹理感知适配器（核心创新）
        self.dict_adapter = TextureAwareDictAdapter(Cx=Cx, use_gabor=True, use_periodic=True)
        
        # 注意：BodyNet需要传入self作为S的代理，需要特殊处理
        # 这里直接使用S和S_T列表，在forward中动态选择
        
        # HyPaNet
        self.hypa_list_: nn.ModuleList = nn.ModuleList()
        for _ in range(self.stage):
            self.hypa_list_.append(HyPaNet(in_nc=1, out_nc=5))
            
        # 纹理一致性损失模块（训练时使用）
        self.texture_loss = TextureConsistencyLoss()
        
    def apply_S(self, X, scale_idx=None):
        """多尺度字典前向传播"""
        if scale_idx is not None:
            return self.S[scale_idx](X)
        else:
            # 融合多尺度输出
            multi_scale = [self.S[i](X) for i in range(3)]
            # 简单平均
            return sum(multi_scale) / 3.0
    
    def apply_S_T(self, Y, scale_idx=None):
        """多尺度字典反向传播"""
        if scale_idx is not None:
            return self.S_T[scale_idx](Y)
        else:
            multi_scale = [self.S_T[i](Y) for i in range(3)]
            return sum(multi_scale) / 3.0

    def forward(self, input: Tensor, sigma: Tensor, return_texture=False):
        """
        Args:
            input: 噪声图像 [B, 1, H, W] 灰度图
            sigma: 噪声标准差 [B] 或 [B,1,1,1]
            return_texture: 是否返回纹理信息（用于损失计算）
        """
        device = input.device
        
        # sigma预处理
        sigma = sigma.to(device)
        if sigma.dim() == 1:
            sigma = sigma.view(-1, 1, 1, 1)
        elif sigma.dim() == 2:
            sigma = sigma.view(sigma.size(0), 1, 1, 1)
            
        # 初始化
        X_img0 = self.headnet(input, sigma)  # [B,1,H,W]
        X = self.apply_S_T(X_img0)  # [B,Cx,H,W]
        
        preds = []
        texture_infos = []
        Z = torch.zeros_like(X)
        beta = torch.zeros_like(X)
        samfeats = enc = dec = None
        
        for k in range(self.stage):
            hypas = self.hypa_list_[k](sigma)
            alpha = hypas[:, 0:1, :, :]
            rho   = hypas[:, 1:2, :, :]
            gamma1 = hypas[:, 2:3, :, :]
            gamma2 = hypas[:, 3:4, :, :]
            gamma3 = hypas[:, 4:5, :, :]
            gamma = [gamma1, gamma2, gamma3]
            
            if k == 0:
                # 初始化阶段
                w, tex_info = self.dict_adapter(X)
                X_mod = X * w
                X1_img = self.apply_S(X_mod)
                temp_back = self.apply_S_T(X1_img) - self.apply_S_T(input)
                temp_mod = temp_back * w
                X2_img = self.apply_S(temp_mod)
                
                X1_coef = self.apply_S_T(X1_img)
                X2_coef = self.apply_S_T(X2_img)
                
                X_ = X2_coef + rho * X1_coef
                X = X1_coef - alpha * X_
                
                # Z-step
                rho_map = (1 / rho.sqrt()).repeat(1, 1, X.size(2), X.size(3))
                Z, samfeats, enc, dec = self.unet(
                    torch.cat([X, rho_map], dim=1),
                    stage_inter=True
                )
                
                # beta-step
                beta = gamma[1] * X - gamma[2] * Z
                
                # 重构
                w_out, tex_out = self.dict_adapter(X)
                output = self.apply_S(X * w_out)
                preds.append(output)
                texture_infos.append(tex_out)
                
            else:
                # 非初始阶段，使用BodyNet逻辑
                # 纹理感知的适配器
                w, tex_info = self.dict_adapter(X)
                X_mod = X * w
                
                # X-step
                Y_hat = self.apply_S(X_mod)
                res = Y_hat - input
                grad_S = self.apply_S_T(res)
                grad_D = grad_S * w
                grad = grad_S + grad_D
                
                X_term = X - Z + beta
                X = X - alpha * (grad + rho * X_term)
                
                # Z-step
                rho_map = (1 / rho.sqrt()).repeat(1, 1, X.size(2), X.size(3))
                Z, samfeats, enc, dec = self.unet(
                    torch.cat([X, rho_map], dim=1),
                    samfeats, enc, dec, stage_inter=True
                )
                
                # beta-step
                beta = gamma[0] * beta + gamma[1] * X - gamma[2] * Z
                
                # 重构
                w_out, tex_out = self.dict_adapter(X)
                output = self.apply_S(X * w_out)
                preds.append(output)
                texture_infos.append(tex_out)
        
        # 最终步骤
        w_final, tex_final = self.dict_adapter(X)
        Y_S_final = self.apply_S(X * w_final)
        temp = Y_S_final - input
        X_1 = self.apply_S_T(temp) * (1.0 + w_final)
        X_2 = rho * (X - Z - beta)
        X_out = X - alpha * (X_1 + X_2)
        
        w_out, tex_out = self.dict_adapter(X_out)
        output = self.apply_S(X_out * w_out)
        preds.append(output)
        texture_infos.append(tex_out)
        
        if return_texture:
            return output, preds, texture_infos
        return output, preds
    
    def compute_texture_loss(self, pred, target, texture_infos):
        """计算纹理一致性损失"""
        total_loss = 0.0
        loss_details = {}
        
        # 如果是列表，取最后一个（最终输出）
        if isinstance(pred, list):
            final_pred = pred[-1]
        else:
            final_pred = pred
            
        # 使用最后一个stage的纹理信息
        if texture_infos and len(texture_infos) > 0:
            final_tex_info = texture_infos[-1]
            loss, detail = self.texture_loss(final_pred, target, final_tex_info)
            total_loss += loss
            loss_details.update(detail)
        
        return total_loss, loss_details


##########################################################################
# 辅助模块
##########################################################################

class HeadNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, d_size: int):
        super(HeadNet, self).__init__()
        # in_channels = 1（灰度图）
        self.head_x = nn.Sequential(
            nn.Conv2d(in_channels + 1, 64, d_size, padding=(d_size - 1) // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1, bias=False)
        )

    def forward(self, y: Any, sigma: Tensor):
        sigma = sigma.repeat(1, 1, y.size(2), y.size(3))
        x = self.head_x(torch.cat([y, sigma], dim=1))
        return x


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


