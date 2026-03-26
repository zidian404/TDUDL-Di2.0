import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, List, Tuple, Optional
from Net.restormer_arch import Restormer11

##########################################################################
# 1. 基础卷积工具函数
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
    return nn.Conv2d(in_chn, out_chn, kernel_size, stride=stride, 
                     padding=(kernel_size - 1) // 2, bias=bias)

def conv_up(in_chn, out_chn, kernel_size, stride=2, bias=False):
    return nn.ConvTranspose2d(in_chn, out_chn, kernel_size, stride=stride, 
                              padding=(kernel_size - 1) // 2, 
                              output_padding=stride-1, bias=bias)

class ST(nn.Module):
    """软阈值算子"""
    def __init__(self):
        super(ST, self).__init__()
    def forward(self, x, t, samfeats=None, enc_in=None, dec_in=None):
        return x.sign() * F.relu(x.abs() - t), samfeats, enc_in, dec_in

##########################################################################
# 2. 矩阵运算算子（支持batch维度）
##########################################################################

def apply_Di(X: Tensor, D_i: Tensor) -> Tensor:
    """X: [B, Cx, H, W], D_i: [B, Cy, Cx, k, k] -> [B, Cy, H, W]"""
    B, Cx, H, W = X.shape
    B2, Cy, Cx2, k, k2 = D_i.shape
    assert B2 == B and Cx2 == Cx and k == k2
    
    patches = F.unfold(X, kernel_size=k, padding=k // 2)  # [B, Cx*k*k, N]
    patches = patches.transpose(1, 2)  # [B, N, Cx*k*k]
    K_flat = D_i.view(B, Cy, Cx * k * k)  # [B, Cy, Cx*k*k]
    Y_flat = torch.bmm(K_flat, patches.transpose(1, 2))  # [B, Cy, N]
    return F.fold(Y_flat, output_size=(H, W), kernel_size=1)

def apply_Di_T(Y: Tensor, D_i: Tensor) -> Tensor:
    """Y: [B, Cy, H, W], D_i: [B, Cy, Cx, k, k] -> [B, Cx, H, W]"""
    B, Cy, H, W = Y.shape
    B2, Cy2, Cx, k, k2 = D_i.shape
    assert B2 == B and Cy2 == Cy and k == k2
    
    patches = F.unfold(Y, kernel_size=k, padding=k // 2)  # [B, Cy*k*k, N]
    patches = patches.transpose(1, 2)  # [B, N, Cy*k*k]
    K_T = D_i.permute(0, 2, 1, 3, 4).contiguous()  # [B, Cx, Cy, k, k]
    K_T_flat = K_T.view(B, Cx, Cy * k * k)  # [B, Cx, Cy*k*k]
    X_flat = torch.bmm(K_T_flat, patches.transpose(1, 2))  # [B, Cx, N]
    return F.fold(X_flat, output_size=(H, W), kernel_size=1)

##########################################################################
# 3. 修复后的增强版字典生成器
##########################################################################

class EnhancedDictGenerator(nn.Module):
    """
    融合静态和动态信息的增强版字典生成器
    - 静态分支：基于输入图像提取纹理特征
    - 动态分支：基于当前系数自适应调整
    - 可学习融合：自动平衡两者贡献
    """
    def __init__(self, in_channels: int, Cx: int, Cy: int, k_size: int, 
                 hidden_dim: int = 64, use_residual: bool = True):
        super(EnhancedDictGenerator, self).__init__()
        self.Cx = Cx
        self.Cy = Cy
        self.k = k_size
        self.use_residual = use_residual
        
        # === 静态分支：基于输入图像 ===
        self.static_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        self.static_pool = nn.AdaptiveAvgPool2d(1)
        self.static_mlp = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, Cy * Cx * k_size * k_size)
        )
        
        # === 动态分支：基于当前系数 ===
        self.dynamic_pool = nn.AdaptiveAvgPool2d(1)
        self.dynamic_mlp = nn.Sequential(
            nn.Conv2d(Cx, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, Cy * Cx * k_size * k_size, 1)
        )
        
        # === 融合机制 ===
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        
        # === 修复后的残差连接 ===
        if use_residual:
            # 使用全局池化+全连接层，避免维度不匹配
            self.residual_pool = nn.AdaptiveAvgPool2d(1)
            self.residual_fc = nn.Sequential(
                nn.Linear(Cx, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, Cy * Cx * k_size * k_size)
            )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, y: Tensor, X: Tensor) -> Tuple[Tensor, dict]:
        """
        y: 输入噪声图像 [B, Cy, H, W]
        X: 当前系数 [B, Cx, H, W]
        返回: D_dict [B, Cy, Cx, k, k], 以及统计信息
        """
        B = y.size(0)
        
        # === 静态分支：基于输入图像 ===
        static_feat = self.static_extractor(y)  # [B, 64, H, W]
        
        # 空间注意力增强
        avg_out = torch.mean(static_feat, dim=1, keepdim=True)
        max_out, _ = torch.max(static_feat, dim=1, keepdim=True)
        attn = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        static_feat_attn = static_feat * attn
        
        static_pooled = self.static_pool(static_feat_attn).view(B, -1)  # [B, 64]
        static_dict_flat = self.static_mlp(static_pooled)  # [B, Cy*Cx*k*k]
        D_static = static_dict_flat.view(B, self.Cy, self.Cx, self.k, self.k)
        
        # === 动态分支：基于当前系数 ===
        dynamic_pooled = self.dynamic_pool(X)  # [B, Cx, 1, 1]
        dynamic_dict_flat = self.dynamic_mlp(dynamic_pooled)  # [B, Cy*Cx*k*k, 1, 1]
        dynamic_dict_flat = dynamic_dict_flat.squeeze(-1).squeeze(-1)  # [B, Cy*Cx*k*k]
        D_dynamic = dynamic_dict_flat.view(B, self.Cy, self.Cx, self.k, self.k)
        
        # === 自适应融合 ===
        alpha = torch.sigmoid(self.fusion_weight)
        
        # 基础融合
        D_fused = alpha * D_static + (1 - alpha) * D_dynamic
        
        # === 修复后的残差连接 ===
        if self.use_residual:
            # 全局池化并生成残差
            X_pooled = self.residual_pool(X).view(B, self.Cx)  # [B, Cx]
            residual_flat = self.residual_fc(X_pooled)  # [B, Cy*Cx*k*k]
            residual = residual_flat.view(B, self.Cy, self.Cx, self.k, self.k)
            D_fused = D_fused + 0.1 * residual
        
        # 返回统计信息
        stats = {
            'fusion_alpha': alpha.item(),
            'static_norm': D_static.norm().item(),
            'dynamic_norm': D_dynamic.norm().item(),
            'fused_norm': D_fused.norm().item()
        }
        
        return D_fused, stats


##########################################################################
# 4. HeadNet 和 HyPaNet
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
        x = (x - 0.098) / 0.0566
        return self.mlp(x) + 1e-6

##########################################################################
# 5. BodyNet
##########################################################################

class BodyNet(nn.Module):
    def __init__(self, unet, S, S_T):
        super(BodyNet, self).__init__()
        self.unet = unet
        self.S = S
        self.S_T = S_T

    def forward(self, X_in, Y, Z, beta, alpha, rho, gamma, 
                Di_batch, samfeats, enc, dec):
        # X-step
        Y_S = self.S(X_in)
        Y_D = apply_Di(X_in, Di_batch)
        Y_hat = Y_S + Y_D
        
        res = Y_hat - Y
        grad = self.S_T(res) + apply_Di_T(res, Di_batch)
        X_out = X_in - alpha * (grad + rho * (X_in - Z + beta))
        
        # Z-step
        rho_map = (1 / rho.sqrt()).repeat(1, 1, X_out.size(2), X_out.size(3))
        Z, samfeats, enc_, dec_ = self.unet(
            torch.cat([X_out, rho_map], dim=1), 
            samfeats, enc, dec, stage_inter=True
        )
        
        # beta-step
        beta = gamma[0] * beta + gamma[1] * X_out - gamma[2] * Z
        
        return X_out, Z, beta, samfeats, enc_, dec_


##########################################################################
# 6. 主模型：融合增强版
##########################################################################

class denoise_Net_hybrid_admm(nn.Module):
    """
    融合版ADMM去噪网络
    - 结合静态字典和动态字典
    - 可学习融合机制
    - 支持字典统计信息监控
    """
    def __init__(self, opt):
        super(denoise_Net_hybrid_admm, self).__init__()
        
        self.n_channels = opt["n_channels"]
        self.d_size = opt["d_size"]
        self.stage = opt["stage"]
        self.m_channels = 16  # 系数域通道数
        
        # 基础模块
        self.headnet = HeadNet(self.n_channels, self.n_channels, 3)
        self.unet = Restormer11(
            inp_channels=self.m_channels + 1,
            out_channels=self.m_channels,
            dim=self.m_channels
        )
        
        # 共享字典 S 和 S_T
        self.S = default_conv(self.m_channels, self.n_channels, self.d_size)
        self.S_T = default_conv(self.n_channels, self.m_channels, self.d_size)
        
        # 增强版字典生成器（融合静态+动态）
        self.dict_generator = EnhancedDictGenerator(
            in_channels=self.n_channels,
            Cx=self.m_channels,
            Cy=self.n_channels,
            k_size=self.d_size,
            hidden_dim=128,
            use_residual=True  # 现在可以安全使用残差连接
        )
        
        # BodyNet
        self.body = BodyNet(self.unet, self.S, self.S_T)
        
        # 超参数网络
        self.hypa_list = nn.ModuleList([
            HyPaNet(in_nc=1, out_nc=5) for _ in range(self.stage)
        ])
        
        # 存储字典统计信息
        self.dict_stats = []
        
    def forward(self, input, sigma, return_stats=False):
        """
        input: [B, Cy, H, W] 噪声图像
        sigma: [B] 或 [B,1,1,1] 噪声水平
        return_stats: 是否返回字典统计信息
        """
        device = input.device
        sigma = sigma.view(sigma.size(0), 1, 1, 1).to(device)
        
        # 初始化 X
        X = self.S_T(self.headnet(input, sigma))
        
        preds = []
        Z = torch.zeros_like(X)
        beta = torch.zeros_like(X)
        samfeats = enc = dec = None
        
        # 存储统计信息
        all_stats = [] if return_stats else None
        
        for k in range(self.stage):
            # 获取超参数
            hypas = self.hypa_list[k](sigma)
            alpha = hypas[:, 0:1, :, :]
            rho = hypas[:, 1:2, :, :]
            gamma = [hypas[:, 2:3], hypas[:, 3:4], hypas[:, 4:5]]
            
            # 生成字典（融合静态和动态）
            Di_batch, dict_stats = self.dict_generator(input, X)
            
            if return_stats:
                all_stats.append(dict_stats)
            
            if k == 0:
                # 初始步：使用增强的字典初始化
                X1_img = self.S(X) + apply_Di(X, Di_batch)
                temp = (self.S_T(X1_img) + apply_Di_T(input, Di_batch)) - self.S_T(input)
                X_ = self.S_T(self.S(temp) + apply_Di(temp, Di_batch)) + rho * self.S_T(X1_img)
                X = self.S_T(X1_img) - alpha * X_
                
                rho_m = (1 / rho.sqrt()).repeat(1, 1, X.size(2), X.size(3))
                Z, samfeats, enc, dec = self.unet(
                    torch.cat([X, rho_m], dim=1), stage_inter=True
                )
                beta = gamma[1] * X - gamma[2] * Z
            else:
                X, Z, beta, samfeats, enc, dec = self.body(
                    X, input, Z, beta, alpha, rho, gamma, 
                    Di_batch, samfeats, enc, dec
                )
            
            # 记录中间结果
            preds.append(self.S(X) + apply_Di(X, Di_batch))
        
        # Final Step
        Di_batch_final, final_stats = self.dict_generator(input, X)
        if return_stats:
            all_stats.append(final_stats)
            
        res_f = (self.S(X) + apply_Di(X, Di_batch_final)) - input
        grad_f = self.S_T(res_f) + apply_Di_T(res_f, Di_batch_final)
        X_out = X - alpha * (grad_f + rho * (X - Z - beta))
        
        final_out = self.S(X_out) + apply_Di(X_out, Di_batch_final)
        preds.append(final_out)
        
        if return_stats:
            return final_out, preds, all_stats
        return final_out, preds
    
    def get_dict_stats(self):
        """返回字典生成器的统计信息"""
        return self.dict_stats


##########################################################################
# 7. 便捷函数：用于训练监控
##########################################################################

def analyze_dict_behavior(model, input, sigma):
    """分析字典生成器的行为"""
    model.eval()
    with torch.no_grad():
        output, preds, stats = model(input, sigma, return_stats=True)
        
        print("=" * 50)
        print("Dictionary Generator Analysis:")
        for i, s in enumerate(stats):
            print(f"  Stage {i}: alpha={s['fusion_alpha']:.3f}, "
                  f"static={s['static_norm']:.3f}, "
                  f"dynamic={s['dynamic_norm']:.3f}")
        print("=" * 50)
        
    return output, stats


# 使用示例
if __name__ == "__main__":
    # 配置
    opt = {
        "n_channels": 1,
        "d_size": 3,
        "stage": 3,
    }
    
    # 创建模型
    model = denoise_Net_hybrid_admm(opt)
    
    # 测试前向传播
    batch_size = 4
    input = torch.randn(batch_size, 1, 128, 128)
    sigma = torch.full((batch_size,), 25, dtype=torch.float32)
    
    output, preds = model(input, sigma)
    print(f"Input shape: {input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of predictions: {len(preds)}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


# 添加别名以保持兼容性
denoise_Net_admm_restormer = denoise_Net_hybrid_admm