# visualize_dictionary.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import os

# 如果没有 Restormer11，创建一个简化的版本用于可视化
try:
    from Net.restormer_arch import Restormer11
except ImportError:
    print("警告: 未找到 Restormer11，使用简化版本")
    # 简化的 Restormer11（仅用于演示字典结构）
    class Restormer11(nn.Module):
        def __init__(self, inp_channels=17, out_channels=16, dim=16):
            super().__init__()
            self.conv = nn.Conv2d(inp_channels, out_channels, 3, padding=1)
        def forward(self, x, samfeats=None, enc=None, dec=None, stage_inter=False):
            if stage_inter:
                return self.conv(x), None, None, None
            return self.conv(x)

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
    return nn.Conv2d(in_chn, out_chn, kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=bias)

def conv_up(in_chn, out_chn, kernel_size, stride=2, bias=False):
    return nn.ConvTranspose2d(in_chn, out_chn, kernel_size, stride=stride, 
                              padding=(kernel_size - 1) // 2, output_padding=stride-1, bias=bias)

class ST(nn.Module):
    def __init__(self):
        super(ST, self).__init__()
    def forward(self, x, t, samfeats=None, enc_in=None, dec_in=None):
        return x.sign() * F.relu(x.abs() - t), samfeats, enc_in, dec_in

##########################################################################
# 2. 矩阵运算算子
##########################################################################

def apply_Di(X, D_i):
    B, Cx, H, W = X.shape
    _, Cy, _, k, _ = D_i.shape
    patches = F.unfold(X, kernel_size=k, padding=k // 2)
    patches = patches.transpose(1, 2)
    K_flat = D_i.view(B, Cy, Cx * k * k)
    Y_flat = torch.bmm(K_flat, patches.transpose(1, 2))
    return F.fold(Y_flat, output_size=(H, W), kernel_size=1)

def apply_Di_T(Y, D_i):
    B, Cy, H, W = Y.shape
    _, _, Cx, k, _ = D_i.shape
    patches = F.unfold(Y, kernel_size=k, padding=k // 2)
    patches = patches.transpose(1, 2)
    K_T = D_i.permute(0, 2, 1, 3, 4).contiguous()
    K_T_flat = K_T.view(B, Cx, Cy * k * k)
    X_flat = torch.bmm(K_T_flat, patches.transpose(1, 2))
    return F.fold(X_flat, output_size=(H, W), kernel_size=1)

##########################################################################
# 3. 子网络定义
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

class DiGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, m_channels):
        super(DiGenerator, self).__init__()
        self.out_c, self.in_c, self.k = out_channels, m_channels, k_size
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
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

class BodyNet(nn.Module):
    def __init__(self, unet, S, S_T):
        super(BodyNet, self).__init__()
        self.unet = unet
        self.S, self.S_T = S, S_T

    def forward(self, X_in, Y, Z, beta, alpha, rho, gamma, Di_batch, samfeats, enc, dec):
        res = (self.S(X_in) + apply_Di(X_in, Di_batch)) - Y
        grad = self.S_T(res) + apply_Di_T(res, Di_batch)
        X_out = X_in - alpha * (grad + rho * (X_in - Z + beta))
        rho_map = (1 / rho.sqrt()).repeat(1, 1, X_out.size(2), X_out.size(3))
        Z, samfeats, enc_, dec_ = self.unet(torch.cat([X_out, rho_map], dim=1), samfeats, enc, dec, stage_inter=True)
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
        
        self.S = default_conv(self.m_channels, self.n_channels, self.d_size)
        self.S_T = default_conv(self.n_channels, self.m_channels, self.d_size)
        
        self.di_gen = DiGenerator(self.n_channels, self.n_channels, self.d_size, self.m_channels)
        
        self.body = BodyNet(self.unet, self.S, self.S_T)
        self.hypa_list = nn.ModuleList([HyPaNet(in_nc=1, out_nc=5) for _ in range(self.stage)])

    def forward(self, input, sigma):
        Di_batch = self.di_gen(input)
        sigma = sigma.view(sigma.size(0), 1, 1, 1).to(input.device)
        X = self.S_T(self.headnet(input, sigma))
        preds = []
        Z = torch.zeros_like(X)
        beta = torch.zeros_like(X)
        samfeats = enc = dec = None

        for k in range(self.stage):
            hypas = self.hypa_list[k](sigma)
            alpha, rho = hypas[:, 0:1], hypas[:, 1:2]
            gamma = [hypas[:, 2:3], hypas[:, 3:4], hypas[:, 4:5]]

            if k == 0:
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
            
            preds.append(self.S(X) + apply_Di(X, Di_batch))

        res_f = (self.S(X) + apply_Di(X, Di_batch)) - input
        grad_f = self.S_T(res_f) + apply_Di_T(res_f, Di_batch)
        X_out = X - alpha * (grad_f + rho * (X - Z - beta))
        
        final_out = self.S(X_out) + apply_Di(X_out, Di_batch)
        preds.append(final_out)

        return final_out, preds

##########################################################################
# 可视化函数
##########################################################################

def visualize_all_dictionaries(model, input_tensor, save_dir='dictionary_vis'):
    """完整可视化所有字典组件"""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        # 获取字典
        S_weight = model.S.weight.data.cpu().numpy()
        S_T_weight = model.S_T.weight.data.cpu().numpy()
        Di_batch = model.di_gen(input_tensor)
        Di_weight = Di_batch[0].cpu().numpy()
        
        print("=" * 60)
        print("字典信息")
        print("=" * 60)
        print(f"S:       {S_weight.shape}")
        print(f"S_T:     {S_T_weight.shape}")
        print(f"Di:      {Di_weight.shape}")
        print("=" * 60)
        
        # 可视化
        plot_dictionary_grid(S_weight, f'{save_dir}/S_dictionary.png', 'S Dictionary (Static)')
        plot_dictionary_grid(S_T_weight, f'{save_dir}/S_T_dictionary.png', 'S_T Dictionary (Static)')
        plot_dictionary_grid(Di_weight, f'{save_dir}/Di_dictionary.png', 'Di Dictionary (Dynamic)')
        
        # 统计分析
        analyze_dictionaries([S_weight, S_T_weight, Di_weight], 
                            ['S', 'S_T', 'Di'], save_dir)
        
        # 对比单个原子
        compare_single_atom(S_weight, Di_weight, save_dir)
        
        return S_weight, S_T_weight, Di_weight

def plot_dictionary_grid(dict_weight, save_path, title):
    """绘制字典网格"""
    if dict_weight.ndim == 4:
        out_c, in_c, k, _ = dict_weight.shape
    else:
        return
    
    # 限制显示数量
    show_out = min(6, out_c)
    show_in = min(6, in_c)
    
    fig, axes = plt.subplots(show_in, show_out, figsize=(show_out*1.5, show_in*1.5))
    
    if show_in == 1 and show_out == 1:
        axes = np.array([[axes]])
    elif show_in == 1 or show_out == 1:
        axes = axes.reshape(show_in, show_out)
    
    for i in range(show_in):
        for j in range(show_out):
            kernel = dict_weight[j, i]
            # 归一化
            vmin, vmax = np.percentile(kernel, [2, 98])
            kernel_disp = np.clip((kernel - vmin) / (vmax - vmin + 1e-8), 0, 1)
            
            ax = axes[i, j] if show_in > 1 and show_out > 1 else (axes[i] if show_in > 1 else axes[j])
            ax.imshow(kernel_disp, cmap='gray', interpolation='nearest')
            ax.set_title(f'{i}→{j}', fontsize=8)
            ax.axis('off')
    
    plt.suptitle(f'{title}\n{out_c}×{in_c}×{k}×{k}', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def analyze_dictionaries(dicts, names, save_dir):
    """统计分析字典"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (d, name) in enumerate(zip(dicts, names)):
        values = d.flatten()
        axes[idx].hist(values, bins=50, alpha=0.7, color=['blue','green','red'][idx], edgecolor='black')
        axes[idx].set_title(f'{name}\nMean={values.mean():.4f}, Std={values.std():.4f}')
        axes[idx].set_xlabel('Weight')
        axes[idx].set_ylabel('Frequency')
        axes[idx].axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 稀疏度
    def gini(d):
        d_flat = np.abs(d.flatten())
        sorted_d = np.sort(d_flat)
        n = len(sorted_d)
        cumsum = np.cumsum(sorted_d).astype(float)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, [gini(d) for d in dicts], color=['blue','green','red'])
    ax.set_ylabel('Gini Sparsity')
    ax.set_title('Dictionary Sparsity (higher = sparser)')
    ax.set_ylim([0, 1])
    for bar, val in zip(bars, [gini(d) for d in dicts]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/sparsity.png', dpi=150, bbox_inches='tight')
    plt.close()

def compare_single_atom(S_weight, Di_weight, save_dir):
    """对比单个原子"""
    # 取第一个原子
    S_atom = S_weight[0, 0]
    Di_atom = Di_weight[0, 0]
    combined = S_atom + Di_atom
    
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    
    atoms = [S_atom, Di_atom, combined, Di_atom - S_atom]
    titles = ['S (Static)', 'Di (Dynamic)', 'S + Di', 'Di - S']
    cmaps = ['RdBu', 'RdBu', 'RdBu', 'RdBu']
    
    for idx, (atom, title, cmap) in enumerate(zip(atoms, titles, cmaps)):
        vmax = max(abs(atom.min()), abs(atom.max()))
        im = axes[idx].imshow(atom, cmap=cmap, vmin=-vmax, vmax=vmax, interpolation='nearest')
        axes[idx].set_title(title)
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/atom_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/atom_comparison.png")

def visualize_input_dependence(model, device, save_dir='dictionary_vis'):
    """可视化不同输入对Di的影响"""
    model.eval()
    Di_list = []
    inputs = []
    
    # 生成不同类型的输入
    with torch.no_grad():
        # 1. 随机噪声
        inputs.append(torch.randn(1, model.n_channels, 128, 128).to(device))
        # 2. 常数图
        inputs.append(torch.ones(1, model.n_channels, 128, 128).to(device) * 0.5)
        # 3. 渐变图
        grad = torch.linspace(0, 1, 128).view(1, 1, 128, 1).repeat(1, model.n_channels, 1, 128).to(device)
        inputs.append(grad)
        
        for inp in inputs:
            Di_batch = model.di_gen(inp)
            Di_list.append(Di_batch[0].cpu().numpy())
    
    # 可视化对比
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    titles = ['Random Noise', 'Constant', 'Gradient']
    
    for idx, (inp, Di) in enumerate(zip(inputs, Di_list)):
        # 显示输入
        inp_disp = inp[0, 0].cpu().numpy()
        axes[0, idx].imshow(inp_disp, cmap='gray')
        axes[0, idx].set_title(f'Input: {titles[idx]}')
        axes[0, idx].axis('off')
        
        # 显示Di均值
        Di_mean = Di.mean(axis=(0, 1))
        axes[1, idx].imshow(Di_mean, cmap='plasma', interpolation='nearest')
        axes[1, idx].set_title(f'Di Mean Pattern\nStd={Di.std():.4f}')
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/input_dependence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/input_dependence.png")

##########################################################################
# 主程序
##########################################################################

if __name__ == "__main__":
    # 配置参数
    opt = {
        "n_channels": 1,   # 灰度图
        "d_size": 3,       # 字典核大小
        "stage": 3
    }
    
    print("创建模型...")
    model = denoise_Net_admm_restormer(opt)
    
    # 加载预训练权重（如果有）
    checkpoint_path = "checkpoints/best.pth"  # 修改为你的路径
    if os.path.exists(checkpoint_path):
        print(f"加载权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        print("加载成功！")
    else:
        print(f"警告: 未找到 {checkpoint_path}")
        print("使用随机初始化权重")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # 创建测试输入
    test_input = torch.randn(1, 1, 256, 256).to(device)
    print(f"输入形状: {test_input.shape}")
    
    # 运行可视化
    print("\n开始字典可视化...")
    S, S_T, Di = visualize_all_dictionaries(model, test_input, save_dir='dictionary_vis')
    
    # 输入依赖性分析
    visualize_input_dependence(model, device, save_dir='dictionary_vis')
    
    print("\n" + "=" * 60)
    print("可视化完成！结果保存在 'dictionary_vis' 目录")
    print("=" * 60)
    print(f"\n生成的文件:")
    print("  - S_dictionary.png: 静态字典S")
    print("  - S_T_dictionary.png: 静态字典S_T")
    print("  - Di_dictionary.png: 动态字典Di")
    print("  - statistics.png: 权重分布统计")
    print("  - sparsity.png: 稀疏度对比")
    print("  - atom_comparison.png: 单个原子对比")
    print("  - input_dependence.png: 输入依赖性")