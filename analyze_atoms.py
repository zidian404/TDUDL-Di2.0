import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
import os

def visualize_and_analyze_atoms(model, save_dir='atom_analysis', device='cuda'):
    """
    完整的原子可视化和分析函数
    
    Args:
        model: 训练好的模型
        save_dir: 保存结果的目录
        device: 设备
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    
    model.eval()
    model.to(device)
    
    print("=" * 60)
    print("开始原子分析...")
    print("=" * 60)
    
    # 1. 尝试获取原子
    atoms = None
    n_atoms = None
    atom_size = None
    
    # 方法1: 如果模型有 get_atoms 方法
    if hasattr(model, 'get_atoms'):
        atoms = model.get_atoms()
        print("✅ 通过 get_atoms() 获取原子")
    
    # 方法2: 如果模型有 dictionary 属性 (常见于字典学习方法)
    elif hasattr(model, 'dictionary'):
        atoms = model.dictionary
        print("✅ 通过 dictionary 属性获取原子")
    
    # 方法3: 如果模型有编码器权重 (自动编码器)
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'weight'):
        # 假设第一层卷积的权重作为原子
        atoms = model.encoder.weight.data
        print("✅ 通过 encoder 第一层权重获取原子")
    
    # 方法4: 遍历模型参数查找可解释的字典
    else:
        print("⚠️ 未找到明显的原子结构，尝试从卷积层提取...")
        conv_layers = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) and module.weight.shape[1] <= 3:
                conv_layers.append((name, module))
        
        if conv_layers:
            # 使用第一个卷积层的权重
            name, conv = conv_layers[0]
            atoms = conv.weight.data
            print(f"✅ 从卷积层 '{name}' 提取原子, shape: {atoms.shape}")
    
    if atoms is None:
        print("❌ 无法自动提取原子，请手动指定模型中的原子层")
        return None
    
    # 解析原子形状
    if atoms.dim() == 4:  # [out_channels, in_channels, height, width]
        n_atoms = atoms.shape[0]
        atom_size = atoms.shape[2]
        # 如果是多通道，取平均或第一个通道
        if atoms.shape[1] > 1:
            atoms = atoms.mean(dim=1)  # [n_atoms, h, w]
        else:
            atoms = atoms.squeeze(1)   # [n_atoms, h, w]
    elif atoms.dim() == 3:  # [n_atoms, height, width]
        n_atoms = atoms.shape[0]
        atom_size = atoms.shape[1]
    else:
        print(f"❌ 无法识别的原子形状: {atoms.shape}")
        return None
    
    print(f"\n📊 原子统计:")
    print(f"  - 原子数量: {n_atoms}")
    print(f"  - 原子尺寸: {atom_size}x{atom_size}")
    print(f"  - 数据类型: {atoms.dtype}")
    print(f"  - 值范围: [{atoms.min().item():.3f}, {atoms.max().item():.3f}]")
    
    # 2. 原子归一化可视化
    atoms_norm = (atoms - atoms.min()) / (atoms.max() - atoms.min() + 1e-8)
    
    # 3. 计算原子统计特性
    atom_stats = {
        'mean': atoms.mean(dim=(1,2)).cpu().numpy(),
        'std': atoms.std(dim=(1,2)).cpu().numpy(),
        'sparsity': (atoms.abs() < 0.1).float().mean(dim=(1,2)).cpu().numpy(),
        'energy': (atoms ** 2).sum(dim=(1,2)).cpu().numpy(),
    }
    
    # 4. 计算原子间的相似度
    atoms_flat = atoms.view(n_atoms, -1)
    atoms_norm_l2 = atoms_flat / (atoms_flat.norm(dim=1, keepdim=True) + 1e-8)
    similarity_matrix = torch.mm(atoms_norm_l2, atoms_norm_l2.t()).cpu().numpy()
    
    # 移除对角线
    np.fill_diagonal(similarity_matrix, 0)
    max_similarity = similarity_matrix.max()
    mean_similarity = similarity_matrix.mean()
    
    atom_stats['max_similarity'] = max_similarity
    atom_stats['mean_similarity'] = mean_similarity
    
    print(f"\n📈 原子质量指标:")
    print(f"  - 平均能量: {atom_stats['energy'].mean():.3f}")
    print(f"  - 能量标准差: {atom_stats['energy'].std():.3f}")
    print(f"  - 平均稀疏度 (|x|<0.1): {atom_stats['sparsity'].mean():.3f}")
    print(f"  - 原子间最大相似度: {max_similarity:.3f}")
    print(f"  - 原子间平均相似度: {mean_similarity:.3f}")
    
    # 判断原子质量
    print(f"\n🔍 原子质量诊断:")
    
    issues = []
    
    # 检查是否过于相似
    if max_similarity > 0.95:
        issues.append(f"⚠️ 原子高度相似 (max sim={max_similarity:.3f})，可能存在冗余")
    elif max_similarity > 0.8:
        issues.append(f"⚡ 原子有一定相似度 (max sim={max_similarity:.3f})")
    else:
        print(f"✅ 原子多样性良好 (max sim={max_similarity:.3f})")
    
    # 检查是否过于稀疏或过于稠密
    mean_sparsity = atom_stats['sparsity'].mean()
    if mean_sparsity > 0.9:
        issues.append(f"⚠️ 原子过于稀疏 (sparsity={mean_sparsity:.3f})，可能全是0")
    elif mean_sparsity < 0.1:
        issues.append(f"⚠️ 原子过于稠密 (sparsity={mean_sparsity:.3f})，信息熵低")
    else:
        print(f"✅ 原子稀疏度合理 (sparsity={mean_sparsity:.3f})")
    
    # 检查能量分布
    energy_cv = atom_stats['energy'].std() / (atom_stats['energy'].mean() + 1e-8)
    if energy_cv < 0.2:
        issues.append(f"⚠️ 原子能量过于均匀 (CV={energy_cv:.3f})，可能未特化")
    elif energy_cv > 1.0:
        issues.append(f"⚠️ 原子能量差异过大 (CV={energy_cv:.3f})，部分原子主导")
    else:
        print(f"✅ 原子能量分布合理 (CV={energy_cv:.3f})")
    
    if not issues:
        print("✅ 所有指标正常，原子质量良好！")
    
    # 5. 可视化
    print(f"\n🎨 生成可视化图表...")
    
    # 5.1 原子网格可视化
    n_cols = min(8, int(np.sqrt(n_atoms)))
    n_rows = min(8, (n_atoms + n_cols - 1) // n_cols)
    n_display = min(n_atoms, 64)  # 最多显示64个
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    if n_display < n_rows * n_cols:
        for i in range(n_display, n_rows * n_cols):
            axes.flat[i].axis('off')
    
    for i in range(n_display):
        row = i // n_cols
        col = i % n_cols
        if n_rows > 1:
            ax = axes[row, col]
        else:
            ax = axes[col]
        
        atom_img = atoms_norm[i].cpu().numpy()
        ax.imshow(atom_img, cmap='RdBu', interpolation='nearest')
        ax.axis('off')
        ax.set_title(f'{i}', fontsize=8)
    
    plt.suptitle(f'Learned Atoms (n={n_atoms}, size={atom_size})', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path / 'atoms_grid.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 保存: atoms_grid.png")
    
    # 5.2 相似度矩阵热力图
    if n_atoms <= 100:  # 避免太大
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(similarity_matrix[:50, :50], cmap='hot', vmin=0, vmax=1)
        ax.set_xlabel('Atom Index')
        ax.set_ylabel('Atom Index')
        ax.set_title('Atom Similarity Matrix')
        plt.colorbar(im, ax=ax)
        plt.savefig(save_path / 'atom_similarity_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 保存: atom_similarity_matrix.png")
    
    # 5.3 原子统计分布
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 能量分布
    axes[0, 0].hist(atom_stats['energy'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Atom Energy')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Energy Distribution (CV={energy_cv:.3f})')
    axes[0, 0].axvline(atom_stats['energy'].mean(), color='r', linestyle='--', label=f'Mean: {atom_stats["energy"].mean():.2f}')
    axes[0, 0].legend()
    
    # 稀疏度分布
    axes[0, 1].hist(atom_stats['sparsity'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Sparsity (|x|<0.1 ratio)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Sparsity Distribution (mean={mean_sparsity:.3f})')
    
    # 相似度分布（不包括对角线）
    sim_flat = similarity_matrix[similarity_matrix > 0]
    if len(sim_flat) > 0:
        axes[1, 0].hist(sim_flat, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Pairwise Similarity')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'Similarity Distribution (max={max_similarity:.3f}, mean={mean_similarity:.3f})')
        axes[1, 0].axvline(mean_similarity, color='r', linestyle='--', label=f'Mean: {mean_similarity:.3f}')
        axes[1, 0].legend()
    
    # 均值 vs 标准差
    axes[1, 1].scatter(atom_stats['mean'], atom_stats['std'], alpha=0.6, s=20)
    axes[1, 1].set_xlabel('Mean Value')
    axes[1, 1].set_ylabel('Standard Deviation')
    axes[1, 1].set_title('Mean vs Std per Atom')
    axes[1, 1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path / 'atom_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 保存: atom_statistics.png")
    
    # 5.4 最激活的原子（如果模型有编码器）
    if hasattr(model, 'encode') or hasattr(model, 'encoder'):
        print(f"\n🔬 测试原子对真实图像的反应...")
        
        # 创建测试图像（随机纹理）
        test_img = torch.randn(1, 1, 128, 128).to(device)
        
        try:
            with torch.no_grad():
                if hasattr(model, 'encode'):
                    coeffs = model.encode(test_img)
                elif hasattr(model, 'encoder'):
                    coeffs = model.encoder(test_img)
                else:
                    coeffs = None
                
                if coeffs is not None:
                    # 计算系数统计
                    if coeffs.dim() > 2:
                        coeffs = coeffs.view(coeffs.size(0), coeffs.size(1), -1).mean(dim=2)
                    
                    activation = coeffs.abs().mean(dim=0).cpu().numpy()
                    top_k = 10
                    top_indices = np.argsort(activation)[-top_k:][::-1]
                    
                    print(f"  Top-{top_k} 最激活的原子: {top_indices}")
                    
                    # 可视化最激活的原子
                    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
                    for i, idx in enumerate(top_indices[:10]):
                        row, col = i // 5, i % 5
                        ax = axes[row, col]
                        atom_img = atoms_norm[idx].cpu().numpy()
                        ax.imshow(atom_img, cmap='RdBu')
                        ax.set_title(f'Atom {idx} (act={activation[idx]:.3f})')
                        ax.axis('off')
                    
                    plt.suptitle('Most Activated Atoms', fontsize=14)
                    plt.tight_layout()
                    plt.savefig(save_path / 'most_activated_atoms.png', dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"  ✅ 保存: most_activated_atoms.png")
        except Exception as e:
            print(f"  ⚠️ 无法测试原子激活: {e}")
    
    # 6. 保存原子数据
    atom_data = {
        'atoms': atoms.cpu().numpy(),
        'n_atoms': n_atoms,
        'atom_size': atom_size,
        'statistics': {
            'energy': atom_stats['energy'].tolist(),
            'sparsity': atom_stats['sparsity'].tolist(),
            'mean_similarity': float(mean_similarity),
            'max_similarity': float(max_similarity),
            'energy_cv': float(energy_cv),
        }
    }
    np.save(save_path / 'atoms_data.npy', atom_data)
    print(f"  ✅ 保存: atoms_data.npy")
    
    # 7. 生成报告（修复编码问题）
    report = f"""
    ========================================
    原子分析报告
    ========================================
    
    基本信息:
      - 原子数量: {n_atoms}
      - 原子尺寸: {atom_size}x{atom_size}
    
    质量指标:
      - 最大相似度: {max_similarity:.4f}
      - 平均相似度: {mean_similarity:.4f}
      - 能量变异系数: {energy_cv:.4f}
      - 平均稀疏度: {mean_sparsity:.4f}
    
    诊断结果:
    """
    
    for issue in issues:
        report += f"\n      {issue}"
    
    if not issues:
        report += "\n      [OK] 原子质量良好！"
    
    report += f"""
    
    建议:
    """
    
    # 根据指标给出建议
    if max_similarity > 0.9:
        report += "      - 减少原子数量或增加正交正则化\n"
    if mean_sparsity > 0.8:
        report += "      - 原子过于稀疏，检查是否有梯度消失问题\n"
    elif mean_sparsity < 0.2:
        report += "      - 原子过于稠密，增加稀疏正则化 (L1)\n"
    if energy_cv < 0.2:
        report += "      - 所有原子能量相近，尝试降低学习率\n"
    
    report += "    ========================================\n"
    
    # 使用 utf-8 编码保存报告，避免 Windows GBK 错误
    with open(save_path / 'atom_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print(f"\n✨ 分析完成！结果保存在: {save_path}")
    print("=" * 60)
    
    return atom_data


# 如果直接运行这个脚本，分析已训练好的模型
if __name__ == '__main__':
    import sys
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 检查是否需要导入模型定义
    # 注意：这里需要根据你的项目结构调整路径
    try:
        # 尝试导入你的模型定义
        sys.path.append(os.path.dirname(__file__))
        import Net.denoise_net as net
        import utils.utils_option as option
        
        # 加载配置
        json_path = "./options/train_options.json"
        if os.path.exists(json_path):
            opt = option.parse(json_path, is_train=True)
            
            # 初始化模型
            print("初始化模型...")
            model = net.denoise_Net_admm_restormer(opt)
            
            # 加载训练好的权重
            model_path = "./model_save/model_best.pth"  # 或者 model_latest.pth
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint['state_dict'])
                model.to(device)
                model.eval()
                print(f"✅ 加载模型: {model_path}")
                
                # 分析原子
                atom_data = visualize_and_analyze_atoms(
                    model, 
                    save_dir='./atom_analysis', 
                    device=device
                )
                print("分析完成！")
            else:
                print(f"❌ 找不到模型文件: {model_path}")
                print("请确保模型文件存在，或修改 model_path 变量")
        else:
            print(f"❌ 找不到配置文件: {json_path}")
            print("请确保在正确的目录下运行此脚本")
            
    except ImportError as e:
        print(f"❌ 导入模型定义失败: {e}")
        print("请确保在项目根目录下运行此脚本")
        print("或者手动修改代码中的导入路径")