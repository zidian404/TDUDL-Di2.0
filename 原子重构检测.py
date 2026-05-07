import torch
import torch.nn.functional as F
import os
import numpy as np
from glob import glob
import cv2
import Net.denoise_net as net
import utils.utils_option as option
import utils.utils_image as image

def img_to_tensor(img_uint):
    """将 uint8 图像转换为 normalized tensor"""
    img_float = img_uint.astype(np.float32) / 255.0
    if img_float.ndim == 2:
        img_float = img_float[:, :, np.newaxis]
    img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).float()
    return img_tensor

def synthesis_by_atoms(X, atoms, gates, k=3):
    B, C_coef, H, W = X.shape
    N, C_img, _, ksize, _ = atoms.shape
    assert ksize == k
    pad = k // 2
    outs = []
    for b in range(B):
        x_b = X[b:b+1]
        y_b = torch.zeros(1, C_img, H, W, device=x_b.device)
        for n in range(N):
            y_n = F.conv2d(x_b, atoms[n], padding=pad)
            y_b = y_b + gates[b, n] * y_n
        outs.append(y_b)
    return torch.cat(outs, dim=0)

def verify_model():
    opt = option.parse("./options/test_options.json", is_train=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = net.denoise_Net_admm_restormer(opt).to(device)
    
    path = opt["pretained_path"]["path"] if isinstance(opt["pretained_path"], dict) else opt["pretained_path"]
    print(f"Loading model from: {path}")
    state = torch.load(path, map_location="cpu")
    if "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    test_dir = opt["test"]["dataroot_H"]
    img_paths = glob(os.path.join(test_dir, "*"))
    if len(img_paths) == 0:
        print(f"❌ 找不到图像在: {test_dir}")
        return
    img_path = img_paths[0]
    print(f"Using test image: {img_path}")
    
    img_uint = image.imread_uint(img_path, n_channels=1)
    img = img_to_tensor(img_uint).to(device).unsqueeze(0)
    sigma = torch.full((1, 1), 0.0, device=device)
    
    with torch.no_grad():
        result = model(img, sigma, return_sparse=True, return_atom_weights=True)
        
        if len(result) == 4:
            final_out, _, X, gates = result
        elif len(result) == 3:
            final_out, X, gates = result
        else:
            print(f"⚠️ 模型返回 {len(result)} 个值")
            return
        
        atoms = model.atom_bank()
        
        print(f"\n=== 基本信息 ===")
        print(f"final_out shape: {final_out.shape}")
        print(f"X shape: {X.shape}")
        print(f"gates shape: {gates.shape}")
        print(f"atoms shape: {atoms.shape}")
        
        print(f"\n{'='*20} Gates 分布诊断 {'='*20}")
        if gates.dim() == 2:
            B, N = gates.shape
            for b in range(min(B, 3)):
                gate_b = gates[b]
                nonzero = (gate_b.abs() > 1e-6).sum().item()
                topk_vals, topk_idx = torch.topk(gate_b, min(5, N))
                print(f"图像 {b}: 非零gate数={nonzero}")
                print(f"  Top-5 原子: {topk_idx.tolist()}")
                print(f"  Top-5 权重: {[f'{v:.4f}' for v in topk_vals.tolist()]}")
                if nonzero == 1:
                    print(f"  ⚠️ 警告：gate 严重集中在单个原子！")
                elif topk_vals[0].item() > 0.5:
                    print(f"  ⚠️ 警告：top-1 权重过高 ({topk_vals[0].item():.4f})")
                else:
                    print(f"  ✅ 良好：多个原子协作")
        
        print(f"\n{'='*20} 原子相似度诊断 {'='*20}")
        atoms_flat = atoms.view(atoms.shape[0], -1)
        similarity = F.cosine_similarity(atoms_flat.unsqueeze(1), atoms_flat.unsqueeze(0), dim=2)
        N = similarity.shape[0]
        mask = ~torch.eye(N, dtype=bool, device=similarity.device)
        off_diag_sim = similarity[mask]
        print(f"原子间平均相似度: {off_diag_sim.mean().item():.4f}")
        print(f"原子间最大相似度: {off_diag_sim.max().item():.4f}")
        
        entropy = -(gates[0] * torch.log(gates[0] + 1e-8)).sum().item()
        print(f"\nGate 熵值: {entropy:.4f} (最大可能: {np.log(N):.4f})")
        if entropy > np.log(N) * 0.7:
            print(f"  ✅ 良好：gate 分布均匀")
        elif entropy < np.log(N) * 0.3:
            print(f"  ⚠️ 警告：gate 分布过于集中")

if __name__ == "__main__":
    verify_model()