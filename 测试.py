import os
import csv
import torch
import numpy as np
import torch.utils.data as data
import matplotlib.pyplot as plt

import Net.denoise_net as net
from utils.dataset_admm import get_data
import utils.utils_option as option
import utils.utils_image as image


def to_4d(x):
    if x.dim() == 3:
        x = x.unsqueeze(1)
    return x


def tensor_to_numpy_img(x):
    """
    输入:
        x: torch.Tensor [1,C,H,W] or [C,H,W]
    输出:
        numpy 图像，适合 matplotlib 显示
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu()

    if x.dim() == 4:
        x = x[0]

    if x.dim() == 3 and x.size(0) == 1:
        return x.squeeze(0).numpy()

    if x.dim() == 3 and x.size(0) == 3:
        return x.permute(1, 2, 0).numpy()

    return x.numpy()


def normalize_for_show(img):
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-12:
        return np.zeros_like(img, dtype=np.float32)
    return (img - mn) / (mx - mn)


def save_compare_figure(inp, out, gt, save_path, psnr_val, ssim_val):
    err = np.abs(out.astype(np.float32) - gt.astype(np.float32))
    err_show = normalize_for_show(err)

    plt.figure(figsize=(18, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(inp, cmap='gray' if inp.ndim == 2 else None)
    plt.title("Input")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(out, cmap='gray' if out.ndim == 2 else None)
    plt.title(f"Output\nPSNR={psnr_val:.6f}\nSSIM={ssim_val:.6f}")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(gt, cmap='gray' if gt.ndim == 2 else None)
    plt.title("GT")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(err_show, cmap='jet')
    plt.title("Abs Error")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    json_path = "./options/train_options.json"
    opt = option.parse(json_path, is_train=False)

    save_root = "./visual_eval_results"
    save_img_dir = os.path.join(save_root, "images")
    os.makedirs(save_img_dir, exist_ok=True)

    csv_path = os.path.join(save_root, "metrics.csv")
    txt_path = os.path.join(save_root, "summary.txt")

    # 数据
    valid_set = get_data(opt, 'valid')
    if isinstance(valid_set, list):
        test_loaders = [
            data.DataLoader(
                dataset=ds,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            ) for ds in valid_set
        ]
    else:
        test_loaders = [
            data.DataLoader(
                dataset=valid_set,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
        ]

    # 模型
    model = net.denoise_Net_admm_restormer(opt).to(device)

    ckpt_path = os.path.join(opt['model_save'], 'model_best.pth')
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()

    rows = []
    psnr_all = []
    ssim_all = []

    global_idx = 0

    with torch.no_grad():
        for loader_idx, test_loader in enumerate(test_loaders):
            for batch_idx, batch in enumerate(test_loader):

                # 兼容返回 (H, L) 或 (H, L, extra)
                if len(batch) == 3:
                    img_H, img_L, extra = batch
                    extra_info = str(extra)
                else:
                    img_H, img_L = batch
                    extra_info = "None"

                img_H = to_4d(img_H).to(device)
                img_L = to_4d(img_L).to(device)

                # ========== 修复点：添加 sigma 参数 ==========
                # 创建 sigma 张量（噪声水平为 0）
                sigma = torch.tensor([0.0]).to(device)
                
                # 调用模型时传入 sigma
                output, preds = model(img_L, sigma)
                # ===========================================
                
                if isinstance(output, (list, tuple)):
                    output = output[0]
                output = to_4d(output)

                # 转 uint 做评估（确保和你原工程一致）
                out_u = image.tensor2uint(output)
                gt_u = image.tensor2uint(img_H)
                in_u = image.tensor2uint(img_L)

                psnr_val = image.calculate_psnr(out_u, gt_u)
                ssim_val = image.calculate_ssim(out_u, gt_u)

                psnr_all.append(psnr_val)
                ssim_all.append(ssim_val)

                print(
                    f"[{global_idx:04d}] loader={loader_idx} batch={batch_idx} "
                    f"PSNR(raw)={psnr_val:.6f}, SSIM(raw)={ssim_val:.6f}"
                )

                # 保存并排图
                fig_path = os.path.join(save_img_dir, f"sample_{global_idx:04d}.png")
                save_compare_figure(
                    inp=in_u,
                    out=out_u,
                    gt=gt_u,
                    save_path=fig_path,
                    psnr_val=psnr_val,
                    ssim_val=ssim_val
                )

                rows.append([
                    global_idx,
                    loader_idx,
                    batch_idx,
                    psnr_val,
                    ssim_val,
                    extra_info,
                    fig_path
                ])

                global_idx += 1

    # 保存 CSV
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index", "loader_idx", "batch_idx",
            "psnr_raw", "ssim_raw", "extra_info", "fig_path"
        ])
        writer.writerows(rows)

    # 汇总
    psnr_mean = float(np.mean(psnr_all)) if len(psnr_all) > 0 else 0.0
    psnr_min = float(np.min(psnr_all)) if len(psnr_all) > 0 else 0.0
    psnr_max = float(np.max(psnr_all)) if len(psnr_all) > 0 else 0.0

    ssim_mean = float(np.mean(ssim_all)) if len(ssim_all) > 0 else 0.0
    ssim_min = float(np.min(ssim_all)) if len(ssim_all) > 0 else 0.0
    ssim_max = float(np.max(ssim_all)) if len(ssim_all) > 0 else 0.0

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Total samples: {len(psnr_all)}\n")
        f.write(f"PSNR mean: {psnr_mean:.6f}\n")
        f.write(f"PSNR min : {psnr_min:.6f}\n")
        f.write(f"PSNR max : {psnr_max:.6f}\n")
        f.write(f"SSIM mean: {ssim_mean:.6f}\n")
        f.write(f"SSIM min : {ssim_min:.6f}\n")
        f.write(f"SSIM max : {ssim_max:.6f}\n")

    print("\n===== Summary =====")
    print(f"Total samples: {len(psnr_all)}")
    print(f"PSNR mean: {psnr_mean:.6f}")
    print(f"PSNR min : {psnr_min:.6f}")
    print(f"PSNR max : {psnr_max:.6f}")
    print(f"SSIM mean: {ssim_mean:.6f}")
    print(f"SSIM min : {ssim_min:.6f}")
    print(f"SSIM max : {ssim_max:.6f}")
    print(f"\nCSV saved to: {csv_path}")
    print(f"Summary saved to: {txt_path}")
    print(f"Images saved to: {save_img_dir}")