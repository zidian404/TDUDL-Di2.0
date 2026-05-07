from typing import List
import torch.utils.data as data
import torch
import time
import os
from tqdm import tqdm
import logging
from torch import optim
import matplotlib.pyplot as plt
from math import log
import signal
import sys
import numpy as np
import torch.nn.functional as F

import Net.denoise_net as net
from utils.dataset_admm import get_data
from utils.loss_function import loss_function
import utils.utils_option as option
import utils.utils_image as image
from utils import utils_logger


def adjust_learning_rate(opt, epo, lr_ini, max_epoch):
    P1 = 50
    P2 = 200 - P1
    if epo < P1:
        lr = lr_ini * (0.65 ** (epo // max(1, int(P1 // log(0.1, 0.65)))))
    else:
        lr = lr_ini * 0.1 * (0.85 ** ((epo - P1) // max(1, int(P2 // log(0.1, 0.85)))))
    for param_group in opt.param_groups:
        param_group['lr'] = lr


def create_blend_window(tile_size, overlap, power=2.0, device='cpu'):
    """
    2D cosine-like blending window.
    Center weight is high, boundary weight is low, reducing tile seams.
    """
    if overlap <= 0:
        return torch.ones((tile_size, tile_size), dtype=torch.float32, device=device)

    overlap = min(overlap, tile_size // 2)
    w = torch.ones(tile_size, dtype=torch.float32, device=device)
    ramp = torch.linspace(0, 1, overlap, dtype=torch.float32, device=device)
    ramp = 0.5 - 0.5 * torch.cos(torch.pi * ramp)
    ramp = ramp.pow(power)

    w[:overlap] = ramp
    w[-overlap:] = torch.flip(ramp, dims=[0])
    window_2d = torch.outer(w, w)
    return window_2d.clamp_min(1e-6)


def tiled_inference(model, img_L, noise_level, tile_size=768, tile_overlap=128, device='cuda'):
    """
    Robust tiled inference for large images.
    - overlapping tiles
    - smooth blending window
    - border-safe coordinates
    - output normalized by accumulated weights
    """
    model.eval()

    if img_L.dim() == 3:
        img_L = img_L.unsqueeze(0)
    if noise_level.dim() == 0:
        noise_level = noise_level.view(1)
    elif noise_level.dim() > 1:
        noise_level = noise_level.view(noise_level.shape[0], -1)[:, 0]

    original_dim3 = False
    if len(img_L.shape) == 3:
        original_dim3 = True
        img_L = img_L.unsqueeze(0)

    B, C, H, W = img_L.shape
    assert B == 1, "Validation tiled inference currently expects batch_size=1"

    if H <= tile_size and W <= tile_size:
        with torch.no_grad():
            output, _ = model(img_L.to(device, non_blocking=True), noise_level.to(device, non_blocking=True))
            if isinstance(output, (list, tuple)):
                output = output[0]
            return output.clamp(0, 1)

    stride = tile_size - tile_overlap
    if stride <= 0:
        raise ValueError("tile_overlap must be smaller than tile_size")

    y_coords = list(range(0, max(H - tile_size + 1, 1), stride))
    x_coords = list(range(0, max(W - tile_size + 1, 1), stride))
    if len(y_coords) == 0:
        y_coords = [0]
    if len(x_coords) == 0:
        x_coords = [0]
    if y_coords[-1] != H - tile_size:
        y_coords.append(max(H - tile_size, 0))
    if x_coords[-1] != W - tile_size:
        x_coords.append(max(W - tile_size, 0))
    y_coords = sorted(set(int(y) for y in y_coords))
    x_coords = sorted(set(int(x) for x in x_coords))

    blend_window = create_blend_window(tile_size, tile_overlap, power=2.0, device=device)
    blend_window = blend_window.unsqueeze(0).unsqueeze(0)

    output_acc = torch.zeros((1, C, H, W), dtype=torch.float32, device=device)
    weight_acc = torch.zeros((1, 1, H, W), dtype=torch.float32, device=device)

    with torch.no_grad():
        for y in y_coords:
            for x in x_coords:
                h_start, h_end = y, y + tile_size
                w_start, w_end = x, x + tile_size

                tile = img_L[:, :, h_start:h_end, w_start:w_end].to(device, non_blocking=True)
                out_tile, _ = model(tile, noise_level.to(device, non_blocking=True))
                if isinstance(out_tile, (list, tuple)):
                    out_tile = out_tile[0]
                out_tile = out_tile.float()

                output_acc[:, :, h_start:h_end, w_start:w_end] += out_tile * blend_window
                weight_acc[:, :, h_start:h_end, w_start:w_end] += blend_window

    final_output = output_acc / weight_acc.clamp_min(1e-6)
    if original_dim3:
        final_output = final_output.squeeze(0)
    return final_output.clamp(0, 1)


def validate_with_tiling(model, test_loaders, test_sigmas, criterion, device, logger,
                         tile_size=768, tile_overlap=128, save_vis=False, vis_dir=None):
    model.eval()
    test_loss = 0.0
    test_psnr = 0.0
    test_ssim = 0.0
    total_batches = 0
    valid_batches = 0

    skip_stats = {
        'input_nan': 0,
        'output_nan': 0,
        'loss_nan': 0,
        'other_error': 0,
    }
    val_results = []

    with torch.no_grad():
        for loader_idx, test_loader in enumerate(test_loaders):
            current_sigma = test_sigmas[loader_idx] if loader_idx < len(test_sigmas) else 25
            print(f"\n处理验证集 {loader_idx+1}/{len(test_loaders)} (噪声水平: {current_sigma})")

            val_set_loss = 0.0
            val_set_psnr = 0.0
            val_set_ssim = 0.0
            val_set_batches = 0

            for batch_idx, (img_H, img_L, noise_level) in enumerate(tqdm(test_loader, desc=f"验证集 {loader_idx+1}")):
                total_batches += 1

                if torch.isnan(img_L).any() or torch.isinf(img_L).any():
                    skip_stats['input_nan'] += 1
                    continue

                try:
                    img_H_device = img_H.to(device, non_blocking=True)
                    test_out = tiled_inference(
                        model=model,
                        img_L=img_L,
                        noise_level=noise_level,
                        tile_size=tile_size,
                        tile_overlap=tile_overlap,
                        device=device
                    )

                    if torch.isnan(test_out).any() or torch.isinf(test_out).any():
                        skip_stats['output_nan'] += 1
                        continue

                    if test_out.dim() == 3:
                        test_out = test_out.unsqueeze(0)
                    if img_H_device.dim() == 3:
                        img_H_device = img_H_device.unsqueeze(0)

                    current_loss = criterion(test_out, img_H_device).item()
                    if np.isnan(current_loss) or np.isinf(current_loss):
                        skip_stats['loss_nan'] += 1
                        continue

                    val_set_loss += current_loss
                    test_loss += current_loss
                    valid_batches += 1
                    val_set_batches += 1

                    test_out_uint = image.tensor2uint(test_out)
                    img_H_uint = image.tensor2uint(img_H_device)
                    current_psnr = image.calculate_psnr(test_out_uint, img_H_uint)
                    current_ssim = image.calculate_ssim(test_out_uint, img_H_uint)

                    val_set_psnr += current_psnr
                    val_set_ssim += current_ssim
                    test_psnr += current_psnr
                    test_ssim += current_ssim

                except Exception as e:
                    logger.error(f"验证阶段分块推理出错: {e}")
                    skip_stats['other_error'] += 1
                    continue

            if val_set_batches > 0:
                val_avg_loss = val_set_loss / val_set_batches
                val_avg_psnr = val_set_psnr / val_set_batches
                val_avg_ssim = val_set_ssim / val_set_batches
                val_results.append({
                    'sigma': current_sigma,
                    'loss': val_avg_loss,
                    'psnr': val_avg_psnr,
                    'ssim': val_avg_ssim,
                    'batches': val_set_batches
                })
                print(f"验证集{loader_idx+1} (σ={current_sigma}) 完成: {val_set_batches} batches, PSNR: {val_avg_psnr:.2f}, SSIM: {val_avg_ssim:.4f}")

    return {
        'valid_batches': valid_batches,
        'total_batches': total_batches,
        'skip_stats': skip_stats,
        'val_results': val_results,
        'avg_loss': test_loss / valid_batches if valid_batches > 0 else None,
        'avg_psnr': test_psnr / valid_batches if valid_batches > 0 else None,
        'avg_ssim': test_ssim / valid_batches if valid_batches > 0 else None,
    }


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"Using GPU for training! Device: {os.environ.get('CUDA_VISIBLE_DEVICES')}\n")
    else:
        print("Using CPU for training.\n")

    json_path = "./options/train_options.json"
    opt = option.parse(json_path, is_train=True)

    logger_name = 'train' + time.strftime('%Y_%m_%d_%H-%M-%S', time.localtime())
    utils_logger.logger_info(logger_name, os.path.join(opt['log_path'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    print("加载数据集...")
    train_set = get_data(opt, 'train')
    valid_set = get_data(opt, 'valid')

    train_loader = data.DataLoader(
        dataset=train_set,
        batch_size=opt.get('batch_size', 4),
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    test_loaders: List[data.DataLoader] = []
    test_sigmas: List[float] = []
    for valid_dataset in valid_set:
        test_loaders.append(data.DataLoader(
            dataset=valid_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=True
        ))
        if hasattr(valid_dataset, 'sigma'):
            sigma_val = valid_dataset.sigma
            if isinstance(sigma_val, (list, tuple)):
                sigma_val = sigma_val[0] if len(sigma_val) > 0 else 25
            test_sigmas.append(float(sigma_val))
        else:
            test_sigmas.append(25.0)

    print("初始化模型...")
    model = net.denoise_Net_admm_restormer(opt)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'])
    criterion = loss_function(opt['loss_function_index'])

    total = sum([param.nelement() for param in model.parameters()])
    logger.info(f"Number of parameter: {total / 1e6 :.2f}M")
    logger.info("start training...")

    start = time.time()
    loss_train = []
    test__loss = []
    test__psnr = []
    test__ssim = []
    best_psnr = 0
    best_epoch = 0
    batch_accumulation = max(1, opt.get('batch_accumulation', 1))
    max_accumulation = 0
    psnr_val_rgb = 0

    tile_size = opt.get('tile_size', 768)
    tile_overlap = opt.get('tile_overlap', 128)
    eval_num = opt.get('eval_num', 5)

    if opt.get("pretained_path", {}).get("index"):
        state = torch.load(opt['pretained_path']["path"], map_location=device)
        model.load_state_dict(state['state_dict'], strict=False)

    reduce_schedule = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.85, patience=5, threshold=1e-3,
        threshold_mode='abs', cooldown=0, min_lr=0, eps=1e-8
    )

    def signal_handler(sig, frame):
        print("\n训练被中断，正在保存当前进度...")
        torch.save({
            'epoch': locals().get('epoch', 0),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss_train': loss_train,
            'test__psnr': test__psnr,
            'test__ssim': test__ssim,
            'best_psnr': best_psnr,
            'best_epoch': best_epoch,
        }, os.path.join(opt['model_save'], 'model_interrupted.pth'))
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    for epoch in range(0, opt["max_epoch"]):
        try:
            if epoch < 200:
                adjust_learning_rate(optimizer, epoch, opt['lr'], opt['max_epoch'])
            else:
                reduce_schedule.step(psnr_val_rgb)

            model.train()
            optimizer.zero_grad()
            loss_epoch = 0
            batch_count = 0
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')

            for batch_idx, (img_H, img_L, noise_level) in enumerate(pbar):
                img_H = img_H.to(device, non_blocking=True)
                img_L = img_L.to(device, non_blocking=True)
                noise_level = noise_level.to(device, non_blocking=True)

                output, preds = model(img_L, noise_level)
                if isinstance(output, (list, tuple)):
                    output = output[0]
                if output.dim() == 3:
                    output = output.unsqueeze(1)
                if img_H.dim() == 3:
                    img_H = img_H.unsqueeze(1)

                loss = criterion(output, img_H) / batch_accumulation
                loss.backward()

                if (batch_idx + 1) % batch_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                loss_epoch += loss.item() * batch_accumulation
                batch_count += 1

                if batch_idx % 10 == 0:
                    pbar.set_postfix({
                        'Loss': f'{loss_epoch / max(batch_count,1):.4f}',
                        'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
                    })

            if len(train_loader) % batch_accumulation != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            avg_loss = loss_epoch / len(train_loader)
            loss_train.append(avg_loss)
            logger.info(f"epoch:[{epoch + 1}/{opt['max_epoch']}], 平均loss: {avg_loss:.4f}")

            if (epoch + 1) % eval_num == 0:
                print(f"\n开始验证阶段 (重叠分块推理) - Epoch {epoch+1}")
                val_stats = validate_with_tiling(
                    model=model,
                    test_loaders=test_loaders,
                    test_sigmas=test_sigmas,
                    criterion=criterion,
                    device=device,
                    logger=logger,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                )

                if val_stats['valid_batches'] > 0:
                    print("\n验证完成统计:")
                    print(f"总batch数: {val_stats['total_batches']}, 有效batch数: {val_stats['valid_batches']}")
                    print(f"跳过统计: {val_stats['skip_stats']}")
                    for i, result in enumerate(val_stats['val_results']):
                        print(f"  验证集{i+1} (σ={result['sigma']}): PSNR: {result['psnr']:.2f}, SSIM: {result['ssim']:.4f}, Loss: {result['loss']:.4f}")

                    avg_test_loss = val_stats['avg_loss']
                    psnr_val_rgb = val_stats['avg_psnr']
                    ssim_val_rgb = val_stats['avg_ssim']

                    if psnr_val_rgb > best_psnr:
                        best_psnr = psnr_val_rgb
                        best_epoch = epoch + 1
                        max_accumulation = 0
                        torch.save({
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'best_psnr': best_psnr,
                        }, os.path.join(opt['model_save'], 'model_best.pth'))
                        logger.info(f'保存最佳模型，PSNR: {best_psnr:.4f}')
                    else:
                        max_accumulation += 1

                    logger.info(f'[epoch {epoch+1} PSNR: {psnr_val_rgb:.4f} Best_PSNR: {best_psnr:.4f} (epoch {best_epoch})]')
                    logger.info(f'Test metrics - Loss: {avg_test_loss:.4f}, SSIM: {ssim_val_rgb:.4f}, 有效batch数: {val_stats["valid_batches"]}')

                    test__loss.append(avg_test_loss)
                    test__psnr.append(psnr_val_rgb)
                    test__ssim.append(ssim_val_rgb)
                else:
                    logger.warning('没有有效的验证数据，跳过这个验证周期')

            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, os.path.join(opt['model_save'], f'model_epoch_{epoch+1}.pth'))
                logger.info(f'保存检查点: model_epoch_{epoch+1}.pth')

            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(opt['model_save'], 'model_latest.pth'))

            if max_accumulation >= 10:
                logger.info(f'早停于epoch {epoch+1}')
                break

        except Exception as e:
            print(f"训练出错: {e}, 正在保存进度...")
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss_train': loss_train,
                'test__psnr': test__psnr,
                'test__ssim': test__ssim,
                'best_psnr': best_psnr,
                'best_epoch': best_epoch
            }, os.path.join(opt['model_save'], 'model_error.pth'))
            logger.error(f"训练出错，进度已保存: {e}")
            raise e

    end = time.time()
    training_hours = (end - start) / 3600
    logger.info(f'总训练时间: {training_hours:.2f}小时')

    if len(loss_train) > 0:
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(loss_train)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)

        if len(test__loss) > 0:
            plt.subplot(2, 2, 2)
            eval_epochs = [i * eval_num for i in range(1, len(test__loss) + 1)]
            plt.plot(eval_epochs, test__loss, 'o-')
            plt.title("Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True)

            plt.subplot(2, 2, 3)
            plt.plot(eval_epochs, test__psnr, 'o-')
            plt.title("Val PSNR")
            plt.xlabel("Epoch")
            plt.ylabel("PSNR (dB)")
            plt.grid(True)

            plt.subplot(2, 2, 4)
            plt.plot(eval_epochs, test__ssim, 'o-')
            plt.title("Val SSIM")
            plt.xlabel("Epoch")
            plt.ylabel("SSIM")
            plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(opt['log_path'], 'training_curves.png'), dpi=150)
        logger.info('训练曲线已保存')