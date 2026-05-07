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
import sys
import numpy as np
import torch.nn.functional as F

# 假设这些模块已存在于您的项目中
# 请确保这些文件在您的项目路径中是可导入的
import Net.denoise_net as net
from utils.dataset_admm import get_data
from utils.loss_function import loss_function
import utils.utils_option as option
import utils.utils_image as image
from utils import utils_logger


def adjust_learning_rate(opt, epo, lr_ini, max_epoch):
    """根据 epoch 调整学习率"""
    P1 = 50
    P2 = 200 - P1
    if epo < P1:
        # P1 之前使用 0.65 的指数衰减
        lr = lr_ini * (0.65 ** (epo // (P1 // log(0.1, 0.65))))
    else:
        # P1 之后使用 0.1 * 0.85 的指数衰减
        lr = lr_ini * 0.1 * (0.85 ** ((epo - P1) // (P2 // log(0.1, 0.85))))

    for param_group in opt.param_groups:
        param_group['lr'] = lr


def tiled_inference(model, img_L, noise_level, tile_size=1024, tile_overlap=64, device='cuda'):
    
    model.eval()
    
    # 确保输入是 (B=1, C, H, W)
    if img_L.dim() == 3:
        img_L = img_L.unsqueeze(0)
    
    B, C, H, W = img_L.shape
    
    # 目标输出图像的容器 (在 CPU 上以节省 GPU 内存)
    output_image = torch.zeros_like(img_L, device='cpu')
    weights = torch.zeros((H, W), device='cpu')
    
    # --- 辅助函数：创建加权模板 ---
    def get_tile_weights(size, overlap):
        """创建用于平滑重叠区域的权重图"""
        w = torch.ones((size, size))
        
        # 线性衰减（左/上）
        if overlap > 0:
            fade_in = torch.linspace(0, 1, overlap)
            w[:overlap, :] *= fade_in.view(-1, 1)
            w[:, :overlap] *= fade_in.view(1, -1)
        
        # 线性衰减（右/下）
        if overlap > 0:
            fade_out = torch.linspace(1, 0, overlap)
            w[-overlap:, :] *= fade_out.view(-1, 1)
            w[:, -overlap:] *= fade_out.view(1, -1)
            
        return w.to(device)

    # --- 计算平铺坐标 ---
    stride = tile_size - tile_overlap
    
    # 计算 y 轴坐标，确保最后一块覆盖到图像底部
    y_coords = np.arange(0, H, stride)
    if y_coords[-1] + tile_size > H:
        y_coords = np.append(y_coords[:-1], H - tile_size)
    y_coords = np.unique(y_coords[y_coords >= 0])
    
    # 计算 x 轴坐标，确保最后一块覆盖到图像右侧
    x_coords = np.arange(0, W, stride)
    if x_coords[-1] + tile_size > W:
        x_coords = np.append(x_coords[:-1], W - tile_size)
    x_coords = np.unique(x_coords[x_coords >= 0])
    
    tile_weights = get_tile_weights(tile_size, tile_overlap)
    
    # --- 遍历所有分块进行推理 ---
    for y in y_coords:
        for x in x_coords:
            h_start, h_end = int(y), int(y) + tile_size
            w_start, w_end = int(x), int(x) + tile_size
            
            # 裁剪分块 (输入仍在 CPU，只将 tile 传到 GPU)
            tile_L = img_L[:, :, h_start:h_end, w_start:w_end].to(device, non_blocking=True)
            
            # 模型推理
            with torch.no_grad():
                # 噪声水平也传到 device 上
                output_tile, _ = model(tile_L, noise_level.to(device, non_blocking=True))
                
                # 处理模型返回元组/列表的情况
                if isinstance(output_tile, (list, tuple)):
                    output_tile = output_tile[0]
                
            # 将加权结果累加到最终图像 (移回 CPU)
            # unsqueeze(0).unsqueeze(0) 确保权重和输出的维度匹配 (1, C, H, W)
            weighted_output = output_tile.cpu() * tile_weights.cpu().unsqueeze(0).unsqueeze(0)
            output_image[:, :, h_start:h_end, w_start:w_end] += weighted_output
            
            # 更新权重图
            weights[h_start:h_end, w_start:w_end] += tile_weights.cpu()

    # 归一化：将累加的结果除以权重总和
    weights = weights.unsqueeze(0).unsqueeze(0).clamp(min=1e-6) # [1, 1, H, W]
    final_output = output_image.to(device) / weights.to(device)

    return final_output.clamp(0, 1) # 确保输出在 [0, 1] 范围内


if __name__ == '__main__':
    # 基本设置
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"Using GPU for training! Device: {os.environ.get('CUDA_VISIBLE_DEVICES')}\n")
    else:
        print("Using CPU for training.\n")
            
    # 配置
    json_path = "./options/train_options.json"
    opt = option.parse(json_path, is_train=True)
    logger_name = 'train' + time.strftime('%Y_%m_%d_%H-%M-%S', time.localtime())
    utils_logger.logger_info(logger_name, os.path.join(opt['log_path'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    # 数据集
    print("加载数据集...")
    train_set = get_data(opt, 'train')
    valid_set = get_data(opt, 'valid')
    
    # 数据加载配置
    train_loader = data.DataLoader(
        dataset=train_set, 
        batch_size=opt.get('batch_size', 4),  # 默认使用 4
        shuffle=True,                  
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    
    test_loaders: List[data.DataLoader] = []
    for valid in valid_set:
        test_loaders.append(data.DataLoader(
            dataset=valid, 
            batch_size=1,
            shuffle=False, 
            num_workers=0,  
            drop_last=False, 
            pin_memory=True
        ))

    # 模型
    print("初始化模型...")
    model = net.denoise_Net_admm_restormer(opt) 
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'])
    criterion = loss_function(opt['loss_function_index'])
    
    total = sum([param.nelement() for param in model.parameters()])
    logger.info(f"Number of parameter: {total / 1e6 :.2f}M")
    logger.info("start training...")

    # 训练记录
    start = time.time()
    loss_train = []
    test__loss = []
    test__psnr = []
    test__ssim = []
    best_psnr = 0
    best_epoch = 0
    batch_accumulation = 1 
    max_accumulation = 0
    
    if opt.get("pretained_path", {}).get("index"):
         state = torch.load(opt['pretained_path']["path"])
         model.load_state_dict(state['state_dict'], strict=False)

    # 学习率调度器
    reduce_schedule = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.85, patience=5, threshold=1e-3, 
        threshold_mode='abs', cooldown=0, min_lr=0, eps=1e-8
    )

    eval_num = 5 

    for epoch in range(0, opt["max_epoch"]):
        try:
            # 学习率调整策略
            if epoch < 200:
                adjust_learning_rate(optimizer, epoch, opt['lr'], opt['max_epoch'])
            else:
                if 'psnr_val_rgb' not in vars():
                    psnr_val_rgb = 0
                reduce_schedule.step(psnr_val_rgb)

            # 训练阶段
            model.train()
            loss_epoch = 0
            batch_count = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
            
            for batch_idx, (img_H, img_L, noise_level) in enumerate(pbar):
                img_H = img_H.to(device, non_blocking=True)
                img_L = img_L.to(device, non_blocking=True)
                noise_level = noise_level.to(device, non_blocking=True)
                
                output, preds = model(img_L, noise_level)

                if isinstance(output, (list, tuple)):
                    output = output[0]
                
                # 确保维度正确 (B, C, H, W)
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
                
                if batch_idx % 20 == 0:
                    pbar.set_postfix({'Loss': f'{loss_epoch/batch_count:.4f}', 'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'})

            avg_loss = loss_epoch / len(train_loader)
            loss_train.append(avg_loss)
            logger.info(f"epoch:[{epoch + 1}/{opt['max_epoch']}], 平均loss: {avg_loss:.4f}")

            #### Evaluation - 使用分块推理 (Tiled Inference) ####
            should_eval = (epoch + 1) % eval_num == 0
            
            if should_eval:
                print(f"\n开始验证阶段 (分块推理) - Epoch {epoch+1}")
                model.eval()
                test_loss = 0
                test_psnr = 0
                test_ssim = 0
                total_batches = 0
                valid_batches = 0
                
                skip_stats = {'input_nan': 0, 'output_nan': 0, 'loss_nan': 0, 'other_error': 0}
                val_results = []
                
                # 分块参数：使用一个合理的值，您可以根据您的模型和GPU显存进行调整
                TILE_SIZE = 512 
                TILE_OVERLAP = 32
                
                with torch.no_grad():
                    for loader_idx, test_loader in enumerate(test_loaders):
                        print(f"\n处理验证集 {loader_idx+1}/{len(test_loaders)} (噪音水平: {opt['valid']['sigma'][loader_idx]})")
                        val_set_batches = 0
                        val_set_loss = 0
                        val_set_psnr = 0
                        val_set_ssim = 0
                        
                        for batch_idx, (img_H, img_L, noise_level) in enumerate(tqdm(test_loader, 0)):
                            total_batches += 1
                            
                            # 鲁棒性检查：输入数据
                            if torch.isnan(img_L).any() or torch.isinf(img_L).any():
                                skip_stats['input_nan'] += 1
                                continue
                            
                            # 真实值转移到 GPU
                            img_H_device = img_H.to(device, non_blocking=True)
                            
                            try:
                                # *** 调用分块推理函数 ***
                                test_out = tiled_inference(
                                    model, 
                                    img_L,                     # 噪声图 (通常在 CPU)
                                    noise_level,               # 噪声水平 (通常在 CPU)
                                    tile_size=TILE_SIZE, 
                                    tile_overlap=TILE_OVERLAP,
                                    device=device
                                )
                                
                                # 检查模型输出
                                if torch.isnan(test_out).any() or torch.isinf(test_out).any():
                                    skip_stats['output_nan'] += 1
                                    continue
                                    
                                # 确保维度正确 (B, C, H, W)
                                if test_out.dim() == 3:
                                    test_out = test_out.unsqueeze(0)
                                if img_H_device.dim() == 3:
                                    img_H_device = img_H_device.unsqueeze(0)
                                    
                                current_loss = criterion(test_out, img_H_device).item()
                                
                                # 检查loss
                                if np.isnan(current_loss) or np.isinf(current_loss):
                                    skip_stats['loss_nan'] += 1
                                    continue
                                    
                                val_set_loss += current_loss
                                test_loss += current_loss
                                valid_batches += 1
                                val_set_batches += 1
                                
                                # 计算 PSNR/SSIM
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
                        
                        # 记录每个验证集的结果
                        if val_set_batches > 0:
                            val_avg_loss = val_set_loss / val_set_batches
                            val_avg_psnr = val_set_psnr / val_set_batches
                            val_avg_ssim = val_set_ssim / val_set_batches
                            val_results.append({
                                'sigma': opt['valid']['sigma'][loader_idx],
                                'loss': val_avg_loss,
                                'psnr': val_avg_psnr,
                                'ssim': val_avg_ssim,
                                'batches': val_set_batches
                            })
                            print(f"验证集{loader_idx+1} (σ={opt['valid']['sigma'][loader_idx]}) 完成: {val_set_batches} batches, PSNR: {val_avg_psnr:.2f}, SSIM: {val_avg_ssim:.4f}")

                # 打印验证统计 
                print(f"\n验证完成统计:")
                print(f"总batch数: {total_batches}, 有效batch数: {valid_batches}")
                for i, result in enumerate(val_results):
                    print(f"  验证集{i+1} (σ={result['sigma']}): PSNR: {result['psnr']:.2f}, SSIM: {result['ssim']:.4f}, Loss: {result['loss']:.4f}")

                # 确保有有效的验证结果
                if valid_batches == 0:
                    logger.info("警告：没有有效的验证数据，跳过这个验证周期")
                    continue
                    
                avg_test_loss = test_loss / valid_batches
                psnr_val_rgb = test_psnr / valid_batches
                ssim_val_rgb = test_ssim / valid_batches
                
                # 保存最佳模型逻辑
                if psnr_val_rgb > best_psnr:
                    max_accumulation = 0
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch + 1
                    torch.save({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }, os.path.join(opt['model_save'], "model_best.pth"))
                    logger.info(f'保存最佳模型，PSNR: {best_psnr:.4f}')
                else:
                    max_accumulation += 1

                logger.info(f'[epoch {epoch+1} PSNR: {psnr_val_rgb:.4f} Best_PSNR: {best_psnr:.4f} (epoch {best_epoch})]')
                logger.info(f'Test metrics - Loss: {avg_test_loss:.4f}, SSIM: {ssim_val_rgb:.4f}, 有效batch数: {valid_batches}')
                
                test__loss.append(avg_test_loss)
                test__psnr.append(psnr_val_rgb)
                test__ssim.append(ssim_val_rgb)

            # 保存检查点和最新模型
            if (epoch + 1) % 10 == 0:
                torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, 
                            os.path.join(opt['model_save'], f'model_epoch_{epoch+1}.pth'))
                logger.info(f'保存检查点: model_epoch_{epoch+1}.pth')

            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, 
                        os.path.join(opt['model_save'], 'model_latest.pth'))
            
            # 早停
            if max_accumulation == 5:
                logger.info(f'早停于epoch {epoch+1}')
                break

        except Exception as e:
            print(f"训练出错: {e}, 正在保存进度...")
            torch.save({
                'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 
                'loss_train': loss_train, 'test__psnr': test__psnr, 'test__ssim': test__ssim, 
                'best_psnr': best_psnr, 'best_epoch': best_epoch
            }, os.path.join(opt['model_save'], 'model_error.pth'))
            logger.error(f"训练出错，进度已保存: {e}")
            raise e

    # 训练完成
    end = time.time()
    training_hours = (end - start) / 3600
    logger.info(f'总训练时间: {training_hours:.2f}小时')
    
    # 绘图
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(loss_train)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.subplot(2, 2, 2)
    # 注意：测试指标是在验证时记录的，因此绘图的 x 轴需要匹配记录的次数
    plt.plot(range(eval_num, len(test__loss) * eval_num + eval_num, eval_num), test__loss)
    plt.title("Test Loss (Evaluated per 10 Epochs)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.subplot(2, 2, 3)
    plt.plot(range(eval_num, len(test__psnr) * eval_num + eval_num, eval_num), test__psnr)
    plt.title("Test PSNR (Evaluated per 10 Epochs)")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    
    plt.subplot(2, 2, 4)
    plt.plot(range(eval_num, len(test__ssim) * eval_num + eval_num, eval_num), test__ssim)
    plt.title("Test SSIM (Evaluated per 10 Epochs)")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    
    plt.tight_layout()
    plt.savefig(os.path.join(opt['log_path'], 'training_curves.png'))
    logger.info('训练曲线已保存')
