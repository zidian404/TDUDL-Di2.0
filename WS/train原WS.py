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

# 🔥 新模型导入（无ids版本）
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
        lr = lr_ini * (0.65 ** (epo // (P1 // log(0.1, 0.65))))
    else:
        lr = lr_ini * 0.1 * (0.85 ** ((epo - P1) // (P2 // log(0.1, 0.85))))
    for param_group in opt.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    # 基本设置
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda')
    print(f"Using GPU for training!\n")
        
    # 🔥 解析配置（无需全局id映射）
    json_path = "./options/train_options.json"
    opt = option.parse(json_path, is_train=True)
    
    # 日志
    logger_name = 'train' + time.strftime('%Y_%m_%d_%H-%M-%S', time.localtime())
    utils_logger.logger_info(logger_name, os.path.join(opt['log_path'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    # 🔥 数据集加载（无需path2id，只返回3元组）
    print("加载数据集...")
    train_set = get_data(opt, 'train')
    valid_set = get_data(opt, 'valid')
    
    logger.info(f"训练集大小: {len(train_set)}")
    logger.info(f"验证集数量: {len(valid_set)} (每个 sigma 一个 Dataset)")
    
    # 数据加载器
    train_loader = data.DataLoader(
        dataset=train_set, 
        batch_size=4,
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

    # 🔥 模型初始化（新版本，无n_samples）
    print("初始化模型...")
    model = net.denoise_Net_admm_restormer(opt)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'])
    criterion = loss_function(opt['loss_function_index'])
    
    total = sum([param.nelement() for param in model.parameters()])
    logger.info(f"Number of parameter: {total / 1e6 :.2f}M")
    logger.info("start training...")

    # 训练记录变量
    start = time.time()
    loss_train = []
    test__loss = []
    test__psnr = []
    test__ssim = []
    best_psnr = 0
    best_epoch = 0
    max_accumulation = 0
    psnr_val_rgb = 0

    # 预训练加载
    if opt.get("pretained_path", {}).get("index"):
        state = torch.load(opt['pretained_path']["path"], weights_only=False)
        model.load_state_dict(state['state_dict'], strict=False)
        print("✅ 加载预训练权重")

    # 学习率调度器
    reduce_schedule = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.85, patience=5,
        threshold=1e-3, threshold_mode='abs', min_lr=0, eps=1e-8
    )

    eval_num = 10

    def signal_handler(sig, frame):
        print(f"\n训练被中断！正在保存当前进度...")
        torch.save({
            'epoch': locals().get('epoch', 0),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_psnr': best_psnr
        }, os.path.join(opt['model_save'], 'model_interrupted.pth'))
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    for epoch in range(0, opt["max_epoch"]):
        try:
            # 学习率调整
            if epoch < 200:
                adjust_learning_rate(optimizer, epoch, opt['lr'], opt['max_epoch'])
            else:
                reduce_schedule.step(psnr_val_rgb)

            # 训练阶段：只解包3个元素，无ids
            model.train()
            loss_epoch = 0
            batch_count = 0
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
            
            for batch_idx, (img_H, img_L, noise_level) in enumerate(pbar):
                img_H = img_H.to(device, non_blocking=True)
                img_L = img_L.to(device, non_blocking=True)
                noise_level = noise_level.to(device, non_blocking=True)
                
                output, preds = model(img_L, noise_level)  # 只传2个参数

                if isinstance(output, (list, tuple)):
                    output = output[0]
                if output.dim() == 3:
                    output = output.unsqueeze(1)
                if img_H.dim() == 3:
                    img_H = img_H.unsqueeze(1)

                loss = criterion(output, img_H)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                loss_epoch += loss.item()
                batch_count += 1
                
                if batch_idx % 10 == 0:
                    pbar.set_postfix({'Loss': f'{loss_epoch/batch_count:.4f}'})

            avg_loss = loss_epoch / len(train_loader)
            loss_train.append(avg_loss)
            logger.info(f"epoch:[{epoch + 1}/{opt['max_epoch']}], 平均loss: {avg_loss:.4f}")

            # 🔥 完整验证阶段：带图片裁剪 + 错误统计 + sigma安全索引
            if (epoch + 1) % eval_num == 0:
                print(f"\n开始验证阶段 - Epoch {epoch+1}")
                model.eval()
                test_loss = 0
                test_psnr = 0
                test_ssim = 0
                total_batches = 0
                valid_batches = 0
                skip_stats = {
                    'input_nan': 0,
                    'output_nan': 0, 
                    'loss_nan': 0,
                    'other_error': 0,
                    'image_cropped': 0
                }
                
                val_results = []
                
                with torch.no_grad():
                    for loader_idx, test_loader in enumerate(test_loaders):
                        # 🔥 安全sigma索引
                        sigma_list = opt.get('valid', {}).get('sigma', [25])
                        sigma_idx = loader_idx % len(sigma_list)
                        sigma_val = sigma_list[sigma_idx]
                        
                        print(f"\n处理验证集 {loader_idx+1}/{len(test_loaders)} (噪音水平: {sigma_val})")
                        val_set_loss = 0
                        val_set_psnr = 0
                        val_set_ssim = 0
                        val_set_batches = 0  # 🔥 定义变量
                        
                        for batch_idx, (v_img_H, v_img_L, v_noise) in enumerate(test_loader):
                            total_batches += 1
                            val_set_batches += 1
                            batch_start_time = time.time()  # 🔥 定义变量
                            
                            # 检查输入数据
                            if torch.isnan(v_img_L).any() or torch.isinf(v_img_L).any():
                                skip_stats['input_nan'] += 1
                                continue
                            
                            # 🔥 图片尺寸裁剪：如果太大就中心裁剪到256x256
                            original_shape = v_img_L.shape
                            max_size = 256
                            
                            if v_img_L.shape[2] > max_size or v_img_L.shape[3] > max_size:
                                H, W = v_img_L.shape[2], v_img_L.shape[3]
                                start_h = (H - max_size) // 2
                                start_w = (W - max_size) // 2
                                v_img_L = v_img_L[:, :, start_h:start_h+max_size, start_w:start_w+max_size]
                                v_img_H = v_img_H[:, :, start_h:start_h+max_size, start_w:start_w+max_size]
                                skip_stats['image_cropped'] += 1
                            
                            # 数据到GPU
                            v_img_H = v_img_H.to(device, non_blocking=True)
                            v_img_L = v_img_L.to(device, non_blocking=True)
                            v_noise = v_noise.to(device, non_blocking=True)
                            
                            try:
                                # 🔥 模型调用：只传2个参数
                                v_out, _ = model(v_img_L, v_noise)
                                
                                if torch.isnan(v_out).any() or torch.isinf(v_out).any():
                                    skip_stats['output_nan'] += 1
                                    continue
                                    
                                if isinstance(v_out, (list, tuple)):
                                    v_out = v_out[0]
                                if v_out.dim() == 3:
                                    v_out = v_out.unsqueeze(1)
                                if v_img_H.dim() == 3:
                                    v_img_H = v_img_H.unsqueeze(1)
                                    
                                current_loss = criterion(v_out, v_img_H).item()
                                
                                if np.isnan(current_loss) or np.isinf(current_loss):
                                    skip_stats['loss_nan'] += 1
                                    continue
                                    
                                val_set_loss += current_loss
                                test_loss += current_loss
                                valid_batches += 1
                                
                                v_out_u = image.tensor2uint(v_out)
                                v_img_H_u = image.tensor2uint(v_img_H)
                                
                                current_psnr = image.calculate_psnr(v_out_u, v_img_H_u)
                                current_ssim = image.calculate_ssim(v_out_u, v_img_H_u)
                                
                                val_set_psnr += current_psnr
                                val_set_ssim += current_ssim
                                test_psnr += current_psnr
                                test_ssim += current_ssim
                                
                                # 进度显示
                                if val_set_batches % 50 == 0:
                                    avg_psnr = val_set_psnr / val_set_batches
                                    avg_ssim = val_set_ssim / val_set_batches
                                    print(f"验证集{loader_idx+1}进度: {val_set_batches} batches, PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")
                                
                            except Exception as e:
                                skip_stats['other_error'] += 1
                                print(f"验证batch {batch_idx} 出错: {e}")
                                continue
                        
                        # 记录每个验证集结果
                        if val_set_batches > 0:
                            val_avg_loss = val_set_loss / val_set_batches
                            val_avg_psnr = val_set_psnr / val_set_batches
                            val_avg_ssim = val_set_ssim / val_set_batches
                            val_results.append({
                                'sigma': sigma_val,
                                'loss': val_avg_loss,
                                'psnr': val_avg_psnr,
                                'ssim': val_avg_ssim,
                                'batches': val_set_batches
                            })
                            print(f"验证集{loader_idx+1} (σ={sigma_val}) 完成: {val_set_batches} batches, PSNR: {val_avg_psnr:.2f}, SSIM: {val_avg_ssim:.4f}")

                # 验证统计
                print(f"\n验证完成统计:")
                print(f"总batch数: {total_batches}")
                print(f"有效batch数: {valid_batches}")
                print(f"图像裁剪: {skip_stats['image_cropped']}")
                print(f"跳过统计: {skip_stats}")

                if valid_batches == 0:
                    logger.warning("警告：没有有效的验证数据")
                    continue
                    
                avg_test_loss = test_loss / valid_batches
                psnr_val_rgb = test_psnr / valid_batches
                ssim_val_rgb = test_ssim / valid_batches
                
                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch + 1
                    torch.save(
                        {'state_dict': model.state_dict()},
                        os.path.join(opt['model_save'], "model_best.pth")
                    )
                    print(f"*** Best Model Saved! Best PSNR: {best_psnr:.4f} ***")
                    max_accumulation = 0
                else:
                    max_accumulation += 1

                logger.info(f'[epoch {epoch+1} PSNR: {psnr_val_rgb:.4f} Best: {best_psnr:.4f}]')
                test__loss.append(avg_test_loss)
                test__psnr.append(psnr_val_rgb)
                test__ssim.append(ssim_val_rgb)

            # 定期保存
            if (epoch + 1) % 10 == 0:
                torch.save(
                    {'epoch': epoch+1, 'state_dict': model.state_dict()},
                    os.path.join(opt['model_save'], f'model_epoch_{epoch+1}.pth')
                )

            # 最新模型
            torch.save(
                {'epoch': epoch+1, 'state_dict': model.state_dict()},
                os.path.join(opt['model_save'], 'model_latest.pth')
            )
            
            if max_accumulation >= 10:
                print(f"Early stopping at epoch {epoch+1}")
                break

        except Exception as e:
            print(f"训练出错: {e}")
            torch.save(
                {'state_dict': model.state_dict(), 'error': str(e)},
                os.path.join(opt['model_save'], 'model_error.pth')
            )
            logger.error(f"训练出错: {e}")
            raise e

    # 训练完成
    end = time.time()
    training_hours = (end - start) / 3600
    logger.info(f'总训练时间: {training_hours:.2f}小时')
    
    # 绘图
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(loss_train, label='Train Loss')
    plt.title("Training Loss")
    
    if test__psnr:
        plt.subplot(2, 2, 2)
        plt.plot(test__psnr, label='Test PSNR')
        plt.title("Test PSNR")
        
        plt.subplot(2, 2, 3)
        plt.plot(test__ssim, label='Test SSIM')
        plt.title("Test SSIM")
    
    plt.tight_layout()
    plt.savefig(os.path.join(opt['log_path'], 'training_curves.png'))
    print("训练结束，曲线已保存到 training_curves.png")