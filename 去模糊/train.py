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


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for training: {device}\n")

    json_path = "./options/train_options.json"
    opt = option.parse(json_path, is_train=True)

    logger_name = 'train' + time.strftime('%Y_%m_%d_%H-%M-%S', time.localtime())
    utils_logger.logger_info(logger_name, os.path.join(opt['log_path'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    print("加载数据集...")
    train_set = get_data(opt, 'train')
    valid_set = get_data(opt, 'valid')

    logger.info(f"训练集大小: {len(train_set)}")
    logger.info(f"验证集数量: {len(valid_set)}")

    train_loader = data.DataLoader(
        dataset=train_set,
        batch_size=opt.get('batch_size', 4),
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
    max_accumulation = 0
    psnr_val_rgb = 0

    if opt.get("pretained_path", {}).get("index"):
        state = torch.load(opt['pretained_path']["path"], map_location=device, weights_only=False)
        model.load_state_dict(state['state_dict'], strict=False)
        print("✅ 加载预训练权重")

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
            if epoch < 200:
                adjust_learning_rate(optimizer, epoch, opt['lr'], opt['max_epoch'])
            else:
                reduce_schedule.step(psnr_val_rgb)

            model.train()
            loss_epoch = 0
            batch_count = 0
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')

            for batch_idx, (img_H, img_L) in enumerate(pbar):
                img_H = img_H.to(device, non_blocking=True)
                img_L = img_L.to(device, non_blocking=True)

                blur_level = torch.full((img_L.size(0), 1), 0.01, device=device, dtype=img_L.dtype)

                output, preds = model(img_L, blur_level)

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
                    pbar.set_postfix({'Loss': f'{loss_epoch / batch_count:.4f}'})

            avg_loss = loss_epoch / len(train_loader)
            loss_train.append(avg_loss)
            logger.info(f"epoch:[{epoch + 1}/{opt['max_epoch']}], 平均loss: {avg_loss:.4f}")

            if (epoch + 1) % eval_num == 0:
                print(f"\n开始验证阶段 - Epoch {epoch+1}")
                model.eval()
                test_loss = 0
                test_psnr = 0
                test_ssim = 0
                valid_batches = 0

                with torch.no_grad():
                    for loader_idx, test_loader in enumerate(test_loaders):
                        val_set_loss = 0
                        val_set_psnr = 0
                        val_set_ssim = 0
                        val_set_batches = 0

                        for batch_idx, (v_img_H, v_img_L) in enumerate(test_loader):
                            v_img_H = v_img_H.to(device, non_blocking=True)
                            v_img_L = v_img_L.to(device, non_blocking=True)

                            blur_level = torch.full((v_img_L.size(0), 1), 0.01, device=device, dtype=v_img_L.dtype)

                            try:
                                v_out, _ = model(v_img_L, blur_level)

                                if isinstance(v_out, (list, tuple)):
                                    v_out = v_out[0]
                                if v_out.dim() == 3:
                                    v_out = v_out.unsqueeze(1)
                                if v_img_H.dim() == 3:
                                    v_img_H = v_img_H.unsqueeze(1)

                                current_loss = criterion(v_out, v_img_H).item()
                                val_set_loss += current_loss
                                test_loss += current_loss
                                valid_batches += 1
                                val_set_batches += 1

                                v_out_u = image.tensor2uint(v_out)
                                v_img_H_u = image.tensor2uint(v_img_H)

                                current_psnr = image.calculate_psnr(v_out_u, v_img_H_u)
                                current_ssim = image.calculate_ssim(v_out_u, v_img_H_u)

                                val_set_psnr += current_psnr
                                val_set_ssim += current_ssim
                                test_psnr += current_psnr
                                test_ssim += current_ssim

                            except Exception as e:
                                print(f"验证batch {batch_idx} 出错: {e}")
                                continue

                        if val_set_batches > 0:
                            val_avg_loss = val_set_loss / val_set_batches
                            val_avg_psnr = val_set_psnr / val_set_batches
                            val_avg_ssim = val_set_ssim / val_set_batches
                            print(f"验证集{loader_idx+1}完成: {val_set_batches} batches, "
                                  f"Loss: {val_avg_loss:.4f}, PSNR: {val_avg_psnr:.2f}, SSIM: {val_avg_ssim:.4f}")

                if valid_batches == 0:
                    logger.warning("警告：没有有效的验证数据")
                    continue

                avg_test_loss = test_loss / valid_batches
                psnr_val_rgb = test_psnr / valid_batches
                ssim_val_rgb = test_ssim / valid_batches

                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
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

            if (epoch + 1) % 10 == 0:
                torch.save(
                    {'epoch': epoch + 1, 'state_dict': model.state_dict()},
                    os.path.join(opt['model_save'], f'model_epoch_{epoch+1}.pth')
                )

            torch.save(
                {'epoch': epoch + 1, 'state_dict': model.state_dict()},
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

    end = time.time()
    training_hours = (end - start) / 3600
    logger.info(f'总训练时间: {training_hours:.2f}小时')

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