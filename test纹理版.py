from typing import Dict, List
import torch.utils.data as data
import torch, cv2
import time
import os
import logging
from torchsummary import summary
from glob import glob
from prettytable import PrettyTable
from torch import cuda
import numpy as np
import random
import copy
from thop import profile
from scipy import linalg
from collections import OrderedDict

# 🔥 修改：导入增强版模型
import Net.denoise_net_enhanced as net  # 改成你的增强版模型文件名
from utils.dataset_admm import get_data
import utils.utils_option as option
from utils.dataset_admm import dataset_admm_denose
import utils.utils_image as image
from utils import utils_logger

# ------------------------
# 辅助函数: 安全模型前向传播（适配增强版）
# ------------------------
def safe_forward(model, img_L, noise_level, return_texture=False):
    """安全的模型前向传播，包含NaN/Inf检查（适配增强版）"""
    with torch.no_grad():
        if torch.isnan(img_L).any() or torch.isinf(img_L).any():
            img_L = torch.nan_to_num(img_L, nan=0.0, posinf=1.0, neginf=-1.0)
            
        if torch.isnan(noise_level).any() or torch.isinf(noise_level).any():
            noise_level = torch.nan_to_num(noise_level, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 🔥 增强版模型调用
        if return_texture:
            test_out, preds, texture_infos = model(img_L, noise_level, return_texture=True)
        else:
            test_out, preds = model(img_L, noise_level, return_texture=False)
        
        if torch.isnan(test_out).any() or torch.isinf(test_out).any():
            test_out = torch.nan_to_num(test_out, nan=0.0, posinf=1.0, neginf=-1.0)
            
        return test_out, preds

# ------------------------
# 主函数
# ------------------------
if __name__ == '__main__':

    gpus = ','.join([str(i) for i in [0]])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 0:
        print(f"\n\nLet's use {torch.cuda.device_count()} GPU!\n\n")
    
    seed_ = 1234
    random.seed(seed_)
    np.random.seed(seed_)
    torch.manual_seed(seed_)
    cuda.manual_seed_all(seed_)
    
    # ------------------------
    #       option_setting
    # ------------------------
    json_path = "./options/test_options_enhanced.json"  # 🔥 使用新的配置文件
    opt = option.parse(json_path, is_train=False)
    
    # logger
    logger_name = 'test_enhanced_' + time.strftime('%Y_%m_%d_%H-%M-%S', time.localtime())
    utils_logger.logger_info(
        logger_name, os.path.join(opt['log_path'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    # -------------------------
    #           dataset
    # ------------------------
    names = []
    test_data_path = opt['test']['dataroot_H']
    for name in sorted(glob(os.path.join(test_data_path, '*'))):
        names.append(os.path.basename(name))
    print(f"Found {len(names)} test images in path: {test_data_path}")
    
    print("Loading test datasets...")
    test_set = get_data(opt, 'test')
    print(f"Loaded {len(test_set)} test sets (one for each image * sigma combination)")
    
    test_loaders: List[data.DataLoader[dataset_admm_denose]] = []
    for i, valid in enumerate(test_set):
        loader = data.DataLoader(
            dataset=valid, 
            batch_size=1, 
            shuffle=False, 
            num_workers=2,  # 🔥 减少num_workers避免问题
            drop_last=False,  # 🔥 改为False，保留所有数据
            pin_memory=True
        )
        test_loaders.append(loader)
    
    print(f"Total {len(test_loaders)} DataLoaders created.")

    # -------------------------
    #           model
    # ------------------------
    print("Loading enhanced model...")
    # 🔥 使用增强版模型
    model = net.denoise_Net_admm_restormer_enhanced(opt)
    pretained_path = opt["pretained_path"]
    
    if not os.path.exists(pretained_path):
        print(f"ERROR: Model file not found: {pretained_path}")
        exit(1)
        
    print(f"Loading pretrained model from: {pretained_path}")
    state = torch.load(pretained_path, map_location='cpu')
    
    if 'state_dict' in state:
        state_dict = state['state_dict']
    else:
        state_dict = state
    
    # 移除可能的'module.'前缀
    if all(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    # 加载模型
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Model loaded! Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params/1e6:.2f}M")
    
    model.cuda()
    model.eval()

    # -------------------------
    #            test
    # ------------------------
    
    avg_psnrs: Dict[str, List[float]] = {}
    avg_ssims: Dict[str, List[float]] = {}
    
    sigma_size = len(opt['test']['sigma'])
    total_inference_time = 0
    total_batches = 0

    print(f"\nStarting full testing with {len(test_loaders)} loaders...")
    print(f"Sigma levels: {opt['test']['sigma']}")
    
    # 循环遍历所有 DataLoader
    for loader_idx, test_loader in enumerate(test_loaders):
        
        avg_psnr = 0.
        avg_ssim = 0.
        
        # 使用索引确定当前测试的是哪张图和哪个sigma
        image_index = loader_idx // sigma_size
        sigma_level = opt['test']['sigma'][loader_idx % sigma_size]
        dataset_name = names[image_index] if image_index < len(names) else f"Unknown_Dataset_{image_index}"
        
        print(f'-> Processing: {dataset_name}, sigma={sigma_level}')
        
        loader_inference_time = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch_idx, (img_H, img_L, noise_level) in enumerate(test_loader):
                batch_count += 1
                
                img_H = img_H.cuda()
                img_L = img_L.cuda()
                noise_level = noise_level.cuda()

                # 🔥 增强版前向传播（测试时不需纹理信息）
                start_time = time.time()
                test_out, _ = safe_forward(model, img_L, noise_level, return_texture=False)
                batch_time = time.time() - start_time
                
                total_inference_time += batch_time
                loader_inference_time += batch_time
                total_batches += 1

                # 计算指标
                test_out_np = image.tensor2uint(test_out)
                img_H_np = image.tensor2uint(img_H)
                
                psnr_ = image.calculate_psnr(test_out_np, img_H_np)
                ssim_ = image.calculate_ssim(test_out_np, img_H_np)
                avg_psnr += psnr_
                avg_ssim += ssim_
                
                # 打印前几个batch的结果
                if batch_idx < 3:
                    print(f'   Batch {batch_idx+1}: PSNR={psnr_:.2f}dB, SSIM={ssim_:.4f}')

        # 计算该loader的平均值
        if batch_count > 0:
            avg_psnr = round(avg_psnr / batch_count, 2)
            avg_ssim = round(avg_ssim * 100 / batch_count, 2)
            avg_time = loader_inference_time / batch_count
            
            print(f'   Completed: PSNR={avg_psnr}, SSIM={avg_ssim}, Time={avg_time:.4f}s per batch')
            
            # 存储结果
            if dataset_name not in avg_psnrs:
                avg_psnrs[dataset_name] = []
                avg_ssims[dataset_name] = []
                
            avg_psnrs[dataset_name].append(avg_psnr)
            avg_ssims[dataset_name].append(avg_ssim)

    # -------------------------
    #       输出最终结果
    # ------------------------
    if total_batches > 0:
        avg_inference_time = total_inference_time / total_batches
        logger.info(f'Average inference time (per batch): {avg_inference_time:.4f} s')
        print(f'\nAverage inference time (per batch): {avg_inference_time:.4f} s')

    # 输出表格
    header = ['Dataset'] + [f'σ={s}' for s in opt['test']['sigma']]
    
    t_psnr = PrettyTable(header)
    for key, value in avg_psnrs.items():
        t_psnr.add_row([key] + value)
    
    t_ssim = PrettyTable(header)
    for key, value in avg_ssims.items():
        t_ssim.add_row([key] + value)

    logger.info(f"Test PSNR:\n{t_psnr}")
    logger.info(f"Test SSIM:\n{t_ssim}")
    
    print(f"\n{'='*60}")
    print(f"Final Test Results (Enhanced Model)")
    print(f"{'='*60}")
    print(f"Test PSNR:\n{t_psnr}")
    print(f"\nTest SSIM:\n{t_ssim}")
    
    # 总体平均PSNR/SSIM计算
    print(f"\n{'='*60}")
    print(f"Overall Average Results")
    print(f"{'='*60}")
    
    for sigma_idx, sigma_level in enumerate(opt['test']['sigma']):
        sigma_key = f'σ={sigma_level}'
        
        psnrs_at_sigma = [vals[sigma_idx] for vals in avg_psnrs.values() if len(vals) > sigma_idx]
        ssims_at_sigma = [vals[sigma_idx] for vals in avg_ssims.values() if len(vals) > sigma_idx]
        
        if psnrs_at_sigma:
            avg_psnr_val = sum(psnrs_at_sigma) / len(psnrs_at_sigma)
            avg_ssim_val = sum(ssims_at_sigma) / len(ssims_at_sigma)
            print(f"  {sigma_key}: PSNR={avg_psnr_val:.2f}dB, SSIM={avg_ssim_val:.4f}")
    
    print(f"\nTesting completed! Processed {total_batches} batches across {len(avg_psnrs)} images.")