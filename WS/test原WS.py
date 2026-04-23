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
import Net.denoise_net as net
from utils.dataset_admm import get_data
import utils.utils_option as option
from utils.dataset_admm import dataset_admm_denose
import utils.utils_image as image
from utils import utils_logger

# ------------------------
# 辅助函数: 安全模型前向传播
# ------------------------
def safe_forward(model, img_L, noise_level):
    """安全的模型前向传播，包含NaN/Inf检查"""
    with torch.no_grad():
        if torch.isnan(img_L).any() or torch.isinf(img_L).any():
            img_L = torch.nan_to_num(img_L, nan=0.0, posinf=1.0, neginf=-1.0)
            
        if torch.isnan(noise_level).any() or torch.isinf(noise_level).any():
            noise_level = torch.nan_to_num(noise_level, nan=0.0, posinf=1.0, neginf=-1.0)
            
        test_out, aaa = model(img_L, noise_level)
        
        if torch.isnan(test_out).any() or torch.isinf(test_out).any():
            test_out = torch.nan_to_num(test_out, nan=0.0, posinf=1.0, neginf=-1.0)
            
        return test_out, aaa

# ------------------------
# 主函数
# ------------------------
if __name__ == '__main__':

    gpus = ','.join([str(i) for i in [0, 1, 2, 3]])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 0:
        print(f"\n\nLet's use {torch.cuda.device_count()} GPU!\n\n")
    
    seed_=1234
    random.seed(seed_)
    np.random.seed(seed_)
    torch.manual_seed(seed_)
    cuda.manual_seed_all(seed_)
    
    # ------------------------
    #       option_setting
    # ------------------------
    json_path = "./options/test_options.json"
    opt = option.parse(json_path, is_train=False)
    
    # logger
    logger_name = 'test'+time.strftime('%Y_%m_%d_%H-%M-%S', time.localtime())
    utils_logger.logger_info(
        logger_name, os.path.join(opt['log_path'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    # -------------------------
    #           dataset
    # ------------------------
    # --- 检查点 1: 文件名列表 ---
    names = []
    test_data_path = opt['test']['dataroot_H']
    for name in sorted(glob(os.path.join(test_data_path, '*'))):
        names.append(os.path.basename(name))
    print(f"Found {len(names)} test images in path: {test_data_path}")
    
    print("Loading test datasets...")
    test_set = get_data(opt, 'test')
    # --- 检查点 2: 数据集数量 ---
    print(f"Loaded {len(test_set)} test sets (one for each image * sigma combination)")
    
    test_loaders: List[data.DataLoader[dataset_admm_denose]] = []
    for i, valid in enumerate(test_set):
        loader = data.DataLoader(dataset=valid, batch_size=1, shuffle=False, num_workers=4, drop_last=True, pin_memory=True)
        test_loaders.append(loader)
    
    print(f"Total {len(test_loaders)} DataLoaders created.")

    # -------------------------
    #           model
    # ------------------------
    print("Loading model...")
    model = net.denoise_Net_admm_restormer(opt)
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

                start_time = time.time()
                test_out, _ = safe_forward(model, img_L, noise_level)
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

        # --------------------
        # 计算该loader的平均值 (每个 loader 只有一个 batch/image)
        # --------------------
        if batch_count > 0:
            avg_psnr = round(avg_psnr / batch_count, 2)
            avg_ssim = round(avg_ssim * 100 / batch_count, 2)
            avg_time = loader_inference_time / batch_count
            
            print(f'   Completed: PSNR={avg_psnr}, SSIM={avg_ssim}, Time={avg_time:.4f}s per batch/image')
            
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
        logger.info(f'Average inference time (per batch/image): {avg_inference_time:.4f} s')
        print(f'\nAverage inference time (per batch/image): {avg_inference_time:.4f} s')

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
    
    print(f"\nFinal Test Results:")
    print(f"Test PSNR:\n{t_psnr}")
    print(f"\nTest SSIM:\n{t_ssim}")
    
    print(f"\nTesting completed! Processed {total_batches} batches across {len(avg_psnrs)} datasets.")
    
    # -------------------------
    # 新增: 总体平均 PSNR/SSIM 计算 (数据集平均值)
    # -------------------------
    
    final_avg_psnr: Dict[str, float] = {}
    final_avg_ssim: Dict[str, float] = {}
    
    for sigma_idx, sigma_level in enumerate(opt['test']['sigma']):
        sigma_key = f'σ={sigma_level}'
        
        # 收集该sigma级别下所有图像的PSNR和SSIM
        psnrs_at_sigma = [vals[sigma_idx] for vals in avg_psnrs.values() if len(vals) > sigma_idx]
        ssims_at_sigma = [vals[sigma_idx] for vals in avg_ssims.values() if len(vals) > sigma_idx]
        
        # 计算平均值
        if psnrs_at_sigma:
            final_avg_psnr[sigma_key] = round(sum(psnrs_at_sigma) / len(psnrs_at_sigma), 2)
        if ssims_at_sigma:
            final_avg_ssim[sigma_key] = round(sum(ssims_at_sigma) / len(ssims_at_sigma), 2)

    t_overall_psnr = PrettyTable(['Metric'] + list(final_avg_psnr.keys()))
    t_overall_psnr.add_row(['Overall Avg. PSNR'] + list(final_avg_psnr.values()))
    
    t_overall_ssim = PrettyTable(['Metric'] + list(final_avg_ssim.keys()))
    t_overall_ssim.add_row(['Overall Avg. SSIM'] + list(final_avg_ssim.values()))
    
    print(f"\nOverall Average Results (Dataset Average):")
    print(f"Overall PSNR:\n{t_overall_psnr}")
    print(f"Overall SSIM:\n{t_overall_ssim}")