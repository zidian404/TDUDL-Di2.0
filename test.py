from typing import Dict, List, Tuple
import torch.utils.data as data
import torch
import time
import os
import logging
from glob import glob
from prettytable import PrettyTable
from torch import cuda
import numpy as np
import random

import Net.denoise_net as net
from utils.dataset_admm import get_data
import utils.utils_option as option
from utils.dataset_admm import dataset_admm_denose
import utils.utils_image as image
from utils import utils_logger


def safe_forward(model, img_L, blur_level=None):
    if torch.isnan(img_L).any() or torch.isinf(img_L).any():
        img_L = torch.nan_to_num(img_L, nan=0.0, posinf=1.0, neginf=-1.0)

    if blur_level is None:
        blur_level = torch.full((img_L.size(0), 1), 0.01, device=img_L.device, dtype=img_L.dtype)
    if blur_level.dim() == 1:
        blur_level = blur_level.unsqueeze(1)
    blur_level = blur_level.to(dtype=img_L.dtype)

    with torch.no_grad():
        test_out, aux = model(img_L, blur_level)

        if torch.isnan(test_out).any() or torch.isinf(test_out).any():
            test_out = torch.nan_to_num(test_out, nan=0.0, posinf=1.0, neginf=-1.0)

        return test_out, aux


if __name__ == '__main__':
    gpus = ','.join([str(i) for i in [0]])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    if torch.cuda.device_count() > 0:
        print(f"\n\nLet's use {torch.cuda.device_count()} GPU!\n\n")

    seed_ = 1234
    random.seed(seed_)
    np.random.seed(seed_)
    torch.manual_seed(seed_)
    cuda.manual_seed_all(seed_)

    json_path = "./options/test_options.json"
    opt = option.parse(json_path, is_train=False)

    logger_name = 'test' + time.strftime('%Y_%m_%d_%H-%M-%S', time.localtime())
    utils_logger.logger_info(
        logger_name, os.path.join(opt['log_path'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    names = []
    test_data_path = opt['test']['dataroot_H']
    for name in sorted(glob(os.path.join(test_data_path, '*'))):
        names.append(os.path.basename(name))
    print(f"Found {len(names)} test images in path: {test_data_path}")

    print("Loading test datasets...")
    test_set = get_data(opt, 'test')
    print(f"Loaded {len(test_set)} test sets")

    test_loaders: List[data.DataLoader] = []
    for valid in test_set:
        loader = data.DataLoader(
            dataset=valid,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=True
        )
        test_loaders.append(loader)

    print(f"Total {len(test_loaders)} DataLoaders created.")

    print("Loading model...")
    model = net.denoise_Net_admm_restormer(opt)
    pretrained_path = opt["pretained_path"]

    if not os.path.exists(pretrained_path):
        print(f"ERROR: Model file not found: {pretrained_path}")
        exit(1)

    print(f"Loading pretrained model from: {pretrained_path}")
    state = torch.load(pretrained_path, map_location='cpu')

    if 'state_dict' in state:
        state_dict = state['state_dict']
    else:
        state_dict = state

    if all(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Model loaded! Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")

    model = model.cuda()
    model.eval()

    motion_kernel_size_list = opt['test'].get("motion_kernel_size_list", [9])
    motion_angle_list = opt['test'].get("motion_angle_list", [90])
    defocus_radius_list = opt['test'].get("defocus_radius_list", [0, 1, 2])

    combo_results: Dict[Tuple[str, int, int], Tuple[float, float]] = {}
    total_inference_time = 0
    total_batches = 0

    combo_per_image = len(motion_kernel_size_list) * len(motion_angle_list) * len(defocus_radius_list)

    print(f"\nStarting testing with {len(test_loaders)} loaders...")

    for loader_idx, test_loader in enumerate(test_loaders):
        image_index = loader_idx // combo_per_image
        rel_idx = loader_idx % combo_per_image

        mk_idx = rel_idx // (len(motion_angle_list) * len(defocus_radius_list))
        remain = rel_idx % (len(motion_angle_list) * len(defocus_radius_list))
        ma_idx = remain // len(defocus_radius_list)
        dr_idx = remain % len(defocus_radius_list)

        dataset_name = names[image_index] if image_index < len(names) else f"Unknown_{image_index}"
        motion_kernel_size = motion_kernel_size_list[mk_idx]
        motion_angle = motion_angle_list[ma_idx]
        defocus_radius = defocus_radius_list[dr_idx]

        print(f"-> Processing: {dataset_name}, motion_k={motion_kernel_size}, angle={motion_angle}, defocus_r={defocus_radius}")

        avg_psnr = 0.0
        avg_ssim = 0.0
        batch_count = 0
        loader_inference_time = 0

        with torch.no_grad():
            for img_H, img_L in test_loader:
                batch_count += 1
                img_H = img_H.cuda()
                img_L = img_L.cuda()

                blur_level = torch.full((img_L.size(0), 1), 0.01, device=img_L.device, dtype=img_L.dtype)

                start_time = time.time()
                test_out, _ = safe_forward(model, img_L, blur_level)
                batch_time = time.time() - start_time

                total_inference_time += batch_time
                loader_inference_time += batch_time
                total_batches += 1

                test_out_np = image.tensor2uint(test_out)
                img_H_np = image.tensor2uint(img_H)

                psnr_ = image.calculate_psnr(test_out_np, img_H_np)
                ssim_ = image.calculate_ssim(test_out_np, img_H_np)

                avg_psnr += psnr_
                avg_ssim += ssim_

        if batch_count > 0:
            avg_psnr = round(avg_psnr / batch_count, 2)
            avg_ssim = round(avg_ssim / batch_count, 4)
            avg_time = loader_inference_time / batch_count

            combo_results[(dataset_name, motion_kernel_size, defocus_radius)] = (avg_psnr, avg_ssim)
            print(f"   Completed: PSNR={avg_psnr}, SSIM={avg_ssim}, Time={avg_time:.4f}s")

    if total_batches > 0:
        avg_inference_time = total_inference_time / total_batches
        logger.info(f'Average inference time (per batch/image): {avg_inference_time:.4f} s')
        print(f'\nAverage inference time (per batch/image): {avg_inference_time:.4f} s')

    header = ['Dataset'] + [f'M{mk}_D{dr}' for mk in motion_kernel_size_list for dr in defocus_radius_list]

    t_psnr = PrettyTable(header)
    t_ssim = PrettyTable(header)

    for name in names:
        row_psnr = [name]
        row_ssim = [name]
        for mk in motion_kernel_size_list:
            for dr in defocus_radius_list:
                key = (name, mk, dr)
                if key in combo_results:
                    row_psnr.append(combo_results[key][0])
                    row_ssim.append(combo_results[key][1])
                else:
                    row_psnr.append('--')
                    row_ssim.append('--')

        t_psnr.add_row(row_psnr)
        t_ssim.add_row(row_ssim)

    logger.info(f"Test PSNR:\n{t_psnr}")
    logger.info(f"Test SSIM:\n{t_ssim}")

    print(f"\nFinal Test Results:")
    print(f"Test PSNR:\n{t_psnr}")
    print(f"\nTest SSIM:\n{t_ssim}")

    overall_psnr = []
    overall_ssim = []

    for _, v in combo_results.items():
        overall_psnr.append(v[0])
        overall_ssim.append(v[1])

    if overall_psnr:
        print(f"\nOverall Avg. PSNR: {sum(overall_psnr)/len(overall_psnr):.2f}")
    if overall_ssim:
        print(f"Overall Avg. SSIM: {sum(overall_ssim)/len(overall_ssim):.4f}")