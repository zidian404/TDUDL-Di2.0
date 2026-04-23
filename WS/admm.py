import torch.utils.data as data
import os
import torch
from typing import List
import numpy as np
from utils import utils_image as util
import random
from copy import deepcopy
from glob import glob
# Z = scipy.linalg.solve_sylvester(a, b, q)


class dataset_admm_denose(data.Dataset):

    # 修正：在测试模式下，接收单个图像路径 image_path 和单个 sigma 值 single_sigma
    def __init__(self, opt, task, image_path=None, single_sigma=None):

        self.opt = opt
        self.task = task
        self.n_channels = opt['n_channels']
        
        if task == 'train':
            # 训练模式：从文件夹加载所有路径，用于随机裁剪
            self.img_paths = util.get_img_paths(self.opt['dataroot_H'])
            self.sigma = opt['sigma']
            if 'H_size' in opt:
                self.patch_size = opt['H_size']
        else:
            # 测试模式：只加载一个文件，使用一个固定的 sigma
            # 修正：self.img_paths 变为包含单个路径的列表
            self.img_paths = [image_path] if image_path else []
            self.sigma = single_sigma
            # 注意：在测试模式下，我们不需要 patch_size

    def __getitem__(self, index):
        
        # get H image
        # 在测试模式下，index 永远是 0
        img_path = self.img_paths[index]
        img_H = util.imread_uint(img_path, self.n_channels)

        H, W = img_H.shape[:2]
        
        if self.task == 'train':
            # crop (随机裁剪path_size*path_size大小图片)
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # augmentation（随机通过旋转等操作进行数据强化）
            patch_H = util.augment_img(patch_H, mode=np.random.randint(0, 8))

            # HWC to CHW, numpy(uint) to tensor
            img_H = util.uint2tensor3(patch_H)
            img_L: torch.Tensor = img_H.clone()

            # get noise level
            noise_level: torch.FloatTensor = torch.FloatTensor(
                [np.random.uniform(self.sigma[0], self.sigma[1])]) / 255.0

            # add noise
            noise = torch.randn(img_L.size()).mul_(noise_level).float()
            img_L.add_(noise)

        else:
            # 测试逻辑：处理大图并添加噪声
            img_H = util.uint2single(img_H)
            img_L = np.copy(img_H)

            # add noise
            # 修正：确保测试时 sigma 是单个数值
            np.random.seed(seed=0)
            img_L += np.random.normal(0, self.sigma / 255.0, img_L.shape)

            noise_level = torch.FloatTensor([self.sigma / 255.0])

            img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)
            h, w = img_H.size()[-2:]
            top = slice(0, h // 8 * 8)
            left = slice(0, (w // 8 * 8))
            img_H = img_H[..., top, left]
            img_L = img_L[..., top, left]


        return img_H, img_L, noise_level

    def __len__(self):
        # 修正：训练模式返回所有图像数，测试模式返回 1 (因为一个 Dataset 对象只代表一张图)
        return len(self.img_paths)


def get_data(opt, task):
    if task == 'train':
        opt_ = opt[task]
        dataset = dataset_admm_denose(opt_, task)
        return dataset
    else:
        # 修正后的测试模式数据加载逻辑
        datasets: List[dataset_admm_denose] = []
        opt_ = opt[task]
        
        # 1. 获取所有单个图像的完整路径
        paths = sorted(glob(os.path.join(opt_['dataroot_H'], '*')))
        sigmas = opt_['sigma']

        # 2. 为每张图片和每个 sigma 级别创建一个独立的 Dataset 对象
        for path in paths:
            for sigma in sigmas:
                # 3. 创建一个新的 opt 字典用于传递通用配置
                opt_subset = deepcopy(opt_) 
                
                # 4. 创建 Dataset 对象，传入单个图像路径和 sigma 值
                datasets.append(dataset_admm_denose(
                    opt_subset, 
                    task, 
                    image_path=path,        # 单个图像路径
                    single_sigma=sigma      # 单个 sigma 值
                ))

        return datasets