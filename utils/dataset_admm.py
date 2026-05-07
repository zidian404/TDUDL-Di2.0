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

        return datasets# utils/dataset_admm.py
import torch.utils.data as data
import os
import torch
from typing import List, Union
import numpy as np
from utils import utils_image as util
import random
from copy import deepcopy
from glob import glob


class dataset_admm_denose(data.Dataset):

    def __init__(self, opt, task, image_path=None, single_sigma=None):
        """
        数据集类
        
        Args:
            opt: 配置参数
            task: 'train' 或 'test'
            image_path: 测试模式下的单个图像路径
            single_sigma: 测试模式下的单个噪声水平
        """
        self.opt = opt
        self.task = task
        self.n_channels = opt.get('n_channels', 1)
        
        if task == 'train':
            # 训练模式：从文件夹加载所有路径
            self.img_paths = util.get_img_paths(self.opt['dataroot_H'])
            # 确保 sigma 是一个包含两个元素的列表
            sigma_config = self.opt.get('sigma', [0, 55])
            if isinstance(sigma_config, (int, float)):
                self.sigma = [0, sigma_config]
            elif isinstance(sigma_config, (list, tuple)) and len(sigma_config) == 1:
                self.sigma = [0, sigma_config[0]]
            elif isinstance(sigma_config, (list, tuple)) and len(sigma_config) >= 2:
                self.sigma = [sigma_config[0], sigma_config[1]]
            else:
                self.sigma = [0, 55]
            
            if 'H_size' in opt:
                self.patch_size = opt['H_size']
        else:
            # 测试模式：只加载一个文件
            self.img_paths = [image_path] if image_path else []
            # 确保 sigma 是单个数值
            if single_sigma is not None:
                self.sigma = single_sigma
            else:
                # 如果传入了列表，取第一个值
                sigma_config = opt.get('sigma', 25)
                if isinstance(sigma_config, (list, tuple)):
                    self.sigma = sigma_config[0] if len(sigma_config) > 0 else 25
                else:
                    self.sigma = sigma_config

    def __getitem__(self, index):
        """获取数据项"""
        
        # 获取图像路径
        if index >= len(self.img_paths):
            raise IndexError(f"Index {index} out of range for {len(self.img_paths)} images")
        
        img_path = self.img_paths[index]
        img_H = util.imread_uint(img_path, self.n_channels)

        H, W = img_H.shape[:2]
        
        if self.task == 'train':
            # ========== 训练模式 ==========
            # 随机裁剪
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # 数据增强
            patch_H = util.augment_img(patch_H, mode=np.random.randint(0, 8))

            # HWC to CHW, numpy(uint) to tensor
            img_H = util.uint2tensor3(patch_H)
            img_L: torch.Tensor = img_H.clone()

            # 获取噪声水平 - 修复索引问题
            if len(self.sigma) >= 2:
                noise_value = np.random.uniform(self.sigma[0], self.sigma[1])
            else:
                noise_value = self.sigma[0] if isinstance(self.sigma, (list, tuple)) else self.sigma
            
            noise_level: torch.FloatTensor = torch.FloatTensor([noise_value]) / 255.0

            # 添加噪声
            noise = torch.randn(img_L.size()).mul_(noise_level).float()
            img_L.add_(noise)

        else:
            # ========== 测试模式 ==========
            # 转换为单精度
            img_H = util.uint2single(img_H)
            img_L = np.copy(img_H)

            # 添加噪声 - 确保 sigma 是数值
            if isinstance(self.sigma, (list, tuple)):
                sigma_value = self.sigma[0]
            else:
                sigma_value = self.sigma
            
            np.random.seed(seed=0)
            img_L += np.random.normal(0, sigma_value / 255.0, img_L.shape)

            noise_level = torch.FloatTensor([sigma_value / 255.0])

            # 转换为 tensor
            img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)
            
            # 裁剪到 8 的倍数
            h, w = img_H.size()[-2:]
            top = slice(0, h // 8 * 8)
            left = slice(0, w // 8 * 8)
            img_H = img_H[..., top, left]
            img_L = img_L[..., top, left]

        return img_H, img_L, noise_level

    def __len__(self):
        return len(self.img_paths)


def get_data(opt, task):
    """
    获取数据加载器
    
    Args:
        opt: 配置参数
        task: 'train' 或 'test'
    
    Returns:
        训练模式返回单个 Dataset，测试模式返回 Dataset 列表
    """
    if task == 'train':
        # 训练模式
        opt_ = opt[task]
        dataset = dataset_admm_denose(opt_, task)
        return dataset
    
    else:
        # 测试模式 - 修复逻辑
        datasets: List[dataset_admm_denose] = []
        opt_ = opt[task]
        
        # 获取所有图像路径
        dataroot = opt_.get('dataroot_H', '')
        if not dataroot:
            raise ValueError("dataroot_H not specified for test mode")
        
        paths = sorted(glob(os.path.join(dataroot, '*')))
        
        if not paths:
            raise ValueError(f"No images found in {dataroot}")
        
        # 获取 sigma 配置
        sigma_config = opt_.get('sigma', [25])
        
        # 确保 sigma 是可迭代的列表
        if isinstance(sigma_config, (int, float)):
            sigmas = [sigma_config]
        elif isinstance(sigma_config, (list, tuple)):
            sigmas = list(sigma_config)
        else:
            sigmas = [25]
        
        # 为每张图片和每个 sigma 级别创建一个 Dataset 对象
        for path in paths:
            for sigma in sigmas:
                # 创建新的 opt 副本（避免共享引用）
                opt_subset = deepcopy(opt_)
                
                # 创建 Dataset 对象
                dataset = dataset_admm_denose(
                    opt_subset, 
                    task, 
                    image_path=path,
                    single_sigma=sigma
                )
                datasets.append(dataset)
        
        return datasets


# ========== 可选：添加便捷的数据加载函数 ==========
def create_dataloader(opt, task, batch_size=1, num_workers=0, shuffle=False):
    """
    创建 DataLoader
    
    Args:
        opt: 配置参数
        task: 'train' 或 'test'
        batch_size: 批次大小
        num_workers: 工作进程数
        shuffle: 是否打乱
    
    Returns:
        DataLoader 对象
    """
    from torch.utils.data import DataLoader
    
    if task == 'train':
        dataset = get_data(opt, task)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        return dataloader
    else:
        # 测试模式：返回多个 DataLoader 的列表
        datasets = get_data(opt, task)
        dataloaders = []
        for dataset in datasets:
            dataloader = DataLoader(
                dataset,
                batch_size=1,  # 测试模式 batch_size 固定为 1
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
            dataloaders.append(dataloader)
        return dataloaders


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 测试配置
    opt = {
        'train': {
            'dataroot_H': 'path/to/train/data',
            'n_channels': 1,
            'sigma': [0, 55],  # 噪声范围 0-55
            'H_size': 128      # 裁剪大小
        },
        'test': {
            'dataroot_H': 'path/to/test/data',
            'n_channels': 1,
            'sigma': [15, 25, 50]  # 测试多个噪声水平
        }
    }
    
    # 测试训练模式
    print("Testing training mode...")
    train_dataset = get_data(opt, 'train')
    if train_dataset:
        print(f"Training dataset size: {len(train_dataset)}")
        img_H, img_L, noise = train_dataset[0]
        print(f"Image shape: {img_H.shape}, Noise level: {noise}")
    
    # 测试测试模式
    print("\nTesting test mode...")
    test_datasets = get_data(opt, 'test')
    print(f"Number of test datasets: {len(test_datasets)}")
    for i, dataset in enumerate(test_datasets):
        print(f"  Dataset {i}: {len(dataset)} image(s)")