import torch.utils.data as data
import os
import torch
from typing import List
import numpy as np
from utils import utils_image as util
import random
from copy import deepcopy
from glob import glob
import cv2


def motion_blur_kernel(kernel_size=15, angle=0):
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    kernel[center, :] = 1.0
    M = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    kernel = kernel / (kernel.sum() + 1e-8)
    return kernel


def defocus_blur_kernel(radius=3):
    if radius <= 0:
        return np.array([[1.0]], dtype=np.float32)

    kernel_size = radius * 2 + 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = radius

    for i in range(kernel_size):
        for j in range(kernel_size):
            if (i - center) ** 2 + (j - center) ** 2 <= radius ** 2:
                kernel[i, j] = 1.0

    kernel = kernel / (kernel.sum() + 1e-8)
    return kernel


def apply_kernel(img, kernel):
    if img.ndim == 3:
        out = np.zeros_like(img)
        for c in range(img.shape[2]):
            out[:, :, c] = cv2.filter2D(img[:, :, c], -1, kernel)
    else:
        out = cv2.filter2D(img, -1, kernel)
    return out


def apply_motion_blur(img, kernel_size=9, angle=90):
    kernel = motion_blur_kernel(kernel_size=kernel_size, angle=angle)
    return apply_kernel(img, kernel)


def apply_defocus_blur(img, radius=2):
    kernel = defocus_blur_kernel(radius=radius)
    return apply_kernel(img, kernel)


def apply_motion_defocus_mix(img, motion_kernel_size=9, motion_angle=90, defocus_radius=2):
    out = apply_motion_blur(img, kernel_size=motion_kernel_size, angle=motion_angle)
    out = apply_defocus_blur(out, radius=defocus_radius)
    return out


class dataset_admm_denose(data.Dataset):
    def __init__(
        self,
        opt,
        task,
        image_path=None,
        single_motion_kernel_size=None,
        single_motion_angle=None,
        single_defocus_radius=None
    ):
        self.opt = opt
        self.task = task
        self.n_channels = opt['n_channels']
        self.blur_type = opt.get('blur_type', 'motion_defocus_mix')

        self.motion_kernel_size_list = opt.get('motion_kernel_size_list', [9])
        self.motion_angle_list = opt.get('motion_angle_list', [90])
        self.defocus_radius_list = opt.get('defocus_radius_list', [0, 1, 2])

        self.single_motion_kernel_size = single_motion_kernel_size
        self.single_motion_angle = single_motion_angle
        self.single_defocus_radius = single_defocus_radius

        if task == 'train':
            self.img_paths = util.get_img_paths(self.opt['dataroot_H'])
            if 'H_size' in opt:
                self.patch_size = opt['H_size']
        else:
            self.img_paths = [image_path] if image_path else []

    def _get_train_blur_params(self):
        motion_kernel_size = random.choice(self.motion_kernel_size_list)
        if motion_kernel_size % 2 == 0:
            motion_kernel_size += 1

        motion_angle = random.choice(self.motion_angle_list)
        defocus_radius = random.choice(self.defocus_radius_list)

        return motion_kernel_size, motion_angle, defocus_radius

    def _get_valid_blur_params(self):
        motion_kernel_size = self.single_motion_kernel_size if self.single_motion_kernel_size is not None else self.motion_kernel_size_list[0]
        if motion_kernel_size % 2 == 0:
            motion_kernel_size += 1

        motion_angle = self.single_motion_angle if self.single_motion_angle is not None else self.motion_angle_list[0]
        defocus_radius = self.single_defocus_radius if self.single_defocus_radius is not None else self.defocus_radius_list[0]

        return motion_kernel_size, motion_angle, defocus_radius

    def _apply_blur(self, img, motion_kernel_size, motion_angle, defocus_radius):
        if self.blur_type == 'motion_defocus_mix':
            return apply_motion_defocus_mix(
                img,
                motion_kernel_size=motion_kernel_size,
                motion_angle=motion_angle,
                defocus_radius=defocus_radius
            )
        elif self.blur_type == 'motion':
            return apply_motion_blur(img, kernel_size=motion_kernel_size, angle=motion_angle)
        elif self.blur_type == 'defocus':
            return apply_defocus_blur(img, radius=defocus_radius)
        else:
            raise ValueError(f"Unsupported blur_type: {self.blur_type}")

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_H = util.imread_uint(img_path, self.n_channels)

        H, W = img_H.shape[:2]

        if self.task == 'train':
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            patch_H = util.augment_img(patch_H, mode=np.random.randint(0, 8))

            motion_kernel_size, motion_angle, defocus_radius = self._get_train_blur_params()
            patch_L = self._apply_blur(
                patch_H,
                motion_kernel_size=motion_kernel_size,
                motion_angle=motion_angle,
                defocus_radius=defocus_radius
            )

            img_H = util.uint2tensor3(patch_H)
            img_L = util.uint2tensor3(patch_L)

        else:
            img_H = util.uint2single(img_H)
            img_L = np.copy(img_H)

            motion_kernel_size, motion_angle, defocus_radius = self._get_valid_blur_params()

            img_L = self._apply_blur(
                img_L,
                motion_kernel_size=motion_kernel_size,
                motion_angle=motion_angle,
                defocus_radius=defocus_radius
            )

            img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)

            h, w = img_H.size()[-2:]
            top = slice(0, h // 8 * 8)
            left = slice(0, (w // 8 * 8))
            img_H = img_H[..., top, left]
            img_L = img_L[..., top, left]

        return img_H, img_L

    def __len__(self):
        return len(self.img_paths)


def get_data(opt, task):
    if task == 'train':
        opt_ = opt[task]
        dataset = dataset_admm_denose(opt_, task)
        return dataset
    else:
        datasets: List[dataset_admm_denose] = []
        opt_ = opt[task]

        paths = sorted(glob(os.path.join(opt_['dataroot_H'], '*')))
        motion_kernel_size_list = opt_.get('motion_kernel_size_list', [9])
        motion_angle_list = opt_.get('motion_angle_list', [90])
        defocus_radius_list = opt_.get('defocus_radius_list', [0, 1, 2])

        for path in paths:
            for motion_kernel_size in motion_kernel_size_list:
                for motion_angle in motion_angle_list:
                    for defocus_radius in defocus_radius_list:
                        opt_subset = deepcopy(opt_)
                        datasets.append(dataset_admm_denose(
                            opt_subset,
                            task,
                            image_path=path,
                            single_motion_kernel_size=motion_kernel_size,
                            single_motion_angle=motion_angle,
                            single_defocus_radius=defocus_radius
                        ))

        return datasets