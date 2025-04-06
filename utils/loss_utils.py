#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=False):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        #return ssim_map.mean(1).mean(1).mean(1)
        return ssim_map


from torchvision.transforms import functional as TF


# 计算亮度权值图
"""def compute_brightness_weight(image):
    # 转换为灰度图
    gray_image = TF.rgb_to_grayscale(image)  # 输出为 (1, H, W)

    # 归一化亮度到 [0, 1]，防止全黑图像
    normalized_brightness = gray_image / (gray_image.max() + 1e-8)

    return normalized_brightness"""


def compute_contrast_loss(predicted_image, e):
    # 确保predicted_image是一个torch.Tensor
    if not isinstance(predicted_image, torch.Tensor):
        raise ValueError("predicted_image must be a torch.Tensor")

    # 计算每个通道的方差
    variances = torch.var(predicted_image, dim=(1, 2), unbiased=False)


    # 计算损失函数
    loss = (variances - e) ** 2
    contrast_loss = loss.mean()  # 对所有通道的损失求平均

    return contrast_loss
def compute_contrast_part_loss(predicted_image, e):
    # 确保predicted_image是一个torch.Tensor
    if not isinstance(predicted_image, torch.Tensor):
        raise ValueError("predicted_image must be a torch.Tensor")

    # 计算需要裁剪的尺寸，以确保可以被4整除
    height, width = predicted_image.shape[1], predicted_image.shape[2]
    new_height = height - (height % 4)
    new_width = width - (width % 4)

    # 裁剪图像
    predicted_image = predicted_image[:, :new_height, :new_width]

    # 将图像划分为4x4的块
    patches = F.unfold(predicted_image, kernel_size=(4, 4), stride=(4, 4))

    # 重新排列维度，以便每个块的通道数在一起
    patches = patches.view(predicted_image.shape[0], -1, 16)  # 假设每个块有16个元素

    # 计算每个块的方差
    variances = torch.var(patches, dim=2, unbiased=False)

    # 计算损失函数
    loss = (variances - e) ** 2

    # 对所有块的损失求平均
    contrast_loss = loss.mean()

    return contrast_loss

def compute_brightness_weight(image):
    """
    根据图像亮度生成权值图，亮度高的地方权值低，亮度低的地方权值高。
    :param image: 输入图像 (C, H, W)，假设是 RGB 格式，范围 [0, 1]
    :return: 权值图 (1, H, W)，范围 [1, 10]
    """
    import torchvision.transforms.functional as TF

    # 转换为灰度图
    gray_image = TF.rgb_to_grayscale(image)  # 输出为 (1, H, W)

    # 归一化亮度到 [0, 1]，防止全黑图像
    normalized_brightness = gray_image / (gray_image.max() + 1e-8)

    # 反转亮度（亮度高的地方权值低，亮度低的地方权值高）
    inverted_brightness = 1 - normalized_brightness

    return  inverted_brightness


# 计算逐像素 SSIM 值
def compute_weighted_ssim(image, gt_image):
    """
    计算加权 SSIM 损失
    :param image: 输入图像 (C, H, W)，范围 [0, 1]
    :param gt_image: GT 图像 (C, H, W)，范围 [0, 1]
    :return: 加权 SSIM 损失 (标量)
    """
    # 逐像素计算 SSIM 热图 (H, W)
    ssim_map = ssim(image, gt_image, size_average=False)  # 不求均值

    # 生成亮度权值图 (1, H, W)
    brightness_weight = compute_brightness_weight(gt_image)  # 假设权值基于 GT 图像

    # 对 SSIM 热图加权 (H, W)
    weighted_ssim_map = ssim_map * brightness_weight.squeeze(0)

    # 计算加权后的 SSIM 损失（取均值）
    weighted_ssim_loss = 1.0 - weighted_ssim_map.mean()

    return weighted_ssim_loss


def ssim2(img1, img2, window_size=11):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean(0)

def get_img_grad_weight(img, beta=2.0):
    _, hd, wd = img.shape 
    bottom_point = img[..., 2:hd,   1:wd-1]
    top_point    = img[..., 0:hd-2, 1:wd-1]
    right_point  = img[..., 1:hd-1, 2:wd]
    left_point   = img[..., 1:hd-1, 0:wd-2]
    grad_img_x = torch.mean(torch.abs(right_point - left_point), 0, keepdim=True)
    grad_img_y = torch.mean(torch.abs(top_point - bottom_point), 0, keepdim=True)
    grad_img = torch.cat((grad_img_x, grad_img_y), dim=0)
    grad_img, _ = torch.max(grad_img, dim=0)
    grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min())
    grad_img = torch.nn.functional.pad(grad_img[None,None], (1,1,1,1), mode='constant', value=1.0).squeeze()
    return grad_img

def lncc(ref, nea):
    # ref_gray: [batch_size, total_patch_size]
    # nea_grays: [batch_size, total_patch_size]
    bs, tps = nea.shape
    patch_size = int(np.sqrt(tps))

    ref_nea = ref * nea
    ref_nea = ref_nea.view(bs, 1, patch_size, patch_size)
    ref = ref.view(bs, 1, patch_size, patch_size)
    nea = nea.view(bs, 1, patch_size, patch_size)
    ref2 = ref.pow(2)
    nea2 = nea.pow(2)

    # sum over kernel
    filters = torch.ones(1, 1, patch_size, patch_size, device=ref.device)
    padding = patch_size // 2
    ref_sum = F.conv2d(ref, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea_sum = F.conv2d(nea, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref2_sum = F.conv2d(ref2, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea2_sum = F.conv2d(nea2, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref_nea_sum = F.conv2d(ref_nea, filters, stride=1, padding=padding)[:, :, padding, padding]

    # average over kernel
    ref_avg = ref_sum / tps
    nea_avg = nea_sum / tps

    cross = ref_nea_sum - nea_avg * ref_sum
    ref_var = ref2_sum - ref_avg * ref_sum
    nea_var = nea2_sum - nea_avg * nea_sum

    cc = cross * cross / (ref_var * nea_var + 1e-8)
    ncc = 1 - cc
    ncc = torch.clamp(ncc, 0.0, 2.0)
    ncc = torch.mean(ncc, dim=1, keepdim=True)
    mask = (ncc < 0.9)
    return ncc, mask