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
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal, getProjectionMatrixCenterShift
import copy
from PIL import Image
from utils.general_utils import PILtoTorch
import os, cv2
import torch.nn.functional as F
from transformers import ViTFeatureExtractor, ViTModel


def dilate(bin_img, ksize=6):
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out

def erode(bin_img, ksize=12):
    out = 1 - dilate(1 - bin_img, ksize)
    return out

class FeatureExtractorCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=6):
        super(FeatureExtractorCNN, self).__init__()
        # 定义一个简单的CNN架构，保持分辨率不变
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # 输出尺寸保持不变
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        """self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 输出尺寸保持不变
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)  # 输出尺寸保持不变
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)"""

        # 可以根据需要添加更多的卷积层

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        """x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))"""
        return x  # 输出的特征图尺寸与输入相同


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy,
                 image_width, image_height,
                 image_path, image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0,
                 ncc_scale=1.0,
                 preload_img=True, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.nearest_id = []
        self.nearest_names = []
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_path = image_path

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image, self.image_gray, self.mask = None, None, None
        self.feature_map = None
        self.preload_img = preload_img
        self.ncc_scale = ncc_scale
        if self.preload_img:
            image = Image.open(self.image_path)
            resized_image = image.resize((image_width, image_height))
            resized_image_rgb = PILtoTorch(resized_image)
            if ncc_scale != 1.0:
                resized_image = image.resize((int(image_width/ncc_scale), int(image_height/ncc_scale)))
            resized_image_gray = resized_image.convert('L')
            resized_image_gray = PILtoTorch(resized_image_gray)

            self.original_image = resized_image_rgb[:3, ...].clamp(0.0, 1.0).to(self.data_device)
            self.image_gray = resized_image_gray.clamp(0.0, 1.0).to(self.data_device)
            self.mask = torch.ones_like(self.image_gray).to(self.data_device)
            # 初始化CNN模型
            """self.cnn_model = FeatureExtractorCNN(in_channels=3, out_channels=3).to(self.data_device)
            self.cnn_model.eval()  # 设置为评估模式
            input_image = self.original_image.unsqueeze(0)  # 形状: [1, 3, H, W]
            feature_map = self.cnn_model(input_image)  # 形状: [1, 64, H, W]
            feature_map = feature_map.squeeze(0)
            self.feature_map = feature_map"""

            # for DTU
        mask_path = image_path.replace("images", "mask")[:-10]
        mask_path = mask_path + image_path[-7:]
        if os.path.exists(mask_path):
            self.mask = torch.tensor(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)).to(self.data_device).squeeze()/255
            self.mask = erode(self.mask[None,None].float()).squeeze()
            self.mask = torch.nn.functional.interpolate(self.mask[None,None], size=(image_height,image_width), mode='bilinear', align_corners=False).squeeze()
            self.mask = (self.mask < 0.5).to(self.data_device)

        self.image_width = image_width
        self.image_height = image_height
        self.resolution = (image_width, image_height)
        self.Fx = fov2focal(FoVx, self.image_width)
        self.Fy = fov2focal(FoVy, self.image_height)
        self.Cx = 0.5 * self.image_width
        self.Cy = 0.5 * self.image_height

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.plane_mask, self.non_plane_mask = None, None

    def get_image(self):
        if self.preload_img:
            return self.original_image.cuda(), self.image_gray.cuda(),self.mask.cuda()
        else:
            image = Image.open(self.image_path)
            resized_image = image.resize((self.image_width, self.image_height))
            resized_image_rgb = PILtoTorch(resized_image)
            if self.ncc_scale != 1.0:
                resized_image = image.resize((int(self.image_width/self.ncc_scale), int(self.image_height/self.ncc_scale)))
            resized_image_gray = resized_image.convert('L')
            resized_image_gray = PILtoTorch(resized_image_gray)
            gt_image = resized_image_rgb[:3, ...].clamp(0.0, 1.0)
            gt_image_gray = resized_image_gray.clamp(0.0, 1.0)
            mask = torch.ones_like(gt_image_gray)
            return gt_image.cuda(), gt_image_gray.cuda(), mask.cuda()

    def get_calib_matrix_nerf(self, scale=1.0):
        intrinsic_matrix = torch.tensor([[self.Fx/scale, 0, self.Cx/scale], [0, self.Fy/scale, self.Cy/scale], [0, 0, 1]]).float()
        extrinsic_matrix = self.world_view_transform.transpose(0,1).contiguous() # cam2world
        return intrinsic_matrix, extrinsic_matrix

    def get_rays(self, scale=1.0):
        W, H = int(self.image_width/scale), int(self.image_height/scale)
        ix, iy = torch.meshgrid(
            torch.arange(W), torch.arange(H), indexing='xy')
        rays_d = torch.stack(
                    [(ix-self.Cx/scale) / self.Fx * scale,
                    (iy-self.Cy/scale) / self.Fy * scale,
                    torch.ones_like(ix)], -1).float().cuda()
        return rays_d

    def get_k(self, scale=1.0):
        K = torch.tensor([[self.Fx / scale, 0, self.Cx / scale],
                        [0, self.Fy / scale, self.Cy / scale],
                        [0, 0, 1]]).cuda()
        return K

    def get_inv_k(self, scale=1.0):
        K_T = torch.tensor([[scale/self.Fx, 0, -self.Cx/self.Fx],
                            [0, scale/self.Fy, -self.Cy/self.Fy],
                            [0, 0, 1]]).cuda()
        return K_T

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

def sample_cam(cam_l: Camera, cam_r: Camera):
    cam = copy.copy(cam_l)

    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam_l.R.transpose()
    Rt[:3, 3] = cam_l.T
    Rt[3, 3] = 1.0

    Rt2 = np.zeros((4, 4))
    Rt2[:3, :3] = cam_r.R.transpose()
    Rt2[:3, 3] = cam_r.T
    Rt2[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    C2W2 = np.linalg.inv(Rt2)
    w = np.random.rand()
    pose_c2w_at_unseen =  w * C2W + (1 - w) * C2W2
    Rt = np.linalg.inv(pose_c2w_at_unseen)
    cam.R = Rt[:3, :3]
    cam.T = Rt[:3, 3]

    cam.world_view_transform = torch.tensor(getWorld2View2(cam.R, cam.T, cam.trans, cam.scale)).transpose(0, 1).cuda()
    cam.projection_matrix = getProjectionMatrix(znear=cam.znear, zfar=cam.zfar, fovX=cam.FoVx, fovY=cam.FoVy).transpose(0,1).cuda()
    cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
    cam.camera_center = cam.world_view_transform.inverse()[3, :3]
    return cam
