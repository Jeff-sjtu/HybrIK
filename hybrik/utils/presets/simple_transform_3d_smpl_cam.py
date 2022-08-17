import math
import random

import cv2
import numpy as np
import torch

from ..bbox import _box_to_center_scale, _center_scale_to_box
from ..transforms import (addDPG, affine_transform, flip_joints_3d, flip_thetas, flip_xyz_joints_3d,
                          get_affine_transform, im_to_torch, batch_rodrigues_numpy, flip_twist,
                          rotmat_to_quat_numpy, rotate_xyz_jts, rot_aa, flip_cam_xyz_joints_3d)
from ..pose_utils import get_intrinsic_metrix

s_coco_2_smpl_jt = [
    -1, 11, 12,
    -1, 13, 14,
    -1, 15, 16,
    -1, -1, -1,
    -1, -1, -1,
    -1,
    5, 6,
    7, 8,
    9, 10,
    -1, -1
]

s_coco_2_h36m_jt = [
    -1,
    -1, 13, 15,
    -1, 14, 16,
    -1, -1,
    0, -1,
    5, 7, 9,
    6, 8, 10
]

s_coco_2_smpl_jt_2d = [
    -1, -1, -1,
    -1, 13, 14,
    -1, 15, 16,
    -1, -1, -1,
    -1, -1, -1,
    -1,
    5, 6,
    7, 8,
    9, 10,
    -1, -1
]

smpl_parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
                16, 17, 18, 19, 20, 21]


left_bones_idx = [
    (0, 1), (1, 4), (4, 7), (12, 13),
    (13, 16), (16, 18), (18, 20)
]

right_bones_idx = [
    (0, 2), (2, 5), (5, 8), (12, 14),
    (14, 17), (17, 19), (19, 21)
]

skeleton_29 = [ 
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), # 5
    (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12), # 11
    (9, 13), (9, 14), (12, 15), (13, 16), (14, 17), (16, 18), # 17
    (17, 19), (18, 20), (19, 21), (20, 22), (21, 23), (15, 24), # 23
    (22, 25), (23, 26), (10, 27), (11, 28) # 27
]

skeleton_3dhp = np.array([(-1, -1)] * 28).astype(int)
skeleton_3dhp[ [6, 7, 17, 18, 19, 20] ] = np.array([
        (19, 20), (24, 25), (9, 10), (14, 15), (10, 11), (15, 16)
    ]).astype(int)

class SimpleTransform3DSMPLCam(object):
    """Generation of cropped input person, pose coords, smpl parameters.

    Parameters
    ----------
    img: torch.Tensor
        A tensor with shape: `(3, h, w)`.
    label: dict
        A dictionary with 4 keys:
            `bbox`: [xmin, ymin, xmax, ymax]
            `joints_3d`: numpy.ndarray with shape: (n_joints, 2),
                    including position and visible flag
            `width`: image width
            `height`: image height
    dataset:
        The dataset to be transformed, must include `joint_pairs` property for flipping.
    scale_factor: int
        Scale augmentation.
    input_size: tuple
        Input image size, as (height, width).
    output_size: tuple
        Heatmap size, as (height, width).
    rot: int
        Ratation augmentation.
    train: bool
        True for training trasformation.
    """

    def __init__(self, dataset, scale_factor, color_factor, occlusion, add_dpg,
                 input_size, output_size, depth_dim, bbox_3d_shape,
                 rot, sigma, train, loss_type='MSELoss', scale_mult=1.25, focal_length=1000, two_d=False,
                 root_idx=0, get_paf=False):
        if two_d:
            self._joint_pairs = dataset.joint_pairs
        else:
            self._joint_pairs_17 = dataset.joint_pairs_17
            self._joint_pairs_24 = dataset.joint_pairs_24
            self._joint_pairs_29 = dataset.joint_pairs_29

        self._scale_factor = scale_factor
        self._color_factor = color_factor
        self._occlusion = occlusion
        self._rot = rot
        self._add_dpg = add_dpg

        self._input_size = input_size
        self._heatmap_size = output_size

        self._sigma = sigma
        self._train = train
        self._loss_type = loss_type
        self._aspect_ratio = float(input_size[1]) / input_size[0]  # w / h
        self._feat_stride = np.array(input_size) / np.array(output_size)

        self.pixel_std = 1

        self.bbox_3d_shape = dataset.bbox_3d_shape
        self._scale_mult = scale_mult
        self.two_d = two_d

        # convert to unit: meter
        self.depth_factor2meter = self.bbox_3d_shape[2] if self.bbox_3d_shape[2] < 500 else self.bbox_3d_shape[2]*1e-3

        self.focal_length = focal_length
        self.root_idx = root_idx

        self.get_paf = get_paf
        # self.use_camera = use_camera

        if train:
            self.num_joints_half_body = dataset.num_joints_half_body
            self.prob_half_body = dataset.prob_half_body

            self.upper_body_ids = dataset.upper_body_ids
            self.lower_body_ids = dataset.lower_body_ids

    def test_transform(self, src, bbox):
        xmin, ymin, xmax, ymax = bbox
        center, scale = _box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio, scale_mult=self._scale_mult)
        scale = scale * 1.0

        input_size = self._input_size
        inp_h, inp_w = input_size
        trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
        img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        bbox = _center_scale_to_box(center, scale)

        img = im_to_torch(img)
        # mean
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)

        # std
        img[0].div_(0.225)
        img[1].div_(0.224)
        img[2].div_(0.229)

        img_center = np.array([float(src.shape[1]) * 0.5, float(src.shape[0]) * 0.5])

        return img, bbox, img_center

    def _integral_target_generator(self, joints_3d, num_joints, patch_height, patch_width):
        target_weight = np.ones((num_joints, 3), dtype=np.float32)
        target_weight[:, 0] = joints_3d[:, 0, 1]
        target_weight[:, 1] = joints_3d[:, 0, 1]
        target_weight[:, 2] = joints_3d[:, 0, 1]

        target = np.zeros((num_joints, 3), dtype=np.float32)
        target[:, 0] = joints_3d[:, 0, 0] / patch_width - 0.5
        target[:, 1] = joints_3d[:, 1, 0] / patch_height - 0.5
        target[:, 2] = joints_3d[:, 2, 0] / self.bbox_3d_shape[0]

        target_weight[target[:, 0] > 0.5] = 0
        target_weight[target[:, 0] < -0.5] = 0
        target_weight[target[:, 1] > 0.5] = 0
        target_weight[target[:, 1] < -0.5] = 0
        target_weight[target[:, 2] > 0.5] = 0
        target_weight[target[:, 2] < -0.5] = 0

        target = target.reshape((-1))
        target_weight = target_weight.reshape((-1))
        return target, target_weight

    def _integral_uvd_target_generator(self, joints_3d, num_joints, patch_height, patch_width):

        target_weight = np.ones((num_joints, 3), dtype=np.float32)
        target_weight[:, 0] = joints_3d[:, 0, 1]
        target_weight[:, 1] = joints_3d[:, 0, 1]
        target_weight[:, 2] = joints_3d[:, 0, 1]

        target = np.zeros((num_joints, 3), dtype=np.float32)
        target[:, 0] = joints_3d[:, 0, 0] / patch_width - 0.5
        target[:, 1] = joints_3d[:, 1, 0] / patch_height - 0.5
        target[:, 2] = joints_3d[:, 2, 0] / self.bbox_3d_shape[2]

        target_weight[target[:, 0] > 0.5] = 0
        target_weight[target[:, 0] < -0.5] = 0
        target_weight[target[:, 1] > 0.5] = 0
        target_weight[target[:, 1] < -0.5] = 0
        target_weight[target[:, 2] > 0.5] = 0
        target_weight[target[:, 2] < -0.5] = 0

        target = target.reshape((-1))
        target_weight = target_weight.reshape((-1))
        return target, target_weight

    def _integral_xyz_target_generator(self, joints_3d, joints_3d_vis, num_joints):
        target_weight = np.ones((num_joints, 3), dtype=np.float32)
        target_weight[:, 0] = joints_3d_vis[:, 0]
        target_weight[:, 1] = joints_3d_vis[:, 1]
        target_weight[:, 2] = joints_3d_vis[:, 2]

        target = np.zeros((num_joints, 3), dtype=np.float32)
        target[:, 0] = joints_3d[:, 0] / self.bbox_3d_shape[0]
        target[:, 1] = joints_3d[:, 1] / self.bbox_3d_shape[1]
        target[:, 2] = joints_3d[:, 2] / self.bbox_3d_shape[2]

        # if self.bbox_3d_shape[0] < 1000:
        #     print(self.bbox_3d_shape, target)
        
        # assert (target[0] == 0).all(), f'{target}, {self.bbox_3d_shape}'

        target = target.reshape((-1))
        target_weight = target_weight.reshape((-1))
        return target, target_weight

    def __call__(self, src, label):
        if self.two_d:
            bbox = list(label['bbox'])
            joint_img = label['joint_img'].copy()
            joints_vis = label['joint_vis'].copy()
            joint_cam = label['joint_cam'].copy()
            self.num_joints = joint_img.shape[0]

            gt_joints = np.zeros((self.num_joints, 3, 2), dtype=np.float32)
            gt_joints[:, :, 0] = joint_img
            gt_joints[:, :, 1] = joints_vis

            imgwidth, imght = label['width'], label['height']
            assert imgwidth == src.shape[1] and imght == src.shape[0]
            self.num_joints = gt_joints.shape[0]

            input_size = self._input_size

            if self._add_dpg and self._train:
                bbox = addDPG(bbox, imgwidth, imght)

            xmin, ymin, xmax, ymax = bbox
            center, scale = _box_to_center_scale(
                xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio, scale_mult=self._scale_mult)
            
            xmin, ymin, xmax, ymax = _center_scale_to_box(center, scale)

            # half body transform
            if self._train and (np.sum(joints_vis[:, 0]) > self.num_joints_half_body and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    gt_joints[:, :, 0], joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    center, scale = c_half_body, s_half_body

            # rescale
            if self._train:
                sf = self._scale_factor
                scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            else:
                scale = scale * 1.0

            # rotation
            if self._train:
                rf = self._rot
                r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0
            else:
                r = 0

            if self._train and self._occlusion:
                while True:
                    area_min = 0.0
                    area_max = 0.7
                    synth_area = (random.random() * (area_max - area_min) + area_min) * (xmax - xmin) * (ymax - ymin)

                    ratio_min = 0.3
                    ratio_max = 1 / 0.3
                    synth_ratio = (random.random() * (ratio_max - ratio_min) + ratio_min)

                    synth_h = math.sqrt(synth_area * synth_ratio)
                    synth_w = math.sqrt(synth_area / synth_ratio)
                    synth_xmin = random.random() * ((xmax - xmin) - synth_w - 1) + xmin
                    synth_ymin = random.random() * ((ymax - ymin) - synth_h - 1) + ymin

                    if synth_xmin >= 0 and synth_ymin >= 0 and synth_xmin + synth_w < imgwidth and synth_ymin + synth_h < imght:
                        synth_xmin = int(synth_xmin)
                        synth_ymin = int(synth_ymin)
                        synth_w = int(synth_w)
                        synth_h = int(synth_h)
                        src[synth_ymin:synth_ymin + synth_h, synth_xmin:synth_xmin + synth_w, :] = np.random.rand(synth_h, synth_w, 3) * 255
                        break
            
            joint_cam = joint_cam.reshape(-1 , 3)
            joints_xyz = joint_cam - joint_cam[[self.root_idx]].copy() # the root index of mpii_3d is 4 !!!

            joints = gt_joints
            if random.random() > 0.5 and self._train:
                # src, fliped = random_flip_image(src, px=0.5, py=0)
                # if fliped[0]:
                assert src.shape[2] == 3
                src = src[:, ::-1, :]

                joints = flip_joints_3d(joints, imgwidth, self._joint_pairs)
                joints_xyz = flip_xyz_joints_3d(joints_xyz, self._joint_pairs)
                center[0] = imgwidth - center[0] - 1
            
            joints_xyz = rotate_xyz_jts(joints_xyz, r)

            inp_h, inp_w = input_size
            trans = get_affine_transform(center, scale, r, [inp_w, inp_h])
            img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)

            # deal with joints visibility
            for i in range(self.num_joints):
                if joints[i, 0, 1] > 0.0:
                    joints[i, 0:2, 0] = affine_transform(joints[i, 0:2, 0], trans)

            trans_inv = get_affine_transform(center, scale, r, [inp_w, inp_h], inv=True).astype(np.float32)
            intrinsic_param = get_intrinsic_metrix(label['f'], label['c'], inv=True).astype(np.float32) if 'f' in label.keys() else np.zeros((3, 3)).astype(np.float32)
            joint_root = label['root_cam'].astype(np.float32) if 'root_cam' in label.keys() else np.zeros((3)).astype(np.float32)
            depth_factor = np.array([self.bbox_3d_shape[2]]).astype(np.float32) if self.bbox_3d_shape else np.zeros((1)).astype(np.float32)

            # generate training targets
            target, target_weight = self._integral_target_generator(joints, self.num_joints, inp_h, inp_w)
            target_xyz, target_xyz_weight = self._integral_xyz_target_generator(joints_xyz, joints_vis, len(joints_vis))

            target_weight *= joints_vis.reshape(-1)
            bbox = _center_scale_to_box(center, scale)

            cam_scale, cam_trans, cam_valid, cam_error, new_uvd = self.calc_cam_scale_trans2(
                                        target_xyz.reshape(-1, 3).copy(), 
                                        target.reshape(-1, 3).copy(), 
                                        target_weight.reshape(-1, 3).copy())

        else:
            bbox = list(label['bbox'])
            joint_img_17 = label['joint_img_17'].copy()
            joint_relative_17 = label['joint_relative_17'].copy()
            joint_cam_17 = label['joint_cam_17'].copy()
            joints_vis_17 = label['joint_vis_17'].copy()
            joint_img_29 = label['joint_img_29'].copy()
            joint_cam_29 = label['joint_cam_29'].copy()
            joints_vis_29 = label['joint_vis_29'].copy()
            # root_cam = label['root_cam'].copy()
            # root_depth = root_cam[2] / self.bbox_3d_shape[2]
            fx, fy = label['f'].copy()

            beta = label['beta'].copy()
            theta = label['theta'].copy()

            assert not (theta<1e-3).all(), label

            if 'twist_phi' in label.keys():
                twist_phi = label['twist_phi'].copy()
                twist_weight = label['twist_weight'].copy()
            else:
                twist_phi = np.zeros((23, 2))
                twist_weight = np.zeros((23, 2))

            gt_joints_17 = np.zeros((17, 3, 2), dtype=np.float32)
            gt_joints_17[:, :, 0] = joint_img_17.copy()
            gt_joints_17[:, :, 1] = joints_vis_17.copy()
            gt_joints_29 = np.zeros((29, 3, 2), dtype=np.float32)
            gt_joints_29[:, :, 0] = joint_img_29.copy()
            gt_joints_29[:, :, 1] = joints_vis_29.copy()

            imgwidth, imght = src.shape[1], src.shape[0]

            input_size = self._input_size

            if self._add_dpg and self._train:
                bbox = addDPG(bbox, imgwidth, imght)

            xmin, ymin, xmax, ymax = bbox
            center, scale = _box_to_center_scale(
                xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio, scale_mult=self._scale_mult)

            xmin, ymin, xmax, ymax = _center_scale_to_box(center, scale)

            # half body transform
            if self._train and (np.sum(joints_vis_17[:, 0]) > self.num_joints_half_body and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    gt_joints_17[:, :, 0], joints_vis_17
                )

                if c_half_body is not None and s_half_body is not None:
                    center, scale = c_half_body, s_half_body

            # rescale
            if self._train:
                sf = self._scale_factor
                scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            else:
                scale = scale * 1.0

            # rotation
            if self._train:
                rf = self._rot
                r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0
            else:
                r = 0

            if self._train and self._occlusion:
                while True:
                    area_min = 0.0
                    area_max = 0.3
                    synth_area = (random.random() * (area_max - area_min) + area_min) * (xmax - xmin) * (ymax - ymin)

                    ratio_min = 0.5
                    ratio_max = 1 / 0.5
                    synth_ratio = (random.random() * (ratio_max - ratio_min) + ratio_min)

                    synth_h = math.sqrt(synth_area * synth_ratio)
                    synth_w = math.sqrt(synth_area / synth_ratio)
                    synth_xmin = random.random() * ((xmax - xmin) - synth_w - 1) + xmin
                    synth_ymin = random.random() * ((ymax - ymin) - synth_h - 1) + ymin

                    if synth_xmin >= 0 and synth_ymin >= 0 and synth_xmin + synth_w < imgwidth and synth_ymin + synth_h < imght:
                        synth_xmin = int(synth_xmin)
                        synth_ymin = int(synth_ymin)
                        synth_w = int(synth_w)
                        synth_h = int(synth_h)
                        src[synth_ymin:synth_ymin + synth_h, synth_xmin:synth_xmin + synth_w, :] = np.random.rand(synth_h, synth_w, 3) * 255
                        break

            joints_17_uvd = gt_joints_17
            joints_29_uvd = gt_joints_29

            joint_cam_17_xyz = joint_cam_17
            joints_cam_24_xyz = joint_cam_29[:24]


            if random.random() > 0.75 and self._train:
                assert src.shape[2] == 3
                src = src[:, ::-1, :]

                joints_17_uvd = flip_joints_3d(joints_17_uvd, imgwidth, self._joint_pairs_17)
                joints_29_uvd = flip_joints_3d(joints_29_uvd, imgwidth, self._joint_pairs_29)
                joint_cam_17_xyz = flip_cam_xyz_joints_3d(joint_cam_17_xyz, self._joint_pairs_17)
                joints_cam_24_xyz = flip_cam_xyz_joints_3d(joints_cam_24_xyz, self._joint_pairs_24)
                theta = flip_thetas(theta, self._joint_pairs_24)
                twist_phi, twist_weight = flip_twist(twist_phi, twist_weight, self._joint_pairs_24)
                center[0] = imgwidth - center[0] - 1

            # rotate global theta
            theta[0, :3] = rot_aa(theta[0, :3], r)

            theta_rot_mat = batch_rodrigues_numpy(theta)
            theta_quat = rotmat_to_quat_numpy(theta_rot_mat).reshape(24 * 4)

            # rotate xyz joints
            joint_cam_17_xyz = rotate_xyz_jts(joint_cam_17_xyz, r)
            joints_17_xyz = joint_cam_17_xyz - joint_cam_17_xyz[:1].copy()
            joints_cam_24_xyz = rotate_xyz_jts(joints_cam_24_xyz, r)
            joints_24_xyz = joints_cam_24_xyz - joints_cam_24_xyz[:1].copy()

            inp_h, inp_w = input_size
            trans = get_affine_transform(center, scale, r, [inp_w, inp_h])
            trans_inv = get_affine_transform(center, scale, r, [inp_w, inp_h], inv=True).astype(np.float32)
            intrinsic_param = get_intrinsic_metrix(label['f'], label['c'], inv=True).astype(np.float32) if 'f' in label.keys() else np.zeros((3, 3)).astype(np.float32)
            joint_root = label['root_cam'].astype(np.float32) if 'root_cam' in label.keys() else np.zeros((3)).astype(np.float32)
            depth_factor = np.array([self.bbox_3d_shape[2]]).astype(np.float32) if self.bbox_3d_shape else np.zeros((1)).astype(np.float32)

            img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
            # affine transform
            for i in range(17):
                if joints_17_uvd[i, 0, 1] > 0.0:
                    joints_17_uvd[i, 0:2, 0] = affine_transform(joints_17_uvd[i, 0:2, 0], trans)

            for i in range(29):
                if joints_29_uvd[i, 0, 1] > 0.0:
                    joints_29_uvd[i, 0:2, 0] = affine_transform(joints_29_uvd[i, 0:2, 0], trans)
            
            target_smpl_weight = torch.ones(1).float()
            theta_24_weights = np.ones((24, 4))

            theta_24_weights = theta_24_weights.reshape(24 * 4)

            # generate training targets
            target_uvd_29, target_weight_29 = self._integral_uvd_target_generator(joints_29_uvd, 29, inp_h, inp_w)
            target_xyz_17, target_weight_17 = self._integral_xyz_target_generator(joints_17_xyz, joints_vis_17, 17)
            target_xyz_24, target_weight_24 = self._integral_xyz_target_generator(joints_24_xyz, joints_vis_29[:24, :], 24)

            target_weight_29 *= joints_vis_29.reshape(-1)
            target_weight_24 *= joints_vis_29[:24, :].reshape(-1)
            target_weight_17 *= joints_vis_17.reshape(-1)
            bbox = _center_scale_to_box(center, scale)

            tmp_uvd_24 = target_uvd_29.reshape(-1, 3)[:24]
            tmp_uvd_24_weight = target_weight_29.reshape(-1, 3)[:24]

            if self.focal_length > 0:
                cam_scale, cam_trans, cam_valid, cam_error, new_uvd = self.calc_cam_scale_trans2(
                                                                target_xyz_24.reshape(-1, 3).copy(), 
                                                                tmp_uvd_24.copy(), 
                                                                tmp_uvd_24_weight.copy())
            
                target_uvd_29 = (target_uvd_29 * target_weight_29).reshape(-1, 3)
            else:
                cam_scale = 1
                cam_trans = np.zeros(2)
                cam_valid = 0
                cam_error = 0

        assert img.shape[2] == 3
        if self._train:
            c_high = 1 + self._color_factor
            c_low = 1 - self._color_factor
            img[:, :, 0] = np.clip(img[:, :, 0] * random.uniform(c_low, c_high), 0, 255)
            img[:, :, 1] = np.clip(img[:, :, 1] * random.uniform(c_low, c_high), 0, 255)
            img[:, :, 2] = np.clip(img[:, :, 2] * random.uniform(c_low, c_high), 0, 255)

        img = im_to_torch(img)
        # mean
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)

        # std
        img[0].div_(0.225)
        img[1].div_(0.224)
        img[2].div_(0.229)

        img_center = np.array([float(imgwidth) * 0.5, float(imght) * 0.5])

        if self.two_d:
            output = {
                'type': '2d_data',
                'image': img,
                'target': torch.from_numpy(target.reshape(-1)).float(),
                'target_weight': torch.from_numpy(target_weight).float(),
                'trans_inv': torch.from_numpy(trans_inv).float(),
                'intrinsic_param': torch.from_numpy(intrinsic_param).float(),
                'joint_root': torch.from_numpy(joint_root).float(),
                'depth_factor': torch.from_numpy(depth_factor).float(),
                'bbox': torch.Tensor(bbox),
                'camera_scale': torch.from_numpy(np.array([cam_scale])).float(),
                'camera_trans': torch.from_numpy(cam_trans).float(),
                'camera_valid': cam_valid,
                'target_xyz': torch.from_numpy(target_xyz).float(),
                'target_xyz_weight': torch.from_numpy(target_xyz_weight).float(),
                'camera_error': cam_error,
                'img_center': torch.from_numpy(img_center).float()
            }

        else:
            output = {
                'type': '3d_data_w_smpl',
                'image': img,
                'target_theta': torch.from_numpy(theta_quat).float(),
                'target_theta_weight': torch.from_numpy(theta_24_weights).float(),
                'target_beta': torch.from_numpy(beta).float(),
                'target_smpl_weight': target_smpl_weight,
                'target_uvd_29': torch.from_numpy(target_uvd_29.reshape(-1)).float(),
                'target_xyz_24': torch.from_numpy(target_xyz_24).float(),
                'target_weight_29': torch.from_numpy(target_weight_29).float(),
                'target_weight_24': torch.from_numpy(target_weight_24).float(),
                'target_xyz_17': torch.from_numpy(target_xyz_17).float(),
                'target_weight_17': torch.from_numpy(target_weight_17).float(),
                'target_xyz_weight_24': torch.from_numpy(target_weight_24).float(),
                'trans_inv': torch.from_numpy(trans_inv).float(),
                'intrinsic_param': torch.from_numpy(intrinsic_param).float(),
                'joint_root': torch.from_numpy(joint_root).float(),
                'depth_factor': torch.from_numpy(depth_factor).float(),
                'bbox': torch.Tensor(bbox),
                # 'target_depth': torch.ones(1).float() * target_depth,
                # 'target_depth_coeff': torch.ones(1).float() * target_depth_coeff,
                # 'target_depth_weight': torch.ones(1).float(),
                'target_twist': torch.from_numpy(twist_phi).float(),
                'target_twist_weight': torch.from_numpy(twist_weight).float(),
                'camera_scale': torch.from_numpy(np.array([cam_scale])).float(),
                'camera_trans': torch.from_numpy(cam_trans).float(),
                'camera_valid': cam_valid,
                'camera_error': cam_error,
                'img_center': torch.from_numpy(img_center).float()
            }
        return output

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self._aspect_ratio * h:
            h = w * 1.0 / self._aspect_ratio
        elif w < self._aspect_ratio * h:
            w = h * self._aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    def calc_cam_scale_trans2(self, xyz_29, uvd_29, uvd_weight):

        f = self.focal_length

        # unit: meter
        # the equation to be solved: 
        # u * 256 / f * (z + f/256 * 1/scale) = x + tx
        # v * 256 / f * (z + f/256 * 1/scale) = y + ty

        weight = (uvd_weight.sum(axis=-1, keepdims=True) >= 3.0) * 1.0 # 24 x 1
        # assert weight.sum() >= 2, 'too few valid keypoints to calculate cam para'

        if weight.sum() < 2:
            # print('bad data')
            return 0, np.zeros(2), 0.0, -1, uvd_29

        xyz_29 = xyz_29 * self.depth_factor2meter # convert to meter
        new_uvd = uvd_29.copy()

        num_joints = len(uvd_29)

        Ax = np.zeros((num_joints, 3))
        Ax[:, 1] = -1
        Ax[:, 0] = uvd_29[:, 0]

        Ay = np.zeros((num_joints, 3))
        Ay[:, 2] = -1
        Ay[:, 0] = uvd_29[:, 1]

        Ax = Ax * weight
        Ay = Ay * weight

        A = np.concatenate([Ax, Ay], axis=0)

        bx = (xyz_29[:, 0] - 256 * uvd_29[:, 0] / f * xyz_29[:, 2]) * weight[:, 0]
        by = (xyz_29[:, 1] - 256 * uvd_29[:, 1] / f * xyz_29[:, 2]) * weight[:, 0]
        b = np.concatenate([bx, by], axis=0)

        A_s = np.dot(A.T, A)
        b_s = np.dot(A.T, b)

        cam_para = np.linalg.solve(A_s, b_s)

        trans = cam_para[1:]
        scale = 1.0 / cam_para[0]

        target_camera = np.zeros(3)
        target_camera[0] = scale
        target_camera[1:] = trans

        backed_projected_xyz = self.back_projection(uvd_29, target_camera, f)
        backed_projected_xyz[:, 2] = backed_projected_xyz[:, 2] * self.depth_factor2meter
        diff = np.sum((backed_projected_xyz-xyz_29)**2, axis=-1) * weight[:, 0]
        diff = np.sqrt(diff).sum() / (weight.sum()+1e-6) * 1000 # roughly mpjpe > 70
        # print(scale, trans, diff)
        if diff < 70:
            new_uvd = self.projection(xyz_29, target_camera, f)
            return scale, trans, 1.0, diff, new_uvd * uvd_weight
        else:
            return scale, trans, 0.0, diff, new_uvd

    def projection(self, xyz, camera, f):
        # xyz: unit: meter, u = f/256 * (x+dx) / (z+dz)
        transl = camera[1:3]
        scale = camera[0]
        z_cam = xyz[:, 2:] + f / (256.0 * scale) # J x 1
        uvd = np.zeros_like(xyz)
        uvd[:, 2] = xyz[:, 2] / self.bbox_3d_shape[2]
        uvd[:, :2] = f / 256.0 * (xyz[:, :2] + transl) / z_cam
        return uvd
    
    def back_projection(self, uvd, pred_camera, focal_length=5000.):
        camScale = pred_camera[:1].reshape(1, -1)
        camTrans = pred_camera[1:].reshape(1, -1)

        camDepth = focal_length / (256 * camScale)

        pred_xyz = np.zeros_like(uvd)
        pred_xyz[:, 2] = uvd[:, 2].copy()
        pred_xyz[:, :2] = (uvd[:, :2] * 256 / focal_length) * (pred_xyz[:, 2:]*self.depth_factor2meter + camDepth) - camTrans

        return pred_xyz


def _box_to_center_scale_nosquare(x, y, w, h, aspect_ratio=1.0, scale_mult=1.5):
    """Convert box coordinates to center and scale.
    adapted from https://github.com/Microsoft/human-pose-estimation.pytorch
    """
    pixel_std = 1
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    if center[0] != -1:
        scale = scale * scale_mult
    return center, scale
