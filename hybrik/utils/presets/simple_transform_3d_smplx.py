import math
import random

import cv2
import numpy as np
import torch

from ..bbox import _box_to_center_scale, _center_scale_to_box
from ..transforms import (addDPG, affine_transform, flip_joints_3d, flip_thetas,
                          get_affine_transform, im_to_torch, batch_rodrigues_numpy, flip_twist,
                          rotmat_to_quat_numpy, rotate_xyz_jts, rot_aa, flip_cam_xyz_joints_3d)
from ..pose_utils import get_intrinsic_metrix
from hybrik.models.layers.smplx.body_models import SMPLXLayer
# from hybrik.models.layers.smplx.joint_names import JOINT_NAMES


smplx_layer_neutral = SMPLXLayer(
    model_path='model_files/smplx/SMPLX_NEUTRAL.npz',
    num_betas=10,
    use_pca=False,
    age='kid',
    kid_template_path='model_files/smplx_kid_template.npy',
)

smplx_layer_female = SMPLXLayer(
    model_path='model_files/smplx/SMPLX_FEMALE.npz',
    num_betas=10,
    use_pca=False,
    age='kid',
    kid_template_path='model_files/smplx_kid_template.npy',
)

smplx_layer_male = SMPLXLayer(
    model_path='model_files/smplx/SMPLX_MALE.npz',
    num_betas=10,
    use_pca=False,
    age='kid',
    kid_template_path='model_files/smplx_kid_template.npy',
)


class SimpleTransform3DSMPLX(object):
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
                 root_idx=0, return_vertices=False, rand_bbox_shift=False):
        # if two_d:
        #     self._joint_pairs = dataset.joint_pairs
        # else:
        #     self._joint_pairs_17 = dataset.joint_pairs_17
        #     self._joint_pairs_24 = dataset.joint_pairs_24
        #     self._joint_pairs_29 = dataset.joint_pairs_29
        self.body_joint_pairs = [
            (1, 2), (4, 5), (7, 8), (10, 11),
            (13, 14), (16, 17), (18, 19), (20, 21),
            (23, 24)
        ]
        self.hand_joint_pairs = [
            (25 + i, 40 + i) for i in range(40 - 25)
        ]
        self.leaf_joint_pairs = [
            (55, 56), (58, 59)
        ] + [(60 + i, 65 + i) for i in range(5)]

        self.joint_pairs_71 = self.body_joint_pairs + self.hand_joint_pairs + self.leaf_joint_pairs
        self.joint_pairs_55 = self.body_joint_pairs + self.hand_joint_pairs
        # self.joint_pairs_hybrik = self.joint_pairs + self.leaf_joint_pairs
        # self.joint_pairs_full = []

        # for i, name in enumerate(JOINT_NAMES):
        #     if i >= 127:
        #         break
        #     if 'left' in name:
        #         left_idx = i
        #         right_name = name.replace('left', 'right')
        #         right_idx = JOINT_NAMES.index(right_name)
        #         self.joint_pairs_full.append((left_idx, right_idx))

        # for pair in self.joint_pairs:
        #     assert pair in self.joint_pairs_full

        self._scale_factor = scale_factor
        self._color_factor = color_factor
        self._occlusion = occlusion
        self._rot = rot
        self._add_dpg = add_dpg
        self.rand_bbox_shift = rand_bbox_shift

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
        # self.kinematic = dataset.kinematic
        self.two_d = two_d

        self.focal_length = focal_length
        self.root_idx = root_idx

        if train:
            self.num_joints_half_body = dataset.num_joints_half_body
            self.prob_half_body = dataset.prob_half_body

            self.upper_body_ids = dataset.upper_body_ids
            self.lower_body_ids = dataset.lower_body_ids

        self.return_vertices = return_vertices
        self.update_beta = True

        if self.update_beta:
            # smplx_layer_neutral = SMPLXLayer(
            #     model_path='model_files/smplx/SMPLX_NEUTRAL.npz',
            #     num_betas=10,
            #     use_pca=False,
            #     age='kid',
            #     kid_template_path='model_files/smplx_kid_template.npy',
            # )

            # smplx_layer_female = SMPLXLayer(
            #     model_path='model_files/smplx/SMPLX_FEMALE.npz',
            #     num_betas=10,
            #     use_pca=False,
            #     age='kid',
            #     kid_template_path='model_files/smplx_kid_template.npy',
            # )

            # smplx_layer_male = SMPLXLayer(
            #     model_path='model_files/smplx/SMPLX_MALE.npz',
            #     num_betas=10,
            #     use_pca=False,
            #     age='kid',
            #     kid_template_path='model_files/smplx_kid_template.npy',
            # )

            smplx_layers_dict = {
                'male': smplx_layer_male,
                'female': smplx_layer_female,
                'neutral': smplx_layer_neutral
            }

            self.shapedirs_dict = {}
            self.shapedirs_norm_dict = {}
            self.normed_shapedirs_dict = {}
            self.v_template_dict = {}
            self.kid_shapedir_dict = {}

            for gender, layer in smplx_layers_dict.items():
                shape_dir = torch.cat([layer.shapedirs, layer.expr_dirs], dim=-1)  # num_v x 3 x 21
                num_v = shape_dir.shape[0]

                self.shapedirs_dict[gender] = shape_dir
                self.kid_shapedir_dict[gender] = layer.shapedirs[:, :, [-1]]

                shape_dir_nokid = layer.shapedirs[:, :, :10]  # num_v x 3 x 10
                shape_dir_reshaped = shape_dir_nokid.reshape(shape_dir.shape[0] * 3, 10)
                shape_disps_norm = torch.norm(shape_dir_reshaped, dim=0)
                normed_shape_disps = shape_dir_reshaped / shape_disps_norm

                suppose_eye = torch.matmul(normed_shape_disps.T, normed_shape_disps)
                torch_eyes = torch.eye(10)
                diff = torch.abs(suppose_eye - torch_eyes)
                assert (diff < 1e-4).all(), diff

                self.shapedirs_norm_dict[gender] = shape_disps_norm.reshape(10)  # 10
                self.normed_shapedirs_dict[gender] = normed_shape_disps.reshape(num_v, 3, 10)  # num_v x 3 x 10
                self.v_template_dict[gender] = layer.v_template  # v_num x 3

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

        # target_weight[target[:, 0] > 0.5] = 0
        # target_weight[target[:, 0] < -0.5] = 0
        # target_weight[target[:, 1] > 0.5] = 0
        # target_weight[target[:, 1] < -0.5] = 0
        # target_weight[target[:, 2] > 0.5] = 0
        # target_weight[target[:, 2] < -0.5] = 0

        target = target.reshape((-1))
        target_weight = target_weight.reshape((-1))
        return target, target_weight

    def _integral_uvd_target_generator(self, joints_3d, patch_height, patch_width):

        num_joints = len(joints_3d)
        target_weight = np.ones((num_joints, 3), dtype=np.float32)
        target_weight[:, 0] = joints_3d[:, 0, 1]
        target_weight[:, 1] = joints_3d[:, 1, 1]
        target_weight[:, 2] = joints_3d[:, 2, 1]

        target = np.zeros((num_joints, 3), dtype=np.float32)
        target[:, 0] = joints_3d[:, 0, 0] / patch_width - 0.5
        target[:, 1] = joints_3d[:, 1, 0] / patch_height - 0.5
        target[:, 2] = joints_3d[:, 2, 0] / self.bbox_3d_shape[2]

        # target_weight[target[:, 0] > 0.5] = 0
        # target_weight[target[:, 0] < -0.5] = 0
        # target_weight[target[:, 1] > 0.5] = 0
        # target_weight[target[:, 1] < -0.5] = 0
        # target_weight[target[:, 2] > 0.5] = 0
        # target_weight[target[:, 2] < -0.5] = 0

        target = target.reshape((-1))
        target_weight = target_weight.reshape((-1))
        return target, target_weight

    def _integral_xyz_target_generator(self, joints_3d, joints_3d_vis):
        num_joints = len(joints_3d)
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

        bbox = list(label['bbox'])
        joint_img = label['joint_img'].copy()
        joint_vis = label['joint_vis'].copy()
        joint_xyz = label['joint_xyz'].copy()
        joint_xyz_vis = np.ones_like(joint_xyz)

        beta = label['beta'].copy()
        beta_kid = label['beta_kid'].copy() if 'beta_kid' in label else np.zeros(1)
        expression = label['expression'].copy()
        theta_full = label['theta_full'].copy()

        twist_phi = label['twist_phi'].copy()
        twist_weight = label['twist_weight'].copy()

        gt_joints = np.stack([joint_img, joint_vis], axis=-1).copy()

        imgwidth, imght = src.shape[1], src.shape[0]

        input_size = self._input_size

        if self._add_dpg and self._train:
            bbox = addDPG(bbox, imgwidth, imght)

        xmin, ymin, xmax, ymax = bbox
        center, scale = _box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio, scale_mult=self._scale_mult)

        if self.rand_bbox_shift:
            if self._train:
                rand_shift = 0.15 * (scale / 1.3) * np.random.randn(2)

                center = center + rand_shift

        xmin, ymin, xmax, ymax = _center_scale_to_box(center, scale)

        # half body transform
        # half_body_flag = False
        # if False:
        assert joint_vis.shape[0] == 71, joint_vis.shape
        if self._train and ((np.sum(joint_vis[:24, 0]) > self.num_joints_half_body) and (np.random.rand() < self.prob_half_body)):
            self.num_joints = 71
            c_half_body, s_half_body = self.half_body_transform(
                gt_joints[:, :, 0], joint_vis
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
            # rf = 0 # no rotation when 3d data
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0
            # r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
        else:
            r = 0

        if self._train and self._occlusion and False:
            # print(xmax, xmin, ymax, ymin)
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

        joints_uvd = gt_joints

        if random.random() > 0.5 and self._train:
            # if False:
            assert src.shape[2] == 3
            src = src[:, ::-1, :]

            joints_uvd = flip_joints_3d(joints_uvd, imgwidth, self.joint_pairs_71)
            # joint_full = flip_cam_xyz_joints_3d(joint_full, self.joint_pairs_71)
            joint_xyz = flip_cam_xyz_joints_3d(joint_xyz, self.joint_pairs_71)

            theta_full = flip_thetas(theta_full, self.joint_pairs_55)
            twist_phi, twist_weight = flip_twist(twist_phi, twist_weight, self.joint_pairs_55)
            center[0] = imgwidth - center[0] - 1

        # rotate global theta
        theta_full[0, :3] = rot_aa(theta_full[0, :3], r)

        theta_rot_mat = batch_rodrigues_numpy(theta_full)
        theta_quat = rotmat_to_quat_numpy(theta_rot_mat).reshape(-1)
        theta_full = theta_rot_mat.reshape(55 * 9)

        # rotate xyz joints
        # joint_full = rotate_xyz_jts(joint_full, r)
        # joint_full = joint_full - joint_full[:1].copy()
        joint_xyz = rotate_xyz_jts(joint_xyz, r)
        joint_xyz = joint_xyz - joint_xyz[[0], :].copy()

        inp_h, inp_w = input_size
        trans = get_affine_transform(center, scale, r, [inp_w, inp_h])
        trans_inv = get_affine_transform(center, scale, r, [inp_w, inp_h], inv=True).astype(np.float32)
        intrinsic_param = get_intrinsic_metrix(label['f'], label['c'], inv=True).astype(np.float32) if 'f' in label.keys() else np.zeros((3, 3)).astype(np.float32)
        joint_root = label['root_cam'].astype(np.float32) if 'root_cam' in label.keys() else np.zeros((3)).astype(np.float32)
        depth_factor = np.array([self.bbox_3d_shape[2]]).astype(np.float32) if self.bbox_3d_shape else np.zeros((1)).astype(np.float32)

        img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        # affine transform
        for i in range(len(joints_uvd)):
            if joints_uvd[i, 0, 1] > 0.0:
                joints_uvd[i, 0:2, 0] = affine_transform(joints_uvd[i, 0:2, 0], trans)

        target_smpl_weight = torch.ones(1).float()
        theta_weights = np.ones(theta_quat.shape)

        is_kid = label['is_kid']
        gender = label['gender']
        if is_kid and gender == 'female':
            gendered_smplx_layer = smplx_layer_neutral
        elif gender == 'male':
            gendered_smplx_layer = smplx_layer_male
        elif gender == 'female':
            gendered_smplx_layer = smplx_layer_female
        else:
            gendered_smplx_layer = smplx_layer_neutral

        if 'beta_kid' in label:
            beta_full = np.concatenate([beta, beta_kid], axis=0)
        else:
            beta_full = beta

        beta_full_torch = torch.from_numpy(beta_full).float().unsqueeze(0)
        expression_torch = torch.from_numpy(expression).float().unsqueeze(0)
        if self.return_vertices:
            gt_output = gendered_smplx_layer.forward_simple(
                betas=beta_full_torch,
                expression=expression_torch,
                full_pose=torch.from_numpy(theta_full).float().reshape(1, 55, 9),
                return_verts=True,
                root_align=True
            )

            gt_vertices = gt_output.vertices

        if self.update_beta:
            beta_new, _ = self.correct_gendered_beta(beta_full_torch, expression_torch, [gender])
            beta_new = beta_new[0, :10]
        else:
            beta_new = torch.from_numpy(beta).float()

        # generate training targets
        target_uvd, target_weight_uvd = self._integral_uvd_target_generator(joints_uvd, inp_h, inp_w)
        # target_xyz_full, target_weight_full = self._integral_xyz_target_generator(joint_full, joint_full_vis)
        target_xyz, target_weight_xyz = self._integral_xyz_target_generator(
            joint_xyz, joint_xyz_vis)
        # print(joints_hybrik)

        bbox = _center_scale_to_box(center, scale)

        if self.focal_length > 0:
            img_center = np.array([float(imgwidth) * 0.5, float(imght) * 0.5])
            # img_center_bbox_coord =  affine_transform(img_center, trans) # 0-255
            # img_center_bbox_coord = img_center_bbox_coord - 128.0 # -128 - 128

            xyz_uvd_weight = target_weight_uvd.reshape(-1, 3)[:55] * target_weight_xyz.reshape(-1, 3)[:55]
            cam_scale, cam_trans, cam_valid, cam_error, _ = self.calc_cam_scale_trans2(
                target_xyz.reshape(-1, 3)[:55].copy(),
                target_uvd.reshape(-1, 3)[:55].copy(),
                xyz_uvd_weight.copy())
        else:
            assert 1 == 2
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

        # print(target_uvd)

        output = {
            'type': '3d_data_w_smpl',
            'image': img,
            'target_theta_full': torch.from_numpy(theta_full).float(),
            'target_theta_quat': torch.from_numpy(theta_quat).float(),
            'target_theta_weight': torch.from_numpy(theta_weights).float(),
            'target_beta': beta_new,
            'target_expression': torch.from_numpy(expression).float(),
            'target_smpl_weight': target_smpl_weight,
            'target_uvd': torch.from_numpy(target_uvd.reshape(-1)).float(),
            'target_weight_uvd': torch.from_numpy(target_weight_uvd.reshape(-1)).float(),
            'target_xyz': torch.from_numpy(target_xyz).float(),
            'target_weight_xyz': torch.from_numpy(target_weight_xyz).float(),
            'trans_inv': torch.from_numpy(trans_inv).float(),
            'intrinsic_param': torch.from_numpy(intrinsic_param).float(),
            'joint_root': torch.from_numpy(joint_root).float(),
            'depth_factor': torch.from_numpy(depth_factor).float(),
            'bbox': torch.Tensor(bbox),
            'target_twist': torch.from_numpy(twist_phi).float(),
            'target_twist_weight': torch.from_numpy(twist_weight).float(),
            'camera_scale': torch.from_numpy(np.array([cam_scale])).float(),
            'camera_trans': torch.from_numpy(cam_trans).float(),
            'camera_valid': cam_valid,
            'camera_error': cam_error,
            'img_center': torch.from_numpy(img_center).float(),
            'target_beta_kid': torch.from_numpy(beta_kid).float(),
            # 'target_vertices': gt_vertices.reshape(10475, 3)
        }
        if self.return_vertices:
            output['target_vertices'] = gt_vertices.reshape(10475, 3)

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

        # do not take into account 2.2
        # the equation to be solved:
        # u * 256 / f * (z + f/256 * 1/scale) = x + tx
        # v * 256 / f * (z + f/256 * 1/scale) = y + ty

        weight = (uvd_weight.sum(axis=-1, keepdims=True) >= 3.0) * 1.0  # 24 x 1
        # assert weight.sum() >= 2, 'too few valid keypoints to calculate cam para'

        if weight.sum() < 2:
            # print('bad data')
            return 1, np.zeros(2), 0.0, -1, uvd_29

        xyz_29 = xyz_29 * 2.2  # convert to meter
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

        backed_projected_xyz = back_projection(uvd_29, target_camera, f)
        backed_projected_xyz[:, 2] = backed_projected_xyz[:, 2] * 2.2
        diff = np.sum((backed_projected_xyz - xyz_29)**2, axis=-1) * weight[:, 0]
        diff = np.sqrt(diff).sum() / (weight.sum() + 1e-6) * 1000  # roughly mpjpe > 70
        # print(scale, trans, diff)
        if diff < 70:
            new_uvd = self.projection(xyz_29, target_camera, f)
            return scale, trans, 1.0, diff, new_uvd * uvd_weight
        else:
            return scale, trans, 0.0, diff, new_uvd

    def calc_cam_scale_trans_refined(self, xyz_29, uv_29, uvd_weight, img_center):

        # the equation to be solved:
        # u_256 / f * (1-cx/u) * (z + tz) = x + tx
        #   -> (u - cx) * (z * 1/f + tz/f) = x + tx
        #
        # v_256 / f * (1-cy/v) * (z + tz) = y + ty

        # calculate: tz/f, tx, ty
        # return scale, [tx, ty], is_valid, error, None

        weight = (uvd_weight.sum(axis=-1, keepdims=True) >= 3.0) * 1.0  # 24 x 1
        # assert weight.sum() >= 2, 'too few valid keypoints to calculate cam para'

        xyz_29 = xyz_29 * 2.2
        uv_29_fullsize = uv_29[:, :2] * 256.0
        uv_c_diff = uv_29_fullsize - img_center

        if weight.sum() <= 2:
            # print('bad data')
            return 1, np.zeros(2), 0.0, -1, None

        num_joints = len(uv_29)

        Ax = np.zeros((num_joints, 3))
        Ax[:, 0] = uv_c_diff[:, 0]
        Ax[:, 1] = -1

        Ay = np.zeros((num_joints, 3))
        Ay[:, 0] = uv_c_diff[:, 1]
        Ay[:, 2] = -1

        Ax = Ax * weight
        Ay = Ay * weight

        A = np.concatenate([Ax, Ay], axis=0)

        bx = (xyz_29[:, 0] - uv_c_diff[:, 0] * xyz_29[:, 2] / 1000.0) * weight[:, 0]
        by = (xyz_29[:, 1] - uv_c_diff[:, 1] * xyz_29[:, 2] / 1000.0) * weight[:, 0]
        b = np.concatenate([bx, by], axis=0)

        A_s = np.dot(A.T, A)
        b_s = np.dot(A.T, b)

        cam_para = np.linalg.solve(A_s, b_s)

        # f_estimated = 1.0 / cam_para[0]
        f_estimated = 1000.0
        tz = cam_para[0] * f_estimated
        tx, ty = cam_para[1:]

        target_camera = np.zeros(4)
        target_camera[0] = f_estimated
        target_camera[1:] = np.array([tx, ty, tz])

        backed_projected_xyz = back_projection_matrix(uv_29_fullsize, xyz_29, target_camera, img_center)
        diff = np.sum((backed_projected_xyz - xyz_29)**2, axis=-1) * weight[:, 0]
        diff = np.sqrt(diff).sum() / (weight.sum() + 1e-6) * 1000  # roughly mpjpe > 70

        out = np.zeros(3)
        out[1:] = cam_para[1:]
        out[0] = 1000.0 / 256.0 / tz

        if diff < 60:
            return out[0], out[1:], 1.0, diff, None
        else:
            return out[0], out[1:], 0.0, diff, None

    def projection(self, xyz, camera, f):
        # xyz: unit: meter, u = f/256 * (x+dx) / (z+dz)
        transl = camera[1:3]
        scale = camera[0]
        z_cam = xyz[:, 2:] + f / (256.0 * scale)  # J x 1
        uvd = np.zeros_like(xyz)
        uvd[:, 2] = xyz[:, 2] / self.bbox_3d_shape[2]
        uvd[:, :2] = f / 256.0 * (xyz[:, :2] + transl) / z_cam
        return uvd

    def correct_gendered_beta(self, beta_gender, expression_gender, gender_list):
        # batch_size = 1, gender is a one-element list
        # convert gendered beta to neutral beta, keep expression unchanged, the last element of beta is also unchanged
        beta_last = beta_gender[:, [-1]].clone()
        gender = gender_list[0]
        shape_components = torch.cat([beta_gender, expression_gender], dim=-1)
        v_shaped_gender = self.v_template_dict[gender] + torch.einsum('bl,mkl->bmk', [shape_components, self.shapedirs_dict[gender]])

        # v_shaped_gender = self.v_template_dict[gender] + torch.einsum('bl,mkl->bmk', [beta_gender, self.shapedirs_dict[gender][:, :, :11]])

        expression_v_neutral = torch.einsum('bl,mkl->bmk', [expression_gender, self.shapedirs_dict['neutral'][:, :, 11:]])
        kid_v_neutral = torch.einsum('bl,mkl->bmk', [beta_last, self.kid_shapedir_dict['neutral']])

        v_shaped_neutral_residual = v_shaped_gender - self.v_template_dict['neutral'] - expression_v_neutral - kid_v_neutral

        betas_regressed = torch.einsum('bmk,mkl->bl', [v_shaped_neutral_residual, self.normed_shapedirs_dict['neutral']]) / self.shapedirs_norm_dict['neutral']

        betas_regressed_full = torch.cat([betas_regressed, beta_last], dim=-1)

        shape_components_new = torch.cat([betas_regressed_full, expression_gender], dim=-1)
        original_neutral_v = torch.einsum('bl,mkl->bmk', [shape_components, self.shapedirs_dict['neutral']]) + self.v_template_dict['neutral']
        generated_neutral_v = torch.einsum('bl,mkl->bmk', [shape_components_new, self.shapedirs_dict['neutral']]) + self.v_template_dict['neutral']

        diff1 = (original_neutral_v - v_shaped_gender)**2
        diff1 = torch.sqrt(diff1.sum(dim=-1)).mean()

        diff2 = (generated_neutral_v - v_shaped_gender)**2
        diff2 = torch.sqrt(diff2.sum(dim=-1)).mean()

        # print('diff', diff1, diff2)

        return betas_regressed_full, expression_gender


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


def back_projection(uvd, pred_camera, focal_length=5000.):
    camScale = pred_camera[:1].reshape(1, -1)
    camTrans = pred_camera[1:].reshape(1, -1)

    camDepth = focal_length / (256 * camScale)

    pred_xyz = np.zeros_like(uvd)
    pred_xyz[:, 2] = uvd[:, 2].copy()
    pred_xyz[:, :2] = (uvd[:, :2] * 256 / focal_length) * (pred_xyz[:, 2:] * 2.2 + camDepth) - camTrans

    return pred_xyz


def back_projection_matrix(uv, xyz, pred_camera, img_center):
    # pred_camera: f, tx, ty, tz
    f, tx, ty, tz = pred_camera
    cx, cy = img_center
    intrinsic_inv = np.array([
        [1 / f, 0, -cx / f],
        [0, 1 / f, -cy / f],
        [0, 0, 1]
    ])

    uv_homo = np.ones((len(uv), 3))
    uv_homo[:, :2] = uv

    xyz_cam = np.matmul(uv_homo, intrinsic_inv.T)  # 29 x 3
    abs_z = xyz[:, [2]] + tz  # 29 x 1
    xyz_cam = xyz_cam * abs_z

    pred_xyz = xyz_cam - pred_camera[1:]

    return pred_xyz
