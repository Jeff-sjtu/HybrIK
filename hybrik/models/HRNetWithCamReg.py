from easydict import EasyDict as edict

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from .builder import SPPE
from .layers.smpl.SMPL import SMPL_layer
from .layers.hrnet.hrnet import get_hrnet

from hybrik.utils.transforms import flip_coord


def flip(x):
    assert (x.dim() == 3 or x.dim() == 4)
    dim = x.dim() - 1

    return x.flip(dims=(dim,))


def norm_heatmap(norm_type, heatmap, tau=5, sample_num=1):
    # Input tensor shape: [N,C,...]
    shape = heatmap.shape
    if norm_type == 'softmax':
        heatmap = heatmap.reshape(*shape[:2], -1)
        # global soft max
        heatmap = F.softmax(heatmap, 2)
        return heatmap.reshape(*shape)
    elif norm_type == 'sampling':
        heatmap = heatmap.reshape(*shape[:2], -1)

        eps = torch.rand_like(heatmap)
        log_eps = torch.log(-torch.log(eps))
        gumbel_heatmap = heatmap - log_eps / tau

        gumbel_heatmap = F.softmax(gumbel_heatmap, 2)
        return gumbel_heatmap.reshape(*shape)
    elif norm_type == 'multiple_sampling':

        heatmap = heatmap.reshape(*shape[:2], 1, -1)

        eps = torch.rand(*heatmap.shape[:2], sample_num, heatmap.shape[3], device=heatmap.device)
        log_eps = torch.log(-torch.log(eps))
        gumbel_heatmap = heatmap - log_eps / tau
        gumbel_heatmap = F.softmax(gumbel_heatmap, 3)
        gumbel_heatmap = gumbel_heatmap.reshape(shape[0], shape[1], sample_num, shape[2])

        # [B, S, K, -1]
        return gumbel_heatmap.transpose(1, 2)
    else:
        raise NotImplementedError


@SPPE.register_module
class HRNetSMPLCamReg(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **kwargs):
        super(HRNetSMPLCamReg, self).__init__()
        self._norm_layer = norm_layer
        self.num_joints = kwargs['NUM_JOINTS']
        self.norm_type = kwargs['POST']['NORM_TYPE']
        self.depth_dim = kwargs['EXTRA']['DEPTH_DIM']
        self.height_dim = kwargs['HEATMAP_SIZE'][0]
        self.width_dim = kwargs['HEATMAP_SIZE'][1]
        self.smpl_dtype = torch.float32

        self.preact = get_hrnet(kwargs['HRNET_TYPE'], num_joints=self.num_joints,
                                depth_dim=self.depth_dim,
                                is_train=True, generate_feat=True, generate_hm=False,
                                pretrain=kwargs['HR_PRETRAINED'])
        self.pretrain_hrnet = kwargs['HR_PRETRAINED']

        h36m_jregressor = np.load('./model_files/J_regressor_h36m.npy')
        self.smpl = SMPL_layer(
            './model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
            h36m_jregressor=h36m_jregressor,
            dtype=self.smpl_dtype
        )

        self.joint_pairs_24 = ((1, 2), (4, 5), (7, 8),
                               (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))

        self.joint_pairs_29 = ((1, 2), (4, 5), (7, 8),
                               (10, 11), (13, 14), (16, 17), (18, 19), (20, 21),
                               (22, 23), (25, 26), (27, 28))

        self.root_idx_smpl = 0

        # mean shape
        init_shape = np.load('./model_files/h36m_mean_beta.npy')
        self.register_buffer(
            'init_shape',
            torch.Tensor(init_shape).float())

        init_cam = torch.tensor([0.9])
        self.register_buffer(
            'init_cam',
            torch.Tensor(init_cam).float())

        self.decshape = nn.Linear(2048, 10)
        self.decphi = nn.Linear(2048, 23 * 2)  # [cos(phi), sin(phi)]
        self.deccam = nn.Linear(2048, 1)
        self.decsigma = nn.Linear(2048, 29)

        self.fc_coord = nn.Linear(2048, 29 * 3)

        self.focal_length = kwargs['FOCAL_LENGTH']
        bbox_3d_shape = kwargs['BBOX_3D_SHAPE'] if 'BBOX_3D_SHAPE' in kwargs else (2000, 2000, 2000)
        self.bbox_3d_shape = torch.tensor(bbox_3d_shape).float()
        self.depth_factor = self.bbox_3d_shape[2] * 1e-3
        self.input_size = 256.0

    def _initialize(self):
        self.preact.init_weights(self.pretrain_hrnet)

    def flip_phi(self, pred_phi):
        pred_phi[:, :, 1] = -1 * pred_phi[:, :, 1]

        for pair in self.joint_pairs_24:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0 - 1, dim1 - 1)).long()
            inv_idx = torch.Tensor((dim1 - 1, dim0 - 1)).long()
            pred_phi[:, idx] = pred_phi[:, inv_idx]

        return pred_phi

    def flip_sigma(self, pred_sigma):

        for pair in self.joint_pairs_29:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0, dim1)).long()
            inv_idx = torch.Tensor((dim1, dim0)).long()
            pred_sigma[:, idx] = pred_sigma[:, inv_idx]

        return pred_sigma

    def update_scale(self, pred_uvd, weight, init_scale, pred_shape, pred_phi, **kwargs):
        cam_depth = self.focal_length / (self.input_size * init_scale + 1e-9)
        pred_phi = pred_phi.reshape(-1, 23, 2)

        pred_xyz = torch.zeros_like(pred_uvd)

        if 'bboxes' in kwargs.keys():
            bboxes = kwargs['bboxes']
            img_center = kwargs['img_center']

            cx = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
            cy = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
            w = (bboxes[:, 2] - bboxes[:, 0])
            h = (bboxes[:, 3] - bboxes[:, 1])

            cx = cx - img_center[:, 0]
            cy = cy - img_center[:, 1]
            cx = cx / w
            cy = cy / h

            bbox_center = torch.stack((cx, cy), dim=1).unsqueeze(dim=1)

            pred_xyz[:, :, 2:] = pred_uvd[:, :, 2:].clone()  # unit: (self.depth_factor m)
            pred_xy = ((pred_uvd[:, :, :2] + bbox_center) * self.input_size / self.focal_length) \
                * (pred_xyz[:, :, 2:] * self.depth_factor + cam_depth)  # unit: m

            pred_xyz[:, :, :2] = pred_xy / self.depth_factor  # unit: (self.depth_factor m)

            camera_root = pred_xyz[:, 0, :] * self.depth_factor
            # camera_root[:, 2] += camDepth[:, 0, 0]
        else:
            # copy z
            pred_xyz[:, :, 2:] = pred_uvd[:, :, 2:].clone()  # unit: (self.depth_factor m)
            # back-project xy
            pred_xy = (pred_uvd[:, :, :2] * self.input_size / self.focal_length) \
                * (pred_xyz[:, :, 2:] * self.depth_factor + cam_depth)  # unit: m

            # unit: (self.depth_factor m)
            pred_xyz[:, :, :2] = pred_xy / self.depth_factor

            # unit: m
            camera_root = pred_xyz[:, 0, :] * self.depth_factor
            # camera_root[:, 2] += cam_depth[:, 0, 0]

        pred_xyz = pred_xyz - pred_xyz[:, [0]]

        output = self.smpl.hybrik(
            pose_skeleton=pred_xyz.type(self.smpl_dtype) * self.depth_factor,  # unit: meter
            betas=pred_shape.type(self.smpl_dtype),
            phis=pred_phi.type(self.smpl_dtype),
            global_orient=None,
            return_verts=True
        )

        # unit: m
        pred_xyz24 = output.joints.float()
        pred_xyz24 = pred_xyz24 - pred_xyz24.reshape(-1, 24, 3)[:, [0], :]
        pred_xyz24 = pred_xyz24 + camera_root.unsqueeze(dim=1)

        pred_uvd24 = pred_uvd[:, :24, :].clone()
        if 'bboxes' in kwargs.keys():
            pred_uvd24[:, :, :2] = pred_uvd24[:, :, :2] + bbox_center

        bs = pred_uvd.shape[0]
        # [B, K, 1]
        weight_uv24 = weight[:, :24, :].reshape(bs, 24, 1)

        Ax = torch.zeros((bs, 24, 1), device=pred_uvd.device, dtype=pred_uvd.dtype)
        Ay = torch.zeros((bs, 24, 1), device=pred_uvd.device, dtype=pred_uvd.dtype)

        Ax[:, :, 0] = pred_uvd24[:, :, 0]
        Ay[:, :, 0] = pred_uvd24[:, :, 1]

        Ax = Ax * weight_uv24
        Ay = Ay * weight_uv24

        # [B, 2K, 1]
        A = torch.cat((Ax, Ay), dim=1)

        bx = (pred_xyz24[:, :, 0] - self.input_size * pred_uvd24[:, :, 0] / self.focal_length * pred_xyz24[:, :, 2]) * weight_uv24[:, :, 0]
        by = (pred_xyz24[:, :, 1] - self.input_size * pred_uvd24[:, :, 1] / self.focal_length * pred_xyz24[:, :, 2]) * weight_uv24[:, :, 0]

        # [B, 2K, 1]
        b = torch.cat((bx, by), dim=1)[:, :, None]
        res = torch.inverse(A.transpose(1, 2).bmm(A)).bmm(A.transpose(1, 2)).bmm(b)

        scale = 1.0 / res

        assert scale.shape == init_scale.shape

        return scale

    def forward(self, x, flip_test=False, **kwargs):
        batch_size, _, _, width_dim = x.shape

        # x0 = self.preact(x)
        x0 = self.preact(x)

        x0 = x0.view(x0.size(0), -1)
        init_shape = self.init_shape.expand(batch_size, -1)     # (B, 10,)
        init_cam = self.init_cam.expand(batch_size, -1)  # (B, 1,)

        delta_shape = self.decshape(x0)
        pred_shape = delta_shape + init_shape
        pred_phi = self.decphi(x0)
        pred_camera = self.deccam(x0).reshape(batch_size, -1) + init_cam

        pred_phi = pred_phi.reshape(batch_size, 23, 2)

        out_coord = self.fc_coord(x0).reshape(batch_size, self.num_joints, 3)
        out_sigma = self.decsigma(x0).sigmoid().reshape(batch_size, self.num_joints, 1)

        if flip_test:
            flip_x = flip(x)
            flip_x0 = self.preact(flip_x)

            flip_out_coord = self.fc_coord(flip_x0).reshape(batch_size, self.num_joints, 3)
            flip_out_sigma = self.decsigma(flip_x0).sigmoid().reshape(batch_size, self.num_joints, 1)

            flip_out_coord, flip_out_sigma = flip_coord((flip_out_coord, flip_out_sigma), self.joint_pairs_29, width_dim, shift=True, flatten=False)
            flip_out_coord = flip_out_coord.reshape(batch_size, self.num_joints, 3)
            flip_out_sigma = flip_out_sigma.reshape(batch_size, self.num_joints, 1)

            out_coord = (out_coord + flip_out_coord) / 2
            out_sigma = (out_sigma + flip_out_sigma) / 2

            flip_delta_shape = self.decshape(flip_x0)
            flip_pred_shape = flip_delta_shape + init_shape
            flip_pred_phi = self.decphi(flip_x0)
            flip_pred_camera = self.deccam(flip_x0).reshape(batch_size, -1) + init_cam

            pred_shape = (pred_shape + flip_pred_shape) / 2

            flip_pred_phi = flip_pred_phi.reshape(batch_size, 23, 2)
            flip_pred_phi = self.flip_phi(flip_pred_phi)
            pred_phi = (pred_phi + flip_pred_phi) / 2

            pred_camera = 2 / (1 / flip_pred_camera + 1 / pred_camera)

        maxvals = 1 - out_sigma

        camScale = pred_camera[:, :1].unsqueeze(1)
        # camTrans = pred_camera[:, 1:].unsqueeze(1)

        # print(out.sum(dim=2, keepdim=True))
        # heatmaps = out / out.sum(dim=2, keepdim=True)

        # uvd
        #  -0.5 ~ 0.5
        pred_uvd_jts_29 = out_coord.reshape(batch_size, self.num_joints, 3)

        if not self.training:
            camScale = self.update_scale(
                pred_uvd=pred_uvd_jts_29,
                weight=1 - out_sigma * 5,
                init_scale=camScale,
                pred_shape=pred_shape,
                pred_phi=pred_phi,
                **kwargs)

        camDepth = self.focal_length / (self.input_size * camScale + 1e-9)

        pred_xyz_jts_29 = torch.zeros_like(pred_uvd_jts_29)
        if 'bboxes' in kwargs.keys():
            bboxes = kwargs['bboxes']
            img_center = kwargs['img_center']

            cx = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
            cy = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
            w = (bboxes[:, 2] - bboxes[:, 0])
            h = (bboxes[:, 3] - bboxes[:, 1])

            cx = cx - img_center[:, 0]
            cy = cy - img_center[:, 1]
            cx = cx / w
            cy = cy / h

            bbox_center = torch.stack((cx, cy), dim=1).unsqueeze(dim=1)

            pred_xyz_jts_29[:, :, 2:] = pred_uvd_jts_29[:, :, 2:].clone()  # unit: (self.depth_factor m)
            pred_xy_jts_29_meter = ((pred_uvd_jts_29[:, :, :2] + bbox_center) * self.input_size / self.focal_length) \
                * (pred_xyz_jts_29[:, :, 2:] * self.depth_factor + camDepth)  # unit: m

            pred_xyz_jts_29[:, :, :2] = pred_xy_jts_29_meter / self.depth_factor  # unit: (self.depth_factor m)

            camera_root = pred_xyz_jts_29[:, 0, :] * self.depth_factor
            camera_root[:, 2] += camDepth[:, 0, 0]
        else:
            pred_xyz_jts_29[:, :, 2:] = pred_uvd_jts_29[:, :, 2:].clone()  # unit: (self.depth_factor m)
            pred_xy_jts_29_meter = (pred_uvd_jts_29[:, :, :2] * self.input_size / self.focal_length) \
                * (pred_xyz_jts_29[:, :, 2:] * self.depth_factor + camDepth)  # unit: m

            pred_xyz_jts_29[:, :, :2] = pred_xy_jts_29_meter / self.depth_factor  # unit: (self.depth_factor m)

            camera_root = pred_xyz_jts_29[:, 0, :] * self.depth_factor
            camera_root[:, 2] += camDepth[:, 0, 0]
        # camTrans = camera_root.squeeze(dim=1)[:, :2]

        # if not self.training:
        pred_xyz_jts_29 = pred_xyz_jts_29 - pred_xyz_jts_29[:, [0]]

        pred_xyz_jts_29_flat = pred_xyz_jts_29.reshape(batch_size, -1)

        pred_phi = pred_phi.reshape(batch_size, 23, 2)

        output = self.smpl.hybrik(
            pose_skeleton=pred_xyz_jts_29.type(self.smpl_dtype) * self.depth_factor,  # unit: meter
            betas=pred_shape.type(self.smpl_dtype),
            phis=pred_phi.type(self.smpl_dtype),
            global_orient=None,
            return_verts=True
        )
        pred_vertices = output.vertices.float()
        #  -0.5 ~ 0.5
        pred_xyz_jts_24_struct = output.joints.float() / self.depth_factor
        #  -0.5 ~ 0.5
        pred_xyz_jts_17 = output.joints_from_verts.float() / self.depth_factor
        pred_theta_mats = output.rot_mats.float().reshape(batch_size, 24 * 9)
        pred_xyz_jts_24 = pred_xyz_jts_29[:, :24, :].reshape(batch_size, 72)
        pred_xyz_jts_24_struct = pred_xyz_jts_24_struct.reshape(batch_size, 72)
        pred_xyz_jts_17_flat = pred_xyz_jts_17.reshape(batch_size, 17 * 3)

        transl = camera_root - output.joints.float().reshape(-1, 24, 3)[:, 0, :]

        output = edict(
            pred_phi=pred_phi,
            pred_delta_shape=delta_shape,
            pred_shape=pred_shape,
            pred_theta_mats=pred_theta_mats,
            pred_uvd_jts=pred_uvd_jts_29.reshape(batch_size, -1),
            pred_xyz_jts_29=pred_xyz_jts_29_flat,
            pred_xyz_jts_24=pred_xyz_jts_24,
            pred_xyz_jts_24_struct=pred_xyz_jts_24_struct,
            pred_xyz_jts_17=pred_xyz_jts_17_flat,
            pred_vertices=pred_vertices,
            maxvals=maxvals,
            cam_scale=camScale[:, 0],
            # cam_trans=camTrans[:, 0],
            cam_root=camera_root,
            transl=transl,
            pred_camera=pred_camera,
            pred_sigma=out_sigma,
            scores=1 - out_sigma,
            # uvd_heatmap=torch.stack([hm_x0, hm_y0, hm_z0], dim=2),
            # uvd_heatmap=heatmaps,
            img_feat=x0
        )
        return output

    def forward_gt_theta(self, gt_theta, gt_beta):

        output = self.smpl(
            pose_axis_angle=gt_theta,
            betas=gt_beta,
            global_orient=None,
            return_verts=True
        )

        return output
