import torch
import torch.nn as nn
from easydict import EasyDict as edict
from torch.nn import functional as F

from hybrik.models.layers.smplx.body_models import SMPLXLayer
from hybrik.utils.transforms import flip_coord

from .builder import SPPE
from .layers.hrnet.hrnet import get_hrnet


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
class HRNetSMPLXCamKidReg(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **kwargs):
        super(HRNetSMPLXCamKidReg, self).__init__()
        self._norm_layer = norm_layer
        self.num_joints = kwargs['NUM_JOINTS']
        assert self.num_joints == 71, self.num_joints
        self.norm_type = kwargs['POST']['NORM_TYPE']
        self.depth_dim = kwargs['EXTRA']['DEPTH_DIM']
        self.height_dim = kwargs['HEATMAP_SIZE'][0]
        self.width_dim = kwargs['HEATMAP_SIZE'][1]
        self.smpl_dtype = torch.float32
        self.use_kid = kwargs['EXTRA']['USE_KID']

        self.preact = get_hrnet(kwargs['HRNET_TYPE'], num_joints=70,
                                depth_dim=self.depth_dim,
                                is_train=True, generate_feat=True, generate_hm=False,
                                pretrain=kwargs['HR_PRETRAINED'])
        self.pretrain_hrnet = kwargs['HR_PRETRAINED']

        self.smplx_layer = SMPLXLayer(
            # model_path='model_files/smpl_v1.1.0/smplx/SMPLX_NEUTRAL.npz',
            model_path='model_files/smplx/SMPLX_NEUTRAL.npz',
            num_betas=10,
            use_pca=False,
            age='kid',
            kid_template_path='model_files/smplx_kid_template.npy',
        )

        self.root_idx_smpl = 0
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
        self.joint_pairs = self.body_joint_pairs + self.hand_joint_pairs + self.leaf_joint_pairs

        # init cam
        init_cam = torch.tensor([0.9])
        self.register_buffer(
            'init_cam',
            torch.Tensor(init_cam).float())

        self.decshape = nn.Linear(2048, 21)

        self.decphi = nn.Linear(2048, 54 * 2)  # [cos(phi), sin(phi)]
        self.deccam = nn.Linear(2048, 1)
        self.decsigma = nn.Linear(2048, (self.num_joints - 1))

        self.fc_coord = nn.Linear(2048, (self.num_joints - 1) * 3)
        self.fc_coord_mouthtop = nn.Linear(2048, 1 * 3)
        self.decsigma_mouthtop = nn.Linear(2048, 1)

        self.focal_length = kwargs['FOCAL_LENGTH']
        bbox_3d_shape = kwargs['BBOX_3D_SHAPE'] if 'BBOX_3D_SHAPE' in kwargs else (2200, 2200, 2200)
        self.bbox_3d_shape = torch.tensor(bbox_3d_shape).float()
        self.depth_factor = self.bbox_3d_shape[2] * 1e-3
        self.hand_relative = kwargs['EXTRA'].get('HAND_REL', False)

        self.left_wrist_id = 20
        self.left_fingers_ids = list(range(25, 40))
        self.right_wrist_id = 21
        self.right_fingers_ids = list(range(40, 55))

        self.input_size = 256.0

    def _initialize(self):
        self.preact.init_weights(self.pretrain_hrnet)

    def flip_phi(self, pred_phi):
        pred_phi[:, :, 1] = -1 * pred_phi[:, :, 1]

        for pair in self.joint_pairs:
            dim0, dim1 = pair
            if dim0 >= 55 or dim1 >= 55:
                break
            idx = torch.Tensor((dim0 - 1, dim1 - 1)).long()
            inv_idx = torch.Tensor((dim1 - 1, dim0 - 1)).long()
            pred_phi[:, idx] = pred_phi[:, inv_idx]

        return pred_phi

    def flip_sigma(self, pred_sigma):

        for pair in self.joint_pairs:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0, dim1)).long()
            inv_idx = torch.Tensor((dim1, dim0)).long()
            pred_sigma[:, idx] = pred_sigma[:, inv_idx]

        return pred_sigma

    def update_scale(self, pred_uvd, weight, init_scale, pred_shape_full, pred_phi, **kwargs):
        batch_size = pred_uvd.shape[0]
        cam_depth = self.focal_length / (self.input_size * init_scale + 1e-9)
        pred_phi = pred_phi.reshape(batch_size, -1, 2)

        pred_xyz = torch.zeros_like(pred_uvd)

        weight = weight.clamp_min(0)

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

        pred_xyz = pred_xyz - pred_xyz[:, [0]]

        pred_beta = pred_shape_full[:, :11]
        pred_expression = pred_shape_full[:, 11:]
        output = self.smplx_layer.hybrik(
            betas=pred_beta.type(self.smpl_dtype),
            expression=pred_expression.type(self.smpl_dtype),
            pose_skeleton=pred_xyz.type(self.smpl_dtype) * 2.2,
            phis=pred_phi.type(self.smpl_dtype),
            return_verts=True,
            naive=True
        )

        # unit: m
        pred_xyz55 = output.joints.float()[:, :55]
        pred_xyz55 = pred_xyz55 - pred_xyz55.reshape(batch_size, -1, 3)[:, [0], :]
        pred_xyz55 = pred_xyz55 + camera_root.unsqueeze(dim=1)

        pred_uvd55 = pred_uvd[:, :55, :].clone()
        if 'bboxes' in kwargs.keys():
            pred_uvd55[:, :, :2] = pred_uvd55[:, :, :2] + bbox_center

        bs = pred_uvd.shape[0]
        # [B, K, 1]
        weight_uv55 = weight[:, :55, :].reshape(bs, 55, 1)

        Ax = torch.zeros((bs, 55, 1), device=pred_uvd.device, dtype=pred_uvd.dtype)
        Ay = torch.zeros((bs, 55, 1), device=pred_uvd.device, dtype=pred_uvd.dtype)

        Ax[:, :, 0] = pred_uvd55[:, :, 0]
        Ay[:, :, 0] = pred_uvd55[:, :, 1]

        Ax = Ax * weight_uv55
        Ay = Ay * weight_uv55

        # [B, 2K, 1]
        A = torch.cat((Ax, Ay), dim=1)

        bx = (pred_xyz55[:, :, 0] - self.input_size * pred_uvd55[:, :, 0] / self.focal_length * pred_xyz55[:, :, 2]) * weight_uv55[:, :, 0]
        by = (pred_xyz55[:, :, 1] - self.input_size * pred_uvd55[:, :, 1] / self.focal_length * pred_xyz55[:, :, 2]) * weight_uv55[:, :, 0]

        # [B, 2K, 1]
        b = torch.cat((bx, by), dim=1)[:, :, None]
        res = torch.inverse(A.transpose(1, 2).bmm(A)).bmm(A.transpose(1, 2)).bmm(b)

        scale = 1.0 / res

        assert scale.shape == init_scale.shape

        return scale

    def forward(self, x, flip_test=False, **kwargs):
        batch_size, _, _, width_dim = x.shape

        x0 = self.preact(x)

        x0 = x0.view(x0.size(0), -1)
        init_cam = self.init_cam.expand(batch_size, -1)  # (B, 1,)

        xc = x0

        pred_shape_full = self.decshape(xc)
        pred_beta = pred_shape_full[:, :11]
        pred_expression = pred_shape_full[:, 11:]

        pred_phi = self.decphi(xc)
        pred_camera = self.deccam(xc).reshape(batch_size, -1) + init_cam

        pred_phi = pred_phi.reshape(batch_size, -1, 2)

        out_coord = self.fc_coord(x0).reshape(batch_size, (self.num_joints - 1), 3)
        out_sigma = self.decsigma(x0).sigmoid().reshape(batch_size, (self.num_joints - 1), 1)

        out_coord_mouthtop = self.fc_coord_mouthtop(x0).reshape(batch_size, 1, 3)
        out_sigma_mouthtop = self.decsigma_mouthtop(x0).sigmoid().reshape(batch_size, 1, 1)
        out_coord = torch.cat((out_coord, out_coord_mouthtop), dim=1)
        out_sigma = torch.cat((out_sigma, out_sigma_mouthtop), dim=1)

        if flip_test:
            flip_x = flip(x)
            flip_x0 = self.preact(flip_x)

            flip_out_coord = self.fc_coord(flip_x0).reshape(batch_size, (self.num_joints - 1), 3)
            flip_out_sigma = self.decsigma(flip_x0).sigmoid().reshape(batch_size, (self.num_joints - 1), 1)

            flip_out_coord_mouthtop = self.fc_coord_mouthtop(flip_x0).reshape(batch_size, 1, 3)
            flip_out_sigma_mouthtop = self.decsigma_mouthtop(flip_x0).sigmoid().reshape(batch_size, 1, 1)
            flip_out_coord = torch.cat((flip_out_coord, flip_out_coord_mouthtop), dim=1)
            flip_out_sigma = torch.cat((flip_out_sigma, flip_out_sigma_mouthtop), dim=1)

            flip_out_coord, flip_out_sigma = flip_coord((flip_out_coord, flip_out_sigma), self.joint_pairs, width_dim, shift=True, flatten=False)
            flip_out_coord = flip_out_coord.reshape(batch_size, self.num_joints, 3)
            flip_out_sigma = flip_out_sigma.reshape(batch_size, self.num_joints, 1)

            out_coord = (out_coord + flip_out_coord) / 2
            out_sigma = (out_sigma + flip_out_sigma) / 2

            flip_pred_shape = self.decshape(flip_x0)
            pred_shape_full = (flip_pred_shape + pred_shape_full) / 2

            pred_beta = pred_shape_full[:, :11]
            pred_expression = pred_shape_full[:, 11:]

            flip_pred_phi = self.decphi(flip_x0)
            flip_pred_phi = flip_pred_phi.reshape(batch_size, -1, 2)
            flip_pred_phi = self.flip_phi(flip_pred_phi)
            pred_phi = (pred_phi + flip_pred_phi) / 2

            flip_pred_camera = self.deccam(flip_x0).reshape(batch_size, -1) + init_cam
            pred_camera = 2 / (1 / flip_pred_camera + 1 / pred_camera)

        maxvals = 1 - out_sigma

        pred_uvd_jts = out_coord.reshape(batch_size, self.num_joints, 3)

        if self.hand_relative:
            left_hand_uvd = out_coord[:, self.left_fingers_ids, :]
            right_hand_uvd = out_coord[:, self.right_fingers_ids, :]
            out_coord[:, self.left_fingers_ids, :] = left_hand_uvd + out_coord[:, [self.left_wrist_id], :]
            out_coord[:, self.right_fingers_ids, :] = right_hand_uvd + out_coord[:, [self.right_wrist_id], :]
        else:
            left_hand_uvd = out_coord[:, self.left_fingers_ids, :] - out_coord[:, [self.left_wrist_id], :]
            right_hand_uvd = out_coord[:, self.right_fingers_ids, :] - out_coord[:, [self.right_wrist_id], :]

        camScale = pred_camera[:, :1].unsqueeze(1)

        ice_step = 3
        for _ in range(ice_step):
            try:
                camScale = self.update_scale(
                    pred_uvd=pred_uvd_jts,
                    weight=1 - out_sigma * 10,
                    init_scale=camScale,
                    pred_shape_full=pred_shape_full,
                    pred_phi=pred_phi,
                    **kwargs)
            except RuntimeError:
                pass

        camDepth = self.focal_length / (self.input_size * camScale + 1e-9)

        assert torch.sum(torch.isnan(pred_uvd_jts)) == 0, ('pred_uvd_jts', pred_uvd_jts)
        assert torch.sum(torch.isnan(pred_camera)) == 0, ('pred_camera', pred_camera)
        pred_xyz_jts = torch.zeros_like(pred_uvd_jts)
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

            pred_xyz_jts[:, :, 2:] = pred_uvd_jts[:, :, 2:].clone()  # unit: (self.depth_factor m)
            pred_xy_jts_29_meter = ((pred_uvd_jts[:, :, :2] + bbox_center) * self.input_size / self.focal_length) \
                * (pred_xyz_jts[:, :, 2:] * self.depth_factor + camDepth)  # unit: m

            pred_xyz_jts[:, :, :2] = pred_xy_jts_29_meter / self.depth_factor  # unit: (self.depth_factor m)

            camera_root = pred_xyz_jts[:, 0, :] * self.depth_factor
            camera_root[:, 2] += camDepth[:, 0, 0]
        else:
            pred_xyz_jts[:, :, 2:] = pred_uvd_jts[:, :, 2:].clone()  # unit: (self.depth_factor m)
            pred_xy_jts_29_meter = (pred_uvd_jts[:, :, :2] * self.input_size / self.focal_length) \
                * (pred_xyz_jts[:, :, 2:] * self.depth_factor + camDepth)  # unit: m

            pred_xyz_jts[:, :, :2] = pred_xy_jts_29_meter / self.depth_factor  # unit: (self.depth_factor m)

            camera_root = pred_xyz_jts[:, 0, :] * self.depth_factor
            camera_root[:, 2] += camDepth[:, 0, 0]

        # assert torch.sum(torch.isnan(pred_xyz_jts)) == 0, ('pred_xyz_jts', pred_xyz_jts, pred_camera)
        # old_pred_xyz_jts = pred_xyz_jts.clone()
        pred_xyz_jts = pred_xyz_jts - pred_xyz_jts[:, [0]]
        # assert torch.sum(torch.isnan(pred_xyz_jts)) == 0, ('pred_xyz_jts_aligned',
        #                                                    pred_xyz_jts, old_pred_xyz_jts, pred_uvd_jts, pred_camera, bboxes, img_center, cx, cy)

        pred_phi = pred_phi.reshape(batch_size, -1, 2)

        output = self.smplx_layer.hybrik(
            betas=pred_beta.type(self.smpl_dtype),
            expression=pred_expression.type(self.smpl_dtype),
            pose_skeleton=pred_xyz_jts.type(self.smpl_dtype) * 2.2,
            phis=pred_phi.type(self.smpl_dtype),
            return_verts=True
        )
        pred_vertices = output.vertices.float()
        #  -0.5 ~ 0.5
        pred_xyz_struct_full = output.joints.float() / 2.2
        #  -0.5 ~ 0.5
        pred_xyz_hybrik_struct = self.smplx_layer.get_extended_joints(output.joints[:, :55].clone(), pred_vertices)
        pred_xyz_hybrik_struct = pred_xyz_hybrik_struct / 2.2

        pred_theta_quat = output.theta_quat.float().reshape(batch_size, -1)
        pred_theta_mat = output.rot_mats.float().reshape(batch_size, -1)

        transl = camera_root - output.joints.float().reshape(batch_size, -1, 3)[:, 0, :]

        gt_output = 0
        if 'gt_labels' in kwargs.keys() and kwargs['gt_labels'] is not None:
            gt_labels = kwargs['gt_labels']
            gt_beta = torch.cat([gt_labels['target_beta'], gt_labels['target_beta_kid']], dim=1)
            with torch.no_grad():
                gt_output = self.smplx_layer.forward_simple(
                    betas=gt_beta,
                    expression=gt_labels['target_expression'],
                    full_pose=gt_labels['target_theta_full'].reshape(-1, 55, 9),
                    return_verts=True,
                    root_align=True
                )

        # project
        pred_xyz_full = (pred_xyz_struct_full * 2.2).reshape(-1, 127, 3)
        pred_uv_full = torch.zeros_like(pred_xyz_full[:, :, :2])
        pred_xyz_full = pred_xyz_full + transl[:, None, :]
        pred_uv_full = (pred_xyz_full[:, :, :2] / pred_xyz_full[:, :, 2:]) * self.focal_length / self.input_size

        if 'bboxes' in kwargs.keys():
            pred_uv_full = pred_uv_full - bbox_center

        output = edict(
            pred_phi=pred_phi,
            pred_shape_full=pred_shape_full,
            pred_beta=pred_beta,
            pred_expression=pred_expression,
            pred_theta_quat=pred_theta_quat,
            pred_theta_mat=pred_theta_mat,
            pred_lh_uvd=left_hand_uvd.reshape(batch_size, 15, 3),
            pred_rh_uvd=right_hand_uvd.reshape(batch_size, 15, 3),
            pred_uvd_jts=pred_uvd_jts.reshape(batch_size, -1),
            pred_xyz_hybrik=pred_xyz_jts.reshape(batch_size, -1),
            pred_xyz_hybrik_struct=pred_xyz_hybrik_struct.reshape(batch_size, -1),
            pred_xyz_full=pred_xyz_struct_full.reshape(batch_size, -1),
            pred_uv_full=pred_uv_full,
            pred_vertices=pred_vertices,
            pred_sigma=out_sigma,
            scores=1 - out_sigma,
            maxvals=maxvals,
            cam_scale=camScale[:, 0],
            cam_root=camera_root,
            transl=transl,
            img_feat=x0,
            pred_camera=pred_camera,
            gt_output=gt_output
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
