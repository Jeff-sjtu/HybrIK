import torch
import torch.nn as nn
from easydict import EasyDict as edict
from torch.nn import functional as F

from hybrik.models.layers.smplx.body_models import SMPLXLayer

from .builder import SPPE
from .layers.hrnet.hrnet_25d import get_hrnet25d


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
class HRNetSMPLXCamKid(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **kwargs):
        super(HRNetSMPLXCamKid, self).__init__()
        self._norm_layer = norm_layer
        self.num_joints = kwargs['NUM_JOINTS']
        assert self.num_joints == 71
        self.norm_type = kwargs['POST']['NORM_TYPE']
        self.depth_dim = kwargs['EXTRA']['DEPTH_DIM']
        self.height_dim = kwargs['HEATMAP_SIZE'][0]
        self.width_dim = kwargs['HEATMAP_SIZE'][1]
        self.smpl_dtype = torch.float32
        self.use_kid = kwargs['EXTRA']['USE_KID']

        self.preact = get_hrnet25d(
            kwargs['HRNET_TYPE'], num_joints=71,
            depth_dim=self.depth_dim,
            is_train=True, generate_feat=True, generate_hm=True,
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
        self.decsigma = nn.Linear(2048, 71)

        self.focal_length = kwargs['FOCAL_LENGTH']
        bbox_3d_shape = kwargs['BBOX_3D_SHAPE'] if 'BBOX_3D_SHAPE' in kwargs else (2200, 2200, 2200)
        self.bbox_3d_shape = torch.tensor(bbox_3d_shape).float()
        self.depth_factor = self.bbox_3d_shape[2] * 1e-3
        self.input_size = 256.0

    def _initialize(self):
        self.preact.init_weights(self.pretrain_hrnet)

    def flip_xyz_coord(self, pred_jts, flatten=True):
        if flatten:
            assert pred_jts.dim() == 2
            num_batches = pred_jts.shape[0]
            pred_jts = pred_jts.reshape(num_batches, self.num_joints, 3)
        else:
            assert pred_jts.dim() == 3
            num_batches = pred_jts.shape[0]

        pred_jts[:, :, 0] = - pred_jts[:, :, 0]

        for pair in self.joint_pairs:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0, dim1)).long()
            inv_idx = torch.Tensor((dim1, dim0)).long()
            pred_jts[:, idx] = pred_jts[:, inv_idx]

        if flatten:
            pred_jts = pred_jts.reshape(num_batches, self.num_joints * 3)

        return pred_jts

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

    def flip_heatmap(self, heatmaps, shift=True, flip_last_dim=True):
        if flip_last_dim:
            heatmaps = heatmaps.flip(dims=(heatmaps.dim() - 1,))

        for pair in self.joint_pairs:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0, dim1)).long()
            inv_idx = torch.Tensor((dim1, dim0)).long()
            heatmaps[:, idx] = heatmaps[:, inv_idx]

        if shift:
            if heatmaps.dim() == 3:
                heatmaps[:, :, 1:] = heatmaps[:, :, 0:-1]
            elif heatmaps.dim() == 4:
                heatmaps[:, :, :, 1:] = heatmaps[:, :, :, 0:-1]
            else:
                heatmaps[:, :, :, :, 1:] = heatmaps[:, :, :, :, 0:-1]

        return heatmaps

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

        num_joints = 22
        # unit: m
        pred_xyz55 = output.joints.float()[:, :num_joints]
        pred_xyz55 = pred_xyz55 - pred_xyz55.reshape(batch_size, -1, 3)[:, [0], :]
        pred_xyz55 = pred_xyz55 + camera_root.unsqueeze(dim=1)

        pred_uvd55 = pred_uvd[:, :num_joints, :].clone()
        if 'bboxes' in kwargs.keys():
            pred_uvd55[:, :, :2] = pred_uvd55[:, :, :2] + bbox_center

        bs = pred_uvd.shape[0]
        # [B, K, 1]
        weight_uv55 = weight[:, :num_joints, :].reshape(bs, num_joints, 1)

        Ax = torch.zeros((bs, num_joints, 1), device=pred_uvd.device, dtype=pred_uvd.dtype)
        Ay = torch.zeros((bs, num_joints, 1), device=pred_uvd.device, dtype=pred_uvd.dtype)

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
        batch_size = x.shape[0]

        out_uv, out_z, x0 = self.preact(x)

        out_uv = out_uv.reshape(batch_size, self.num_joints, self.height_dim * self.width_dim)
        out_z = out_z.reshape(batch_size, self.num_joints, self.depth_dim)

        heatmaps_uv = norm_heatmap(self.norm_type, out_uv)
        heatmaps_z = norm_heatmap(self.norm_type, out_z)

        if flip_test:
            flip_x = flip(x)
            flip_out_uv, flip_out_z, flip_x0 = self.preact(flip_x)

            # flip heatmap
            flip_out_uv = flip_out_uv.reshape(batch_size, self.num_joints, self.height_dim, self.width_dim)
            flip_out_z = flip_out_z.reshape(batch_size, self.num_joints, self.depth_dim)
            flip_out_uv = self.flip_heatmap(flip_out_uv)
            flip_out_z = self.flip_heatmap(flip_out_z, shift=False, flip_last_dim=False)

            flip_out_uv = flip_out_uv.reshape(batch_size, self.num_joints, self.height_dim * self.width_dim)
            flip_out_z = flip_out_z.reshape(batch_size, self.num_joints, self.depth_dim)

            flip_heatmaps_uv = norm_heatmap(self.norm_type, flip_out_uv)
            flip_heatmaps_z = norm_heatmap(self.norm_type, flip_out_z)

            heatmaps_uv = (heatmaps_uv + flip_heatmaps_uv) / 2
            heatmaps_z = (heatmaps_z + flip_heatmaps_z) / 2
            # heatmaps_uv = heatmaps_uv
            # heatmaps_z = heatmaps_z

        maxvals_uv, _ = torch.max(heatmaps_uv, dim=2, keepdim=True)
        maxvals_z, _ = torch.max(heatmaps_z, dim=2, keepdim=True)
        maxvals = maxvals_uv * maxvals_z

        heatmaps_uv = heatmaps_uv.reshape((batch_size, self.num_joints, self.height_dim, self.width_dim))

        hm_x0 = heatmaps_uv.sum(axis=2)
        hm_y0 = heatmaps_uv.sum(axis=3)
        hm_z0 = heatmaps_z

        range_tensor = torch.arange(hm_x0.shape[-1], dtype=torch.float32, device=hm_x0.device)
        hm_x = hm_x0 * range_tensor
        hm_y = hm_y0 * range_tensor
        hm_z = hm_z0 * range_tensor

        coord_x = hm_x.sum(dim=2, keepdim=True)
        coord_y = hm_y.sum(dim=2, keepdim=True)
        coord_z = hm_z.sum(dim=2, keepdim=True)

        coord_x = coord_x / float(self.width_dim) - 0.5
        coord_y = coord_y / float(self.height_dim) - 0.5
        coord_z = coord_z / float(self.depth_dim) - 0.5

        #  -0.5 ~ 0.5
        pred_uvd_jts = torch.cat((coord_x, coord_y, coord_z), dim=2)

        x0 = x0.view(x0.size(0), -1)
        init_cam = self.init_cam.expand(batch_size, -1)  # (B, 1,)

        xc = x0

        pred_shape_full = self.decshape(xc)
        pred_beta = pred_shape_full[:, :11]
        pred_expression = pred_shape_full[:, 11:]

        pred_phi = self.decphi(xc)
        pred_camera = self.deccam(xc).reshape(batch_size, -1) + init_cam

        sigma = self.decsigma(xc).reshape(batch_size, self.num_joints, 1).sigmoid()

        pred_phi = pred_phi.reshape(batch_size, -1, 2)

        if flip_test:
            flip_pred_shape = self.decshape(flip_x0)
            pred_shape_full = (flip_pred_shape + pred_shape_full) / 2

            pred_beta = pred_shape_full[:, :11]
            pred_expression = pred_shape_full[:, 11:]

            flip_pred_phi = self.decphi(flip_x0)
            flip_pred_phi = flip_pred_phi.reshape(batch_size, -1, 2)
            flip_pred_phi = self.flip_phi(flip_pred_phi)
            pred_phi = (pred_phi + flip_pred_phi) / 2

            flip_pred_camera = self.deccam(flip_x0).reshape(batch_size, -1) + init_cam
            flip_sigma = self.decsigma(flip_x0).reshape(batch_size, self.num_joints, 1).sigmoid()
            pred_camera = 2 / (1 / flip_pred_camera + 1 / pred_camera)

            flip_sigma = self.flip_sigma(flip_sigma)
            sigma = (sigma + flip_sigma) / 2

        camScale = pred_camera[:, :1].unsqueeze(1)

        # if not self.training:
        ice_step = 3
        for _ in range(ice_step):
            camScale = self.update_scale(
                pred_uvd=pred_uvd_jts,
                weight=1 - sigma * 10,
                init_scale=camScale,
                pred_shape_full=pred_shape_full,
                pred_phi=pred_phi,
                **kwargs)

        camDepth = self.focal_length / (self.input_size * camScale + 1e-9)

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

        pred_xyz_jts = pred_xyz_jts - pred_xyz_jts[:, [0]]

        pred_phi = pred_phi.reshape(batch_size, -1, 2)

        output = self.smplx_layer.hybrik(
            betas=pred_beta.type(self.smpl_dtype),
            expression=pred_expression.type(self.smpl_dtype),
            pose_skeleton=pred_xyz_jts.type(self.smpl_dtype) * 2.2,
            phis=pred_phi.type(self.smpl_dtype),
            return_verts=True,
            naive=False
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
            pred_uvd_jts=pred_uvd_jts.reshape(batch_size, -1),
            pred_xyz_hybrik=pred_xyz_jts.reshape(batch_size, -1),
            pred_xyz_hybrik_struct=pred_xyz_hybrik_struct.reshape(batch_size, -1),
            pred_xyz_full=pred_xyz_struct_full.reshape(batch_size, -1),
            pred_uv_full=pred_uv_full,
            pred_vertices=pred_vertices,
            maxvals=maxvals,
            cam_scale=camScale[:, 0],
            cam_root=camera_root,
            transl=transl,
            img_feat=x0,
            pred_camera=pred_camera,
            sigma=sigma,
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
