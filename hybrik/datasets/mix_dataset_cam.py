import bisect
import random

import numpy as np
import torch
import torch.utils.data as data

from .h36m_smpl import H36mSMPL
from .hp3d import HP3D
from .mscoco import Mscoco

s_mpii_2_smpl_jt = [
    6, 3, 2,
    -1, 4, 1,
    -1, 5, 0,
    -1, -1, -1,
    8, -1, -1,
    -1,
    13, 12,
    14, 11,
    15, 10,
    -1, -1
]
s_3dhp_2_smpl_jt = [
    4, -1, -1, # pelvis, _, _
    -1, 19, 24, # _, left_knee, right_knee
    -1, 20, 25, # _, left_ankle, right_ankle
    -1, -1, -1,  # _, _, _
    5, -1, -1, # neck, _, _
    -1, # _
    9, 14, # left shoulder, right_shoulder
    10, 15, # left elbow, right elbow
    11, 16, # left wrist, right wrist
    -1, -1,
    # 7, # head top
    # -1, -1, # _, _
    # 21, 26 # left foot, right foot
]
s_coco_2_smpl_jt = [
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

neighbour_kpts = [
    [1, 2, 3], [0, 4], [0, 5], [0, 6], # 3
    [1, 7], [2, 8], [3, 9], [4, 10], # 7
    [5, 11], [6, 12, 13, 14], [7], [8], # 11
    [9, 15], [9, 16], [9, 17], [12], # 15
    [13, 18], [14, 19], [16, 20], [17, 21], # 19
    [18, 22], [19, 23], [20], [21], # 23
    [15], [22], [23], [10], [11] # 28
]

s_smpl24_jt_num = 24


class MixDatasetCam(data.Dataset):
    CLASSES = ['person']
    EVAL_JOINTS = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]

    num_joints = 24
    joints_name_17 = (
        'Pelvis',                               # 0
        'L_Hip', 'L_Knee', 'L_Ankle',           # 3
        'R_Hip', 'R_Knee', 'R_Ankle',           # 6
        'Torso', 'Neck',                        # 8
        'Nose', 'Head',                         # 10
        'L_Shoulder', 'L_Elbow', 'L_Wrist',     # 13
        'R_Shoulder', 'R_Elbow', 'R_Wrist',     # 16
    )
    joints_name_24 = (
        'pelvis', 'left_hip', 'right_hip',      # 2
        'spine1', 'left_knee', 'right_knee',    # 5
        'spine2', 'left_ankle', 'right_ankle',  # 8
        'spine3', 'left_foot', 'right_foot',    # 11
        'neck', 'left_collar', 'right_collar',  # 14
        'jaw',                                  # 15
        'left_shoulder', 'right_shoulder',      # 17
        'left_elbow', 'right_elbow',            # 19
        'left_wrist', 'right_wrist',            # 21
        'left_thumb', 'right_thumb'             # 23
    )
    data_domain = set([
        'type',
        'target_theta',
        'target_theta_weight',
        'target_beta',
        'target_smpl_weight',
        'target_uvd_29',
        'target_xyz_24',
        'target_weight_24',
        'target_weight_29',
        'target_xyz_17',
        'target_weight_17',
        'trans_inv',
        'intrinsic_param',
        'joint_root',
        'target_twist',
        'target_twist_weight',
        'depth_factor',
        'target_xyz_weight_24',
        'camera_scale',
        'camera_trans',
        'camera_valid',
        'camera_error'
    ])

    def __init__(self,
                 cfg,
                 train=True):
        self._train = train
        self.heatmap_size = cfg.MODEL.HEATMAP_SIZE
        self.bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))

        if train:
            self.db0 = H36mSMPL(
                cfg=cfg,
                ann_file=cfg.DATASET.SET_LIST[0].TRAIN_SET,
                train=True)
            self.db1 = Mscoco(
                cfg=cfg,
                ann_file=f'person_keypoints_{cfg.DATASET.SET_LIST[1].TRAIN_SET}.json',
                train=True)
            self.db2 = HP3D(
                cfg=cfg,
                ann_file=cfg.DATASET.SET_LIST[2].TRAIN_SET,
                train=True)

            self._subsets = [self.db0, self.db1, self.db2]
            self._2d_length = len(self.db1)
            self._3d_length = len(self.db0) + len(self.db2)
        else:
            self.db0 = H36mSMPL(
                cfg=cfg,
                ann_file=cfg.DATASET.SET_LIST[0].TEST_SET,
                train=train)

            self._subsets = [self.db0]

        self._subset_size = [len(item) for item in self._subsets]
        self._db0_size = len(self.db0)

        if train:
            self.max_db_data_num = max(self._subset_size)
            self.tot_size = 2 * max(self._subset_size)
            self.partition = [0.4, 0.5, 0.1]
        else:
            self.tot_size = self._db0_size
            self.partition = [1]

        self.cumulative_sizes = self.cumsum(self.partition)

        self.joint_pairs_24 = self.db0.joint_pairs_24
        self.joint_pairs_17 = self.db0.joint_pairs_17
        self.root_idx_17 = self.db0.root_idx_17
        self.root_idx_smpl = self.db0.root_idx_smpl
        self.evaluate_xyz_17 = self.db0.evaluate_xyz_17
        self.evaluate_uvd_24 = self.db0.evaluate_uvd_24
        self.evaluate_xyz_24 = self.db0.evaluate_xyz_24

        self.out_size = 64

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            r.append(e + s)
            s += e
        return r

    def __len__(self):
        return self.tot_size

    def __getitem__(self, idx):
        assert idx >= 0
        if self._train:
            p = random.uniform(0, 1)

            dataset_idx = bisect.bisect_right(self.cumulative_sizes, p)

            _db_len = self._subset_size[dataset_idx]

            # last batch: random sampling
            if idx >= _db_len * (self.tot_size // _db_len):
                sample_idx = random.randint(0, _db_len - 1)
            else:  # before last batch: use modular
                sample_idx = idx % _db_len
        else:
            dataset_idx = 0
            sample_idx = idx

        img, target, img_id, bbox = self._subsets[dataset_idx][sample_idx]

        if dataset_idx > 0 and dataset_idx < 3:
            # COCO, 3DHP
            label_jts_origin = target.pop('target')
            label_jts_mask_origin = target.pop('target_weight')

            # label_xyz_origin = target.pop('target_xyz')
            # label_xyz_mask_origin = target.pop('target_xyz_weight')

            label_uvd_29 = torch.zeros(29, 3)
            label_xyz_24 = torch.zeros(24, 3)
            label_uvd_29_mask = torch.zeros(29, 3)
            label_xyz_17 = torch.zeros(17, 3)
            label_xyz_17_mask = torch.zeros(17, 3)
            label_xyz_29_mask = torch.zeros(29, 3)

            if dataset_idx == 1:
                # COCO
                assert label_jts_origin.dim() == 1 and label_jts_origin.shape[0] == 17 * 2, label_jts_origin.shape

                label_jts_origin = label_jts_origin.reshape(17, 2)
                label_jts_mask_origin = label_jts_mask_origin.reshape(17, 2)

                for i in range(s_smpl24_jt_num):
                    id1 = i
                    id2 = s_coco_2_smpl_jt[i]
                    if id2 >= 0:
                        label_uvd_29[id1, :2] = label_jts_origin[id2, :2].clone()
                        label_uvd_29_mask[id1, :2] = label_jts_mask_origin[id2, :2].clone()
            elif dataset_idx == 2:
                # 3DHP
                assert label_jts_origin.dim() == 1 and label_jts_origin.shape[0] == 28 * 3, label_jts_origin.shape

                label_xyz_origin = target.pop('target_xyz').reshape(28, 3)
                label_xyz_mask_origin = target.pop('target_xyz_weight').reshape(28, 3)

                label_jts_origin = label_jts_origin.reshape(28, 3)
                label_jts_mask_origin = label_jts_mask_origin.reshape(28, 3)

                for i in range(s_smpl24_jt_num):
                    id1 = i
                    id2 = s_3dhp_2_smpl_jt[i]
                    if id2 >= 0:
                        label_uvd_29[id1, :3] = label_jts_origin[id2, :3].clone()
                        label_uvd_29_mask[id1, :3] = label_jts_mask_origin[id2, :3].clone()
                        label_xyz_24[id1, :3] = label_xyz_origin[id2, :3].clone()
                        label_xyz_29_mask[id1, :3] = label_xyz_mask_origin[id2, :3].clone()

            label_uvd_29 = label_uvd_29.reshape(-1)
            label_xyz_24 = label_xyz_24.reshape(-1)
            label_uvd_24_mask = label_uvd_29_mask[:24, :].reshape(-1)
            label_uvd_29_mask = label_uvd_29_mask.reshape(-1)
            label_xyz_17 = label_xyz_17.reshape(-1)
            label_xyz_17_mask = label_xyz_17_mask.reshape(-1)
            label_xyz_24_mask = label_xyz_29_mask[:24, :].reshape(-1)

            target['target_uvd_29'] = label_uvd_29
            target['target_xyz_24'] = label_xyz_24
            target['target_weight_24'] = label_uvd_24_mask
            target['target_weight_29'] = label_uvd_29_mask
            target['target_xyz_17'] = label_xyz_17
            target['target_weight_17'] = label_xyz_17_mask
            target['target_theta'] = torch.zeros(24 * 4)
            target['target_beta'] = torch.zeros(10)
            target['target_smpl_weight'] = torch.zeros(1)
            target['target_theta_weight'] = torch.zeros(24 * 4)
            target['target_twist'] = torch.zeros(23, 2)
            target['target_twist_weight'] = torch.zeros(23, 2)
            target['target_xyz_weight_24'] = label_xyz_24_mask
        # else:
        #     assert set(target.keys()).issubset(self.data_domain), (set(target.keys()) - self.data_domain, self.data_domain - set(target.keys()),)
        target.pop('type')

        target['dataset_idx'] = dataset_idx

        return img, target, img_id, bbox
