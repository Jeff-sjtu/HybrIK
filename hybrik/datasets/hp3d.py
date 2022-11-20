"""MPI-INF-3DHP dataset."""
import json
import os

import cv2
import joblib
import numpy as np
import torch.utils.data as data

from hybrik.utils.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy
from hybrik.utils.pose_utils import (cam2pixel_matrix, pixel2cam_matrix,
                                     reconstruction_error)
from hybrik.utils.presets import (SimpleTransform3DSMPL,
                                  SimpleTransform3DSMPLCam)


class HP3D(data.Dataset):
    """ MPI-INF-3DHP dataset.

    Parameters
    ----------
    ann_file: str,
        Path to the annotation json file.
    root: str, default './data/3dhp'
        Path to the 3dhp dataset.
    train: bool, default is True
        If true, will set as training mode.
    skip_empty: bool, default is False
        Whether skip entire image if no valid label is found.
    """
    CLASSES = ['person']
    EVAL_JOINTS = [i - 1 for i in [8, 6, 15, 16, 17, 10, 11, 12, 24, 25, 26, 19, 20, 21, 5, 4, 7]]
    EVAL_JOINTS_17 = [
        14,
        11, 12, 13,
        8, 9, 10,
        15, 1,
        16, 0,
        5, 6, 7,
        2, 3, 4
    ]
    joints_name_17 = (
        'Pelvis',                               # 0
        'L_Hip', 'L_Knee', 'L_Ankle',           # 3
        'R_Hip', 'R_Knee', 'R_Ankle',           # 6
        'Torso', 'Neck',                        # 8
        'Nose', 'Head',                         # 10
        'L_Shoulder', 'L_Elbow', 'L_Wrist',     # 13
        'R_Shoulder', 'R_Elbow', 'R_Wrist',     # 16
    )
    # EVAL_JOINTS = [10, 8, 14, 15, 16, 11, 12, 13, 1, 2, 3, 4, 5, 6, 0, 7, 9]  # h36m -> 3dhp

    # num_joints = 28
    joints_name = ('spine3', 'spine4', 'spine2', 'spine', 'pelvis',                         # 4
                   'neck', 'head', 'head_top',                                              # 7
                   'left_clavicle', 'left_shoulder', 'left_elbow',                          # 10
                   'left_wrist', 'left_hand', 'right_clavicle',                             # 13
                   'right_shoulder', 'right_elbow', 'right_wrist',                          # 16
                   'right_hand', 'left_hip', 'left_knee',                                   # 19
                   'left_ankle', 'left_foot', 'left_toe',                                   # 22
                   'right_hip', 'right_knee', 'right_ankle', 'right_foot', 'right_toe')     # 27
    skeleton = ((0, 2), (1, 0), (2, 3), (3, 4),
                (5, 1), (6, 5), (7, 6), (8, 1), (9, 8), (10, 9),
                (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15),
                (17, 16), (18, 4), (19, 18), (20, 19), (21, 20), (22, 21),
                (23, 4), (24, 23), (25, 24), (26, 25), (27, 26)
                )
    skeleton = (
        (1, 0), (2, 1), (3, 2),         # 2
        (4, 0), (5, 4), (6, 5),         # 5
        (7, 0), (8, 7),                 # 7
        (9, 8), (10, 9),                # 9
        (11, 7), (12, 11), (13, 12),    # 12
        (14, 7), (15, 14), (16, 15),    # 15
    )
    mean_bone_len = None
    test_seqs = (1, 2, 3, 4, 5, 6)
    joint_groups = {'Head': [0], 'Neck': [1], 'Shou': [2, 5], 'Elbow': [3, 6], 'Wrist': [4, 7], 'Hip': [8, 11], 'Knee': [9, 12], 'Ankle': [10, 13]}
    # activity_name full name: ('Standing/Walking','Exercising','Sitting','Reaching/Crouching','On The Floor','Sports','Miscellaneous')
    activity_name = ('Stand', 'Exe', 'Sit', 'Reach', 'Floor', 'Sports', 'Miscell')
    pck_thres = 150
    auc_thres = list(range(0, 155, 5))

    def __init__(self,
                 cfg,
                 ann_file,
                 root='./data/3dhp',
                 train=True,
                 skip_empty=True,
                 dpg=False,
                 lazy_import=False):
        self._cfg = cfg

        self._ann_file = os.path.join(
            root, f'annotation_mpi_inf_3dhp_{ann_file}.json')
        self._lazy_import = lazy_import
        self._root = root
        self._skip_empty = skip_empty
        self._train = train
        self._dpg = dpg

        self.bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
        self._scale_factor = cfg.DATASET.SCALE_FACTOR
        self._color_factor = cfg.DATASET.COLOR_FACTOR
        self._rot = cfg.DATASET.ROT_FACTOR
        self._input_size = cfg.MODEL.IMAGE_SIZE
        self._output_size = cfg.MODEL.HEATMAP_SIZE

        self._occlusion = cfg.DATASET.OCCLUSION

        self._crop = cfg.MODEL.EXTRA.CROP
        self._sigma = cfg.MODEL.EXTRA.SIGMA
        self._depth_dim = cfg.MODEL.EXTRA.DEPTH_DIM

        self._check_centers = False

        self.num_class = len(self.CLASSES)
        self.num_joints = 28 if self._train else 17

        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY

        self.augment = cfg.MODEL.EXTRA.AUGMENT

        self._loss_type = cfg.LOSS['TYPE']
        self.kinematic = cfg.MODEL.EXTRA.get('KINEMATIC', False)

        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        self.lower_body_ids = (0, 1, 2, 3, 4, 5, 6)

        self.root_idx = self.joints_name.index('pelvis') if self._train else self.EVAL_JOINTS.index(self.joints_name.index('pelvis'))

        if cfg.MODEL.EXTRA.PRESET == 'simple_smpl_3d':
            self.transformation = SimpleTransform3DSMPL(
                self, scale_factor=self._scale_factor,
                color_factor=self._color_factor,
                occlusion=False,
                input_size=self._input_size,
                output_size=self._output_size,
                depth_dim=self._depth_dim,
                bbox_3d_shape=self.bbox_3d_shape,
                rot=self._rot, sigma=self._sigma,
                train=self._train, add_dpg=self._dpg,
                loss_type=self._loss_type, two_d=True)
        elif cfg.MODEL.EXTRA.PRESET == 'simple_smpl_3d_cam':
            self.transformation = SimpleTransform3DSMPLCam(
                self, scale_factor=self._scale_factor,
                color_factor=self._color_factor,
                occlusion=False,
                input_size=self._input_size,
                output_size=self._output_size,
                depth_dim=self._depth_dim,
                bbox_3d_shape=self.bbox_3d_shape,
                rot=self._rot, sigma=self._sigma,
                train=self._train, add_dpg=self._dpg,
                loss_type=self._loss_type, two_d=True,
                root_idx=self.root_idx)

        self.root_idx_17 = 0
        self.lshoulder_idx = self.joints_name.index('left_shoulder') if self._train else self.EVAL_JOINTS.index(self.joints_name.index('left_shoulder'))
        self.rshoulder_idx = self.joints_name.index('right_shoulder') if self._train else self.EVAL_JOINTS.index(self.joints_name.index('right_shoulder'))

        self.db = self.load_pt()

    def __getitem__(self, idx):
        # get image id
        img_path = self.db['img_path'][idx]
        img_id = self.db['img_id'][idx]

        # load ground truth, including bbox, keypoints, image size
        label = {}
        for k in self.db.keys():
            label[k] = self.db[k][idx].copy()
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # transform ground truth into training label and apply data augmentation
        target = self.transformation(img, label)

        img = target.pop('image')
        bbox = target.pop('bbox')
        return img, target, img_id, bbox

    def __len__(self):
        return len(self.db['img_path'])

    def load_pt(self):
        if os.path.exists(self._ann_file + '.pt'):
            db = joblib.load(self._ann_file + '.pt', 'r')
        else:
            self._save_pt()
            db = joblib.load(self._ann_file + '.pt', 'r')

        return db

    def _save_pt(self):
        _items, _labels = self._load_jsons()
        keys = list(_labels[0].keys())
        _db = {}
        for k in keys:
            _db[k] = []

        print(f'Generating 3DHP pt: {len(_labels)}...')
        for obj in _labels:
            for k in keys:
                _db[k].append(np.array(obj[k]))

        _db['img_path'] = _items
        for k in keys:
            _db[k] = np.stack(_db[k])
            assert _db[k].shape[0] == len(_labels)

        joblib.dump(_db, self._ann_file + '.pt')

    def _load_jsons(self):
        """Load all image paths and labels from JSON annotation files into buffer."""
        items = []
        labels = []

        with open(self._ann_file, 'r') as fid:
            database = json.load(fid)
        # iterate through the annotations
        for ann_image, ann_annotations in zip(database['images'], database['annotations']):
            ann = dict()
            for k, v in ann_image.items():
                assert k not in ann.keys()
                ann[k] = v
            for k, v in ann_annotations.items():
                ann[k] = v

            image_id = ann['image_id']

            width, height = ann['width'], ann['height']
            xmin, ymin, xmax, ymax = bbox_clip_xyxy(
                bbox_xywh_to_xyxy(ann['bbox']), width, height)

            intrinsic_param = np.array(ann['cam_param']['intrinsic_param'], dtype=np.float32)

            f = np.array([intrinsic_param[0, 0], intrinsic_param[1, 1]], dtype=np.float32)
            c = np.array([intrinsic_param[0, 2], intrinsic_param[1, 2]], dtype=np.float32)

            joint_cam = np.array(ann['keypoints_cam'])

            joint_img = cam2pixel_matrix(joint_cam, intrinsic_param)
            joint_img[:, 2] = joint_img[:, 2] - joint_cam[self.root_idx, 2]
            joint_vis = np.ones((self.num_joints, 3))

            root_cam = joint_cam[self.root_idx]

            abs_path = os.path.join(self._root, 'mpi_inf_3dhp_{}_set'.format('train' if self._train else 'test'), ann['file_name'])

            items.append(abs_path)
            labels.append({
                'bbox': (xmin, ymin, xmax, ymax),
                'img_id': image_id,
                'img_path': abs_path,
                'img_name': ann['file_name'],
                'width': width,
                'height': height,
                'joint_img': joint_img,
                'joint_vis': joint_vis,
                'joint_cam': joint_cam,
                'root_cam': root_cam,
                'intrinsic_param': intrinsic_param,
                'f': f,
                'c': c
            })
            if not self._train:
                labels[-1]['activity_id'] = ann['activity_id']
        return items, labels

    @property
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        hp3d_joint_pairs = ((8, 13), (9, 14), (10, 15), (11, 16), (12, 17),
                            (18, 23), (19, 24), (20, 25), (21, 26), (22, 27))
        return hp3d_joint_pairs
        # return ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))  # h36m pairs

    def _get_box_center_area(self, bbox):
        """Get bbox center"""
        c = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
        area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        return c, area

    def _get_keypoints_center_count(self, keypoints):
        """Get geometric center of all keypoints"""
        keypoint_x = np.sum(keypoints[:, 0, 0] * (keypoints[:, 0, 1] > 0))
        keypoint_y = np.sum(keypoints[:, 1, 0] * (keypoints[:, 1, 1] > 0))
        num = float(np.sum(keypoints[:, 0, 1]))
        return np.array([keypoint_x / num, keypoint_y / num]), num

    def add_thorax(self, joint_coord):
        thorax = (joint_coord[self.lshoulder_idx, :] + joint_coord[self.rshoulder_idx, :]) * 0.5
        thorax = thorax.reshape((1, 3))
        joint_coord = np.concatenate((joint_coord, thorax), axis=0)
        return joint_coord

    def _calc_metric_per_class(self, error, seq_idx_dict):
        seq_mpjpes_list = []
        seq_pck_array_list = []
        seq_auc_array_list = []
        for i in seq_idx_dict.keys():
            seq_error = np.take(error, seq_idx_dict[i], axis=0)
            seq_mpjpes = np.mean(seq_error, axis=0)
            seq_mpjpes = np.concatenate((seq_mpjpes, np.array([np.mean(seq_error)])), 0)
            joint_count = 0
            num_frames = seq_error.shape[0]
            num_joint_groups = len(self.joint_groups.keys())
            num_thres = len(self.auc_thres)
            # calculate pck & auc curve
            seq_pck_curve_array = np.zeros((num_joint_groups + 1, num_thres))
            seq_pck_array = np.zeros((num_joint_groups + 1))
            seq_auc_array = np.zeros((num_joint_groups + 1))
            # transval of joint groups
            for j_idx, j in enumerate(self.joint_groups.keys()):
                seq_jgroup_error = np.take(seq_error, self.joint_groups[j], axis=1)
                # transval of all thresholds
                for t_idx, t in enumerate(self.auc_thres):
                    seq_pck_curve_array[j_idx, t_idx] = np.sum(seq_jgroup_error < t) / (len(self.joint_groups[j]) * num_frames)
                joint_count += len(self.joint_groups[j])
                seq_pck_curve_array[-1, :] += seq_pck_curve_array[j_idx, :] * len(self.joint_groups[j])
                seq_auc_array[j_idx] = 100 * np.sum(seq_pck_curve_array[j_idx]) / num_thres
                seq_pck_array[j_idx] = 100 * np.sum(seq_jgroup_error < self.pck_thres) / (len(self.joint_groups[j]) * num_frames)
                seq_pck_array[-1] += seq_pck_array[j_idx] * len(self.joint_groups[j])
            seq_pck_array[-1] /= joint_count
            seq_pck_curve_array[-1, :] /= joint_count
            seq_auc_array[-1] = 100 * np.sum(seq_pck_curve_array[-1, :]) / num_thres
            seq_mpjpes_list.append(seq_mpjpes)
            seq_pck_array_list.append(seq_pck_array)
            seq_auc_array_list.append(seq_auc_array)
        return seq_mpjpes_list, seq_pck_array_list, seq_auc_array_list

    def evaluate_xyz_17(self, preds, result_dir):
        print('Evaluation start...')
        gts = self._labels
        assert len(gts) == len(preds)
        sample_num = len(gts)
        seq_idx_dict = {k: [] for k in self.test_seqs}
        act_idx_dict = {k: [] for k in range(len(self.activity_name))}

        pred_save = []
        error = np.zeros((sample_num, 17))  # joint error
        error_pa = np.zeros((sample_num, 17))  # joint error
        error_x = np.zeros((sample_num, 17))  # joint error
        error_y = np.zeros((sample_num, 17))  # joint error
        error_z = np.zeros((sample_num, 17))  # joint error
        # error for each sequence
        for n in range(sample_num):
            gt = gts[n]
            img_name = gt['img_name']

            # intrinsic_param = gt['intrinsic_param']
            bbox = gt['bbox']
            gt_3d_root = gt['root_cam']
            gt_3d_kpt = gt['joint_cam']

            gt_3d_kpt = np.take(gt_3d_kpt, self.EVAL_JOINTS_17, axis=0)

            # gt_vis = gt['joint_vis']
            pred_3d_kpt = preds[n]['xyz_17'].copy() * self.bbox_3d_shape[2]

            # root joint alignment
            pred_3d_kpt = pred_3d_kpt - pred_3d_kpt[self.root_idx_17]
            gt_3d_kpt = gt_3d_kpt - gt_3d_kpt[self.root_idx_17]

            # if self.protocol == 1:
            #     # rigid alignment for PA MPJPE (protocol #1)
            pred_3d_kpt_pa = reconstruction_error(pred_3d_kpt, gt_3d_kpt)
            align = False
            if align:
                pred_3d_kpt = pred_3d_kpt_pa
            # exclude thorax
            # pred_3d_kpt = np.take(pred_3d_kpt, self.EVAL_JOINTS, axis=0)
            # pred_3d_kpt_pa = np.take(pred_3d_kpt_pa, self.EVAL_JOINTS, axis=0)
            # gt_3d_kpt = np.take(gt_3d_kpt, self.EVAL_JOINTS, axis=0)

            # error calculate
            error[n] = np.sqrt(np.sum((pred_3d_kpt - gt_3d_kpt)**2, 1))
            error_pa[n] = np.sqrt(np.sum((pred_3d_kpt_pa - gt_3d_kpt)**2, 1))
            error_x[n] = np.abs(pred_3d_kpt[:, 0] - gt_3d_kpt[:, 0])
            error_y[n] = np.abs(pred_3d_kpt[:, 1] - gt_3d_kpt[:, 1])
            error_z[n] = np.abs(pred_3d_kpt[:, 2] - gt_3d_kpt[:, 2])

            # record idx per seq or act
            seq_id = int(img_name.split('/')[-3][2])
            seq_idx_dict[seq_id].append(n)
            act_idx_dict[int(gt['activity_id']) - 1].append(n)

            img_name = gt['img_path']
            # prediction save
            pred_save.append({'img_name': img_name, 'joint_cam': pred_3d_kpt.tolist(
            ), 'bbox': [float(_) for _ in bbox], 'root_cam': gt_3d_root.tolist()})  # joint_cam is root-relative coordinate

        # total error
        tot_err = np.mean(error)
        tot_err_pa = np.mean(error_pa)
        tot_err_x = np.mean(error_x)
        tot_err_y = np.mean(error_y)
        tot_err_z = np.mean(error_z)

        eval_summary = f'PA MPJPE >> tot: {tot_err_pa:2f}; MPJPE >> tot: {tot_err:2f}, x: {tot_err_x:2f}, y: {tot_err_y:.2f}, z: {tot_err_z:2f}\n'

        seq_mpjpes_list, seq_pck_array_list, seq_auc_array_list = self._calc_metric_per_class(error, seq_idx_dict)
        act_mpjpes_list, act_pck_array_list, act_auc_array_list = self._calc_metric_per_class(error, act_idx_dict)
        all_mpjpes_list, all_pck_array_list, all_auc_array_list = self._calc_metric_per_class(error, {0: list(range(sample_num))})

        # Summary mpjpe per sequence
        eval_summary += '#' * 10 + 'MPJPE per sequence\n'
        # eval_summary += ''.join(['MPJPE\t'] + [self.joints_name[j] + ' ' for j in self.EVAL_JOINTS_17] + ['Average\n'])
        total_mpjpe = 0
        for i_idx, i in enumerate(self.test_seqs):
            eval_summary += ''.join([f'TS{i}\t'] + ['{:.2f}\t'.format(seq_mpjpes_list[i_idx][j]) for j in range(seq_mpjpes_list[i_idx].shape[0])] + ['\n'])
            total_mpjpe += seq_mpjpes_list[i_idx][-1]
        total_mpjpe /= len(self.test_seqs)
        eval_summary += f'Avg MPJPE >> tot: {total_mpjpe:2f}\n'

        # Summary pck per sequence
        eval_summary += '#' * 10 + 'PCK per sequence\n'
        # eval_summary += ''.join(['PCK\t'] + [k + '\t' for k in self.joint_groups.keys()] + ['Total\n'])
        total_pck = 0
        for i_idx, i in enumerate(self.test_seqs):
            eval_summary += ''.join([f'TS{i}\t'] + ['{:.2f}\t'.format(seq_pck_array_list[i_idx][j]) for j in range(seq_pck_array_list[i_idx].shape[0])] + ['\n'])
            total_pck += seq_pck_array_list[i_idx][-1]
        total_pck /= len(self.test_seqs)
        eval_summary += f'Avg PCK >> tot: {total_pck:2f}\n'

        # Summary auc per sequence
        eval_summary += '#' * 10 + 'AUC per sequence\n'
        # eval_summary += ''.join(['AUC\t'] + [k + '\t' for k in self.joint_groups.keys()] + ['Total\n'])
        total_auc = 0
        for i_idx, i in enumerate(self.test_seqs):
            eval_summary += ''.join([f'TS{i}\t'] + ['{:.2f}\t'.format(seq_auc_array_list[i_idx][j]) for j in range(seq_auc_array_list[i_idx].shape[0])] + ['\n'])
            total_auc += seq_auc_array_list[i_idx][-1]
        total_auc /= len(self.test_seqs)
        eval_summary += f'Avg AUC >> tot: {total_auc:2f}\n'

        # Summary mpjpe per action
        eval_summary += '#' * 10 + 'MPJPE per action\n'
        # eval_summary += ''.join(['MPJPE\t'] + [self.joints_name[j] + ' ' for j in self.EVAL_JOINTS_17] + ['Average\n'])
        total_mpjpe = 0
        for i_idx, i in enumerate(self.activity_name):
            eval_summary += ''.join([f'{i}\t'] + ['{:.2f}\t'.format(act_mpjpes_list[i_idx][j]) for j in range(act_mpjpes_list[i_idx].shape[0])] + ['\n'])
            total_mpjpe += act_mpjpes_list[i_idx][-1]
        total_mpjpe /= len(self.activity_name)
        eval_summary += ''.join(['All\t'] + ['{:.2f}\t'.format(all_mpjpes_list[0][j]) for j in range(all_mpjpes_list[0].shape[0])] + ['\n'])
        eval_summary += f'Avg MPJPE >> tot: {total_mpjpe:2f}\n'

        # Summary pck per action
        eval_summary += '#' * 10 + 'PCK per action\n'
        # eval_summary += ''.join(['PCK\t'] + [k + '\t' for k in self.joint_groups.keys()] + ['Total\n'])
        total_pck = 0
        for i_idx, i in enumerate(self.activity_name):
            eval_summary += ''.join([f'{i}\t'] + ['{:.2f}\t'.format(act_pck_array_list[i_idx][j]) for j in range(act_pck_array_list[i_idx].shape[0])] + ['\n'])
            total_pck += act_pck_array_list[i_idx][-1]
        total_pck /= len(self.activity_name)
        eval_summary += ''.join(['All\t'] + ['{:.2f}\t'.format(all_pck_array_list[0][j]) for j in range(all_pck_array_list[0].shape[0])] + ['\n'])
        eval_summary += f'Avg PCK >> tot: {total_pck:2f}\n'

        # Summary auc per action
        eval_summary += '#' * 10 + 'AUC per action\n'
        # eval_summary += ''.join(['AUC\t'] + [k + '\t' for k in self.joint_groups.keys()] + ['Total\n'])
        total_auc = 0
        for i_idx, i in enumerate(self.activity_name):
            eval_summary += ''.join([f'{i}\t'] + ['{:.2f}\t'.format(act_auc_array_list[i_idx][j]) for j in range(act_auc_array_list[i_idx].shape[0])] + ['\n'])
            total_auc += act_auc_array_list[i_idx][-1]
        total_auc /= len(self.activity_name)
        eval_summary += ''.join(['All\t'] + ['{:.2f}\t'.format(all_auc_array_list[0][j]) for j in range(all_auc_array_list[0].shape[0])] + ['\n'])
        eval_summary += f'Avg AUC >> tot: {total_auc:2f}\n'

        print(eval_summary)

        # prediction save
        with open(result_dir, 'w') as f:
            json.dump(pred_save, f)
        print("Test result is saved at " + result_dir)
        return tot_err

    def evaluate(self, preds, result_dir):
        print('Evaluation start...')
        gts = self._labels
        assert len(gts) == len(preds)
        sample_num = len(gts)
        seq_idx_dict = {k: [] for k in self.test_seqs}
        act_idx_dict = {k: [] for k in range(len(self.activity_name))}

        pred_save = []
        error = np.zeros((sample_num, len(self.EVAL_JOINTS)))  # joint error
        error_pa = np.zeros((sample_num, len(self.EVAL_JOINTS)))  # joint error
        error_x = np.zeros((sample_num, len(self.EVAL_JOINTS)))  # joint error
        error_y = np.zeros((sample_num, len(self.EVAL_JOINTS)))  # joint error
        error_z = np.zeros((sample_num, len(self.EVAL_JOINTS)))  # joint error
        # error for each sequence
        for n in range(sample_num):
            gt = gts[n]
            img_name = gt['img_name']

            intrinsic_param = gt['intrinsic_param']
            bbox = gt['bbox']
            gt_3d_root = gt['root_cam']
            gt_3d_kpt = gt['joint_cam']

            # gt_vis = gt['joint_vis']

            # restore coordinates to original space
            pred_2d_kpt = preds[n].copy()
            # pred_2d_kpt[:, 0] = pred_2d_kpt[:, 0] / self._output_size[1] * bbox[2] + bbox[0]
            # pred_2d_kpt[:, 1] = pred_2d_kpt[:, 1] / self._output_size[0] * bbox[3] + bbox[1]
            pred_2d_kpt[:, 2] = pred_2d_kpt[:, 2] * self.bbox_3d_shape[0] + gt_3d_root[2]

            # back project to camera coordinate system
            pred_3d_kpt = pixel2cam_matrix(pred_2d_kpt, intrinsic_param)

            # root joint alignment
            pred_3d_kpt = pred_3d_kpt - pred_3d_kpt[self.root_idx]
            gt_3d_kpt = gt_3d_kpt - gt_3d_kpt[self.root_idx]

            # if self.protocol == 1:
            #     # rigid alignment for PA MPJPE (protocol #1)
            pred_3d_kpt_pa = reconstruction_error(pred_3d_kpt, gt_3d_kpt)

            # exclude thorax
            # pred_3d_kpt = np.take(pred_3d_kpt, self.EVAL_JOINTS, axis=0)
            # pred_3d_kpt_pa = np.take(pred_3d_kpt_pa, self.EVAL_JOINTS, axis=0)
            # gt_3d_kpt = np.take(gt_3d_kpt, self.EVAL_JOINTS, axis=0)

            # error calculate
            error[n] = np.sqrt(np.sum((pred_3d_kpt - gt_3d_kpt)**2, 1))
            error_pa[n] = np.sqrt(np.sum((pred_3d_kpt_pa - gt_3d_kpt)**2, 1))
            error_x[n] = np.abs(pred_3d_kpt[:, 0] - gt_3d_kpt[:, 0])
            error_y[n] = np.abs(pred_3d_kpt[:, 1] - gt_3d_kpt[:, 1])
            error_z[n] = np.abs(pred_3d_kpt[:, 2] - gt_3d_kpt[:, 2])

            # record idx per seq or act
            seq_id = int(img_name.split('/')[-3][2])
            seq_idx_dict[seq_id].append(n)
            act_idx_dict[int(gt['activity_id']) - 1].append(n)

            img_name = gt['img_path']
            # prediction save
            pred_save.append({'img_name': img_name, 'joint_cam': pred_3d_kpt.tolist(
            ), 'bbox': [float(_) for _ in bbox], 'root_cam': gt_3d_root.tolist()})  # joint_cam is root-relative coordinate

        # total error
        tot_err = np.mean(error)
        tot_err_pa = np.mean(error_pa)
        tot_err_x = np.mean(error_x)
        tot_err_y = np.mean(error_y)
        tot_err_z = np.mean(error_z)

        eval_summary = f'PA MPJPE >> tot: {tot_err_pa:2f}; MPJPE >> tot: {tot_err:2f}, x: {tot_err_x:2f}, y: {tot_err_y:.2f}, z: {tot_err_z:2f}\n'

        seq_mpjpes_list, seq_pck_array_list, seq_auc_array_list = self._calc_metric_per_class(error, seq_idx_dict)
        act_mpjpes_list, act_pck_array_list, act_auc_array_list = self._calc_metric_per_class(error, act_idx_dict)
        all_mpjpes_list, all_pck_array_list, all_auc_array_list = self._calc_metric_per_class(error, {0: list(range(sample_num))})

        # Summary mpjpe per sequence
        eval_summary += '#' * 10 + 'MPJPE per sequence\n'
        eval_summary += ''.join(['MPJPE\t'] + [self.joints_name[j] + ' ' for j in self.EVAL_JOINTS] + ['Average\n'])
        total_mpjpe = 0
        for i_idx, i in enumerate(self.test_seqs):
            eval_summary += ''.join([f'TS{i}\t'] + ['{:.2f}\t'.format(seq_mpjpes_list[i_idx][j]) for j in range(seq_mpjpes_list[i_idx].shape[0])] + ['\n'])
            total_mpjpe += seq_mpjpes_list[i_idx][-1]
        total_mpjpe /= len(self.test_seqs)
        eval_summary += f'Avg MPJPE >> tot: {total_mpjpe:2f}\n'

        # Summary pck per sequence
        eval_summary += '#' * 10 + 'PCK per sequence\n'
        eval_summary += ''.join(['PCK\t'] + [k + '\t' for k in self.joint_groups.keys()] + ['Total\n'])
        total_pck = 0
        for i_idx, i in enumerate(self.test_seqs):
            eval_summary += ''.join([f'TS{i}\t'] + ['{:.2f}\t'.format(seq_pck_array_list[i_idx][j]) for j in range(seq_pck_array_list[i_idx].shape[0])] + ['\n'])
            total_pck += seq_pck_array_list[i_idx][-1]
        total_pck /= len(self.test_seqs)
        eval_summary += f'Avg PCK >> tot: {total_pck:2f}\n'

        # Summary auc per sequence
        eval_summary += '#' * 10 + 'AUC per sequence\n'
        eval_summary += ''.join(['AUC\t'] + [k + '\t' for k in self.joint_groups.keys()] + ['Total\n'])
        total_auc = 0
        for i_idx, i in enumerate(self.test_seqs):
            eval_summary += ''.join([f'TS{i}\t'] + ['{:.2f}\t'.format(seq_auc_array_list[i_idx][j]) for j in range(seq_auc_array_list[i_idx].shape[0])] + ['\n'])
            total_auc += seq_auc_array_list[i_idx][-1]
        total_auc /= len(self.test_seqs)
        eval_summary += f'Avg AUC >> tot: {total_auc:2f}\n'

        # Summary mpjpe per action
        eval_summary += '#' * 10 + 'MPJPE per action\n'
        eval_summary += ''.join(['MPJPE\t'] + [self.joints_name[j] + ' ' for j in self.EVAL_JOINTS] + ['Average\n'])
        total_mpjpe = 0
        for i_idx, i in enumerate(self.activity_name):
            eval_summary += ''.join([f'{i}\t'] + ['{:.2f}\t'.format(act_mpjpes_list[i_idx][j]) for j in range(act_mpjpes_list[i_idx].shape[0])] + ['\n'])
            total_mpjpe += act_mpjpes_list[i_idx][-1]
        total_mpjpe /= len(self.activity_name)
        eval_summary += ''.join(['All\t'] + ['{:.2f}\t'.format(all_mpjpes_list[0][j]) for j in range(all_mpjpes_list[0].shape[0])] + ['\n'])
        eval_summary += f'Avg MPJPE >> tot: {total_mpjpe:2f}\n'

        # Summary pck per action
        eval_summary += '#' * 10 + 'PCK per action\n'
        eval_summary += ''.join(['PCK\t'] + [k + '\t' for k in self.joint_groups.keys()] + ['Total\n'])
        total_pck = 0
        for i_idx, i in enumerate(self.activity_name):
            eval_summary += ''.join([f'{i}\t'] + ['{:.2f}\t'.format(act_pck_array_list[i_idx][j]) for j in range(act_pck_array_list[i_idx].shape[0])] + ['\n'])
            total_pck += act_pck_array_list[i_idx][-1]
        total_pck /= len(self.activity_name)
        eval_summary += ''.join(['All\t'] + ['{:.2f}\t'.format(all_pck_array_list[0][j]) for j in range(all_pck_array_list[0].shape[0])] + ['\n'])
        eval_summary += f'Avg PCK >> tot: {total_pck:2f}\n'

        # Summary auc per action
        eval_summary += '#' * 10 + 'AUC per action\n'
        eval_summary += ''.join(['AUC\t'] + [k + '\t' for k in self.joint_groups.keys()] + ['Total\n'])
        total_auc = 0
        for i_idx, i in enumerate(self.activity_name):
            eval_summary += ''.join([f'{i}\t'] + ['{:.2f}\t'.format(act_auc_array_list[i_idx][j]) for j in range(act_auc_array_list[i_idx].shape[0])] + ['\n'])
            total_auc += act_auc_array_list[i_idx][-1]
        total_auc /= len(self.activity_name)
        eval_summary += ''.join(['All\t'] + ['{:.2f}\t'.format(all_auc_array_list[0][j]) for j in range(all_auc_array_list[0].shape[0])] + ['\n'])
        eval_summary += f'Avg AUC >> tot: {total_auc:2f}\n'

        print(eval_summary)

        # prediction save
        with open(result_dir, 'w') as f:
            json.dump(pred_save, f)
        print("Test result is saved at " + result_dir)
        return tot_err
