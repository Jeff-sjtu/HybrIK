"""Human3.6M dataset."""
import json
import os

# import scipy.misc
import cv2
import joblib
import numpy as np
import torch.utils.data as data
from hybrik.utils.pose_utils import pixel2cam, reconstruction_error
from hybrik.utils.presets import (SimpleTransform3DSMPL,
                                  SimpleTransform3DSMPLCam)


class H36mSMPL(data.Dataset):
    """ Human3.6M smpl dataset. 17 Human3.6M joints + 29 SMPL joints

    Parameters
    ----------
    ann_file: str,
        Path to the annotation json file.
    root: str, default './data/h36m'
        Path to the h36m dataset.
    train: bool, default is True
        If true, will set as training mode.
    skip_empty: bool, default is False
        Whether skip entire image if no valid label is found.
    """
    CLASSES = ['person']
    EVAL_JOINTS = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]

    num_joints = 17 + 29
    num_thetas = 24
    joints_name_17 = (
        'Pelvis',                               # 0
        'L_Hip', 'L_Knee', 'L_Ankle',           # 3
        'R_Hip', 'R_Knee', 'R_Ankle',           # 6
        'Torso', 'Neck',                        # 8
        'Nose', 'Head',                         # 10
        'L_Shoulder', 'L_Elbow', 'L_Wrist',     # 13
        'R_Shoulder', 'R_Elbow', 'R_Wrist',     # 16
    )
    joints_name_29 = (
        'pelvis', 'left_hip', 'right_hip',      # 2
        'spine1', 'left_knee', 'right_knee',    # 5
        'spine2', 'left_ankle', 'right_ankle',  # 8
        'spine3', 'left_foot', 'right_foot',    # 11
        'neck', 'left_collar', 'right_collar',  # 14
        'jaw',                                  # 15
        'left_shoulder', 'right_shoulder',      # 17
        'left_elbow', 'right_elbow',            # 19
        'left_wrist', 'right_wrist',            # 21
        'left_thumb', 'right_thumb',            # 23
        'head', 'left_middle', 'right_middle',  # 26
        'left_bigtoe', 'right_bigtoe'           # 28
    )
    joints_name_14 = (
        'R_Ankle', 'R_Knee', 'R_Hip',           # 2
        'L_Hip', 'L_Knee', 'L_Ankle',           # 5
        'R_Wrist', 'R_Elbow', 'R_Shoulder',     # 8
        'L_Shoulder', 'L_Elbow', 'L_Wrist',     # 11
        'Neck', 'Head'
    )

    action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases',
                   'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']

    block_list = ['s_09_act_05_subact_02_ca', 's_09_act_10_subact_02_ca', 's_09_act_13_subact_01_ca']

    def __init__(self,
                 cfg,
                 ann_file,
                 root='./data/h36m',
                 train=True,
                 skip_empty=True,
                 dpg=False,
                 lazy_import=False):
        self._cfg = cfg
        self.protocol = cfg.DATASET.PROTOCOL

        self._ann_file = os.path.join(
            root, 'annotations', ann_file + f'_protocol_{self.protocol}.json')
        self._lazy_import = lazy_import
        self._root = root
        self._skip_empty = skip_empty
        self._train = train
        self._dpg = dpg

        self._det_bbox_file = getattr(cfg.DATASET.SET_LIST[0], 'DET_BOX', None)
        self.bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))

        self._scale_factor = cfg.DATASET.SCALE_FACTOR
        self._color_factor = cfg.DATASET.COLOR_FACTOR
        self._rot = cfg.DATASET.ROT_FACTOR
        self._input_size = cfg.MODEL.IMAGE_SIZE
        self._output_size = cfg.MODEL.HEATMAP_SIZE

        self._occlusion = cfg.DATASET.OCCLUSION

        self._crop = cfg.MODEL.EXTRA.CROP
        self._sigma = cfg.MODEL.EXTRA.SIGMA
        self._depth_dim = getattr(cfg.MODEL.EXTRA, 'DEPTH_DIM', None)

        self._check_centers = False

        self.num_class = len(self.CLASSES)
        self.num_joints = cfg.MODEL.NUM_JOINTS

        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY

        self.augment = cfg.MODEL.EXTRA.AUGMENT
        self.dz_factor = cfg.MODEL.EXTRA.get('FACTOR', None)

        self._loss_type = cfg.LOSS['TYPE']

        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        self.lower_body_ids = (0, 1, 2, 3, 4, 5, 6)

        self.kinematic = cfg.MODEL.EXTRA.get('KINEMATIC', False)
        self.classfier = cfg.MODEL.EXTRA.get('WITHCLASSFIER', False)

        self.root_idx_17 = self.joints_name_17.index('Pelvis')
        self.lshoulder_idx_17 = self.joints_name_17.index('L_Shoulder')
        self.rshoulder_idx_17 = self.joints_name_17.index('R_Shoulder')
        self.root_idx_smpl = self.joints_name_29.index('pelvis')
        self.lshoulder_idx_29 = self.joints_name_29.index('left_shoulder')
        self.rshoulder_idx_29 = self.joints_name_29.index('right_shoulder')

        self.db = self.load_pt()

        if cfg.MODEL.EXTRA.PRESET == 'simple_smpl_3d':
            self.transformation = SimpleTransform3DSMPL(
                self, scale_factor=self._scale_factor,
                color_factor=self._color_factor,
                occlusion=self._occlusion,
                input_size=self._input_size,
                output_size=self._output_size,
                depth_dim=self._depth_dim,
                bbox_3d_shape=self.bbox_3d_shape,
                rot=self._rot, sigma=self._sigma,
                train=self._train, add_dpg=self._dpg,
                loss_type=self._loss_type, scale_mult=1)
        elif cfg.MODEL.EXTRA.PRESET == 'simple_smpl_3d_cam':
            self.transformation = SimpleTransform3DSMPLCam(
                self, scale_factor=self._scale_factor,
                color_factor=self._color_factor,
                occlusion=self._occlusion,
                input_size=self._input_size,
                output_size=self._output_size,
                depth_dim=self._depth_dim,
                bbox_3d_shape=self.bbox_3d_shape,
                rot=self._rot, sigma=self._sigma,
                train=self._train, add_dpg=self._dpg,
                loss_type=self._loss_type, scale_mult=1)

    def __getitem__(self, idx):
        # get image id
        img_path = self.db['img_path'][idx]
        img_id = self.db['img_id'][idx]

        # load ground truth, including bbox, keypoints, image size
        label = {}
        for k in self.db.keys():
            label[k] = self.db[k][idx].copy()

        # img = scipy.misc.imread(img_path, mode='RGB')
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # img = load_image(img_path)
        # img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        # transform ground truth into training label and apply data augmentation
        target = self.transformation(img, label)

        img = target.pop('image')
        bbox = target.pop('bbox')
        return img, target, img_id, bbox

    def __len__(self):
        return len(self.db['img_path'])

    def load_pt(self):
        db = joblib.load(self._ann_file + '.pt')
        return db

    @property
    def joint_pairs_17(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return ((1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16))

    @property
    def joint_pairs_24(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))

    @property
    def joint_pairs_29(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23), (25, 26), (27, 28))

    @property
    def bone_pairs(self):
        """Bone pairs which defines the pairs of bone to be swapped
        when the image is flipped horizontally."""
        return ((0, 3), (1, 4), (2, 5), (10, 13), (11, 14), (12, 15))

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

    def evaluate_uvd_24(self, preds, result_dir):
        print('Evaluation start...')
        assert len(self.db['img_id']) == len(preds)
        sample_num = len(self.db['img_id'])

        pred_save = []
        error = np.zeros((sample_num, 24))      # joint error
        error_x = np.zeros((sample_num, 24))    # joint error
        error_y = np.zeros((sample_num, 24))    # joint error
        error_z = np.zeros((sample_num, 24))    # joint error
        # error for each sequence
        error_action = [[] for _ in range(len(self.action_name))]
        for n in range(sample_num):
            image_id = self.db['img_id'][n]
            f = self.db['f'][n]
            c = self.db['c'][n]
            bbox = self.db['bbox'][n]
            gt_3d_root = self.db['root_cam'][n].copy()
            gt_3d_kpt = self.db['joint_cam_29'][n][:24].copy()

            # restore coordinates to original space
            pred_2d_kpt = preds[image_id]['uvd_jts'][:24].copy()
            # pred_2d_kpt[:, 0] = pred_2d_kpt[:, 0] / self._output_size[1] * bbox[2] + bbox[0]
            # pred_2d_kpt[:, 1] = pred_2d_kpt[:, 1] / self._output_size[0] * bbox[3] + bbox[1]
            pred_2d_kpt[:, 2] = pred_2d_kpt[:, 2] * self.bbox_3d_shape[2] + gt_3d_root[2]

            # back project to camera coordinate system
            pred_3d_kpt = pixel2cam(pred_2d_kpt, f, c)

            # root joint alignment
            pred_3d_kpt = pred_3d_kpt - pred_3d_kpt[self.root_idx_smpl]
            gt_3d_kpt = gt_3d_kpt - gt_3d_kpt[self.root_idx_smpl]

            if self.protocol == 1:
                # rigid alignment for PA MPJPE (protocol #1)
                pred_3d_kpt = reconstruction_error(pred_3d_kpt, gt_3d_kpt)

            # error calculate
            error[n] = np.sqrt(np.sum((pred_3d_kpt - gt_3d_kpt)**2, 1))
            error_x[n] = np.abs(pred_3d_kpt[:, 0] - gt_3d_kpt[:, 0])
            error_y[n] = np.abs(pred_3d_kpt[:, 1] - gt_3d_kpt[:, 1])
            error_z[n] = np.abs(pred_3d_kpt[:, 2] - gt_3d_kpt[:, 2])
            img_name = self.db['img_path'][n]
            action_idx = int(img_name[img_name.find(
                'act') + 4:img_name.find('act') + 6]) - 2
            error_action[action_idx].append(error[n].copy())

            # prediction save
            pred_save.append({'image_id': int(image_id), 'joint_cam': pred_3d_kpt.tolist(
            ), 'bbox': bbox.tolist(), 'root_cam': gt_3d_root.tolist()})  # joint_cam is root-relative coordinate

        # total error
        tot_err = np.mean(error)
        tot_err_x = np.mean(error_x)
        tot_err_y = np.mean(error_y)
        tot_err_z = np.mean(error_z)
        metric = 'PA MPJPE' if self.protocol == 1 else 'MPJPE'

        eval_summary = f'UVD_24 Protocol {self.protocol} error ({metric}) >> tot: {tot_err:2f}, x: {tot_err_x:2f}, y: {tot_err_y:.2f}, z: {tot_err_z:2f}\n'

        # error for each action
        for i in range(len(error_action)):
            err = np.mean(np.array(error_action[i]))
            eval_summary += (self.action_name[i] + ': %.2f ' % err)

        print(eval_summary)

        # prediction save
        with open(result_dir, 'w') as f:
            json.dump(pred_save, f)
        print("Test result is saved at " + result_dir)
        return tot_err

    def evaluate_xyz_24(self, preds, result_dir):
        print('Evaluation start...')
        assert len(self.db['img_id']) == len(preds)
        sample_num = len(self.db['img_id'])

        pred_save = []
        error = np.zeros((sample_num, 24))  # joint error
        error_align = np.zeros((sample_num, 24))  # joint error
        error_x = np.zeros((sample_num, 24))  # joint error
        error_y = np.zeros((sample_num, 24))  # joint error
        error_z = np.zeros((sample_num, 24))  # joint error
        # error for each sequence
        error_action = [[] for _ in range(len(self.action_name))]
        for n in range(sample_num):
            image_id = self.db['img_id'][n]
            bbox = self.db['bbox'][n]
            gt_3d_root = self.db['root_cam'][n].copy()
            gt_3d_kpt = self.db['joint_cam_29'][n][:24].copy()

            # gt_vis = self.db['joint_vis']

            # restore coordinates to original space
            pred_3d_kpt = preds[image_id]['xyz_24'].copy() * self.bbox_3d_shape[2]

            # root joint alignment
            pred_3d_kpt = pred_3d_kpt - pred_3d_kpt[self.root_idx_smpl]
            gt_3d_kpt = gt_3d_kpt - gt_3d_kpt[self.root_idx_smpl]

            # rigid alignment for PA MPJPE
            pred_3d_kpt_align = reconstruction_error(pred_3d_kpt.copy(), gt_3d_kpt.copy())

            # error calculate
            error[n] = np.sqrt(np.sum((pred_3d_kpt - gt_3d_kpt)**2, 1))
            error_align[n] = np.sqrt(np.sum((pred_3d_kpt_align - gt_3d_kpt)**2, 1))
            error_x[n] = np.abs(pred_3d_kpt[:, 0] - gt_3d_kpt[:, 0])
            error_y[n] = np.abs(pred_3d_kpt[:, 1] - gt_3d_kpt[:, 1])
            error_z[n] = np.abs(pred_3d_kpt[:, 2] - gt_3d_kpt[:, 2])
            img_name = self.db['img_path'][n]
            action_idx = int(img_name[img_name.find(
                'act') + 4:img_name.find('act') + 6]) - 2
            error_action[action_idx].append(error[n].copy())

            # prediction save
            pred_save.append({'image_id': int(image_id), 'joint_cam': pred_3d_kpt.tolist(
            ), 'bbox': bbox.tolist(), 'root_cam': gt_3d_root.tolist()})  # joint_cam is root-relative coordinate

        # total error
        tot_err = np.mean(error)
        tot_err_align = np.mean(error_align)
        tot_err_x = np.mean(error_x)
        tot_err_y = np.mean(error_y)
        tot_err_z = np.mean(error_z)
        metric = 'PA MPJPE' if self.protocol == 1 else 'MPJPE'

        eval_summary = f'XYZ_24 Protocol {self.protocol} error ({metric}) >> PA-MPJPE: {tot_err_align:2f} | MPJPE: {tot_err:2f}, x: {tot_err_x:2f}, y: {tot_err_y:.2f}, z: {tot_err_z:2f}\n'

        # error for each action
        for i in range(len(error_action)):
            err = np.mean(np.array(error_action[i]))
            eval_summary += (self.action_name[i] + ': %.2f ' % err)

        print(eval_summary)

        # prediction save
        with open(result_dir, 'w') as f:
            json.dump(pred_save, f)
        print("Test result is saved at " + result_dir)
        return tot_err

    def evaluate_xyz_17(self, preds, result_dir):
        print('Evaluation start...')
        assert len(self.db['img_id']) == len(preds)
        sample_num = len(self.db['img_id'])

        pred_save = []
        error = np.zeros((sample_num, len(self.EVAL_JOINTS)))  # joint error
        error_align = np.zeros((sample_num, len(self.EVAL_JOINTS)))  # joint error
        error_x = np.zeros((sample_num, len(self.EVAL_JOINTS)))  # joint error
        error_y = np.zeros((sample_num, len(self.EVAL_JOINTS)))  # joint error
        error_z = np.zeros((sample_num, len(self.EVAL_JOINTS)))  # joint error
        # error for each sequence
        error_action = [[] for _ in range(len(self.action_name))]
        for n in range(sample_num):
            image_id = self.db['img_id'][n]
            bbox = self.db['bbox'][n]
            gt_3d_root = self.db['root_cam'][n].copy()
            gt_3d_kpt = self.db['joint_relative_17'][n].copy()

            # gt_vis = self.db['joint_vis'][n]

            # restore coordinates to original space
            pred_3d_kpt = preds[image_id]['xyz_17'].copy() * self.bbox_3d_shape[2]

            # root joint alignment
            pred_3d_kpt = pred_3d_kpt - pred_3d_kpt[self.root_idx_17]
            gt_3d_kpt = gt_3d_kpt - gt_3d_kpt[self.root_idx_17]

            # rigid alignment for PA MPJPE
            # pred_3d_kpt_align = rigid_align(pred_3d_kpt.copy(), gt_3d_kpt.copy())
            pred_3d_kpt_align = reconstruction_error(pred_3d_kpt.copy(), gt_3d_kpt.copy())
            # pred_3d_kpt_align = pred_3d_kpt_align - pred_3d_kpt_align[self.root_idx_17]

            # select eval 14 joints
            pred_3d_kpt = np.take(pred_3d_kpt, self.EVAL_JOINTS, axis=0)
            gt_3d_kpt = np.take(gt_3d_kpt, self.EVAL_JOINTS, axis=0)
            pred_3d_kpt_align = np.take(pred_3d_kpt_align, self.EVAL_JOINTS, axis=0)

            # error calculate
            error[n] = np.sqrt(np.sum((pred_3d_kpt - gt_3d_kpt)**2, 1))
            error_align[n] = np.sqrt(np.sum((pred_3d_kpt_align - gt_3d_kpt)**2, 1))
            error_x[n] = np.abs(pred_3d_kpt[:, 0] - gt_3d_kpt[:, 0])
            error_y[n] = np.abs(pred_3d_kpt[:, 1] - gt_3d_kpt[:, 1])
            error_z[n] = np.abs(pred_3d_kpt[:, 2] - gt_3d_kpt[:, 2])
            img_name = self.db['img_path'][n]
            action_idx = int(img_name[img_name.find(
                'act') + 4:img_name.find('act') + 6]) - 2
            error_action[action_idx].append(error[n].copy())

            # prediction save
            pred_save.append({'image_id': int(image_id), 'joint_cam': pred_3d_kpt.tolist(
            ), 'bbox': bbox.tolist(), 'root_cam': gt_3d_root.tolist()})  # joint_cam is root-relative coordinate

        # total error
        tot_err = np.mean(error)
        tot_err_align = np.mean(error_align)
        tot_err_x = np.mean(error_x)
        tot_err_y = np.mean(error_y)
        tot_err_z = np.mean(error_z)
        metric = 'PA MPJPE' if self.protocol == 1 else 'MPJPE'

        eval_summary = f'XYZ_14 Protocol {self.protocol} error ({metric}) >> PA-MPJPE: {tot_err_align:2f} | MPJPE: {tot_err:2f}, x: {tot_err_x:2f}, y: {tot_err_y:.2f}, z: {tot_err_z:2f}\n'

        # error for each action
        for i in range(len(error_action)):
            err = np.mean(np.array(error_action[i]))
            eval_summary += (self.action_name[i] + ': %.2f ' % err)

        print(eval_summary)

        # prediction save
        with open(result_dir, 'w') as f:
            json.dump(pred_save, f)
        print("Test result is saved at " + result_dir)
        return tot_err_align
