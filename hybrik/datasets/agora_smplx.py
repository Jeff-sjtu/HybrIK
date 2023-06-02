import json
import os

import cv2
import joblib
import numpy as np
import torch
import torch.utils.data as data
from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix
from tqdm import tqdm

from hybrik.models.layers.smplx.joint_names import JOINT_NAMES
from hybrik.models.layers.smplx.load_body_models import load_models
from hybrik.utils.pose_utils import pixel2cam, reconstruction_error
from hybrik.utils.presets.simple_transform_3d_smplx import \
    SimpleTransform3DSMPLX

(
    smplx_layer_neutral,
    smplx_layer_male,
    smplx_layer_female,
    smplx_layer_neutral_kid,
    smplx_layer_male_kid,
    smplx_layer_female_kid
) = load_models()


class AGORAX(data.Dataset):
    """ AGORA-SMPLX dataset.

    Parameters
    ----------
    ann_file: str,
        Path to the annotation json file.
    root: str, default './data/agora'
        Path to the AGORA dataset.
    train: bool, default is True
        If true, will set as training mode.
    skip_empty: bool, default is False
        Whether skip entire image if no valid label is found.
    """
    CLASSES = ['person']

    bbox_3d_shape = (2.2, 2.2, 2.2)
    joints_names = JOINT_NAMES
    joints_names_hybrik = JOINT_NAMES[:55] + [
        'LBigToe', 'RBigToe', 'mouth_bottom', 'leye', 'reye',   # 59
        'lindex', 'lmiddle', 'lpinky', 'lring', 'lthumb',       # 64
        'rindex', 'rmiddle', 'rpinky', 'rring', 'rthumb',       # 69
        'mouth_top'
    ]

    def __init__(self,
                 cfg,
                 ann_file,
                 root='./data/AGORA',
                 train=True,
                 skip_empty=True,
                 dpg=False,
                 lazy_import=False,
                 high_res_inp=True,
                 return_img_path=False,
                 return_vertices=False,
                 finetune=False):
        self._cfg = cfg

        self._ann_file = ann_file
        self._lazy_import = lazy_import
        self._root = root
        self._skip_empty = skip_empty
        self._train = train
        self._dpg = dpg
        self.return_img_path = return_img_path

        self._scale_factor = cfg.DATASET.SCALE_FACTOR
        self._color_factor = cfg.DATASET.COLOR_FACTOR
        self._rot = cfg.DATASET.ROT_FACTOR
        self._input_size = cfg.MODEL.IMAGE_SIZE
        self._output_size = cfg.MODEL.HEATMAP_SIZE

        self._occlusion = False
        self.use_kid = cfg.DATASET.get('USE_KID', False)
        print('self.use_kid', self.use_kid)

        self._crop = cfg.MODEL.EXTRA.CROP
        self._sigma = cfg.MODEL.EXTRA.SIGMA
        self._depth_dim = cfg.MODEL.EXTRA.DEPTH_DIM

        self.num_class = len(self.CLASSES)

        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY

        self.augment = cfg.MODEL.EXTRA.AUGMENT

        self._loss_type = cfg.LOSS['TYPE']

        self.upper_body_ids = (6, 9, 12, 13, 14, 15, 16,
                               17, 18, 19, 20, 21, 22, 23)
        self.lower_body_ids = (0, 1, 2, 3, 4, 5, 7, 8, 10, 11)

        self.kinematic = cfg.MODEL.EXTRA.get('KINEMATIC', False)
        self.classfier = cfg.MODEL.EXTRA.get('WITHCLASSFIER', False)

        self.root_idx_smpl = 0
        self.root_left_hand = 20
        self.root_right_hand = 21
        self.root_head = 15

        self.focal_length = cfg.DATASET.get('FOCAL_LENGTH', 1000)
        self.high_res_inp = high_res_inp

        self.db = self.load_pt()

        self.finetune = finetune

        self.transformation = SimpleTransform3DSMPLX(
            self, scale_factor=self._scale_factor,
            color_factor=self._color_factor,
            occlusion=False,
            input_size=self._input_size,
            output_size=self._output_size,
            depth_dim=self._depth_dim,
            bbox_3d_shape=self.bbox_3d_shape,
            rot=self._rot, sigma=self._sigma,
            train=self._train, add_dpg=self._dpg,
            loss_type=self._loss_type,
            focal_length=self.focal_length,
            scale_mult=1.0, return_vertices=return_vertices,
            rand_bbox_shift=False
        )

    def __getitem__(self, idx):
        # get image id
        img_path = self.db['img_path'][idx]
        img_id = self.db['img_id'][idx]

        img_path = img_path.split('/')
        assert img_path[3] == 'images', img_path
        if self.high_res_inp:
            img_path[3] = 'images_large'
            basename = img_path[5]
            basename = basename[:-13] + '.png'
            img_path[5] = basename
        else:
            img_path[3] = 'images_small'
        img_path = '/'.join(img_path)

        # load ground truth, including bbox, keypoints, image size
        label = {}
        for k in self.db.keys():
            if type(self.db[k][idx]) is str:
                label[k] = self.db[k][idx]
            else:
                label[k] = self.db[k][idx].copy()

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        if self.high_res_inp:
            assert img.shape[1] == 3840, img.shape
            assert img.shape[0] == 2160, img.shape
        else:
            assert img.shape[1] == 1280, img.shape
            assert img.shape[0] == 720, img.shape

        # transform ground truth into training label and apply data augmentation
        target = self.transformation(img, label)

        img = target.pop('image')
        bbox = target.pop('bbox')

        if self.finetune:
            target['smplx_weight'] = 1.0
            # target.pop('target_theta_full')

            target.pop('type')
            target.pop('camera_error')

            target['dataset_idx'] = 0
        # return_img_path = self.cfg.DATASET.get('FOCAL_LENGTH', 5000)
        if self.return_img_path:
            # temorarily reuse an unused var
            return img, target, (img_id, img_path), bbox
        return img, target, img_id, bbox

    def __len__(self):
        return len(self.db['img_path'])

    def load_pt(self):
        if self.high_res_inp:
            resolution = '_4k_'
        else:
            resolution = '_720p_'
        if os.path.exists(self._ann_file + resolution + '_final.pt'):
            db = joblib.load(self._ann_file + resolution + '_final.pt', 'r')
        else:
            self._save_pt()
            db = joblib.load(self._ann_file + resolution + '_final.pt', 'r')

        return db

    def _save_pt(self):

        if self.high_res_inp:
            resolution = '_4k_'
        else:
            resolution = '_720p_'

        _db = joblib.load(self._ann_file, 'r')
        _items, _labels = self._lazy_load_pt(_db)

        keys = list(_labels[0].keys())
        _db = {}
        for k in keys:
            _db[k] = []

        print(f'Generating AGORA pt: {len(_labels)}...')
        for obj in _labels:
            for k in keys:
                _db[k].append(np.array(obj[k]))

        _db['img_path'] = _items
        for k in keys:
            _db[k] = np.stack(_db[k])
            assert _db[k].shape[0] == len(_labels)

        joblib.dump(_db, self._ann_file + resolution + '_final.pt')

    def _lazy_load_pt(self, db):
        """Load all image paths and labels from json annotation files into buffer."""

        items = []
        labels = []

        db_len = len(db['ann_path'])

        for k, v in db.items():
            assert len(v) == db_len, k

        img_cnt = 0
        kid_cnt = 0

        for idx in tqdm(range(db_len)):
            img_name = db['img_path'][idx]
            ann_path = db['ann_path'][idx]

            focal, pelvis_pos = get_focal(db, idx, img_name)

            # print(ann_path, img_name)
            ann_file = ann_path.split('/')[-1]

            ann_file = ann_file.split('_')
            if 'train' in img_name:
                img_parent_path = os.path.join(self._root, 'images', f'{ann_file[0]}_{ann_file[1]}')
            else:
                img_parent_path = os.path.join(self._root, 'images', 'validation')

            img_path = os.path.join(img_parent_path, img_name)

            beta = np.array(db['shape'][idx]).reshape(10)
            expression = np.array(db['expression'][idx]).reshape(10)
            theta_full = np.array(db['full_pose'][idx]).reshape(-1, 3)
            angle = db['twist_angle'][idx].reshape(-1)
            cos = np.cos(angle)
            sin = np.sin(angle)

            phi = np.stack((cos, sin), axis=1)
            phi_weight = (angle > -10) * 1.0
            phi_weight = np.stack([phi_weight, phi_weight], axis=1)

            # convert to torch
            beta = torch.from_numpy(beta).reshape(1, 10)
            expression = torch.from_numpy(expression).reshape(1, 10)
            theta = torch.from_numpy(theta_full).reshape(1, 55, 3)
            phi = torch.from_numpy(phi).reshape(1, -1, 2)
            theta = axis_angle_to_matrix(theta.reshape(55, 3)).reshape(1, 55, 3, 3)

            gender = db['gender'][idx]
            is_kid = db['is_kid'][idx]

            if is_kid:
                shape_kid = np.array(db['shape_kid'][idx]).reshape(1)
                shape_kid = torch.from_numpy(shape_kid).reshape(1, 1)
                beta = torch.cat((beta, shape_kid), dim=1)
                if gender == 'female':
                    smplx_layer = smplx_layer_female_kid
                elif gender == 'male':
                    smplx_layer = smplx_layer_male_kid
                elif gender == 'neutral':
                    smplx_layer = smplx_layer_neutral_kid
            else:
                if gender == 'female':
                    smplx_layer = smplx_layer_female
                elif gender == 'male':
                    smplx_layer = smplx_layer_male
                elif gender == 'neutral':
                    smplx_layer = smplx_layer_neutral

            output = smplx_layer.forward_simple(
                betas=beta,
                full_pose=theta,
                expression=expression,
                return_verts=True,
                root_align=True
            )

            gt_joints_55 = output.joints_55
            gt_verts = output.vertices

            leaf_vertices = gt_verts[:, smplx_layer.LEAF_INDICES].clone()
            gt_joints_71_xyz = torch.cat([gt_joints_55, leaf_vertices], dim=1)

            gt_joints_71_xyz = gt_joints_71_xyz.cpu().numpy()[0]
            gt_joints_71_xyz = gt_joints_71_xyz + pelvis_pos[None, :]
            rel_71_xyz = gt_joints_71_xyz - gt_joints_71_xyz[[0], :]

            x0 = 1280 / 2
            y0 = 720 / 2
            if self.high_res_inp:
                focal = focal * 3
                x0 = x0 * 3
                y0 = y0 * 3

            gt_joints_71_uv = project(gt_joints_71_xyz, focal, x0, y0)

            joint_img_71 = np.zeros_like(gt_joints_71_xyz)
            joint_img_71[:, :2] = gt_joints_71_uv.copy()
            joint_img_71[:, 2] = rel_71_xyz[:, 2].copy()
            joint_vis_71 = np.ones_like(joint_img_71)

            root_cam = pelvis_pos

            # generate bbox from kpt2d
            # print(joint_2d)
            left, right, upper, lower = \
                gt_joints_71_uv[:, 0].min(), gt_joints_71_uv[:, 0].max(), gt_joints_71_uv[:, 1].min(), gt_joints_71_uv[:, 1].max()

            center = np.array([(left + right) * 0.5, (upper + lower) * 0.5], dtype=np.float32)
            scale = [right - left, lower - upper]

            scale = float(max(scale))

            # rand_norm = np.array([local_random.gauss(mu=0, sigma=1), local_random.gauss(mu=0, sigma=1)])
            # print(rand_norm)
            # rand_shift = 0.05 * scale * rand_norm
            # center = center + rand_shift

            scale = scale * 1.3

            xmin, ymin, xmax, ymax = center[0] - scale * 0.5, center[1] - scale * 0.5, center[0] + scale * 0.5, center[1] + scale * 0.5

            if self.high_res_inp:
                if not (xmin < 3840 - 3 and ymin < 2160 - 3 and xmax > 3 and ymax > 3):
                    continue
            else:
                if not (xmin < 1280 - 3 and ymin < 720 - 3 and xmax > 3 and ymax > 3):
                    continue

            is_valid = db['is_valid'][idx]

            gender = db['gender'][idx]
            is_kid = False
            if self.use_kid:
                is_kid = db['is_kid'][idx]
                beta_kid = np.array(db['shape_kid'][idx])
                kid_cnt += (2 if is_kid else 0)
            else:
                beta_kid = np.zeros(1)

            beta = beta.numpy()[0, :10]
            expression = expression.numpy()[0]
            phi = phi.numpy()[0]

            added_num = 2 if is_kid else 1
            for _ in range(added_num):
                # child ratio x 2
                items.append(img_path)
                labels.append({
                    'bbox': (xmin, ymin, xmax, ymax),
                    'img_id': img_cnt,
                    'img_path': img_path,
                    'img_name': img_name,
                    'is_valid': is_valid,
                    'joint_img': joint_img_71.copy(),
                    'joint_vis': joint_vis_71.copy(),
                    'joint_xyz': rel_71_xyz.copy(),
                    'twist_phi': phi,
                    'twist_weight': phi_weight,
                    'beta': beta,
                    'expression': expression,
                    'theta_full': theta_full,
                    'root_cam': root_cam,
                    'beta_kid': beta_kid,
                    'gender': gender,
                    'is_kid': is_kid,
                    'focal': focal,
                    'pelvis_depth': pelvis_pos
                })

                img_cnt += 1

        print('datalen', db_len, len(items), 'kid', kid_cnt / len(items))
        return items, labels

    def evaluate_uvd_24(self, preds, result_dir):
        print('Evaluation start...')
        assert len(self.db['img_id']) == len(preds)
        sample_num = len(self.db['img_id'])

        pred_save = []
        error = np.zeros((sample_num, 24))  # joint error
        error_x = np.zeros((sample_num, 24))  # joint error
        error_y = np.zeros((sample_num, 24))  # joint error
        error_z = np.zeros((sample_num, 24))  # joint error
        for n in range(sample_num):
            image_id = self.db['img_id'][n]
            f = self.db['f'][n]
            c = self.db['c'][n]
            bbox = self.db['bbox'][n]
            gt_3d_root = self.db['root_cam'][n].copy()
            gt_3d_kpt = self.db['joint_cam_29'][n][:24, :].copy()

            # gt_vis = gt['joint_vis']

            # restore coordinates to original space
            pred_2d_kpt = preds[image_id]['uvd_jts'][:24, :].copy()
            # pred_2d_kpt[:, 0] = pred_2d_kpt[:, 0] / self._output_size[1] * bbox[2] + bbox[0]
            # pred_2d_kpt[:, 1] = pred_2d_kpt[:, 1] / self._output_size[0] * bbox[3] + bbox[1]
            pred_2d_kpt[:, 2] = pred_2d_kpt[:, 2] * self.bbox_3d_shape[2] + gt_3d_root[2]

            # back project to camera coordinate system
            pred_3d_kpt = pixel2cam(pred_2d_kpt, f, c)

            # root joint alignment
            pred_3d_kpt = pred_3d_kpt - pred_3d_kpt[self.root_idx_smpl]
            gt_3d_kpt = gt_3d_kpt - gt_3d_kpt[self.root_idx_smpl]

            # error calculate
            error[n] = np.sqrt(np.sum((pred_3d_kpt - gt_3d_kpt)**2, 1))
            error_x[n] = np.abs(pred_3d_kpt[:, 0] - gt_3d_kpt[:, 0])
            error_y[n] = np.abs(pred_3d_kpt[:, 1] - gt_3d_kpt[:, 1])
            error_z[n] = np.abs(pred_3d_kpt[:, 2] - gt_3d_kpt[:, 2])
            img_name = self.db['img_path'][n]

            # prediction save
            pred_save.append({
                'img_name': str(img_name), 'joint_cam': pred_3d_kpt.tolist(),
                'bbox': bbox.tolist(), 'root_cam': gt_3d_root.tolist()})  # joint_cam is root-relative coordinate

        # total error
        tot_err = np.mean(error) * 1000
        # tot_err_kp = np.mean(error, axis=0) * 1000
        tot_err_x = np.mean(error_x) * 1000
        tot_err_y = np.mean(error_y) * 1000
        tot_err_z = np.mean(error_z) * 1000
        metric = 'MPJPE'

        eval_summary = f'UVD_24 error ({metric}) >> tot: {tot_err:.2f}, x: {tot_err_x:.2f}, y: {tot_err_y:.2f}, z: {tot_err_z:.2f}\n'

        print(eval_summary)
        # print(f'UVD_24 error per joint: {tot_err_kp}')

        # prediction save
        with open(result_dir, 'w') as f:
            json.dump(pred_save, f)
        print("Test result is saved at " + result_dir)
        return tot_err

    def evaluate_xyz_hybrik(self, preds, result_dir, use_struct=False):
        print('Evaluation start...')
        assert len(self.db['img_id']) == len(preds)
        sample_num = len(self.db['img_id'])

        pred_save = []
        error = np.zeros((sample_num, 71))  # joint error
        error_align = np.zeros((sample_num, 71))  # joint error
        error_x = np.zeros((sample_num, 71))  # joint error
        error_y = np.zeros((sample_num, 71))  # joint error
        error_z = np.zeros((sample_num, 71))  # joint error
        error_left_hand_aligned = np.zeros((sample_num, 15))
        error_right_hand_aligned = np.zeros((sample_num, 15))
        error_head_aligned = np.zeros((sample_num, 3))
        for n in range(sample_num):
            image_id = self.db['img_id'][n]
            bbox = self.db['bbox'][n]
            gt_3d_root = self.db['root_cam'][n].copy()
            gt_3d_kpt = self.db['joint_xyz'][n].copy()

            # restore coordinates to original space
            if use_struct:
                pred_3d_kpt = preds[image_id]['xyz_hybrik_struct'].copy() * self.bbox_3d_shape[2]
            else:
                pred_3d_kpt = preds[image_id]['xyz_hybrik'].copy() * self.bbox_3d_shape[2]

            # left hand
            pred_3d_kpt_left_hand = pred_3d_kpt[25:40]
            gt_3d_kpt_left_hand = gt_3d_kpt[25:40]
            pred_3d_kpt_left_hand = pred_3d_kpt_left_hand - pred_3d_kpt[self.root_left_hand]
            gt_3d_kpt_left_hand = gt_3d_kpt_left_hand - gt_3d_kpt[self.root_left_hand]
            # right hand
            pred_3d_kpt_right_hand = pred_3d_kpt[40:55]
            gt_3d_kpt_right_hand = gt_3d_kpt[40:55]
            pred_3d_kpt_right_hand = pred_3d_kpt_right_hand - pred_3d_kpt[self.root_right_hand]
            gt_3d_kpt_right_hand = gt_3d_kpt_right_hand - gt_3d_kpt[self.root_right_hand]
            # head
            pred_3d_kpt_head = pred_3d_kpt[22:25]
            gt_3d_kpt_head = gt_3d_kpt[22:25]
            pred_3d_kpt_head = pred_3d_kpt_head - pred_3d_kpt[self.root_head]
            gt_3d_kpt_head = gt_3d_kpt_head - gt_3d_kpt[self.root_head]

            # root joint alignment
            pred_3d_kpt = pred_3d_kpt - pred_3d_kpt[self.root_idx_smpl]
            gt_3d_kpt = gt_3d_kpt - gt_3d_kpt[self.root_idx_smpl]

            # rigid alignment for PA MPJPE
            pred_3d_kpt_align = reconstruction_error(
                pred_3d_kpt.copy(), gt_3d_kpt.copy())

            # error calculate
            error[n] = np.sqrt(np.sum((pred_3d_kpt - gt_3d_kpt)**2, 1))
            error_left_hand_aligned[n] = np.sqrt(np.sum((pred_3d_kpt_left_hand - gt_3d_kpt_left_hand)**2, 1))
            error_right_hand_aligned[n] = np.sqrt(np.sum((pred_3d_kpt_right_hand - gt_3d_kpt_right_hand)**2, 1))
            error_head_aligned[n] = np.sqrt(np.sum((pred_3d_kpt_head - gt_3d_kpt_head)**2, 1))
            error_align[n] = np.sqrt(np.sum((pred_3d_kpt_align - gt_3d_kpt)**2, 1))
            error_x[n] = np.abs(pred_3d_kpt[:, 0] - gt_3d_kpt[:, 0])
            error_y[n] = np.abs(pred_3d_kpt[:, 1] - gt_3d_kpt[:, 1])
            error_z[n] = np.abs(pred_3d_kpt[:, 2] - gt_3d_kpt[:, 2])
            img_name = self.db['img_path'][n]

            # prediction save
            pred_save.append({
                'img_name': str(img_name), 'joint_cam': pred_3d_kpt.tolist(),
                'bbox': bbox.tolist(), 'root_cam': gt_3d_root.tolist()})  # joint_cam is root-relative coordinate

        # total error
        tot_err = np.mean(error) * 1000
        tot_err_hand_aligned = (np.mean(error_left_hand_aligned) + np.mean(error_right_hand_aligned)) * 1000 / 2
        tot_err_face_aligned = np.mean(error_head_aligned) * 1000
        tot_err_align = np.mean(error_align) * 1000
        tot_err_x = np.mean(error_x) * 1000
        tot_err_y = np.mean(error_y) * 1000
        tot_err_z = np.mean(error_z) * 1000

        body_error = np.mean(error[:, :22]) * 1000
        face_error = np.mean(error[:, 22:25]) * 1000
        hand_error = np.mean(error[:, 25:55]) * 1000

        joint_level_summary = ''
        for i, name in enumerate(self.joints_names_hybrik):
            if i < 30:
                joint_error = np.mean(error[:, i]) * 1000
                joint_level_summary += f'{name}: {joint_error.mean():.1f}, '

        print(joint_level_summary)

        if use_struct:
            eval_summary = f'XYZ_hybrik struct >> tot: {tot_err:.2f}, tot_pa: {tot_err_align:.2f}, x: {tot_err_x:.2f}, y: {tot_err_y:.2f}, z: {tot_err_z:.2f}\n '
        else:
            eval_summary = f'XYZ_hybrik >> tot: {tot_err:.2f}, tot_pa: {tot_err_align:.2f}, x: {tot_err_x:.2f}, y: {tot_err_y:.2f}, z: {tot_err_z:.2f}\n '

        eval_summary += f'body_error: {body_error:.2f}, '
        eval_summary += f'face_error: {face_error:.2f}, face_error_aligned: {tot_err_face_aligned:.2f}, '
        eval_summary += f'hand_error: {hand_error:.2f}, hand_error_aligned: {tot_err_hand_aligned:.2f}\n'
        print(eval_summary)

        # prediction save
        with open(result_dir, 'w') as f:
            json.dump(pred_save, f)
        print("Test result is saved at " + result_dir)
        return tot_err, eval_summary

    def evaluate_pelvis_depth(self, preds, result_dir, use_struct=False):
        print('Evaluation start...')
        assert len(self.db['img_id']) == len(preds)
        sample_num = len(self.db['img_id'])

        pred_save = []
        error = np.zeros((sample_num, 1))  # joint error
        for n in range(sample_num):
            image_id = self.db['img_id'][n]
            bbox = self.db['bbox'][n]

            gt_pelvis_depth = self.db['pelvis_depth'][n].copy()
            gt_pelvis_depth = gt_pelvis_depth[2]
            # if self.high_res_inp:
            #     gt_pelvis_depth = gt_pelvis_depth * 720 / 2160
            focal_i = self.db['focal'][n].copy()
            pred_depth = preds[image_id]['pelvis_depth'].copy()
            pred_depth = pred_depth * focal_i / 1000.0

            # error calculate
            error[n] = np.abs(pred_depth - gt_pelvis_depth) / 3
            img_name = self.db['img_path'][n]

            # prediction save
            pred_save.append({
                'img_name': str(img_name),
                'bbox': bbox.tolist()})  # joint_cam is root-relative coordinate

        # total error
        tot_err = np.mean(error) * 1000

        eval_summary = f'Depth hybrik >> tot: {tot_err:.2f}\n '
        print(eval_summary)

        # prediction save
        with open(result_dir, 'w') as f:
            json.dump(pred_save, f)
        print("Test result is saved at " + result_dir)
        return tot_err, eval_summary


def focalLength_mm2px(focalLength, dslr_sens, focalPoint):
    focal_pixel = (focalLength / dslr_sens) * focalPoint * 2
    return focal_pixel


def get_focal(db, idx, imgpath):

    dslr_sens_width = 36
    dslr_sens_height = 20.25

    imgWidth = 1280
    imgHeight = 720

    if 'hdri' in imgpath:
        ground_plane = [0, 0, 0]
        scene3d = False
        focalLength = 50
        camPosWorld = [0, 0, 170]
        camYaw = 0
        camPitch = 0
    elif 'cam00' in imgpath:
        ground_plane = [0, 0, 0]
        scene3d = True
        focalLength = 18
        camPosWorld = [400, -275, 265]
        camYaw = 135
        camPitch = 30
    elif 'cam01' in imgpath:
        ground_plane = [0, 0, 0]
        scene3d = True
        focalLength = 18
        camPosWorld = [400, 225, 265]
        camYaw = -135
        camPitch = 30
    elif 'cam02' in imgpath:
        ground_plane = [0, 0, 0]
        scene3d = True
        focalLength = 18
        camPosWorld = [-490, 170, 265]
        camYaw = -45
        camPitch = 30
    elif 'cam03' in imgpath:
        ground_plane = [0, 0, 0]
        scene3d = True
        focalLength = 18
        camPosWorld = [-490, -275, 265]
        camYaw = 45
        camPitch = 30
    elif 'ag2' in imgpath:
        ground_plane = [0, 0, 0]
        scene3d = False
        focalLength = 28
        camPosWorld = [0, 0, 170]
        camYaw = 0
        camPitch = 15
    else:
        ground_plane = [0, -1.7, 0]
        scene3d = True
        focalLength = 28
        # camPosWorld = [
        #     df.iloc[i]['camX'],
        #     df.iloc[i]['camY'],
        #     df.iloc[i]['camZ']]
        # camYaw = df.iloc[i]['camYaw']
        camPitch = 0

    cx = imgWidth / 2
    cy = imgHeight / 2

    focalLength_x = focalLength_mm2px(focalLength, dslr_sens_width, cx)
    focalLength_y = focalLength_mm2px(focalLength, dslr_sens_height, cy)

    assert abs(focalLength_x - focalLength_y) < 1e-3, (focalLength_x, focalLength_y)

    joints_xyz_127 = db['gt_joints_3d'][idx]
    pelvis_pos = joints_xyz_127[0]
    # pelvis_pos = None

    return focalLength_x, pelvis_pos


def project(xyz, focal, x0, y0):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    u = x * focal / z + x0
    v = y * focal / z + y0

    uv = np.stack((u, v), axis=1)
    return uv
