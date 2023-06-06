"""Image demo script."""
import argparse
import os
import pickle as pk

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPLX
from hybrik.utils.render_pytorch3d import render_mesh
from hybrik.utils.vis import get_max_iou_box, get_one_box, vis_2d
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm


det_transform = T.Compose([T.ToTensor()])

halpe_wrist_ids = [94, 115]
halpe_left_hand_ids = [
    5, 6, 7,
    9, 10, 11,
    17, 18, 19,
    13, 14, 15,
    1, 2, 3,
]
halpe_right_hand_ids = [
    5, 6, 7,
    9, 10, 11,
    17, 18, 19,
    13, 14, 15,
    1, 2, 3,
]

halpe_lhand_leaves = [
    8, 12, 20, 16, 4
]
halpe_rhand_leaves = [
    8, 12, 20, 16, 4
]


halpe_hand_ids = [i + 94 for i in halpe_left_hand_ids] + [i + 115 for i in halpe_right_hand_ids]
halpe_hand_leaves_ids = [i + 94 for i in halpe_lhand_leaves] + [i + 115 for i in halpe_rhand_leaves]


def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]


def get_video_info(in_file):
    stream = cv2.VideoCapture(in_file)
    assert stream.isOpened(), 'Cannot capture source'
    # self.path = input_source
    datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))
    fps = stream.get(cv2.CAP_PROP_FPS)
    frameSize = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # bitrate = int(stream.get(cv2.CAP_PROP_BITRATE))
    videoinfo = {'fourcc': fourcc, 'fps': fps, 'frameSize': frameSize}
    stream.release()

    return stream, videoinfo, datalen


def recognize_video_ext(ext=''):
    if ext == 'mp4':
        return cv2.VideoWriter_fourcc(*'mp4v'), '.' + ext
    elif ext == 'avi':
        return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
    elif ext == 'mov':
        return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
    else:
        print("Unknow video format {}, will use .mp4 instead of it".format(ext))
        return cv2.VideoWriter_fourcc(*'mp4v'), '.mp4'


def integral_hm(hms):
    # hms: [B, K, H, W]
    B, K, H, W = hms.shape
    hms = hms.sigmoid()
    hms = hms.reshape(B, K, -1)
    hms = hms / hms.sum(dim=2, keepdim=True)
    hms = hms.reshape(B, K, H, W)

    hm_x = hms.sum((2,))
    hm_y = hms.sum((3,))

    w_x = torch.arange(hms.shape[3]).to(hms.device).float()
    w_y = torch.arange(hms.shape[2]).to(hms.device).float()

    hm_x = hm_x * w_x
    hm_y = hm_y * w_y

    coord_x = hm_x.sum(dim=2, keepdim=True)
    coord_y = hm_y.sum(dim=2, keepdim=True)

    coord_x = coord_x / float(hms.shape[3]) - 0.5
    coord_y = coord_y / float(hms.shape[2]) - 0.5

    coord_uv = torch.cat((coord_x, coord_y), dim=2)
    return coord_uv


parser = argparse.ArgumentParser(description='HybrIK Demo')

parser.add_argument('--gpu',
                    help='gpu',
                    default=0,
                    type=int)
# parser.add_argument('--img-path',
#                     help='image name',
#                     default='',
#                     type=str)
parser.add_argument('--video-name',
                    help='video name',
                    default='',
                    type=str)
parser.add_argument('--out-dir',
                    help='output folder',
                    default='',
                    type=str)
parser.add_argument('--save-pk', default=False, dest='save_pk',
                    help='save prediction', action='store_true')
parser.add_argument('--save-img', default=False, dest='save_img',
                    help='save prediction', action='store_true')


opt = parser.parse_args()


cfg_file = './configs/smplx/256x192_hrnet_rle_smplx_kid.yaml'
CKPT = './pretrained_models/hybrikx_rle_hrnet.pth'

cfg = update_config(cfg_file)

cfg['MODEL']['EXTRA']['USE_KID'] = cfg['DATASET'].get('USE_KID', False)
cfg['LOSS']['ELEMENTS']['USE_KID'] = cfg['DATASET'].get('USE_KID', False)


bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]
dummpy_set = edict({
    'joint_pairs_17': None,
    'joint_pairs_24': None,
    'joint_pairs_29': None,
    'bbox_3d_shape': bbox_3d_shape
})

res_keys = [
    'pred_uvd',
    'pred_scores',
    'pred_camera',
    'f',
    'pred_betas',
    'pred_thetas',
    'pred_phi',
    'pred_cam_root',
    # 'features',
    'transl',
    'transl_camsys',
    'bbox',
    'height',
    'width',
    'img_path'
]
res_db = {k: [] for k in res_keys}

transformation = SimpleTransform3DSMPLX(
    dummpy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
    color_factor=cfg.DATASET.COLOR_FACTOR,
    occlusion=cfg.DATASET.OCCLUSION,
    input_size=cfg.MODEL.IMAGE_SIZE,
    output_size=cfg.MODEL.HEATMAP_SIZE,
    depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
    bbox_3d_shape=bbox_3d_shape,
    rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
    train=False, add_dpg=False,
    loss_type=cfg.LOSS['TYPE'])

det_model = fasterrcnn_resnet50_fpn(pretrained=True)

hybrik_model = builder.build_sppe(cfg.MODEL)

print(f'Loading model from {CKPT}...')
save_dict = torch.load(CKPT, map_location='cpu')
if type(save_dict) == dict:
    model_dict = save_dict['model']
    hybrik_model.load_state_dict(model_dict)
else:
    hybrik_model.load_state_dict(save_dict)

det_model.cuda(opt.gpu)
hybrik_model.cuda(opt.gpu)
det_model.eval()
hybrik_model.eval()

print('### Extract Image...')
video_basename = os.path.basename(opt.video_name).split('.')[0]

if not os.path.exists(opt.out_dir):
    os.makedirs(opt.out_dir)
if not os.path.exists(os.path.join(opt.out_dir, 'raw_images')):
    os.makedirs(os.path.join(opt.out_dir, 'raw_images'))
if not os.path.exists(os.path.join(opt.out_dir, 'res_images')) and opt.save_img:
    os.makedirs(os.path.join(opt.out_dir, 'res_images'))
if not os.path.exists(os.path.join(opt.out_dir, 'res_2d_images')) and opt.save_img:
    os.makedirs(os.path.join(opt.out_dir, 'res_2d_images'))

_, info, _ = get_video_info(opt.video_name)
video_basename = os.path.basename(opt.video_name).split('.')[0]

savepath = f'./{opt.out_dir}/res_{video_basename}.mp4'
savepath2d = f'./{opt.out_dir}/res_2d_{video_basename}.mp4'
info['savepath'] = savepath
info['savepath2d'] = savepath2d

write_stream = cv2.VideoWriter(
    *[info[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
write2d_stream = cv2.VideoWriter(
    *[info[k] for k in ['savepath2d', 'fourcc', 'fps', 'frameSize']])
if not write_stream.isOpened():
    print("Try to use other video encoders...")
    ext = info['savepath'].split('.')[-1]
    fourcc, _ext = recognize_video_ext(ext)
    info['fourcc'] = fourcc
    info['savepath'] = info['savepath'][:-4] + _ext
    info['savepath2d'] = info['savepath2d'][:-4] + _ext
    write_stream = cv2.VideoWriter(
        *[info[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
    write2d_stream = cv2.VideoWriter(
        *[info[k] for k in ['savepath2d', 'fourcc', 'fps', 'frameSize']])

assert write_stream.isOpened(), 'Cannot open video for writing'
assert write2d_stream.isOpened(), 'Cannot open video for writing'

os.system(f'ffmpeg -i {opt.video_name} {opt.out_dir}/raw_images/{video_basename}-%06d.png')


files = os.listdir(f'{opt.out_dir}/raw_images')
files.sort()

img_path_list = []

for file in tqdm(files):
    if not os.path.isdir(file) and file[-4:] in ['.jpg', '.png']:

        img_path = os.path.join(opt.out_dir, 'raw_images', file)
        img_path_list.append(img_path)

prev_box = None
renderer = None
smplx_faces = torch.from_numpy(hybrik_model.smplx_layer.faces.astype(np.int32))

print('### Run Model...')
idx = 0
for img_path in tqdm(img_path_list):
    dirname = os.path.dirname(img_path)
    basename = os.path.basename(img_path)

    with torch.no_grad():
        # Run Detection
        input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        det_input = det_transform(input_image).to(opt.gpu)
        det_output = det_model([det_input])[0]

        if prev_box is None:
            tight_bbox = get_one_box(det_output)  # xyxy
            if tight_bbox is None:
                continue
        else:
            tight_bbox = get_one_box(det_output)  # xyxy
            # tight_bbox = get_max_iou_box(det_output, prev_box)  # xyxy

        if tight_bbox is None:
            tight_bbox = prev_box

        prev_box = tight_bbox

        # Run HybrIK
        # bbox: [x1, y1, x2, y2]
        pose_input, bbox, img_center = transformation.test_transform(
            input_image.copy(), tight_bbox)
        pose_input = pose_input.to(opt.gpu)[None, :, :, :]

        '''
        pose_input_192 = pose_input[:, :, :, 32:-32].clone()

        # pose_input_192, bbox192 = al_transformation.test_transform(
        #     input_image.copy(), tight_bbox)
        # pose_input_192 = pose_input_192.to(opt.gpu)[None, :, :, :]
        pose_input_192[:, 0] = pose_input_192[:, 0] * 0.225
        pose_input_192[:, 1] = pose_input_192[:, 1] * 0.224
        pose_input_192[:, 2] = pose_input_192[:, 2] * 0.229
        al_output = alphapose_model(pose_input_192)
        al_uv_jts = integral_hm(al_output).squeeze(0)
        # hand_uv_jts = al_uv_jts[halpe_hand_ids, :]
        al_uv_jts[:, 0] = al_uv_jts[:, 0] * 192 / 256
        wrist_uv_jts = al_uv_jts[halpe_wrist_ids, :]
        hand_uv_jts = al_uv_jts[halpe_hand_ids, :]
        hand_leaf_uv_jts = al_uv_jts[halpe_hand_leaves_ids, :]
        '''
        # vis 2d
        bbox_xywh = xyxy2xywh(bbox)
        '''
        al_fb_pts = al_uv_jts.clone() * bbox_xywh[2]
        al_fb_pts[:, 0] = al_fb_pts[:, 0] + bbox_xywh[0]
        al_fb_pts[:, 1] = al_fb_pts[:, 1] + bbox_xywh[1]
        '''

        pose_output = hybrik_model(
            pose_input, flip_test=True,
            bboxes=torch.from_numpy(np.array(bbox)).to(pose_input.device).unsqueeze(0).float(),
            img_center=torch.from_numpy(img_center).to(pose_input.device).unsqueeze(0).float(),
            # al_hands=hand_uv_jts.to(pose_input.device).unsqueeze(0).float(),
            # al_hands_leaf=hand_leaf_uv_jts.to(pose_input.device).unsqueeze(0).float(),
        )

        uv_jts = pose_output.pred_uvd_jts.reshape(-1, 3)[:, :2]
        # uv_jts[25:55, :2] = hand_uv_jts
        # uv_jts[-10:, :2] = hand_leaf_uv_jts
        transl = pose_output.transl.detach()

        # Visualization
        image = input_image.copy()
        focal = 1000.0
        bbox_xywh = xyxy2xywh(bbox)
        transl_camsys = transl.clone()
        transl_camsys = transl_camsys * 256 / bbox_xywh[2]

        focal = focal / 256 * bbox_xywh[2]

        vertices = pose_output.pred_vertices.detach()

        verts_batch = vertices
        transl_batch = transl

        color_batch = render_mesh(
            vertices=verts_batch, faces=smplx_faces,
            translation=transl_batch,
            focal_length=focal, height=image.shape[0], width=image.shape[1])

        valid_mask_batch = (color_batch[:, :, :, [-1]] > 0)
        image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
        image_vis_batch = (image_vis_batch * 255).cpu().numpy()

        color = image_vis_batch[0]
        valid_mask = valid_mask_batch[0].cpu().numpy()
        input_img = image
        alpha = 0.9
        image_vis = alpha * color[:, :, :3] * valid_mask + (
            1 - alpha) * input_img * valid_mask + (1 - valid_mask) * input_img

        image_vis = image_vis.astype(np.uint8)
        image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)

        if opt.save_img:
            idx += 1
            res_path = os.path.join(opt.out_dir, 'res_images', f'image-{idx:06d}.jpg')
            cv2.imwrite(res_path, image_vis)
        write_stream.write(image_vis)
        '''
        # vis 2d
        pts = uv_jts * bbox_xywh[2]
        pts[:, 0] = pts[:, 0] + bbox_xywh[0]
        pts[:, 1] = pts[:, 1] + bbox_xywh[1]
        image = input_image.copy()
        bbox_img = vis_2d(image, tight_bbox, pts)
        # bbox_img = vis_2d(image, tight_bbox, al_fb_pts)
        bbox_img = cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR)
        write2d_stream.write(bbox_img)

        if opt.save_img:
            res_path = os.path.join(
                opt.out_dir, 'res_2d_images', f'image-{idx:06d}.jpg')
            cv2.imwrite(res_path, bbox_img)
        '''
    if opt.save_pk:
        assert pose_input.shape[0] == 1, 'Only support single batch inference for now'
        pred_uvd_jts = pose_output.pred_uvd_jts.reshape(
            -1, 3).cpu().data.numpy()
        pred_scores = pose_output.maxvals.cpu(
        ).data[:, :29].reshape(29).numpy()
        pred_camera = pose_output.pred_camera.squeeze(
            dim=0).cpu().data.numpy()
        pred_betas = pose_output.pred_shape_full.squeeze(
            dim=0).cpu().data.numpy()
        pred_theta = pose_output.pred_theta_mat.squeeze(
            dim=0).cpu().data.numpy()
        pred_phi = pose_output.pred_phi.squeeze(dim=0).cpu().data.numpy()
        pred_cam_root = pose_output.cam_root.squeeze(dim=0).cpu().numpy()
        img_size = np.array((input_image.shape[0], input_image.shape[1]))

        res_db['pred_uvd'].append(pred_uvd_jts)
        res_db['pred_scores'].append(pred_scores)
        res_db['pred_camera'].append(pred_camera)
        res_db['f'].append(1000.0)
        res_db['pred_betas'].append(pred_betas)
        res_db['pred_thetas'].append(pred_theta)
        res_db['pred_phi'].append(pred_phi)
        res_db['pred_cam_root'].append(pred_cam_root)
        res_db['transl'].append(transl[0].cpu().data.numpy())
        res_db['transl_camsys'].append(transl_camsys[0].cpu().data.numpy())
        res_db['bbox'].append(np.array(bbox))
        res_db['height'].append(img_size[0])
        res_db['width'].append(img_size[1])
        res_db['img_path'].append(img_path)

if opt.save_pk:
    n_frames = len(res_db['img_path'])
    for k in res_db.keys():
        res_db[k] = np.stack(res_db[k])
        assert res_db[k].shape[0] == n_frames

    with open(os.path.join(opt.out_dir, 'res.pk'), 'wb') as fid:
        pk.dump(res_db, fid)

write_stream.release()
write2d_stream.release()
