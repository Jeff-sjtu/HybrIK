"""Image demo script."""
import argparse
import os
import sys

import cv2
import ffmpeg
import numpy as np
import torch
from easydict import EasyDict as edict
from hybrik.models import builder
from hybrik.models.layers.smpl.lbs import rotmat_to_aa
from hybrik.models.smplify.smplify import SMPLify3D
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPL
from hybrik.utils.render import SMPLRenderer
from hybrik.utils.vis import get_max_iou_box, get_one_box, vis_2d, vis_smpl_3d
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm

det_transform = T.Compose([T.ToTensor()])


def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]


def get_video_info(in_file):
    probe = ffmpeg.probe(in_file)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    if video_stream is None:
        print('No video stream found', file=sys.stderr)
        sys.exit(1)
    return video_stream


parser = argparse.ArgumentParser(description='HybrIK Demo')
CKPT = 'pretrained_w_cam_res50.pth'

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
parser.add_argument('--post-process',
                    default=False,
                    dest='post_process',
                    help='post process with SMPLify-X',
                    action='store_true')

opt = parser.parse_args()


cfg_file = 'configs/256x192_adam_lr1e-3-res50_cam_2x_w_pw3d_3dhp.yaml'
cfg = update_config(cfg_file)

bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]
dummpy_set = edict({
    'joint_pairs_17': None,
    'joint_pairs_24': None,
    'joint_pairs_29': None,
    'bbox_3d_shape': bbox_3d_shape
})

transformation = SimpleTransform3DSMPL(
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
hybrik_model.load_state_dict(torch.load(CKPT, map_location='cpu'), strict=False)

det_model.cuda(opt.gpu)
hybrik_model.cuda(opt.gpu)
det_model.eval()
hybrik_model.eval()
smplify = SMPLify3D(step_size=1e-2, smpl=hybrik_model.smpl, device=opt.gpu, num_iters=[5, 20], focal_length=1000)

print('### Extract Image...')
video_basename = os.path.basename(opt.video_name).split('.')[0]

if not os.path.exists(opt.out_dir):
    os.makedirs(opt.out_dir)
if not os.path.exists(os.path.join(opt.out_dir, 'raw_images')):
    os.makedirs(os.path.join(opt.out_dir, 'raw_images'))
if not os.path.exists(os.path.join(opt.out_dir, 'res_images')):
    os.makedirs(os.path.join(opt.out_dir, 'res_images'))
if not os.path.exists(os.path.join(opt.out_dir, 'res_2d_images')):
    os.makedirs(os.path.join(opt.out_dir, 'res_2d_images'))

info = get_video_info(opt.video_name)
bitrate = info['bit_rate']
os.system(f'ffmpeg -i {opt.video_name} {opt.out_dir}/raw_images/{video_basename}-%06d.jpg')


files = os.listdir(f'{opt.out_dir}/raw_images')
files.sort()

img_path_list = []

for file in tqdm(files):
    if not os.path.isdir(file) and file[-4:] in ['.jpg', '.png']:

        img_path = os.path.join(opt.out_dir, 'raw_images', file)
        img_path_list.append(img_path)

prev_box = None

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
            tight_bbox = get_max_iou_box(det_output, prev_box)  # xyxy

        prev_box = tight_bbox

        # Run HybrIK
        pose_input, bbox = transformation.test_transform(img_path, tight_bbox)
        pose_input = pose_input.to(opt.gpu)[None, :, :, :]
        pose_output = hybrik_model(pose_input)
        uv_29 = pose_output.pred_uvd_jts.reshape(29, 3)[:, :2]
        transl = pose_output.transl
        vertices = pose_output.pred_vertices

        if opt.post_process:
            init_pose = pose_output.pred_theta_mats
            init_pose = rotmat_to_aa(init_pose.reshape(-1, 3, 3)).reshape(-1, 24 * 3)
            init_betas = pose_output.pred_shape
            init_cam_t = pose_output.transl

            # keypoints_2d_jts = pose_output.pred_uvd_jts.reshape(-1, 29, 3)[:, :24, :2] * hybrik_model.input_size
            keypoints_3d_jts = pose_output.pred_xyz_jts_24.reshape(-1, 24, 3) * hybrik_model.depth_factor
            keypoints_3d_jts = keypoints_3d_jts + pose_output.cam_root[:, None, :]
            keypoints_3d_conf = 1 - pose_output.maxvals.reshape(-1, 29, 1)[:, :24] * 10
            keypoints_3d = torch.cat((keypoints_3d_jts, keypoints_3d_conf), dim=2)
            # keypoints_2d = torch.cat((keypoints_2d_jts, keypoints_2d_conf), dim=2)
            smplify_output = smplify(init_pose, init_betas, init_cam_t, keypoints_3d)

            # pred_xyz_jts_24_struct = smplify_output.joints
            # # print(pred_xyz_jts_17.shape, smplify_output.joints_from_verts.shape)
            # pred_xyz_jts_17 = smplify_output.joints_from_verts
            # pred_mesh = smplify_output.vertices
            # print(smplify_output.reprojection_loss[0])

            vertices = smplify_output.vertices
            transl = smplify_output.camera_translation

    # Visualization
    image = input_image.copy()
    img_size = (image.shape[0], image.shape[1])
    focal = np.array([1000, 1000])
    bbox_xywh = xyxy2xywh(bbox)
    princpt = [bbox_xywh[0], bbox_xywh[1]]

    renderer = SMPLRenderer(faces=hybrik_model.smpl.faces,
                            img_size=img_size, focal=focal,
                            princpt=princpt)

    transl = transl.detach().cpu().numpy().squeeze()
    transl[2] = transl[2] * 256 / bbox_xywh[2]

    image_vis = vis_smpl_3d(
        vertices, image, cam_root=transl,
        f=focal, c=princpt, renderer=renderer)

    image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)
    # cv2.imshow('image_vis', image_vis)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    idx += 1
    res_path = os.path.join(opt.out_dir, 'res_images', f'image-{idx:06d}.jpg')
    cv2.imwrite(res_path, image_vis)

    # vis 2d
    pts = uv_29 * bbox_xywh[2]
    pts[:, 0] = pts[:, 0] + bbox_xywh[0]
    pts[:, 1] = pts[:, 1] + bbox_xywh[1]
    image = input_image.copy()
    bbox_img = vis_2d(image, tight_bbox, pts)
    bbox_img = cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR)
    res_path = os.path.join(opt.out_dir, 'res_2d_images', f'image-{idx:06d}.jpg')
    cv2.imwrite(res_path, bbox_img)

os.system(f"ffmpeg -r 25 -i ./{opt.out_dir}/res_images/image-%06d.jpg -vcodec mpeg4 -b:v {bitrate} ./{opt.out_dir}/res_{video_basename}.mp4")
os.system(f"ffmpeg -r 25 -i ./{opt.out_dir}/res_2d_images/image-%06d.jpg -vcodec mpeg4 -b:v {bitrate} ./{opt.out_dir}/res_2d_{video_basename}.mp4")
