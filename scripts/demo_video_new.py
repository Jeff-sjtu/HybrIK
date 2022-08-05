"""Image demo script."""
import argparse
import os

import cv2
import joblib
import numpy as np
import torch
from easydict import EasyDict as edict
from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPL
from hybrik.utils.render_pytorch3d import render_mesh
from hybrik.utils.vis import get_max_iou_box, get_one_box, vis_2d
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
    # probe = ffmpeg.probe(in_file)
    # video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    # if video_stream is None:
    #     print('No video stream found', file=sys.stderr)
    #     sys.exit(1)
    # return video_stream


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


parser = argparse.ArgumentParser(description='HybrIK Demo')
CKPT = 'pretrained_w_cam.pth'

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
parser.add_argument('--save-pt', default=False, dest='save_pt',
                    help='save prediction', action='store_true')
parser.add_argument('--not-vis', default=False, dest='not_vis',
                    help='do not visualize', action='store_true')

opt = parser.parse_args()


cfg_file = 'configs/256x192_adam_lr1e-3-res34_smpl_3d_cam_2x_mix.yaml'
cfg = update_config(cfg_file)

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
    'pred_xyz_17',
    'pred_xyz_29',
    'pred_xyz_24_struct',
    'pred_scores',
    'pred_camera',
    'f',
    'pred_betas',
    'pred_thetas',
    'pred_phi',
    'scale_mult',
    'pred_cam_root',
    # 'features',
    'bbox',
    'height',
    'width',
    'img_path'
]
res_db = {k: [] for k in res_keys}

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
hybrik_model.load_state_dict(torch.load(
    CKPT, map_location='cpu'), strict=False)

det_model.cuda(opt.gpu)
hybrik_model.cuda(opt.gpu)
det_model.eval()
hybrik_model.eval()

if not os.path.exists(opt.out_dir):
    os.makedirs(opt.out_dir)
if not os.path.exists(os.path.join(opt.out_dir, 'raw_images')):
    os.makedirs(os.path.join(opt.out_dir, 'raw_images'))
if not os.path.exists(os.path.join(opt.out_dir, 'res_images')):
    os.makedirs(os.path.join(opt.out_dir, 'res_images'))
if not os.path.exists(os.path.join(opt.out_dir, 'res_2d_images')):
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

os.system(
    f'ffmpeg -i {opt.video_name} {opt.out_dir}/raw_images/{video_basename}-%06d.png')

files = os.listdir(f'{opt.out_dir}/raw_images')
files.sort()

img_path_list = []

for file in tqdm(files):
    if not os.path.isdir(file) and file[-4:] in ['.jpg', '.png']:

        img_path = os.path.join(opt.out_dir, 'raw_images', file)
        img_path_list.append(img_path)

prev_box = None
renderer = None
smpl_faces = torch.from_numpy(hybrik_model.smpl.faces.astype(np.int32))

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
        # bbox: [x1, y1, x2, y2]
        pose_input, bbox = transformation.test_transform(
            input_image, tight_bbox)
        pose_input = pose_input.to(opt.gpu)[None, :, :, :]
        pose_output = hybrik_model(pose_input)
        uv_29 = pose_output.pred_uvd_jts.reshape(29, 3)[:, :2]

        if not opt.not_vis:
            # Visualization
            image = input_image.copy()
            # img_size = (image.shape[0], image.shape[1])
            # focal = np.array([1000, 1000])
            focal = 1000.0
            bbox_xywh = xyxy2xywh(bbox)
            princpt = [bbox_xywh[0], bbox_xywh[1]]

            # if renderer is None:
            #     renderer = SMPLRenderer(faces=hybrik_model.smpl.faces,
            #                             img_size=img_size, focal=focal,
            #                             princpt=princpt)

            transl = pose_output.transl.detach()
            # transl[:, 2] = transl[:, 2] * 256 / bbox_xywh[2]
            focal = focal / 256 * bbox_xywh[2]

            px = bbox_xywh[0] - image.shape[1] / 2
            py = bbox_xywh[1] - image.shape[0] / 2
            transl[:, 0] = transl[:, 0] + px * transl[:, 2] / focal
            transl[:, 1] = transl[:, 1] + py * transl[:, 2] / focal

            # image_vis = vis_smpl_3d(
            #     pose_output, image, cam_root=transl,
            #     f=focal, c=princpt, renderer=renderer)

            vertices = pose_output.pred_vertices.detach()

            # image_batch = torch.stack(render_queue['image'], dim=0)
            verts_batch = vertices
            transl_batch = transl

            color_batch = render_mesh(
                vertices=verts_batch, faces=smpl_faces,
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

            write_stream.write(image_vis)

            # idx += 1
            # res_path = os.path.join(opt.out_dir, 'res_images', f'image-{idx:06d}.jpg')
            # cv2.imwrite(res_path, image_vis)

            # color = render_mesh(
            #     vertices=vertices, faces=smpl_faces,
            #     translation=transl,
            #     focal_length=focal, height=image.shape[0], width=image.shape[1])

            # valid_mask = (color[:, :, -1] > 0)[:, :, None]
            # image_vis = color[:, :, :3] * valid_mask
            # image_vis = (image_vis * 255).cpu().numpy().astype(np.uint8)

            # print(image_vis.shape)
            # write_stream.write(image_vis)

            # image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)
            # # cv2.imshow('image_vis', image_vis)
            # # cv2.waitKey(0)
            # # cv2.destroyAllWindows()
            # idx += 1
            # res_path = os.path.join(opt.out_dir, 'res_images', f'image-{idx:06d}.jpg')
            # cv2.imwrite(res_path, image_vis)

            # vis 2d
            pts = uv_29 * bbox_xywh[2]
            pts[:, 0] = pts[:, 0] + bbox_xywh[0]
            pts[:, 1] = pts[:, 1] + bbox_xywh[1]
            image = input_image.copy()
            bbox_img = vis_2d(image, tight_bbox, pts)
            bbox_img = cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR)
            res_path = os.path.join(
                opt.out_dir, 'res_2d_images', f'image-{idx:06d}.jpg')
            # cv2.imwrite(res_path, bbox_img)
            write2d_stream.write(bbox_img)

        if opt.save_pt:
            assert pose_input.shape[0] == 1, 'Only support single batch inference for now'

            pred_xyz_jts_17 = pose_output.pred_xyz_jts_17.reshape(
                17, 3).cpu().data.numpy()
            pred_uvd_jts = pose_output.pred_uvd_jts.reshape(
                -1, 3).cpu().data.numpy()
            pred_xyz_jts_29 = pose_output.pred_xyz_jts_29.reshape(
                -1, 3).cpu().data.numpy()
            pred_xyz_jts_24_struct = pose_output.pred_xyz_jts_24_struct.reshape(
                24, 3).cpu().data.numpy()
            pred_scores = pose_output.maxvals.cpu(
            ).data[:, :29].reshape(29).numpy()
            pred_camera = pose_output.pred_camera.squeeze(
                dim=0).cpu().data.numpy()
            pred_betas = pose_output.pred_shape.squeeze(
                dim=0).cpu().data.numpy()
            pred_theta = pose_output.pred_theta_mats.squeeze(
                dim=0).cpu().data.numpy()
            pred_phi = pose_output.pred_phi.squeeze(dim=0).cpu().data.numpy()
            pred_cam_root = pose_output.cam_root.squeeze(dim=0).cpu().numpy()
            img_size = np.array((input_image.shape[0], input_image.shape[1]))

            res_db['pred_xyz_17'].append(pred_xyz_jts_17)
            res_db['pred_uvd'].append(pred_uvd_jts)
            res_db['pred_xyz_29'].append(pred_xyz_jts_29)
            res_db['pred_xyz_24_struct'].append(pred_xyz_jts_24_struct)
            res_db['pred_scores'].append(pred_scores)
            res_db['pred_camera'].append(pred_camera)
            res_db['f'].append(1000.0)
            res_db['pred_betas'].append(pred_betas)
            res_db['pred_thetas'].append(pred_theta)
            res_db['pred_phi'].append(pred_phi)
            res_db['pred_cam_root'].append(pred_cam_root)
            # res_db['features'].append(img_feat)
            res_db['bbox'].append(np.array(bbox))
            res_db['height'].append(img_size[0])
            res_db['width'].append(img_size[1])
            res_db['img_path'].append(img_path)

write_stream.release()
write2d_stream.release()

# if not opt.not_vis:
#     os.system(f"ffmpeg -r 25 -i ./{opt.out_dir}/res_images/image-%06d.jpg -vcodec mpeg4 -b:v {bitrate} ./{opt.out_dir}/res_{video_basename}.mp4")
#     os.system(f"ffmpeg -r 25 -i ./{opt.out_dir}/res_2d_images/image-%06d.jpg -vcodec mpeg4 -b:v {bitrate} ./{opt.out_dir}/res_2d_{video_basename}.mp4")

if opt.save_pt:

    for k in res_db.keys():
        try:
            v = np.stack(res_db[k], axis=0)
        except Exception:
            v = res_db[k]
            print(k, ' failed')

        res_db[k] = v

    res_db_path = os.path.join(f'./{opt.out_dir}/', f'{video_basename}.pt')
    joblib.dump(res_db, res_db_path)
    print('Prediction is saved in:', res_db_path)
