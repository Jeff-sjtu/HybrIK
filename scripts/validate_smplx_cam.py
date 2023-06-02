"""Validation script."""
import argparse
import os
import pickle as pk
import sys

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from hybrik.datasets import AGORAX
from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.env import init_dist
from hybrik.utils.metrics import NullWriter
from hybrik.utils.transforms import get_func_heatmap_to_coord
from hybrik.utils.vis import vis_uvd_trivial


parser = argparse.ArgumentParser(description='HybrIK Validate')
parser.add_argument('--cfg',
                    help='experiment configure file name',
                    required=True,
                    type=str)
parser.add_argument('--checkpoint',
                    help='checkpoint file name',
                    required=True,
                    type=str)
parser.add_argument('--gpus',
                    help='gpus',
                    type=str)
parser.add_argument('--batch',
                    help='validation batch size',
                    type=int)
parser.add_argument('--flip-test',
                    default=False,
                    dest='flip_test',
                    help='flip test',
                    action='store_true')
parser.add_argument('--flip-shift',
                    default=False,
                    dest='flip_shift',
                    help='flip shift',
                    action='store_true')
parser.add_argument('--visualize',
                    default=False,
                    dest='visualize',
                    help='visualize predictions',
                    action='store_true')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed testing')
parser.add_argument('--dist-url', default='tcp://192.168.1.219:23456', type=str,
                    help='url used to set up distributed testing')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
                    help='job launcher')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed testing')

opt = parser.parse_args()
cfg = update_config(opt.cfg)

cfg['MODEL']['EXTRA']['USE_KID'] = cfg['DATASET'].get('USE_KID', False)
cfg['LOSS']['ELEMENTS']['USE_KID'] = cfg['DATASET'].get('USE_KID', False)

gpus = [int(i) for i in opt.gpus.split(',')]

norm_method = cfg.LOSS.get('norm', 'softmax')

if opt.visualize:
    from hybrik.utils.render_pytorch3d import render_mesh


def validate_gt(m, opt, cfg, gt_val_dataset, heatmap_to_coord, batch_size=32, pred_root=False, test_vertices=False):

    gt_val_sampler = torch.utils.data.distributed.DistributedSampler(
        gt_val_dataset, num_replicas=opt.world_size, rank=opt.rank)

    gt_val_loader = torch.utils.data.DataLoader(
        gt_val_dataset, batch_size=batch_size, shuffle=False, num_workers=5, drop_last=False, sampler=gt_val_sampler)
    kpt_pred = {}
    m.eval()

    hm_shape = cfg.MODEL.get('HEATMAP_SIZE')
    hm_shape = (hm_shape[1], hm_shape[0])
    smplx_faces = torch.from_numpy(m.module.smplx_layer.faces.astype(np.int32))
    # pve_list = []

    for inps, labels, img_ids, bboxes in tqdm(gt_val_loader, dynamic_ncols=True):
        if opt.visualize:
            img_ids, img_paths = img_ids
        if isinstance(inps, list):
            inps = [inp.cuda(opt.gpu) for inp in inps]
        else:
            inps = inps.cuda(opt.gpu)

        for k, _ in labels.items():
            try:
                labels[k] = labels[k].cuda(opt.gpu)
            except AttributeError:
                assert k == 'type'

        # output = m(inps, trans_inv, intrinsic_param, joint_root, depth_factor, (gt_betas, None, None))
        output = m(
            inps, flip_test=opt.flip_test, bboxes=bboxes,
            img_center=labels['img_center'],
        )

        pred_xyz_hybrik = output.pred_xyz_hybrik.reshape(inps.shape[0], -1, 3)
        pred_xyz_hybrik_struct = output.pred_xyz_hybrik_struct.reshape(inps.shape[0], -1, 3)
        pred_uvd_jts = output.pred_uvd_jts.reshape(inps.shape[0], -1, 3)
        # pred_camera = output.pred_camera

        pred_xyz_hybrik = pred_xyz_hybrik.cpu().data.numpy()
        pred_uvd_jts = pred_uvd_jts.cpu().data.numpy()
        pred_xyz_hybrik_struct = pred_xyz_hybrik_struct.cpu().data.numpy()

        pve = np.zeros(pred_xyz_hybrik.shape[0])
        if test_vertices:
            # gt_mesh = output.gt_output.vertices.cpu().numpy()
            gt_mesh = labels['target_vertices'].cpu().numpy()
            pred_mesh = output.pred_vertices.cpu().numpy()
            pve = np.sqrt(np.sum((pred_mesh - gt_mesh) ** 2, 2))
            pve = pve.reshape(pred_mesh.shape[0], -1).mean(axis=-1)
            # pve_list.append(np.mean(pve) * 1000)

        for i in range(pred_xyz_hybrik.shape[0]):
            # bbox = bboxes[i].tolist()
            kpt_pred[int(img_ids[i])] = {
                'xyz_hybrik': pred_xyz_hybrik[i],
                'xyz_hybrik_struct': pred_xyz_hybrik_struct[i],
                'pve': pve[i]
            }

        if opt.visualize:
            visualize(inps, output, img_paths, bboxes, smplx_faces)

    with open(os.path.join('exp', f'test_gt_kpt_rank_{opt.rank}.pkl'), 'wb') as fid:
        pk.dump(kpt_pred, fid, pk.HIGHEST_PROTOCOL)

    torch.distributed.barrier()  # Make sure all JSON files are saved

    if opt.rank == 0:
        kpt_all_pred = {}
        for r in range(opt.world_size):
            with open(os.path.join('exp', f'test_gt_kpt_rank_{r}.pkl'), 'rb') as fid:
                kpt_pred = pk.load(fid)

            os.remove(os.path.join('exp', f'test_gt_kpt_rank_{r}.pkl'))

            kpt_all_pred.update(kpt_pred)

        tot_err_70, eval_summary = gt_val_dataset.evaluate_xyz_hybrik(
            kpt_all_pred, os.path.join('exp', 'test_3d_kpt.json'))

        tot_err_70_struct, eval_summary_struct = gt_val_dataset.evaluate_xyz_hybrik(
            kpt_all_pred, os.path.join('exp', 'test_3d_kpt.json'), use_struct=True)

        mve = 0
        if test_vertices:
            # print(f'PVE: {np.mean(pve_list)}')
            pve_list = [item['pve'] for _, item in kpt_all_pred.items()]
            # print(f'PVE: {np.mean(pve_list)}')
            mve = np.mean(pve_list)
            print(f'PVE: {mve}')

        eval_summaries = [eval_summary, eval_summary_struct, mve]

        return tot_err_70_struct, eval_summaries
    else:
        return None, None


def main():
    if opt.launcher == 'slurm':
        main_worker(None, opt, cfg)
    else:
        ngpus_per_node = torch.cuda.device_count()
        opt.ngpus_per_node = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(opt, cfg))


def main_worker(gpu, opt, cfg):

    if gpu is not None:
        opt.gpu = gpu

    init_dist(opt)

    if not opt.log:
        null_writer = NullWriter()
        sys.stdout = null_writer

    torch.backends.cudnn.benchmark = True

    m = builder.build_sppe(cfg.MODEL)
    old_children_map = m.smplx_layer.children_map.clone()

    print(f'Loading model from {opt.checkpoint}...')
    save_dict = torch.load(opt.checkpoint, map_location='cpu')
    if type(save_dict) == dict:
        model_dict = save_dict['model']
        m.load_state_dict(model_dict, strict=False)
    else:
        m.load_state_dict(save_dict, strict=False)

    m.cuda(opt.gpu)
    m.smplx_layer.children_map = old_children_map.to(m.smplx_layer.children_map.device)

    m = torch.nn.parallel.DistributedDataParallel(m, device_ids=[opt.gpu])

    heatmap_to_coord = get_func_heatmap_to_coord(cfg)

    # gt_val_dataset_hp3d = HP3D(
    #     cfg=cfg,
    #     ann_file='test',
    #     train=False)

    gt_val_dataset_agora = AGORAX(
        cfg=cfg,
        ann_file=cfg.DATASET.SET_LIST[0].TEST_SET,
        train=False,
        high_res_inp=False,
        return_vertices=True,
        return_img_path=opt.visualize)

    print('##### Testing on AGORA #####')
    with torch.no_grad():
        gt_tot_err = validate_gt(m, opt, cfg, gt_val_dataset_agora, heatmap_to_coord, opt.batch, test_vertices=True)
    print(f'##### gt agora err: {gt_tot_err} #####')


def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]


def visualize(inps, output, img_paths, bboxes, smplx_faces):
    f = 1000.0

    pred_vertices = output.pred_vertices.detach()
    transl = output.transl.detach()
    for bs in range(transl.shape[0]):
        img_path = img_paths[bs]
        input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        bbox = bboxes[bs]
        bbox_xywh = xyxy2xywh(bbox)

        focal = f / 256 * bbox_xywh[2]
        verts_batch = pred_vertices[[bs]]
        transl_batch = transl[[bs]]

        color_batch = render_mesh(
            vertices=verts_batch, faces=smplx_faces,
            translation=transl_batch,
            focal_length=focal, height=input_image.shape[0], width=input_image.shape[1])

        valid_mask_batch = (color_batch[:, :, :, [-1]] > 0)
        image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
        image_vis_batch = (image_vis_batch * 255).cpu().numpy()

        color = image_vis_batch[0]
        valid_mask = valid_mask_batch[0].cpu().numpy()

        input_img = input_image
        alpha = 0.9
        image_vis = alpha * color[:, :, :3] * valid_mask + (
            1 - alpha) * input_img * valid_mask + (1 - valid_mask) * input_img

        image_vis = image_vis.astype(np.uint8)
        image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)

        x1, y1, x2, y2 = bbox
        image_vis = cv2.rectangle(image_vis, (int(x1), int(y1)), (int(x2), int(y2)), (154, 201, 219), 5)

        # cv2.imwrite(f'exp/visualize/vis_smpl/{str(idx*batch+i)}.jpg', image_vis)
        save_path = 'exp/visualize/agora_smplx_infer/' + os.path.basename(img_path)
        save_path = save_path.replace('.png', f'_{bs}.png')
        pa_path = save_path.split('/')[:-1]
        if not os.path.exists('/'.join(pa_path)):
            os.makedirs('/'.join(pa_path))

        cv2.imwrite(save_path, image_vis)

        vis_uvd_trivial(
            output.pred_uvd_jts[[bs]].reshape(1, -1, 3).cpu().numpy(),
            # output.pred_uv_struct[[bs]].reshape(1, -1, 2).cpu().numpy(),
            imgs=inps[[bs]].cpu(),
            saved_path=[save_path.replace('.png', '_uv2.png')],
            rescale=True)


if __name__ == "__main__":
    main()
