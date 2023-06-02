import os
import torch
import math

import cv2
import numpy as np
import PIL.Image as pil_img


RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

SOFT_BLUE = (179, 205, 227)
SOFT_RED = (251, 180, 174)
SOFT_GREEN = (204, 235, 197)


def get_one_box(det_output, thrd=0.9):
    max_area = 0
    max_bbox = None

    if det_output['boxes'].shape[0] == 0 or thrd < 1e-5:
        return None

    for i in range(det_output['boxes'].shape[0]):
        bbox = det_output['boxes'][i]
        score = det_output['scores'][i]
        if float(score) < thrd:
            continue
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if float(area) > max_area:
            max_bbox = [float(x) for x in bbox]
            max_area = area

    if max_bbox is None:
        return get_one_box(det_output, thrd=thrd - 0.1)

    return max_bbox


def get_max_iou_box(det_output, prev_bbox, thrd=0.9):
    max_score = 0
    max_bbox = None
    for i in range(det_output['boxes'].shape[0]):
        bbox = det_output['boxes'][i]
        score = det_output['scores'][i]
        # if float(score) < thrd:
        #     continue
        # area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        iou = calc_iou(prev_bbox, bbox)
        iou_score = float(score) * iou
        if float(iou_score) > max_score:
            max_bbox = [float(x) for x in bbox]
            max_score = iou_score
    if max_bbox is None:
        max_bbox = prev_bbox

    return max_bbox


def calc_iou(bbox1, bbox2):
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    box1Area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    box2Area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
    
    iou = interArea / float(box1Area + box2Area - interArea)

    return iou


def vis_bbox(image, bbox):

    x1, y1, x2, y2 = bbox

    bbox_img = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
    bbox_img = pil_img.fromarray(bbox_img[:, :, :3].astype(np.uint8))

    return np.asarray(bbox_img)


def vis_2d(image, bbox, pts):

    x1, y1, x2, y2 = bbox

    image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (154, 201, 219), 5)

    for pt in pts:
        x, y = pt
        image = cv2.circle(image, (int(x), int(y)), 3, (255, 136, 132), 3)
    image = pil_img.fromarray(image[:, :, :3].astype(np.uint8))

    return np.asarray(image)


def vis_smpl_3d(pose_output, img, cam_root, f, c, renderer, color_id=0, cam_rt=np.zeros(3),
                cam_t=np.zeros(3), J_regressor_h36m=None):
    '''
    input theta_mats: np.ndarray (96, )
    input betas: np.ndarray (10, )
    input img: RGB Image array with value in [0, 1]
    input cam_root: np.ndarray (3, )
    input f: np.ndarray (2, )
    input c: np.ndarray (2, )
    '''

    vertices = pose_output.pred_vertices.detach().cpu().numpy().squeeze()
    # J_from_verts_h36m = vertices2joints(J_regressor_h36m, pose_output['vertices'].detach().cpu())

    # cam_for_render = np.hstack([f[0], c])

    # center = pose_output.joints[0][0].cpu().data.numpy()
    # vert_shifted = vertices - center + cam_root
    vert_shifted = vertices + cam_root
    vert_shifted = vert_shifted

    # Render results
    rend_img_overlay = renderer(
        vert_shifted, princpt=c, img=img, do_alpha=True, color_id=color_id, cam_rt=cam_rt, cam_t=cam_t)

    img = pil_img.fromarray(rend_img_overlay[:, :, :3].astype(np.uint8))
    # if len(filename) > 0:
    #     img.save(filename)

    return np.asarray(img)


def get_color_fast(idx):
    color_pool = [RED, GREEN, BLUE, CYAN, YELLOW, ORANGE, PURPLE, WHITE]
    color = color_pool[idx % 8]

    return color


def vis_frame(frame, im_res, tracking, format='coco', add_bbox=False, skeleton=None):
    '''
    frame: frame image
    im_res: im_res of predictions
    format: coco or mpii
    return rendered image
    '''
    # print(im_res)
    kp_num = 17
    '''
    if len(im_res) > 0:
        kp_num = len(im_res[0]['keypoints'])
    '''
    if kp_num == 17:
        if format == 'coco':
            l_pair = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (17, 11), (17, 12),  # Body
                (11, 13), (12, 14), (13, 15), (14, 16)
            ]

            p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                       (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                       (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
            line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                          (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                          (77, 222, 255), (255, 156, 127),
                          (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]
        elif format == 'mpii':
            '''
            l_pair = [
                (8, 9), (11, 12), (11, 10), (2, 1), (1, 0),
                (13, 14), (14, 15), (3, 4), (4, 5),
                (8, 7), (7, 6), (6, 2), (6, 3), (8, 12), (8, 13)
            ]
            '''
            l_pair = (
                (3, 6), (4, 3), (5, 4),
                (2, 6), (1, 2), (0, 1),
                (7, 6), (8, 7), (9, 8),
                (13, 7), (14, 13), (15, 14),
                (12, 7), (11, 12), (10, 11))
            '''
            p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
            line_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
            '''
            p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                       (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                       (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
            line_color = [
                (77, 222, 255), (0, 127, 255), (0, 77, 255),
                (255, 156, 127), (255, 127, 77), (255, 77, 36),
                (0, 215, 255), (0, 255, 204), (0, 134, 255),
                (77, 255, 222), (77, 196, 255), (77, 135, 255),
                (0, 255, 50), (191, 255, 77), (77, 255, 77), (77, 255, 77), (77, 255, 77)]
        elif format == 'h36m':
            l_pair = (
                (0, 1), (0, 2),
                (1, 4), (2, 5), (4, 7), (5, 8),  # leg
                (0, 3), (3, 6), (6, 9),  # spine
                (12, 13), (12, 14),
                (13, 16), (14, 17),
                (16, 18), (18, 20), (17, 19), (19, 21))  # arm
            p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                       (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                       (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255), (0, 255, 255), (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
            line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                          (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                          (77, 222, 255), (255, 156, 127),
                          (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36), (0, 77, 255), (255, 77, 36), (255, 77, 36)]
        else:
            raise NotImplementedError

    if skeleton is not None:
        l_pair = skeleton

    img = frame.copy()
    height, width = img.shape[:2]
    for human in im_res:
        part_line = {}
        keypoints = torch.tensor(human['keypoints']).reshape(-1, 3)
        # assert keypoints.shape[0] == 17, keypoints.shape
        # print(keypoints)
        kp_preds = keypoints[:, :2]
        kp_scores = keypoints[:, 2:3]
        # if kp_num == 17:
        #     kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5, :] + kp_preds[6, :]) / 2, 0)))
        #     kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[5, :] + kp_scores[6, :]) / 2, 0)))
        if tracking:
            color = get_color_fast(int(abs(human['image_id'])))
        else:
            color = BLUE

        # Draw keypoints
        for n in range(kp_scores.shape[0]):
            if kp_scores[n] <= 0.2:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (int(cor_x), int(cor_y))
            bg = img.copy()
            if n < len(p_color):
                if tracking:
                    cv2.circle(bg, (int(cor_x), int(cor_y)), 2, color, -1)
                else:
                    cv2.circle(bg, (int(cor_x), int(cor_y)), 2, p_color[n], -1)
            else:
                if n == 70:
                    cv2.circle(bg, (int(cor_x), int(cor_y)), 1, (255, 136, 132), 2)
                else:
                    cv2.circle(bg, (int(cor_x), int(cor_y)), 1, (255, 255, 255), 2)
            # Now create a mask of logo and create its inverse mask also
            transparency = float(max(0, min(1, kp_scores[n])))
            img = cv2.addWeighted(bg, transparency, img, 1 - transparency, 0)

        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                bg = img.copy()

                X = (start_xy[0], end_xy[0])
                Y = (start_xy[1], end_xy[1])
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                # stickwidth = (kp_scores[start_p] + kp_scores[end_p])+5
                stickwidth = np.sqrt(height * width) / 150
                polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), int(stickwidth)), int(angle), 0, 360, 1)
                if i < len(line_color):
                    if tracking:
                        cv2.fillConvexPoly(bg, polygon, color)
                    else:
                        cv2.fillConvexPoly(bg, polygon, line_color[i])
                else:
                    cv2.line(bg, start_xy, end_xy, (255, 255, 255), 1)
                # transparency = float(max(0, min(1, 0.5 * (kp_scores[start_p] + kp_scores[end_p]))))
                transparency = float(max(0, min(1, min(kp_scores[start_p], kp_scores[end_p]))))
                img = cv2.addWeighted(bg, transparency, img, 1 - transparency, 0)

        if add_bbox:
            xmin, ymin, xmax, ymax = human['bbox']
            xmin, ymin, xmax, ymax = int(xmin + 0.5), int(ymin + 0.5), int(xmax + 0.5), int(ymax + 0.5)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (180, 119, 31), 2)

    return img


def vis_uvd_trivial(uvd_data, uvd_weight=None, imgs=None, idx=0, dataset_name='3dhp', rescale=False, saved_path=None):
    batch = uvd_data.shape[0]
    if rescale:
        imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
    else:
        imgs = imgs.cpu().numpy()

    if uvd_weight is None:
        uvd_weight = np.ones_like(uvd_data)

    if rescale:
        imgs[:, :, :, 0] *= 0.225
        imgs[:, :, :, 1] *= 0.224
        imgs[:, :, :, 2] *= 0.229

        imgs[:, :, :, 0] += 0.406
        imgs[:, :, :, 1] += 0.457
        imgs[:, :, :, 2] += 0.480

    imgs = (imgs * 255).astype(np.uint8)

    image_shape = np.array([imgs.shape[2], imgs.shape[1], 1])

    for i in range(batch):
        # item: [71, 2]
        # uvd_data: [B, 71, 3]
        root_place = uvd_data[i, 0]
        weight = uvd_weight[i].sum(axis=-1) > 0
        uvd_data[i, weight < 1, :] = root_place
        item = uvd_data[i][:, :2]
        # last_one_dim = np.ones((item.shape[0], 1)) * 0.5
        weight = weight * 1.0
        # print(weight.shape, item.shape, uvd_data.shape, uvd_weight.shape)
        item = (item + 0.5) * image_shape[None, :2]
        item = np.concatenate([item, weight[:, None]], axis=-1)
        img = imgs[i]
        item_dict = [{
            'keypoints': item
        }]
        # print((item+0.5) * image_shape)
        new_img = vis_frame(img[:, :, ::-1], item_dict, tracking=False, format='h36m')

        if saved_path is None:
            pa_path = './exp/visualize/demo/'
        else:
            pa_path = saved_path[i].split('/')[:-1]
            pa_path = '/'.join(pa_path)

        if not os.path.exists(pa_path):
            os.makedirs(pa_path)

        # cv2.imwrite(pa_path + str(idx*batch+i) + f'_{dataset_name}2.jpg', new_img)
        cv2.imwrite(saved_path[i], new_img)
