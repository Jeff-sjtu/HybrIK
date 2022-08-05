import cv2
import numpy as np
import PIL.Image as pil_img


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
