"""Pose related transforrmation functions."""

import random

import cv2
import numpy as np
import torch
from torch.nn import functional as F


def rnd(x):
    return max(-2 * x, min(2 * x, np.random.randn(1)[0] * x))


def box_transform(bbox, sf, imgwidth, imght, train):
    """Random scaling."""
    width = bbox[2] - bbox[0]
    ht = bbox[3] - bbox[1]
    if train:
        scaleRate = 0.25 * np.clip(np.random.randn() * sf, - sf, sf)

        bbox[0] = max(0, bbox[0] - width * scaleRate / 2)
        bbox[1] = max(0, bbox[1] - ht * scaleRate / 2)
        bbox[2] = min(imgwidth, bbox[2] + width * scaleRate / 2)
        bbox[3] = min(imght, bbox[3] + ht * scaleRate / 2)
    else:
        scaleRate = 0.25

        bbox[0] = max(0, bbox[0] - width * scaleRate / 2)
        bbox[1] = max(0, bbox[1] - ht * scaleRate / 2)
        bbox[2] = min(imgwidth, max(
            bbox[2] + width * scaleRate / 2, bbox[0] + 5))
        bbox[3] = min(imght, max(bbox[3] + ht * scaleRate / 2, bbox[1] + 5))

    return bbox


def addDPG(bbox, imgwidth, imght):
    """Add dpg for data augmentation, including random crop and random sample."""
    PatchScale = random.uniform(0, 1)
    width = bbox[2] - bbox[0]
    ht = bbox[3] - bbox[1]

    if PatchScale > 0.85:
        ratio = ht / width
        if (width < ht):
            patchWidth = PatchScale * width
            patchHt = patchWidth * ratio
        else:
            patchHt = PatchScale * ht
            patchWidth = patchHt / ratio

        xmin = bbox[0] + random.uniform(0, 1) * (width - patchWidth)
        ymin = bbox[1] + random.uniform(0, 1) * (ht - patchHt)
        xmax = xmin + patchWidth + 1
        ymax = ymin + patchHt + 1
    else:
        xmin = max(
            1, min(bbox[0] + np.random.normal(-0.0142, 0.1158) * width, imgwidth - 3))
        ymin = max(
            1, min(bbox[1] + np.random.normal(0.0043, 0.068) * ht, imght - 3))
        xmax = min(
            max(xmin + 2, bbox[2] + np.random.normal(0.0154, 0.1337) * width), imgwidth - 3)
        ymax = min(
            max(ymin + 2, bbox[3] + np.random.normal(-0.0013, 0.0711) * ht), imght - 3)

    bbox[0] = xmin
    bbox[1] = ymin
    bbox[2] = xmax
    bbox[3] = ymax

    return bbox


def im_to_torch(img):
    """Transform ndarray image to torch tensor.

    Parameters
    ----------
    img: numpy.ndarray
        An ndarray with shape: `(H, W, 3)`.

    Returns
    -------
    torch.Tensor
        A tensor with shape: `(3, H, W)`.

    """
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img


def torch_to_im(img):
    """Transform torch tensor to ndarray image.

    Parameters
    ----------
    img: torch.Tensor
        A tensor with shape: `(3, H, W)`.

    Returns
    -------
    numpy.ndarray
        An ndarray with shape: `(H, W, 3)`.

    """
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0))  # C*H*W
    return img


def load_image(img_path):
    # H x W x C => C x H x W
    return im_to_torch(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))


def to_numpy(tensor):
    # torch.Tensor => numpy.ndarray
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    # numpy.ndarray => torch.Tensor
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def cv_cropBox(img, bbox, input_size):
    """Crop bbox from image by Affinetransform.

    Parameters
    ----------
    img: torch.Tensor
        A tensor with shape: `(3, H, W)`.
    bbox: list or tuple
        [xmin, ymin, xmax, ymax].
    input_size: tuple
        Resulting image size, as (height, width).

    Returns
    -------
    torch.Tensor
        A tensor with shape: `(3, height, width)`.

    """
    xmin, ymin, xmax, ymax = bbox
    xmax -= 1
    ymax -= 1
    resH, resW = input_size

    lenH = max((ymax - ymin), (xmax - xmin) * resH / resW)
    lenW = lenH * resW / resH
    if img.dim() == 2:
        img = img[np.newaxis, :, :]

    box_shape = [ymax - ymin, xmax - xmin]
    pad_size = [(lenH - box_shape[0]) // 2, (lenW - box_shape[1]) // 2]
    # Padding Zeros
    img[:, :ymin, :], img[:, :, :xmin] = 0, 0
    img[:, ymax + 1:, :], img[:, :, xmax + 1:] = 0, 0

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = np.array([xmin - pad_size[1], ymin - pad_size[0]], np.float32)
    src[1, :] = np.array([xmax + pad_size[1], ymax + pad_size[0]], np.float32)
    dst[0, :] = 0
    dst[1, :] = np.array([resW - 1, resH - 1], np.float32)

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    dst_img = cv2.warpAffine(torch_to_im(img), trans,
                             (resW, resH), flags=cv2.INTER_LINEAR)
    if dst_img.ndim == 2:
        dst_img = dst_img[:, :, np.newaxis]

    return im_to_torch(torch.Tensor(dst_img))


def cv_cropBox_rot(img, bbox, input_size, rot):
    """Crop bbox from image by Affinetransform.

    Parameters
    ----------
    img: torch.Tensor
        A tensor with shape: `(3, H, W)`.
    bbox: list or tuple
        [xmin, ymin, xmax, ymax].
    input_size: tuple
        Resulting image size, as (height, width).

    Returns
    -------
    torch.Tensor
        A tensor with shape: `(3, height, width)`.

    """
    xmin, ymin, xmax, ymax = bbox
    xmax -= 1
    ymax -= 1
    resH, resW = input_size
    rot_rad = np.pi * rot / 180

    if img.dim() == 2:
        img = img[np.newaxis, :, :]

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    center = np.array([(xmax + xmin) / 2, (ymax + ymin) / 2])

    src_dir = get_dir([0, (ymax - ymin) * -0.5], rot_rad)
    dst_dir = np.array([0, (resH - 1) * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [(resW - 1) * 0.5, (resH - 1) * 0.5]
    dst[1, :] = np.array([(resW - 1) * 0.5, (resH - 1) * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    dst_img = cv2.warpAffine(torch_to_im(img), trans,
                             (resW, resH), flags=cv2.INTER_LINEAR)
    if dst_img.ndim == 2:
        dst_img = dst_img[:, :, np.newaxis]

    return im_to_torch(torch.Tensor(dst_img))


def fix_cropBox(img, bbox, input_size):
    """Crop bbox from image by Affinetransform.

    Parameters
    ----------
    img: torch.Tensor
        A tensor with shape: `(3, H, W)`.
    bbox: list or tuple
        [xmin, ymin, xmax, ymax].
    input_size: tuple
        Resulting image size, as (height, width).

    Returns
    -------
    torch.Tensor
        A tensor with shape: `(3, height, width)`.

    """
    xmin, ymin, xmax, ymax = bbox
    input_ratio = input_size[0] / input_size[1]
    bbox_ratio = (ymax - ymin) / (xmax - xmin)
    if bbox_ratio > input_ratio:
        # expand width
        cx = (xmax + xmin) / 2
        h = ymax - ymin
        w = h / input_ratio
        xmin = cx - w / 2
        xmax = cx + w / 2
    elif bbox_ratio < input_ratio:
        # expand height
        cy = (ymax + ymin) / 2
        w = xmax - xmin
        h = w * input_ratio
        ymin = cy - h / 2
        ymax = cy + h / 2
    bbox = [int(x) for x in [xmin, ymin, xmax, ymax]]

    return cv_cropBox(img, bbox, input_size), bbox


def fix_cropBox_rot(img, bbox, input_size, rot):
    """Crop bbox from image by Affinetransform.

    Parameters
    ----------
    img: torch.Tensor
        A tensor with shape: `(3, H, W)`.
    bbox: list or tuple
        [xmin, ymin, xmax, ymax].
    input_size: tuple
        Resulting image size, as (height, width).

    Returns
    -------
    torch.Tensor
        A tensor with shape: `(3, height, width)`.

    """
    xmin, ymin, xmax, ymax = bbox
    input_ratio = input_size[0] / input_size[1]
    bbox_ratio = (ymax - ymin) / (xmax - xmin)
    if bbox_ratio > input_ratio:
        # expand width
        cx = (xmax + xmin) / 2
        h = ymax - ymin
        w = h / input_ratio
        xmin = cx - w / 2
        xmax = cx + w / 2
    elif bbox_ratio < input_ratio:
        # expand height
        cy = (ymax + ymin) / 2
        w = xmax - xmin
        h = w * input_ratio
        ymin = cy - h / 2
        ymax = cy + h / 2
    bbox = [int(x) for x in [xmin, ymin, xmax, ymax]]

    return cv_cropBox_rot(img, bbox, input_size, rot), bbox


def get_3rd_point(a, b):
    """Return vector c that perpendicular to (a - b)."""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    """Rotate the point by `rot_rad` degree."""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def cv_cropBoxInverse(inp, bbox, img_size, output_size):
    """Paste the cropped bbox to the original image.

    Parameters
    ----------
    inp: torch.Tensor
        A tensor with shape: `(3, height, width)`.
    bbox: list or tuple
        [xmin, ymin, xmax, ymax].
    img_size: tuple
        Original image size, as (img_H, img_W).
    output_size: tuple
        Cropped input size, as (height, width).
    Returns
    -------
    torch.Tensor
        A tensor with shape: `(3, img_H, img_W)`.

    """
    xmin, ymin, xmax, ymax = bbox
    xmax -= 1
    ymax -= 1
    resH, resW = output_size
    imgH, imgW = img_size

    lenH = max((ymax - ymin), (xmax - xmin) * resH / resW)
    lenW = lenH * resW / resH
    if inp.dim() == 2:
        inp = inp[np.newaxis, :, :]

    box_shape = [ymax - ymin, xmax - xmin]
    pad_size = [(lenH - box_shape[0]) // 2, (lenW - box_shape[1]) // 2]

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = 0
    src[1, :] = np.array([resW - 1, resH - 1], np.float32)
    dst[0, :] = np.array([xmin - pad_size[1], ymin - pad_size[0]], np.float32)
    dst[1, :] = np.array([xmax + pad_size[1], ymax + pad_size[0]], np.float32)

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    dst_img = cv2.warpAffine(torch_to_im(inp), trans,
                             (imgW, imgH), flags=cv2.INTER_LINEAR)
    if dst_img.ndim == 3 and dst_img.shape[2] == 1:
        dst_img = dst_img[:, :, 0]
        return dst_img
    elif dst_img.ndim == 2:
        return dst_img
    else:
        return im_to_torch(torch.Tensor(dst_img))


def cv_rotate(img, rot, input_size):
    """Rotate image by Affinetransform.

    Parameters
    ----------
    img: torch.Tensor
        A tensor with shape: `(3, H, W)`.
    rot: int
        Rotation degree.
    input_size: tuple
        Resulting image size, as (height, width).

    Returns
    -------
    torch.Tensor
        A tensor with shape: `(3, height, width)`.

    """
    resH, resW = input_size
    center = np.array((resW - 1, resH - 1)) / 2
    rot_rad = np.pi * rot / 180

    src_dir = get_dir([0, (resH - 1) * -0.5], rot_rad)
    dst_dir = np.array([0, (resH - 1) * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [(resW - 1) * 0.5, (resH - 1) * 0.5]
    dst[1, :] = np.array([(resW - 1) * 0.5, (resH - 1) * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    dst_img = cv2.warpAffine(torch_to_im(img), trans,
                             (resW, resH), flags=cv2.INTER_LINEAR)
    if dst_img.ndim == 2:
        dst_img = dst_img[:, :, np.newaxis]

    return im_to_torch(torch.Tensor(dst_img))


def count_visible(bbox, joints_3d):
    """Count number of visible joints given bound box."""
    vis = np.logical_and.reduce((
        joints_3d[:, 0, 0] > 0,
        joints_3d[:, 0, 0] > bbox[0],
        joints_3d[:, 0, 0] < bbox[2],
        joints_3d[:, 1, 0] > 0,
        joints_3d[:, 1, 0] > bbox[1],
        joints_3d[:, 1, 0] < bbox[3],
        joints_3d[:, 0, 1] > 0,
        joints_3d[:, 1, 1] > 0
    ))
    return np.sum(vis), vis


def drawGaussian(img, pt, sigma):
    """Draw 2d gaussian on input image.

    Parameters
    ----------
    img: torch.Tensor
        A tensor with shape: `(3, H, W)`.
    pt: list or tuple
        A point: (x, y).
    sigma: int
        Sigma of gaussian distribution.

    Returns
    -------
    torch.Tensor
        A tensor with shape: `(3, H, W)`.

    """
    img = to_numpy(img)
    tmpSize = 3 * sigma
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - tmpSize), int(pt[1] - tmpSize)]
    br = [int(pt[0] + tmpSize + 1), int(pt[1] + tmpSize + 1)]

    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    size = 2 * tmpSize + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return to_torch(img)


def flip(x):
    assert (x.dim() == 3 or x.dim() == 4)
    dim = x.dim() - 1

    return x.flip(dims=(dim,))


def flip_heatmap(heatmap, joint_pairs, shift=False):
    """Flip pose heatmap according to joint pairs.

    Parameters
    ----------
    heatmap : numpy.ndarray
        Heatmap of joints.
    joint_pairs : list
        List of joint pairs.
    shift : bool
        Whether to shift the output.

    Returns
    -------
    numpy.ndarray
        Flipped heatmap.

    """
    assert (heatmap.dim() == 3 or heatmap.dim() == 4)
    out = flip(heatmap)

    for pair in joint_pairs:
        dim0, dim1 = pair
        idx = torch.Tensor((dim0, dim1)).long()
        inv_idx = torch.Tensor((dim1, dim0)).long()
        if out.dim() == 4:
            out[:, idx] = out[:, inv_idx]
        else:
            out[idx] = out[inv_idx]

    if shift:
        if out.dim() == 3:
            out[:, :, 1:] = out[:, :, 0:-1]
        else:
            out[:, :, :, 1:] = out[:, :, :, 0:-1]
    return out


def flip_coord(preds, joint_pairs, width_dim, shift=False, flatten=True):
    """Flip pose heatmap according to joint pairs.

    Parameters
    ----------
    heatmap : numpy.ndarray
        Heatmap of joints.
    joint_pairs : list
        List of joint pairs.
    shift : bool
        Whether to shift the output.

    Returns
    -------
    numpy.ndarray
        Flipped heatmap.

    """
    pred_jts, pred_scores = preds
    if flatten:
        assert pred_jts.dim() == 2 and pred_scores.dim() == 3
        num_batches = pred_jts.shape[0]
        num_joints = pred_jts.shape[1] // 3
        pred_jts = pred_jts.reshape(num_batches, num_joints, 3)
    else:
        assert pred_jts.dim() == 3 and pred_scores.dim() == 3
        num_batches = pred_jts.shape[0]
        num_joints = pred_jts.shape[1]

    # flip
    if shift:
        pred_jts[:, :, 0] = - pred_jts[:, :, 0]
    else:
        pred_jts[:, :, 0] = -1 / width_dim - pred_jts[:, :, 0]

    for pair in joint_pairs:
        dim0, dim1 = pair
        idx = torch.Tensor((dim0, dim1)).long()
        inv_idx = torch.Tensor((dim1, dim0)).long()
        pred_jts[:, idx] = pred_jts[:, inv_idx]
        pred_scores[:, idx] = pred_scores[:, inv_idx]

    pred_jts = pred_jts.reshape(num_batches, num_joints * 3)
    return pred_jts, pred_scores


def flip_uvd_coord(pred_jts, joint_pairs, width_dim, shift=False, flatten=True):
    """Flip pose heatmap according to joint pairs.

    Parameters
    ----------
    heatmap : numpy.ndarray
        Heatmap of joints.
    joint_pairs : list
        List of joint pairs.
    shift : bool
        Whether to shift the output.

    Returns
    -------
    numpy.ndarray
        Flipped heatmap.

    """
    if flatten:
        assert pred_jts.dim() == 2
        num_batches = pred_jts.shape[0]
        num_joints = pred_jts.shape[1] // 3
        pred_jts = pred_jts.reshape(num_batches, num_joints, 3)
    else:
        assert pred_jts.dim() == 3
        num_batches = pred_jts.shape[0]
        num_joints = pred_jts.shape[1]

    # flip
    if shift:
        pred_jts[:, :, 0] = - pred_jts[:, :, 0]
    else:
        pred_jts[:, :, 0] = -1 / width_dim - pred_jts[:, :, 0]

    for pair in joint_pairs:
        dim0, dim1 = pair
        idx = torch.Tensor((dim0, dim1)).long()
        inv_idx = torch.Tensor((dim1, dim0)).long()
        pred_jts[:, idx] = pred_jts[:, inv_idx]

    if flatten:
        pred_jts = pred_jts.reshape(num_batches, num_joints * 3)

    return pred_jts


def flip_xyz_coord(xyz_jts, joint_pairs, flatten=True):
    """Flip pose heatmap according to joint pairs.

    Parameters
    ----------
    xyz_jts : torch.Tensor
        Heatmap of joints.
    joint_pairs : list
        List of joint pairs.
    shift : bool
        Whether to shift the output.

    Returns
    -------
    torch.Tensor
        Flipped heatmap.

    """
    if flatten:
        assert xyz_jts.dim() == 2
        num_batches = xyz_jts.shape[0]
        num_joints = xyz_jts.shape[1] // 3
        xyz_jts = xyz_jts.reshape(num_batches, num_joints, 3)
    else:
        assert xyz_jts.dim() == 3
        num_batches = xyz_jts.shape[0]
        num_joints = xyz_jts.shape[1]

    xyz_jts[:, :, 0] = - xyz_jts[:, :, 0]

    for pair in joint_pairs:
        dim0, dim1 = pair
        idx = torch.Tensor((dim0, dim1)).long()
        inv_idx = torch.Tensor((dim1, dim0)).long()
        xyz_jts[:, idx] = xyz_jts[:, inv_idx]

    if flatten:
        xyz_jts = xyz_jts.reshape(num_batches, num_joints * 3)

    return xyz_jts


def flip_coord_2d(preds, joint_pairs, width_dim, shift=False):
    """Flip pose heatmap according to joint pairs.

    Parameters
    ----------
    heatmap : numpy.ndarray
        Heatmap of joints.
    joint_pairs : list
        List of joint pairs.
    shift : bool
        Whether to shift the output.

    Returns
    -------
    numpy.ndarray
        Flipped heatmap.

    """
    pred_jts, pred_scores = preds
    assert pred_scores.dim() == 3

    # flip
    if shift:
        pred_jts[:, :, 0] = -1 / width_dim - pred_jts[:, :, 0]
    else:
        pred_jts[:, :, 0] = - pred_jts[:, :, 0]

    for pair in joint_pairs:
        dim0, dim1 = pair
        idx = torch.Tensor((dim0, dim1)).long()
        inv_idx = torch.Tensor((dim1, dim0)).long()
        pred_jts[:, idx, :] = pred_jts[:, inv_idx, :]
        pred_scores[:, idx, :] = pred_scores[:, inv_idx, :]

    return pred_jts, pred_scores


def flip_heatmap_coord(preds, joint_pairs, shift=False):
    """Flip pose heatmap and coord_z according to joint pairs.

    Parameters
    ----------
    preds : [[torch.Tensor, torch.Tensor], torch.Tensor]
        [[Heatmap of joints,z coord], score]
    joint_pairs : list
        List of joint pairs.
    shift : bool
        Whether to shift the output.

    Returns
    -------
    numpy.ndarray
        [[Flipped Heatmap of joints,z coord], score]

    """
    pred_jts, pred_scores = preds
    heatmap = pred_jts[0]
    pred_z = pred_jts[1]
    assert heatmap.dim() == 3 or heatmap.dim() == 4
    assert pred_z.dim() == 2 and pred_scores.dim() == 3
    out = flip(heatmap)

    for pair in joint_pairs:
        dim0, dim1 = pair
        idx = torch.Tensor((dim0, dim1)).long()
        inv_idx = torch.Tensor((dim1, dim0)).long()
        if out.dim() == 4:
            out[:, idx] = out[:, inv_idx]
        else:
            out[idx] = out[inv_idx]
        pred_z[:, idx] = pred_z[:, inv_idx]
        pred_scores[:, idx] = pred_scores[:, inv_idx]

    if shift:
        if out.dim() == 3:
            out[:, :, 1:] = out[:, :, 0:-1]
        else:
            out[:, :, :, 1:] = out[:, :, :, 0:-1]
    return [out, pred_z], pred_scores


def flip_heatmap_dz(heatmap, joint_pairs, bone_pairs, num_joints, num_bones, shift=False):
    assert (heatmap.dim() == 3 or heatmap.dim() == 4)
    out = flip(heatmap)

    for pair in joint_pairs:
        dim0, dim1 = pair
        idx = torch.Tensor((dim0, dim1)).long()
        inv_idx = torch.Tensor((dim1, dim0)).long()
        if out.dim() == 4:
            out[:, idx] = out[:, inv_idx]
        else:
            out[idx] = out[inv_idx]

    for pair in bone_pairs:
        dim0, dim1 = pair
        du_idx = torch.Tensor((dim0 + num_joints, dim1 + num_joints)).long()
        du_inv_idx = torch.Tensor((dim1 + num_joints, dim0 + num_joints)).long()
        dv_idx = torch.Tensor((dim0 + num_joints + num_bones, dim1 + num_joints + num_bones)).long()
        dv_inv_idx = torch.Tensor((dim1 + num_joints + num_bones, dim0 + num_joints + num_bones)).long()

        if out.dim() == 4:
            out[:, du_idx] = out[:, du_inv_idx]
            out[:, dv_idx] = out[:, dv_inv_idx]
        else:
            out[du_idx] = out[du_inv_idx]
            out[dv_idx] = out[dv_inv_idx]

    if shift:
        if out.dim() == 3:
            out[:, :, 1:] = out[:, :, 0:-1]
        else:
            out[:, :, :, 1:] = out[:, :, :, 0:-1]
    return out


def flip_coord_bone(preds, joint_pairs, bone_pairs, width_dim, shift=False):
    """Flip pose heatmap according to joint pairs.

    Parameters
    ----------
    heatmap : numpy.ndarray
        Heatmap of joints.
    joint_pairs : list
        List of joint pairs.
    shift : bool
        Whether to shift the output.

    Returns
    -------
    numpy.ndarray
        Flipped heatmap.

    """
    pred_2d, pred_bones, pred_bones_bin, pred_scores = preds
    assert pred_2d.dim() == 2 and pred_scores.dim() == 3
    num_batches = pred_2d.shape[0]
    num_joints = pred_2d.shape[1] // 2
    pred_2d = pred_2d.reshape(num_batches, num_joints, 2)

    # flip
    if shift:
        pred_2d[:, :, 0] = -1 / width_dim - pred_2d[:, :, 0]
    else:
        pred_2d[:, :, 0] = - pred_2d[:, :, 0]

    for pair in joint_pairs:
        dim0, dim1 = pair
        idx = torch.Tensor((dim0, dim1)).long()
        inv_idx = torch.Tensor((dim1, dim0)).long()
        pred_2d[:, idx] = pred_2d[:, inv_idx]
        pred_scores[:, idx] = pred_scores[:, inv_idx]

    pred_2d = pred_2d.reshape(num_batches, num_joints * 2)

    for pair in bone_pairs:
        dim0, dim1 = pair
        idx = torch.Tensor((dim0, dim1)).long()
        inv_idx = torch.Tensor((dim1, dim0)).long()
        pred_bones[:, idx] = pred_bones[:, inv_idx]
        pred_bones_bin[:, idx] = pred_bones_bin[:, inv_idx]

    return pred_2d, pred_bones, pred_bones_bin, pred_scores


def flip_refine_coord(preds, joint_pairs, width_dim, shift=False):
    """Flip pose heatmap according to joint pairs.

    Parameters
    ----------
    heatmap : numpy.ndarray
        Heatmap of joints.
    joint_pairs : list
        List of joint pairs.
    shift : bool
        Whether to shift the output.

    Returns
    -------
    numpy.ndarray
        Flipped heatmap.

    """
    pred_jts, refine_jts, pred_scores = preds
    assert pred_jts.dim() == 2 and pred_scores.dim() == 3
    num_batches = pred_jts.shape[0]
    num_joints = pred_jts.shape[1] // 3
    pred_jts = pred_jts.reshape(num_batches, num_joints, 3)
    refine_jts = refine_jts.reshape(num_batches, num_joints, 3)

    # flip
    if shift:
        pred_jts[:, :, 0] = -1 / width_dim - pred_jts[:, :, 0]
        refine_jts[:, :, 0] = -1 / width_dim - refine_jts[:, :, 0]
    else:
        pred_jts[:, :, 0] = - pred_jts[:, :, 0]
        refine_jts[:, :, 0] = - refine_jts[:, :, 0]

    for pair in joint_pairs:
        dim0, dim1 = pair
        idx = torch.Tensor((dim0, dim1)).long()
        inv_idx = torch.Tensor((dim1, dim0)).long()
        pred_jts[:, idx] = pred_jts[:, inv_idx]
        refine_jts[:, idx] = refine_jts[:, inv_idx]
        pred_scores[:, idx] = pred_scores[:, inv_idx]

    pred_jts = pred_jts.reshape(num_batches, num_joints * 3)
    refine_jts = refine_jts.reshape(num_batches, num_joints * 3)
    return pred_jts, refine_jts, pred_scores


def flip_joints_3d(joints_3d, width, joint_pairs):
    """Flip 3d joints.

    Parameters
    ----------
    joints_3d : numpy.ndarray
        Joints in shape (num_joints, 3, 2)
    width : int
        Image width.
    joint_pairs : list
        List of joint pairs.

    Returns
    -------
    numpy.ndarray
        Flipped 3d joints with shape (num_joints, 3, 2)

    """
    joints = joints_3d.copy()
    # flip horizontally
    joints[:, 0, 0] = width - joints[:, 0, 0] - 1
    # change left-right parts
    for pair in joint_pairs:
        joints[pair[0], :, 0], joints[pair[1], :, 0] = \
            joints[pair[1], :, 0], joints[pair[0], :, 0].copy()
        joints[pair[0], :, 1], joints[pair[1], :, 1] = \
            joints[pair[1], :, 1], joints[pair[0], :, 1].copy()

    joints[:, :, 0] *= joints[:, :, 1]
    return joints


def flip_xyz_joints_3d(joints_3d, joint_pairs):
    """Flip 3d xyz joints.

    Parameters
    ----------
    joints_3d : numpy.ndarray
        Joints in shape (num_joints, 3)
    joint_pairs : list
        List of joint pairs.

    Returns
    -------
    numpy.ndarray
        Flipped 3d joints with shape (num_joints, 3)

    """
    assert joints_3d.ndim in (2, 3)

    joints = joints_3d.copy()
    # flip horizontally
    joints[:, 0] = -1 * joints[:, 0]
    # change left-right parts
    for pair in joint_pairs:
        joints[pair[0], :], joints[pair[1], :] = joints[pair[1], :], joints[pair[0], :].copy()

    return joints


def flip_cam_xyz_joints_3d(joints_3d, joint_pairs):
    """Flip 3d xyz joints.

    Parameters
    ----------
    joints_3d : numpy.ndarray
        Joints in shape (num_joints, 3)
    joint_pairs : list
        List of joint pairs.

    Returns
    -------
    numpy.ndarray
        Flipped 3d joints with shape (num_joints, 3)

    """
    root_jts = joints_3d[:1].copy()
    joints = (joints_3d - root_jts)
    assert joints_3d.ndim in (2, 3)

    # flip horizontally
    joints[:, 0] = -1 * joints[:, 0]

    # change left-right parts
    for pair in joint_pairs:
        joints[pair[0], :], joints[pair[1], :] = joints[pair[1], :], joints[pair[0], :].copy()

    return joints + root_jts


def flip_thetas(thetas, theta_pairs):
    """Flip thetas.

    Parameters
    ----------
    thetas : numpy.ndarray
        Joints in shape (num_thetas, 3)
    theta_pairs : list
        List of theta pairs.

    Returns
    -------
    numpy.ndarray
        Flipped thetas with shape (num_thetas, 3)

    """
    thetas_flip = thetas.copy()
    # reflect horizontally
    thetas_flip[:, 1] = -1 * thetas_flip[:, 1]
    thetas_flip[:, 2] = -1 * thetas_flip[:, 2]
    # thetas_flip[:, 0] = -1 * thetas_flip[:, 0]
    # change left-right parts
    for pair in theta_pairs:
        thetas_flip[pair[0], :], thetas_flip[pair[1], :] = \
            thetas_flip[pair[1], :], thetas_flip[pair[0], :].copy()

    return thetas_flip


def flip_twist(twist_phi, twist_weight, twist_pairs):
    # twist_flip = -1 * twist_phi.copy() # 23 x 2
    twist_flip = np.zeros_like(twist_phi)
    weight_flip = twist_weight.copy()

    twist_flip[:, 0] = twist_phi[:, 0].copy() # cos
    twist_flip[:, 1] = -1 * twist_phi[:, 1].copy() # sin
    for pair in twist_pairs:
        idx0 = pair[0] - 1
        idx1 = pair[1] - 1
        twist_flip[idx0, :], twist_flip[idx1, :] = \
            twist_flip[idx1, :], twist_flip[idx0, :].copy()
        
        weight_flip[idx0, :], weight_flip[idx1, :] = \
            weight_flip[idx1, :], weight_flip[idx0, :].copy()
    
    return twist_flip, weight_flip


def rot_aa(aa, rot):
    """Rotate axis angle parameters."""
    # pose parameters
    R = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                  [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                  [0, 0, 1]])
    # find the rotation of the body in camera frame
    per_rdg, _ = cv2.Rodrigues(aa)
    # apply the global rotation to the global orientation
    resrot, _ = cv2.Rodrigues(np.dot(R, per_rdg))
    aa = (resrot.T)[0]
    return aa


def rotate_xyz_jts(xyz_jts, rot):
    assert xyz_jts.ndim == 2 and xyz_jts.shape[1] == 3
    xyz_jts_new = xyz_jts.copy()

    rot_rad = - np.pi * rot / 180

    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    xyz_jts_new[:, 0] = xyz_jts[:, 0] * cs - xyz_jts[:, 1] * sn
    xyz_jts_new[:, 1] = xyz_jts[:, 0] * sn + xyz_jts[:, 1] * cs
    return xyz_jts_new


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def batch_rodrigues_numpy(rot_vecs, epsilon=1e-8):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: numpy.ndarray Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: numpy.ndarray Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]

    angle = np.linalg.norm(rot_vecs + 1e-8, axis=1, keepdims=True)
    rot_dir = rot_vecs / angle

    cos = np.cos(angle)[:, None, :]
    sin = np.sin(angle)[:, None, :]

    # Bx1 arrays
    rx, ry, rz = np.split(rot_dir, 3, axis=1)
    K = np.zeros((batch_size, 3, 3))
    zeros = np.zeros((batch_size, 1))

    K = np.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], axis=1) \
        .reshape((batch_size, 3, 3))

    ident = np.eye(3)[None, :, :]
    rot_mat = ident + sin * K + (1 - cos) * np.einsum('bij,bjk->bik', K, K)
    return rot_mat


def shuffle_joints(joints, joint_pairs):
    """Shuffle 3d joints.

    Parameters
    ----------
    joints : numpy.ndarray
        Joints in shape (num_joints, 3, 2)
    width : int
        Image width.
    joint_pairs : list
        List of joint pairs.

    Returns
    -------
    numpy.ndarray
        Flipped 3d joints with shape (num_joints, 3, 2)

    """
    joints = joints.copy()
    # change left-right parts
    for pair in joint_pairs:
        joints[pair[0], :], joints[pair[1], :] = joints[pair[1], :], joints[pair[0], :].copy()

    return joints


def norm_heatmap(norm_name, heatmap):
    # Input tensor shape: [N,C,...]
    if isinstance(heatmap, np.ndarray):
        heatmap = torch.from_numpy(heatmap)
    assert isinstance(
        heatmap, torch.Tensor), 'Heatmap to be normalized must be torch.Tensor!'
    shape = heatmap.shape
    if norm_name == 'softmax':
        heatmap = heatmap.reshape(*shape[:2], -1)
        # global soft max
        heatmap = F.softmax(heatmap, 2)
        return heatmap.reshape(*shape)
    elif norm_name == 'sigmoid':
        return heatmap.sigmoid()
    elif norm_name == 'divide_sum':
        heatmap = heatmap.reshape(*shape[:2], -1)
        heatmap = heatmap / heatmap.sum(dim=2, keepdim=True)
        return heatmap.reshape(*shape)
    else:
        raise NotImplementedError


pw3d_recover_z = (
    None, (0,), (1,),
    (2, 0), (3, 1), (4, 2, 0), (5, 3, 1),
    (6,),
    (7,), (8,),
    (9, 7), (10, 8), (11, 9, 7), (12, 10, 8)
)
h36m_recover_z = (
    None, (0,), (1, 0), (2, 1, 0),
    (3,), (4, 3), (5, 4, 3),
    (6,), (7, 6),
    (8, 7, 6), (9, 8, 7, 6),
    (10, 6), (11, 10, 6), (12, 11, 10, 6),
    (13, 6), (14, 13, 6), (15, 14, 13, 6),
    (16, 6)
)
hp3d_recover_z = (
    (0, 2, 3), (1, 0, 2, 3), (2, 3), (3,), None,
    (4, 1, 0, 2, 3), (5, 4, 1, 0, 2, 3), (6, 5, 4, 1, 0, 2, 3), (7, 1, 0, 2, 3), (8, 7, 1, 0, 2, 3), (9, 8, 7, 1, 0, 2, 3),
    (10, 9, 8, 7, 1, 0, 2, 3), (11, 10, 9, 8, 7, 1, 0, 2, 3), (12, 1, 0, 2, 3), (13, 12, 1, 0, 2, 3), (14, 13, 12, 1, 0, 2, 3), (15, 14, 13, 12, 1, 0, 2, 3),
    (16, 15, 14, 13, 12, 1, 0, 2, 3), (17,), (18, 17), (19, 18, 17), (20, 19, 18, 17), (21, 20, 19, 18, 17),
    (22,), (23, 22), (24, 23, 22), (25, 24, 23, 22), (26, 25, 24, 23, 22)
)
recover_z = {
    'pw3d': pw3d_recover_z,
    'h36m': h36m_recover_z,
    'hp3d': hp3d_recover_z
}


def heatmap_to_coord(pred_jts, pred_scores, hm_shape, bbox, output_3d=False, mean_bbox_scale=None):
    # TODO: This cause imbalanced GPU useage, implement cpu version
    hm_width, hm_height = hm_shape

    ndims = pred_jts.dim()
    assert ndims in [2, 3], "Dimensions of input heatmap should be 2 or 3"
    if ndims == 2:
        pred_jts = pred_jts.unsqueeze(0)
        pred_scores = pred_scores.unsqueeze(0)

    coords = pred_jts.cpu().numpy()
    coords = coords.astype(float)
    pred_scores = pred_scores.cpu().numpy()
    pred_scores = pred_scores.astype(float)

    coords[:, :, 0] = (coords[:, :, 0] + 0.5) * hm_width
    coords[:, :, 1] = (coords[:, :, 1] + 0.5) * hm_height

    preds = np.zeros_like(coords)
    # transform bbox to scale
    xmin, ymin, xmax, ymax = bbox
    w = xmax - xmin
    h = ymax - ymin
    center = np.array([xmin + w * 0.5, ymin + h * 0.5])
    scale = np.array([w, h])
    # Transform back
    for i in range(coords.shape[0]):
        for j in range(coords.shape[1]):
            preds[i, j, 0:2] = transform_preds(coords[i, j, 0:2], center, scale,
                                               [hm_width, hm_height])
            if output_3d:
                if mean_bbox_scale is not None:
                    zscale = scale[0] / mean_bbox_scale
                    preds[i, j, 2] = coords[i, j, 2] / zscale
                else:
                    preds[i, j, 2] = coords[i, j, 2]
    # maxvals = np.ones((*preds.shape[:2], 1), dtype=float)
    # score_mul = 1 if norm_name == 'sigmoid' else 5

    return preds, pred_scores


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    target_coords[0:2] = affine_transform(coords[0:2], trans)
    return target_coords


def transform_preds_new(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform_new(center, scale, 0, output_size, inv=1)
    target_coords[0:2] = affine_transform(coords[0:2], trans)
    return target_coords


def get_max_pred(heatmaps):
    num_joints = heatmaps.shape[0]
    width = heatmaps.shape[2]
    heatmaps_reshaped = heatmaps.reshape((num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 1)
    maxvals = np.max(heatmaps_reshaped, 1)

    maxvals = maxvals.reshape((num_joints, 1))
    idx = idx.reshape((num_joints, 1))

    preds = np.tile(idx, (1, 2)).astype(np.float32)

    preds[:, 0] = (preds[:, 0]) % width
    preds[:, 1] = np.floor((preds[:, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_max_pred_batch(batch_heatmaps):
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.max(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_affine_transform_new(center,
                             scale,
                             rot,
                             output_size,
                             shift=np.array([0, 0], dtype=np.float32),
                             inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, (dst_w - 1) * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]
    dst[1, :] = np.array([(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_func_heatmap_to_coord(cfg):
    if cfg.TEST.get('HEATMAP2COORD') == 'coord':
        return heatmap_to_coord
    else:
        raise NotImplementedError


def rotmat_to_quat_numpy(rotmat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    Returns:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    """
    trace = np.einsum('bii->b', rotmat)
    m32 = rotmat[:, 2, 1]
    m23 = rotmat[:, 1, 2]
    m13 = rotmat[:, 0, 2]
    m31 = rotmat[:, 2, 0]
    m21 = rotmat[:, 1, 0]
    m12 = rotmat[:, 0, 1]

    trace = trace + 1
    w = np.sqrt(trace.clip(min=1e-8)) / 2
    x = (m32 - m23) / (4 * w)
    y = (m13 - m31) / (4 * w)
    z = (m21 - m12) / (4 * w)

    return np.stack([w, x, y, z], axis=1)


def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat



def rot_theta(theta, xyz_coord, rot, smpl_parents):
    """Rotate axis angle parameters."""
    theta_0 = theta[:1] # the first theta is global orientation, and should be processed separately
    theta_left = theta[1:]
    smpl_parents_left = smpl_parents[1:]

    angle = np.linalg.norm(theta_left + 1e-8, axis=1, keepdims=True)

    origin_xyz_start = xyz_coord[smpl_parents_left]
    origin_xyz_end = origin_xyz_start + theta_left
    # new_xyz_start = np.einsum('ca,ba->ba', trans_3d, origin_xyz_start)
    # new_xyz_end = np.einsum('ca,ba->ba', trans_3d, origin_xyz_end)
    new_xyz_start = rotate_xyz_jts(origin_xyz_start, rot)
    new_xyz_end = rotate_xyz_jts(origin_xyz_end, rot)
    new_axis = new_xyz_end - new_xyz_start
    new_axis = new_axis / np.linalg.norm(new_axis + 1e-8, axis=1, keepdims=True)

    new_theta_left = new_axis * angle
    new_theta = np.concatenate([theta_0, new_theta_left], axis=0)
    return new_theta

