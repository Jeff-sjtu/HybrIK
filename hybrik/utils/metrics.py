import numpy as np


class NullWriter(object):
    def write(self, arg):
        pass

    def flush(self):
        pass


class DataLogger(object):
    """Average data logger."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.value = 0
        self.sum = 0
        self.cnt = 0
        self.avg = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.cnt += n
        self._cal_avg()

    def _cal_avg(self):
        self.avg = self.sum / self.cnt


def calc_coord_accuracy(pred_jts, labels, label_masks, hm_shape, norm='softmax', num_joints=None, root_idx=None):
    """Calculate integral coordinates accuracy."""
    coords = pred_jts.detach().cpu().numpy()
    coords = coords.astype(float)
    if num_joints is not None:
        coords = coords.reshape(coords.shape[0], num_joints, -1)
        labels = labels.reshape(labels.shape[0], num_joints, -1)
        label_masks = label_masks.reshape(label_masks.shape[0], num_joints, -1)
        coords = coords[:, :, :3].reshape(coords.shape[0], -1)
        labels = labels[:, :, :3].reshape(coords.shape[0], -1)
        label_masks = label_masks[:, :, :3].reshape(coords.shape[0], -1)
    else:
        num_joints = coords.shape[1] // 3

    hm_width, hm_height, hm_depth = hm_shape
    coords = coords.reshape((coords.shape[0], int(coords.shape[1] / 3), 3))

    coords[:, :, 0] = (coords[:, :, 0] + 0.5) * hm_width
    coords[:, :, 1] = (coords[:, :, 1] + 0.5) * hm_height

    labels = labels.cpu().data.numpy().reshape(pred_jts.shape[0], num_joints, 3)
    label_masks = label_masks.cpu().data.numpy().reshape(pred_jts.shape[0], num_joints, 3)

    labels[:, :, 0] = (labels[:, :, 0] + 0.5) * hm_width
    labels[:, :, 1] = (labels[:, :, 1] + 0.5) * hm_height
    labels[:, :, 2] = (labels[:, :, 2] + 0.5) * hm_depth

    coords[:, :, 2] = (coords[:, :, 2] + 0.5) * hm_depth

    if root_idx is not None:
        labels = labels - labels[:, root_idx, :][:, None, :]
        coords = coords - coords[:, root_idx, :][:, None, :]

    coords = coords * label_masks
    labels = labels * label_masks

    norm = np.ones((pred_jts.shape[0], 3)) * np.array([hm_width, hm_height, hm_depth]) / 10

    dists = calc_dist(coords, labels, norm)

    acc = 0
    sum_acc = 0
    cnt = 0
    for i in range(num_joints):
        acc = dist_acc(dists[i])
        if acc >= 0:
            sum_acc += acc
            cnt += 1

    if cnt > 0:
        return sum_acc / cnt
    else:
        return 0


def calc_dist(preds, target, normalize):
    """Calculate normalized distances"""
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))

    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1

    return dists


def dist_acc(dists, thr=0.5):
    """Calculate accuracy with given input distance."""
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def calc_bin_accuracy(pred_bins, label_bins, label_masks):
    if pred_bins.dim() == 3:
        pred_bins = pred_bins.detach().reshape(-1, 2).numpy()
        pred_bins = np.argmax(pred_bins, axis=1)
    else:
        pred_bins = pred_bins.detach().reshape(-1).numpy()
        pred_bins[pred_bins >= 0.5] = 1
        pred_bins[pred_bins < 0.5] = 0

    label_bins = label_bins.reshape(-1).float().numpy()
    label_masks = label_masks.reshape(-1).numpy()

    assert pred_bins.shape == label_bins.shape
    correct = (pred_bins == label_bins) * 1.0
    correct *= label_masks

    return np.sum(correct) / np.sum(label_masks)
