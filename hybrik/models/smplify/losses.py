import torch


def perspective_projection(points, rotation, translation,
                           focal_length, camera_center=None):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.
    if camera_center is not None:
        K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]


def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def angle_prior(pose):
    """
    Angle prior that penalizes unnatural bending of the knees and elbows
    """
    # We subtract 3 because pose does not include the global rotation of the model
    return torch.exp(pose[:, [55 - 3, 58 - 3, 12 - 3, 15 - 3]] * torch.tensor([1., -1., -1, -1.], device=pose.device)) ** 2


def body_fitting_loss(body_pose, betas, model_joints, camera_t, camera_center,
                      joints_2d, joints_conf, pose_prior,
                      focal_length=5000, sigma=100, pose_prior_weight=0,
                      shape_prior_weight=5, angle_prior_weight=0,
                      output='sum'):
    """
    Loss function for body fitting
    """

    batch_size = body_pose.shape[0]
    rotation = torch.eye(3, device=body_pose.device).unsqueeze(0).expand(batch_size, -1, -1)
    projected_joints = perspective_projection(
        points=model_joints, rotation=rotation, translation=camera_t,
        focal_length=focal_length, camera_center=camera_center)

    # Weighted robust reprojection error
    reprojection_error = gmof(projected_joints - joints_2d, sigma)
    reprojection_loss = (joints_conf ** 2) * reprojection_error.sum(dim=-1)

    # Pose prior loss
    pose_prior_loss = (pose_prior_weight ** 2) * pose_prior(body_pose, betas)

    # Angle prior for knees and elbows
    angle_prior_loss = (angle_prior_weight ** 2) * angle_prior(body_pose).sum(dim=-1)

    # Regularizer to prevent betas from taking large values
    shape_prior_loss = (shape_prior_weight ** 2) * (betas ** 2).sum(dim=-1)

    total_loss = reprojection_loss.sum(dim=-1) + pose_prior_loss + angle_prior_loss + shape_prior_loss

    if output == 'sum':
        return total_loss.sum()
    elif output == 'reprojection':
        # return joints_conf * torch.sqrt(((projected_joints - joints_2d) ** 2).sum(dim=-1))
        return reprojection_loss


def body_fitting_loss_3d(body_pose, betas, model_joints, camera_t, camera_center,
                         joints_3d, joints_conf, pose_prior,
                         focal_length=5000, sigma=100, pose_prior_weight=0,
                         shape_prior_weight=5, angle_prior_weight=0,
                         output='sum'):
    """
    Loss function for body fitting
    """

    # batch_size = body_pose.shape[0]

    points = model_joints + camera_t.unsqueeze(1)

    # Weighted robust reprojection error
    jts_error = gmof(points - joints_3d, sigma)
    jts_loss = (joints_conf ** 2) * jts_error.sum(dim=-1)

    # Pose prior loss
    pose_prior_loss = (pose_prior_weight ** 2) * pose_prior(body_pose, betas)

    # Angle prior for knees and elbows
    angle_prior_loss = (angle_prior_weight ** 2) * angle_prior(body_pose).sum(dim=-1)

    # Regularizer to prevent betas from taking large values
    shape_prior_loss = (shape_prior_weight ** 2) * (betas ** 2).sum(dim=-1)

    total_loss = jts_loss.sum(dim=-1) * 100 + pose_prior_loss + angle_prior_loss + shape_prior_loss

    if output == 'sum':
        return total_loss.sum()
    elif output == 'jts':
        # return joints_conf * torch.sqrt(((projected_joints - joints_2d) ** 2).sum(dim=-1))
        return jts_loss


def camera_fitting_loss(model_joints, camera_t, camera_t_est, camera_center, joints_2d, joints_conf,
                        focal_length=5000, depth_loss_weight=100):
    """
    Loss function for camera optimization.
    """

    # Project model joints
    batch_size = model_joints.shape[0]
    rotation = torch.eye(3, device=model_joints.device).unsqueeze(0).expand(batch_size, -1, -1)
    projected_joints = perspective_projection(
        points=model_joints, rotation=rotation, translation=camera_t,
        focal_length=focal_length, camera_center=camera_center)

    gt_joints_ind = [1, 2, 16, 17]
    reprojection_error_gt = (joints_2d[:, gt_joints_ind] - projected_joints[:, gt_joints_ind]) ** 2

    # Check if for each example in the batch all 4 OpenPose detections are valid, otherwise use the GT detections
    # OpenPose joints are more reliable for this task, so we prefer to use them if possible
    reprojection_loss = reprojection_error_gt.sum(dim=(1, 2))

    # Loss that penalizes deviation from depth estimate
    depth_loss = (depth_loss_weight ** 2) * (camera_t[:, 2] - camera_t_est[:, 2]) ** 2

    total_loss = reprojection_loss + depth_loss
    return total_loss.sum()


def camera_fitting_loss_3d(model_joints, camera_t, camera_t_est, camera_center, joints_3d, joints_conf,
                           focal_length=5000, depth_loss_weight=100):
    """
    Loss function for camera optimization.
    """

    # Project model joints
    # batch_size = model_joints.shape[0]
    points = model_joints + camera_t.unsqueeze(1)

    gt_joints_ind = [1, 2, 16, 17]
    jts_error_gt = (joints_3d[:, gt_joints_ind] - points[:, gt_joints_ind]) ** 2

    # Check if for each example in the batch all 4 OpenPose detections are valid, otherwise use the GT detections
    # OpenPose joints are more reliable for this task, so we prefer to use them if possible
    jts_loss = jts_error_gt.sum(dim=(1, 2))

    # Loss that penalizes deviation from depth estimate
    depth_loss = (depth_loss_weight ** 2) * (camera_t[:, 2] - camera_t_est[:, 2]) ** 2

    total_loss = jts_loss * 100 + depth_loss
    return total_loss.sum()
