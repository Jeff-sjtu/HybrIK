import torch

from .losses import camera_fitting_loss, camera_fitting_loss_3d, body_fitting_loss, body_fitting_loss_3d

# For the GMM prior, we use the GMM implementation of SMPLify-X
# https://github.com/vchoutas/smplify-x/blob/master/smplifyx/prior.py
from .prior import MaxMixturePrior

from easydict import EasyDict as edict


class SMPLify():
    """Implementation of single-stage SMPLify."""
    def __init__(self,
                 step_size=1e-2,
                 batch_size=66,
                 num_iters=[10, 100],
                 focal_length=1000,
                 smpl=None,
                 device=torch.device('cuda')):

        # Store options
        self.device = device
        self.focal_length = focal_length
        self.step_size = step_size

        # Ignore the the following joints for the fitting process
        # ign_joints = ['Right Hip', 'Left Hip']
        self.ign_joints = [1, 2]
        self.num_iters = num_iters
        # GMM pose prior
        self.pose_prior = MaxMixturePrior(prior_folder='model_files',
                                          num_gaussians=8,
                                          dtype=torch.float32).to(device)
        # Load SMPL model
        self.smpl = smpl

    def __call__(self, init_pose, init_betas, init_cam_t, keypoints_2d, camera_center=None):
        """Perform body fitting.
        Input:
            init_pose: SMPL pose estimate
            init_betas: SMPL betas estimate
            init_cam_t: Camera translation estimate
            camera_center: Camera center location
            keypoints_2d: Keypoints used for the optimization
        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
            camera_translation: Camera translation
            reprojection_loss: Final joint reprojection loss
        """

        # batch_size = init_pose.shape[0]

        # Make camera translation a learnable parameter
        camera_translation = init_cam_t.clone()

        # Get joint confidence
        joints_2d = keypoints_2d[:, :, :2]
        joints_conf = keypoints_2d[:, :, -1]

        # Split SMPL pose to body pose and global orientation
        body_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        betas = init_betas.detach().clone()

        # Step 1: Optimize camera translation and body orientation
        # Optimize only camera translation and body orientation
        body_pose.requires_grad = False
        betas.requires_grad = False
        global_orient.requires_grad = True
        camera_translation.requires_grad = True

        camera_opt_params = [global_orient, camera_translation]
        camera_optimizer = torch.optim.Adam(camera_opt_params, lr=self.step_size, betas=(0.9, 0.999))

        for i in range(self.num_iters[0]):
            smpl_output = self.smpl(global_orient=global_orient,
                                    pose_axis_angle=body_pose,
                                    betas=betas)
            model_joints = smpl_output.joints
            loss = camera_fitting_loss(model_joints, camera_translation,
                                       init_cam_t, camera_center,
                                       joints_2d, joints_conf, focal_length=self.focal_length)
            camera_optimizer.zero_grad()
            loss.backward()
            camera_optimizer.step()

        # Fix camera translation after optimizing camera
        camera_translation.requires_grad = False

        # Step 2: Optimize body joints
        # Optimize only the body pose and global orientation of the body
        body_pose.requires_grad = True
        betas.requires_grad = True
        global_orient.requires_grad = True
        camera_translation.requires_grad = False
        body_opt_params = [body_pose, betas, global_orient]

        # For joints ignored during fitting, set the confidence to 0
        joints_conf[:, self.ign_joints] = 0.

        body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))
        for i in range(self.num_iters[1]):
            smpl_output = self.smpl(global_orient=global_orient,
                                    pose_axis_angle=body_pose,
                                    betas=betas)
            model_joints = smpl_output.joints[:, :24, :]
            loss = body_fitting_loss(body_pose, betas, model_joints, camera_translation, camera_center,
                                     joints_2d, joints_conf, self.pose_prior,
                                     focal_length=self.focal_length)
            body_optimizer.zero_grad()
            loss.backward()
            body_optimizer.step()

        # Get final loss value
        with torch.no_grad():
            smpl_output = self.smpl(global_orient=global_orient,
                                    pose_axis_angle=body_pose,
                                    betas=betas)
            model_joints = smpl_output.joints
            reprojection_loss = body_fitting_loss(body_pose, betas, model_joints, camera_translation, camera_center,
                                                  joints_2d, joints_conf, self.pose_prior,
                                                  focal_length=self.focal_length,
                                                  output='reprojection')

        vertices = smpl_output.vertices.detach()
        joints = smpl_output.joints.detach()
        joints_from_verts = smpl_output.joints_from_verts.detach()
        pose = torch.cat([global_orient, body_pose], dim=-1).detach()
        betas = betas.detach()

        return edict(
            vertices=vertices,
            joints=joints,
            joints_from_verts=joints_from_verts,
            pose=pose,
            betas=betas,
            camera_translation=camera_translation,
            reprojection_loss=reprojection_loss
        )

    def get_fitting_loss(self, pose, betas, cam_t, camera_center, keypoints_2d):
        """Given body and camera parameters, compute reprojection loss value.
        Input:
            pose: SMPL pose parameters
            betas: SMPL beta parameters
            cam_t: Camera translation
            camera_center: Camera center location
            keypoints_2d: Keypoints used for the optimization
        Returns:
            reprojection_loss: Final joint reprojection loss
        """

        # batch_size = pose.shape[0]

        # Get joint confidence
        joints_2d = keypoints_2d[:, :, :2]
        joints_conf = keypoints_2d[:, :, -1]
        # For joints ignored during fitting, set the confidence to 0
        joints_conf[:, self.ign_joints] = 0.

        # Split SMPL pose to body pose and global orientation
        body_pose = pose[:, 3:]
        global_orient = pose[:, :3]

        with torch.no_grad():
            smpl_output = self.smpl(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas, return_full_pose=True)
            model_joints = smpl_output.joints
            reprojection_loss = body_fitting_loss(body_pose, betas, model_joints, cam_t, camera_center,
                                                  joints_2d, joints_conf, self.pose_prior,
                                                  focal_length=self.focal_length,
                                                  output='reprojection')

        return reprojection_loss


class SMPLify3D:
    """Implementation of single-stage SMPLify (3D-keypoint version)."""

    def __init__(self,
                 step_size=1e-2,
                 batch_size=66,
                 num_iters=[10, 100],
                 focal_length=1000,
                 smpl=None,
                 device=torch.device('cuda')):

        # Store options
        self.device = device
        self.focal_length = focal_length
        self.step_size = step_size

        # Ignore the the following joints for the fitting process
        # ign_joints = ['Right Hip', 'Left Hip']
        self.ign_joints = [1, 2]
        self.num_iters = num_iters
        # GMM pose prior
        self.pose_prior = MaxMixturePrior(prior_folder='model_files',
                                          num_gaussians=8,
                                          dtype=torch.float32).to(device)
        # Load SMPL model
        self.smpl = smpl

    def __call__(self, init_pose, init_betas, init_cam_t, keypoints_3d, camera_center=None):
        """Perform body fitting.
        Input:
            init_pose: SMPL pose estimate
            init_betas: SMPL betas estimate
            init_cam_t: Camera translation estimate
            camera_center: Camera center location
            keypoints_3d: Keypoints in camera coordinates used for the optimization
        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
            camera_translation: Camera translation
            reprojection_loss: Final joint reprojection loss
        """

        # batch_size = init_pose.shape[0]

        # Make camera translation a learnable parameter
        camera_translation = init_cam_t.clone()

        # Get joint confidence
        joints_3d = keypoints_3d[:, :, :3]
        joints_conf = keypoints_3d[:, :, -1]

        # Split SMPL pose to body pose and global orientation
        body_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        betas = init_betas.detach().clone()

        # Step 1: Optimize camera translation and body orientation
        # Optimize only camera translation and body orientation
        body_pose.requires_grad = False
        betas.requires_grad = False
        global_orient.requires_grad = True
        camera_translation.requires_grad = True

        camera_opt_params = [global_orient, camera_translation]
        camera_optimizer = torch.optim.Adam(
            camera_opt_params, lr=self.step_size, betas=(0.9, 0.999))

        for i in range(self.num_iters[0]):
            smpl_output = self.smpl(global_orient=global_orient,
                                    pose_axis_angle=body_pose,
                                    betas=betas)
            model_joints = smpl_output.joints
            loss = camera_fitting_loss_3d(
                model_joints, camera_translation,
                init_cam_t, camera_center,
                joints_3d, joints_conf, focal_length=self.focal_length)
            camera_optimizer.zero_grad()
            loss.backward()
            camera_optimizer.step()

        # Fix camera translation after optimizing camera
        camera_translation.requires_grad = False

        # Step 2: Optimize body joints
        # Optimize only the body pose and global orientation of the body
        body_pose.requires_grad = True
        betas.requires_grad = True
        global_orient.requires_grad = True
        camera_translation.requires_grad = False
        body_opt_params = [body_pose, betas, global_orient]

        # For joints ignored during fitting, set the confidence to 0
        joints_conf[:, self.ign_joints] = 0.

        body_optimizer = torch.optim.Adam(
            body_opt_params, lr=self.step_size, betas=(0.9, 0.999))
        for i in range(self.num_iters[1]):
            smpl_output = self.smpl(global_orient=global_orient,
                                    pose_axis_angle=body_pose,
                                    betas=betas)
            model_joints = smpl_output.joints[:, :24, :]
            loss = body_fitting_loss_3d(
                body_pose, betas, model_joints, camera_translation, camera_center,
                joints_3d, joints_conf, self.pose_prior,
                focal_length=self.focal_length)
            body_optimizer.zero_grad()
            loss.backward()
            body_optimizer.step()

        # Get final loss value
        with torch.no_grad():
            smpl_output = self.smpl(global_orient=global_orient,
                                    pose_axis_angle=body_pose,
                                    betas=betas)
            model_joints = smpl_output.joints
            reprojection_loss = body_fitting_loss_3d(body_pose, betas, model_joints, camera_translation, camera_center,
                                                     joints_3d, joints_conf, self.pose_prior,
                                                     focal_length=self.focal_length,
                                                     output='reprojection')

        vertices = smpl_output.vertices.detach()
        joints = smpl_output.joints.detach()
        joints_from_verts = smpl_output.joints_from_verts.detach()
        pose = torch.cat([global_orient, body_pose], dim=-1).detach()
        betas = betas.detach()

        return edict(
            vertices=vertices,
            joints=joints,
            joints_from_verts=joints_from_verts,
            pose=pose,
            betas=betas,
            camera_translation=camera_translation,
            reprojection_loss=reprojection_loss
        )
