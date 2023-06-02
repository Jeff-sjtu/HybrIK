# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import, division, print_function

import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.transforms.rotation_conversions import (axis_angle_to_matrix,
                                                       matrix_to_axis_angle)

from .utils import Tensor, rot_mat_to_euler


def find_dynamic_lmk_idx_and_bcoords(
    vertices: Tensor,
    pose: Tensor,
    dynamic_lmk_faces_idx: Tensor,
    dynamic_lmk_b_coords: Tensor,
    neck_kin_chain: List[int],
    pose2rot: bool = True,
) -> Tuple[Tensor, Tensor]:
    ''' Compute the faces, barycentric coordinates for the dynamic landmarks
        To do so, we first compute the rotation of the neck around the y-axis
        and then use a pre-computed look-up table to find the faces and the
        barycentric coordinates that will be used.
        Special thanks to Soubhik Sanyal (soubhik.sanyal@tuebingen.mpg.de)
        for providing the original TensorFlow implementation and for the LUT.
        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices
        pose: torch.tensor Bx(Jx3), dtype = torch.float32
            The current pose of the body model
        dynamic_lmk_faces_idx: torch.tensor L, dtype = torch.long
            The look-up table from neck rotation to faces
        dynamic_lmk_b_coords: torch.tensor Lx3, dtype = torch.float32
            The look-up table from neck rotation to barycentric coordinates
        neck_kin_chain: list
            A python list that contains the indices of the joints that form the
            kinematic chain of the neck.
        dtype: torch.dtype, optional
        Returns
        -------
        dyn_lmk_faces_idx: torch.tensor, dtype = torch.long
            A tensor of size BxL that contains the indices of the faces that
            will be used to compute the current dynamic landmarks.
        dyn_lmk_b_coords: torch.tensor, dtype = torch.float32
            A tensor of size BxL that contains the indices of the faces that
            will be used to compute the current dynamic landmarks.
    '''

    dtype = vertices.dtype
    batch_size = vertices.shape[0]

    if pose2rot:
        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1,
                                     neck_kin_chain)
        rot_mats = batch_rodrigues(
            aa_pose.view(-1, 3)).view(batch_size, -1, 3, 3)
    else:
        rot_mats = torch.index_select(
            pose.view(batch_size, -1, 3, 3), 1, neck_kin_chain)

    rel_rot_mat = torch.eye(
        3, device=vertices.device, dtype=dtype).unsqueeze_(dim=0).repeat(
            batch_size, 1, 1)
    for idx in range(len(neck_kin_chain)):
        rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

    y_rot_angle = torch.round(
        torch.clamp(-rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi,
                    max=39)).to(dtype=torch.long)
    neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
    mask = y_rot_angle.lt(-39).to(dtype=torch.long)
    neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
    y_rot_angle = (neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle)

    dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx,
                                           0, y_rot_angle)
    dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords,
                                          0, y_rot_angle)

    return dyn_lmk_faces_idx, dyn_lmk_b_coords


def vertices2landmarks(
    vertices: Tensor,
    faces: Tensor,
    lmk_faces_idx: Tensor,
    lmk_bary_coords: Tensor
) -> Tensor:
    ''' Calculates landmarks by barycentric interpolation
        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices
        faces: torch.tensor Fx3, dtype = torch.long
            The faces of the mesh
        lmk_faces_idx: torch.tensor L, dtype = torch.long
            The tensor with the indices of the faces used to calculate the
            landmarks.
        lmk_bary_coords: torch.tensor Lx3, dtype = torch.float32
            The tensor of barycentric coordinates that are used to interpolate
            the landmarks
        Returns
        -------
        landmarks: torch.tensor BxLx3, dtype = torch.float32
            The coordinates of the landmarks for each mesh in the batch
    '''
    # Extract the indices of the vertices for each face
    # BxLx3
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device

    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
        batch_size, -1, 3)

    lmk_faces += torch.arange(
        batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts

    lmk_vertices = vertices.view(-1, 3)[lmk_faces].view(
        batch_size, -1, 3, 3)

    landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
    return landmarks


def lbs(
    betas: Tensor,
    pose: Tensor,
    v_template: Tensor,
    shapedirs: Tensor,
    posedirs: Tensor,
    J_regressor: Tensor,
    parents: Tensor,
    lbs_weights: Tensor,
    pose2rot: bool = True,
):
    # ) -> Tuple[Tensor, Tensor]:
    ''' Performs Linear Blend Skinning with the given shape and pose parameters
        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional
        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    '''

    batch_size = max(betas.shape[0], pose.shape[0])
    device, dtype = betas.device, betas.dtype

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(
            pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped
    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed


def hybrik(
    betas: Tensor,
    pose_skeleton: Tensor,
    phis: Tensor,
    v_template: Tensor,
    shapedirs: Tensor,
    posedirs: Tensor,
    J_regressor: Tensor,
    parents: Tensor,
    children: Tensor,
    lbs_weights: Tensor,
    leaf_indices: Tensor,
    leaf_thetas: Tensor = None,
    train: bool = True,
    use_hand_pca: bool = False,
    lhand_filter_matrix: Tensor = None,
    rhand_filter_matrix: Tensor = None,
    naive=False
):

    # parents should add leaf joints
    batch_size = max(betas.shape[0], pose_skeleton.shape[0])
    device, dtype = betas.device, betas.dtype
    num_theta = phis.shape[1] + 1

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    if leaf_thetas is not None:
        rest_J = vertices2joints(J_regressor, v_shaped)
    else:
        rest_J_inner = vertices2joints(J_regressor, v_shaped)

        leaf_vertices = v_shaped[:, leaf_indices].clone()
        rest_J = torch.cat([rest_J_inner, leaf_vertices], dim=1)

    # 3. Get the rotation matrics
    if train:
        naive = True
    rot_mats, rotate_rest_pose = batch_inverse_kinematics_transform_naive(
        pose_skeleton.clone(), phis.clone(),
        rest_J.clone(), children, parents, dtype=dtype, train=train,
        leaf_thetas=leaf_thetas,
        naive=naive)

    if use_hand_pca:
        rot_aa = matrix_to_axis_angle(rot_mats.reshape(batch_size, -1, 3, 3))
        lhand_aa = rot_aa[:, 25:40].reshape(batch_size, 45)
        rhand_aa = rot_aa[:, 40:55].reshape(batch_size, 45)

        lhand_aa = torch.einsum('bi,ij->bj', [lhand_aa, lhand_filter_matrix])
        rhand_aa = torch.einsum('bi,ij->bj', [rhand_aa, rhand_filter_matrix])
        lhand_rotmat = axis_angle_to_matrix(lhand_aa.reshape(batch_size, 15, 3))
        rhand_rotmat = axis_angle_to_matrix(rhand_aa.reshape(batch_size, 15, 3))

        rot_mats[:, 25:40] = lhand_rotmat
        rot_mats[:, 40:55] = rhand_rotmat

    test_joints = True
    if test_joints:
        J_transformed, A = batch_rigid_transform(
            rot_mats, rest_J[:, :num_theta].clone(), parents[:num_theta], dtype=dtype)
    else:
        J_transformed = None

    ident = torch.eye(3, dtype=dtype, device=device)
    pose_feature = (rot_mats[:, 1:] - ident).view([batch_size, -1])
    pose_offsets = torch.matmul(pose_feature, posedirs) \
        .view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped.detach()

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed, rot_mats


def batch_inverse_kinematics_transform_naive(
        pose_skeleton,
        phis,
        rest_pose,
        children, parents, dtype=torch.float32, train=False,
        leaf_thetas=None, naive=False):
    """
    Applies a batch of inverse kinematics transfoirm to the joints
    Parameters
    ----------
    pose_skeleton : torch.tensor BxNx3
        Locations of estimated pose skeleton.
    global_orient : torch.tensor Bx1x3x3
        Tensor of global rotation matrices
    phis : torch.tensor BxNx2
        The rotation on bone axis parameters
    rest_pose : torch.tensor Bx(N+1)x3
        Locations of rest_pose. (Template Pose)
    children: dict
        The dictionary that describes the kinematic chidrens for the model
    parents : torch.tensor Bx(N+1)
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32
    Returns
    -------
    rot_mats: torch.tensor Bx(N+1)x3x3
        The rotation matrics of each joints
    rel_transforms : torch.tensor Bx(N+1)x4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """
    batch_size = pose_skeleton.shape[0]
    device = pose_skeleton.device

    rel_rest_pose = rest_pose.clone()
    rel_rest_pose[:, 1:] -= rest_pose[:, parents[1:]].clone()
    rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

    # rotate the T pose
    rotate_rest_pose = torch.zeros_like(rel_rest_pose)
    # set up the root
    rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

    rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
    rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, parents[1:]].clone()
    rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

    # the predicted final pose
    final_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)
    final_pose_skeleton = final_pose_skeleton - final_pose_skeleton[:, 0:1] + rel_rest_pose[:, 0:1]

    rel_rest_pose = rel_rest_pose
    rel_pose_skeleton = rel_pose_skeleton
    final_pose_skeleton = final_pose_skeleton
    rotate_rest_pose = rotate_rest_pose

    assert phis.dim() == 3
    phis = phis / (torch.norm(phis, dim=2, keepdim=True) + 1e-8)

    if train:
        global_orient_mat = batch_get_pelvis_orient(
            rel_pose_skeleton.clone(), rel_rest_pose.clone(), parents, children, dtype)
    else:
        global_orient_mat = batch_get_pelvis_orient_svd(
            rel_pose_skeleton.clone(), rel_rest_pose.clone(), parents, children, dtype)

    rot_mat_chain = [global_orient_mat]
    rot_mat_local = [global_orient_mat]

    representative_children = {
        9: 12,
        15: 70,
        20: 25,
        21: 40
    }
    # leaf nodes rot_mats
    if leaf_thetas is not None:
        assert NotImplementedError

    for i in range(1, parents.shape[0]):
        if children[i] == -1:
            # leaf nodes
            if leaf_thetas is not None:
                raise NotImplementedError

        elif children[i] < -1:
            # elif children[i] < -1:
            # more than one child

            # adaptive
            rotate_rest_pose[:, i] = rotate_rest_pose[:, parents[i]] + torch.matmul(
                rot_mat_chain[parents[i]],
                rel_rest_pose[:, i]
            )

            # naive
            child_final_loc = torch.matmul(
                rot_mat_chain[parents[i]].transpose(1, 2),
                rel_pose_skeleton[:, children[i]])

            if i == 15 and False:
                # only use eyes to recover HEAD orientation
                c1 = 23
                c2 = 24

                if train:
                    children_final_loc = []
                    children_rest_loc = []
                else:
                    middle_final_loc = (rel_pose_skeleton[:, c1] + rel_pose_skeleton[:, c2]) / 2
                    middle_rel_rest = (rel_rest_pose[:, c1] + rel_rest_pose[:, c2]) / 2

                    children_final_loc = [middle_final_loc]
                    children_rest_loc = [middle_rel_rest]
                c_list = [c1, c2]
                random.shuffle(c_list)
                for c in c_list:
                    children_final_loc.append(rel_pose_skeleton[:, c].clone())

                    children_rest_loc.append(rel_rest_pose[:, c].clone())

            else:
                # original
                spine_child = [representative_children[i]]  # choose one "main child"
                for c in range(1, parents.shape[0]):
                    if parents[c] == i and c not in spine_child:
                        if i == 15 and c == 22:
                            continue
                        spine_child.append(c)

                children_final_loc = []
                children_rest_loc = []
                for c in spine_child:
                    children_final_loc.append(rel_pose_skeleton[:, c].clone())
                    children_rest_loc.append(rel_rest_pose[:, c].clone())

                c_list = spine_child

            if train:
                # if True:
                rot_mat = batch_get_children_orient(
                    children_final_loc, children_rest_loc,
                    rot_mat_chain[parents[i]], c_list, dtype)
            else:
                rot_mat = batch_get_children_orient_svd(
                    children_final_loc, children_rest_loc,
                    rot_mat_chain[parents[i]], c_list, dtype)

            rot_mat_chain.append(
                torch.matmul(
                    rot_mat_chain[parents[i]],
                    rot_mat)
            )
            rot_mat_local.append(rot_mat)
        else:
            if children[i] < -1:
                # choose one child and continues training
                children[i] = representative_children[i]

            # (B, 3, 1)
            rotate_rest_pose[:, i] = rotate_rest_pose[:, parents[i]] + torch.matmul(
                rot_mat_chain[parents[i]],
                rel_rest_pose[:, i]
            )
            if i == 22 and not naive:
                assert children[i] == 57
                # child == 57, mouth bottom

                child_final_loc_bottom = torch.matmul(
                    rot_mat_chain[parents[i]].transpose(1, 2),
                    rel_pose_skeleton[:, children[i]])

                child_final_loc_top = torch.matmul(
                    rot_mat_chain[parents[i]].transpose(1, 2),
                    rel_pose_skeleton[:, 70])

                _, sin1, cos1 = vectors2aa(
                    child_final_loc_top, child_final_loc_bottom)

                child_rest_loc_bottom = rel_rest_pose[:, children[i]]
                child_rest_loc_top = rel_rest_pose[:, 70]
                child_rest_loc_top[:, 0] = child_rest_loc_bottom[:, 0]

                rot2, axis2, _, _ = vectors2rotmat2(
                    child_rest_loc_top, child_rest_loc_bottom, dtype=child_final_loc_top.dtype)

                # Convert location revolve to rot_mat by rodrigues
                # (B, 1, 1)
                rx, ry, rz = torch.split(axis2, 1, dim=1)
                zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)

                K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
                    .view((batch_size, 3, 3))
                ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
                rot1 = ident + sin1 * K + (1 - cos1) * torch.bmm(K, K)

                rot_mat = torch.matmul(
                    rot1,
                    rot2.transpose(1, 2),
                )
                rot_aa = matrix_to_axis_angle(rot_mat)
                angle = torch.norm(rot_aa, dim=1, keepdim=True)
                axis = rot_aa / angle
                angle = (angle - 10 / 180 * 3.14).clamp_min(0)
                rot_aa = axis * angle
                rot_mat = axis_angle_to_matrix(rot_aa)

                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[parents[i]],
                    rot_mat))
                rot_mat_local.append(rot_mat)

                continue

            # if i in [16, 17] and not naive:
            #     wrist_pos = final_pose_skeleton[:, children[children[i]]]
            #     elbow_pos_1 = final_pose_skeleton[:, children[i]]
            #     shoulder_pos = rotate_rest_pose[:, i]

            #     r2 = torch.norm(rel_rest_pose[:, children[i]], dim=1, keepdim=True)
            #     r0 = torch.norm(rel_rest_pose[:, children[children[i]]], dim=1, keepdim=True)
            #     r = torch.norm(wrist_pos - shoulder_pos, dim=1, keepdim=True)

            #     k = (r * r + r0 * r0 - r2 * r2) / (2 * r * r + 1e-6)
            #     k = k.clamp(min=0, max=1)

            #     vec_a = wrist_pos - shoulder_pos
            #     vec_a = vec_a / (torch.norm(vec_a, dim=1, keepdim=True) + 1e-6)
            #     vec_b = elbow_pos_1 - shoulder_pos

            #     vec_c = vec_b - torch.sum(vec_a * vec_b, dim=1, keepdim=True) * vec_a
            #     vec_c = vec_c / (torch.norm(vec_c, dim=1, keepdim=True) + 1e-6)

            #     k2 = torch.sqrt((r0 * r0 - k * k * r * r).clamp_min(0))
            #     vec_c = k2 * vec_c

            #     elbow_pos_final = (wrist_pos - shoulder_pos) * (1 - k) + shoulder_pos + vec_c

            #     child_final_loc = elbow_pos_final - rotate_rest_pose[:, i]

            #     child_final_loc = torch.matmul(
            #         rot_mat_chain[parents[i]].transpose(1, 2),
            #         child_final_loc)
            if i in [18, 19, 4, 5] and not naive:
                # (B, 3, 1)
                child_final_loc = final_pose_skeleton[:, children[i]] - rotate_rest_pose[:, i]

                child_final_loc = torch.matmul(
                    rot_mat_chain[parents[i]].transpose(1, 2),
                    child_final_loc)
            else:
                # naive
                child_final_loc = torch.matmul(
                    rot_mat_chain[parents[i]].transpose(1, 2),
                    rel_pose_skeleton[:, children[i]])

            child_rest_loc = rel_rest_pose[:, children[i]]
            # (B, 1, 1)
            child_final_norm = torch.norm(child_final_loc, dim=1, keepdim=True)
            child_rest_norm = torch.norm(child_rest_loc, dim=1, keepdim=True)

            # child_final_norm = torch.norm(child_final_loc, dim=1, keepdim=True)

            # (B, 3, 1)
            axis = torch.cross(child_rest_loc, child_final_loc, dim=1)
            axis_norm = torch.norm(axis, dim=1, keepdim=True)

            # (B, 1, 1)
            cos = torch.sum(child_rest_loc * child_final_loc, dim=1, keepdim=True) / (child_rest_norm * child_final_norm + 1e-8)
            sin = axis_norm / (child_rest_norm * child_final_norm + 1e-8)

            # (B, 3, 1)
            axis = axis / (axis_norm + 1e-8)

            # Convert location revolve to rot_mat by rodrigues
            # (B, 1, 1)
            rx, ry, rz = torch.split(axis, 1, dim=1)
            zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)

            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
                .view((batch_size, 3, 3))
            ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
            rot_mat_loc = ident + sin * K + (1 - cos) * torch.bmm(K, K)

            # Convert spin to rot_mat
            # (B, 3, 1)
            spin_axis = child_rest_loc / child_rest_norm
            # (B, 1, 1)
            rx, ry, rz = torch.split(spin_axis, 1, dim=1)
            zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)
            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
                .view((batch_size, 3, 3))
            ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
            # (B, 1, 1)
            cos, sin = torch.split(phis[:, i - 1], 1, dim=1)
            cos = torch.unsqueeze(cos, dim=2)
            sin = torch.unsqueeze(sin, dim=2)
            rot_mat_spin = ident + sin * K + (1 - cos) * torch.bmm(K, K)
            rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)

            rot_mat_chain.append(torch.matmul(
                rot_mat_chain[parents[i]],
                rot_mat))
            rot_mat_local.append(rot_mat)

    # (B, K + 1, 3, 3)
    rot_mats = torch.stack(rot_mat_local, dim=1)

    return rot_mats, rotate_rest_pose.squeeze(-1)


def batch_get_pelvis_orient_svd(rel_pose_skeleton, rel_rest_pose, parents, children, dtype):
    # if isinstance(rel_pose_skeleton, list):
    #     device = rel_pose_skeleton[0].device
    # else:
    #     device = rel_pose_skeleton.device

    pelvis_child = []
    for i in range(parents.shape[0]):
        if parents[i] == 0 and i not in pelvis_child:
            pelvis_child.append(i)

    rest_mat = []
    target_mat = []
    for child in pelvis_child:
        rest_mat.append(rel_rest_pose[:, child].clone())
        target_mat.append(rel_pose_skeleton[:, child].clone())

    rest_mat = torch.cat(rest_mat, dim=2)
    target_mat = torch.cat(target_mat, dim=2)
    S = rest_mat.bmm(target_mat.transpose(1, 2))

    mask_zero = S.sum(dim=(1, 2))

    S_non_zero = S[mask_zero != 0].reshape(-1, 3, 3)

    U, Sig, V = torch.svd(S_non_zero)
    # U, Sig, V = torch.svd(S_non_zero.cpu())
    # U = U.to(device=device)
    # V = V.to(device=device)

    assert (Sig[:, 0] >= Sig[:, 1]).all() and (Sig[:, 1] >= Sig[:, 2]).all(), Sig

    rot_mat = torch.zeros_like(S)
    rot_mat[mask_zero == 0] = torch.eye(3, device=S.device)

    # rot_mat_non_zero = torch.bmm(V, U.transpose(1, 2))

    det_u_vt = torch.det(torch.bmm(V, U.transpose(1, 2)))
    det_modify_mat = torch.eye(3, device=U.device).unsqueeze(0).expand(U.shape[0], -1, -1).clone()
    det_modify_mat[:, 2, 2] = det_u_vt
    rot_mat_non_zero = torch.bmm(torch.bmm(V, det_modify_mat), U.transpose(1, 2))

    rot_mat[mask_zero != 0] = rot_mat_non_zero

    assert torch.sum(torch.isnan(rot_mat)) == 0, ('rot_mat', rot_mat)

    return rot_mat


def batch_get_pelvis_orient(rel_pose_skeleton, rel_rest_pose, parents, children, dtype):
    batch_size = rel_pose_skeleton.shape[0]
    device = rel_pose_skeleton.device

    pelvis_child = []
    for i in range(parents.shape[0]):
        if parents[i] == 0 and i not in pelvis_child:
            pelvis_child.append(i)

    spine_final_loc = rel_pose_skeleton[:, pelvis_child[-1]].clone()  # TODO: may get worse performance if choose other child ?
    spine_rest_loc = rel_rest_pose[:, pelvis_child[-1]].clone()
    spine_norm = torch.norm(spine_final_loc, dim=1, keepdim=True)
    spine_norm = spine_final_loc / (spine_norm + 1e-8)

    rot_mat_spine = vectors2rotmat(spine_rest_loc, spine_final_loc, dtype)

    assert torch.sum(torch.isnan(rot_mat_spine)
                     ) == 0, ('rot_mat_spine', rot_mat_spine, spine_final_loc, rel_pose_skeleton)
    center_final_loc = 0
    center_rest_loc = 0
    for child in pelvis_child:
        if child == pelvis_child[-1]:
            continue
        center_final_loc = center_final_loc + rel_pose_skeleton[:, child].clone()
        center_rest_loc = center_rest_loc + rel_rest_pose[:, child].clone()

    center_final_loc = center_final_loc / (len(pelvis_child) - 1)
    center_rest_loc = center_rest_loc / (len(pelvis_child) - 1)

    center_rest_loc = torch.matmul(rot_mat_spine, center_rest_loc)

    center_final_loc = center_final_loc - torch.sum(center_final_loc * spine_norm, dim=1, keepdim=True) * spine_norm
    center_rest_loc = center_rest_loc - torch.sum(center_rest_loc * spine_norm, dim=1, keepdim=True) * spine_norm

    center_final_loc_norm = torch.norm(center_final_loc, dim=1, keepdim=True)
    center_rest_loc_norm = torch.norm(center_rest_loc, dim=1, keepdim=True)

    # print(center_final_loc_norm, center_rest_loc_norm)

    # (B, 3, 1)
    axis = torch.cross(center_rest_loc, center_final_loc, dim=1)
    axis_norm = torch.norm(axis, dim=1, keepdim=True)

    # (B, 1, 1)
    cos = torch.sum(center_rest_loc * center_final_loc, dim=1, keepdim=True) / \
        (center_rest_loc_norm * center_final_loc_norm + 1e-8)
    sin = axis_norm / (center_rest_loc_norm * center_final_loc_norm + 1e-8)

    assert torch.sum(torch.isnan(cos)
                     ) == 0, ('cos', cos)
    assert torch.sum(torch.isnan(sin)
                     ) == 0, ('sin', sin)
    # (B, 3, 1)
    axis = axis / (axis_norm + 1e-8)

    # Convert location revolve to rot_mat by rodrigues
    # (B, 1, 1)
    rx, ry, rz = torch.split(axis, 1, dim=1)
    zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)

    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat_center = ident + sin * K + (1 - cos) * torch.bmm(K, K)

    rot_mat = torch.matmul(rot_mat_center, rot_mat_spine)

    return rot_mat


def batch_get_children_orient_svd(rel_pose_skeleton, rel_rest_pose, rot_mat_chain_parent, children_list, dtype):
    # if isinstance(rel_pose_skeleton, list):
    #     device = rel_pose_skeleton[0].device
    # else:
    #     device = rel_pose_skeleton.device

    rest_mat = []
    target_mat = []
    for c, child in enumerate(children_list):
        if isinstance(rel_pose_skeleton, list):
            target = rel_pose_skeleton[c].clone()
            template = rel_rest_pose[c].clone()
        else:
            target = rel_pose_skeleton[:, child].clone()
            template = rel_rest_pose[:, child].clone()

        target = torch.matmul(
            rot_mat_chain_parent.transpose(1, 2),
            target)

        target_mat.append(target)
        rest_mat.append(template)

    rest_mat = torch.cat(rest_mat, dim=2)
    target_mat = torch.cat(target_mat, dim=2)
    S = rest_mat.bmm(target_mat.transpose(1, 2))

    U, Sig, V = torch.svd(S)
    # U, Sig, V = torch.svd(S.cpu())
    # U = U.to(device=device)
    # V = V.to(device=device)

    # assert (Sig[:, 0] >= Sig[:, 1]).all() and (Sig[:, 1] >= Sig[:, 2]).all(), Sig

    det_u_vt = torch.det(torch.bmm(V, U.transpose(1, 2)))
    det_modify_mat = torch.eye(3, device=U.device).unsqueeze(0).expand(U.shape[0], -1, -1).clone()
    det_modify_mat[:, 2, 2] = det_u_vt
    rot_mat = torch.bmm(torch.bmm(V, det_modify_mat), U.transpose(1, 2))
    assert torch.sum(torch.isnan(rot_mat)) == 0, ('3children rot_mat', rot_mat)

    return rot_mat


def batch_get_children_orient(rel_pose_skeleton, rel_rest_pose, rot_mat_chain_parent, children_list, dtype):
    # the first child is the main-align joint
    assert isinstance(rel_pose_skeleton, list)
    batch_size = rel_pose_skeleton[0].shape[0]
    device = rel_pose_skeleton[0].device

    spine_final_loc = rel_pose_skeleton[0].clone()

    spine_final_loc = torch.matmul(
        rot_mat_chain_parent.transpose(1, 2),
        spine_final_loc)

    spine_rest_loc = rel_rest_pose[0].clone()

    spine_norm = spine_final_loc / torch.norm(spine_final_loc, dim=1, keepdim=True)
    rot_mat_spine = vectors2rotmat(spine_rest_loc, spine_final_loc, dtype)

    center_final_loc = 0
    center_rest_loc = 0
    for c, child in enumerate(children_list):
        if c == 0:
            continue
        if isinstance(rel_pose_skeleton, list):
            center_final_loc += rel_pose_skeleton[c].clone()
            center_rest_loc += rel_rest_pose[c].clone()
        else:
            center_final_loc += rel_pose_skeleton[:, child].clone()
            center_rest_loc += rel_rest_pose[:, child].clone()

    center_final_loc /= (len(children_list) - 1)
    center_rest_loc /= (len(children_list) - 1)

    center_final_loc = torch.matmul(
        rot_mat_chain_parent.transpose(1, 2),
        center_final_loc)

    center_rest_loc = torch.matmul(rot_mat_spine, center_rest_loc)

    center_final_loc = center_final_loc - torch.sum(center_final_loc * spine_norm, dim=1, keepdim=True) * spine_norm
    center_rest_loc = center_rest_loc - torch.sum(center_rest_loc * spine_norm, dim=1, keepdim=True) * spine_norm

    center_final_loc_norm = torch.norm(center_final_loc, dim=1, keepdim=True)
    center_rest_loc_norm = torch.norm(center_rest_loc, dim=1, keepdim=True)

    # (B, 3, 1)
    axis = torch.cross(center_rest_loc, center_final_loc, dim=1)

    # (B, 1, 1)
    cos = torch.sum(center_rest_loc * center_final_loc, dim=1, keepdim=True) / \
        (center_rest_loc_norm * center_final_loc_norm + 1e-8)
    sin = torch.norm(axis, dim=1, keepdim=True) / (center_rest_loc_norm * center_final_loc_norm + 1e-8)

    # (B, 3, 1)
    axis = axis / (torch.norm(axis, dim=1, keepdim=True) + 1e-8)

    # Convert location revolve to rot_mat by rodrigues
    # (B, 1, 1)
    rx, ry, rz = torch.split(axis, 1, dim=1)
    zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)

    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat_center = ident + sin * K + (1 - cos) * torch.bmm(K, K)

    rot_mat = torch.matmul(rot_mat_center, rot_mat_spine)

    return rot_mat


def vectors2rotmat(vec_rest, vec_final, dtype):
    batch_size = vec_final.shape[0]
    device = vec_final.device

    # (B, 1, 1)
    vec_final_norm = torch.norm(vec_final, dim=1, keepdim=True)
    vec_rest_norm = torch.norm(vec_rest, dim=1, keepdim=True)

    # (B, 3, 1)
    axis = torch.cross(vec_rest, vec_final, dim=1)
    axis_norm = torch.norm(axis, dim=1, keepdim=True)

    # (B, 1, 1)
    cos = torch.sum(vec_rest * vec_final, dim=1, keepdim=True) / (vec_rest_norm * vec_final_norm + 1e-8)
    sin = axis_norm / (vec_rest_norm * vec_final_norm + 1e-8)

    # (B, 3, 1)
    axis = axis / (axis_norm + 1e-8)

    # Convert location revolve to rot_mat by rodrigues
    # (B, 1, 1)
    rx, ry, rz = torch.split(axis, 1, dim=1)
    zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)

    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat_loc = ident + sin * K + (1 - cos) * torch.bmm(K, K)

    return rot_mat_loc


def vectors2rotmat2(vec_rest, vec_final, dtype):
    batch_size = vec_final.shape[0]
    device = vec_final.device

    # (B, 1, 1)
    vec_final_norm = torch.norm(vec_final, dim=1, keepdim=True)
    vec_rest_norm = torch.norm(vec_rest, dim=1, keepdim=True)

    # (B, 3, 1)
    axis = torch.cross(vec_rest, vec_final, dim=1)
    axis_norm = torch.norm(axis, dim=1, keepdim=True)

    # (B, 1, 1)
    cos = torch.sum(vec_rest * vec_final, dim=1, keepdim=True) / (vec_rest_norm * vec_final_norm + 1e-8)
    sin = axis_norm / (vec_rest_norm * vec_final_norm + 1e-8)

    # (B, 3, 1)
    axis = axis / (axis_norm + 1e-8)

    # Convert location revolve to rot_mat by rodrigues
    # (B, 1, 1)
    rx, ry, rz = torch.split(axis, 1, dim=1)
    zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)

    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat_loc = ident + sin * K + (1 - cos) * torch.bmm(K, K)

    return rot_mat_loc, axis, cos, sin


def vectors2aa(vec_rest, vec_final):
    # (B, 1, 1)
    vec_final_norm = torch.norm(vec_final, dim=1, keepdim=True)
    vec_rest_norm = torch.norm(vec_rest, dim=1, keepdim=True)

    # (B, 3, 1)
    axis = torch.cross(vec_rest, vec_final, dim=1)
    axis_norm = torch.norm(axis, dim=1, keepdim=True)

    # (B, 1, 1)
    cos = torch.sum(vec_rest * vec_final, dim=1, keepdim=True) / (vec_rest_norm * vec_final_norm + 1e-8)
    sin = axis_norm / (vec_rest_norm * vec_final_norm + 1e-8)

    # (B, 3, 1)
    axis = axis / (axis_norm + 1e-8)

    return axis, sin, cos


def lbs_get_twist(
    betas: Tensor,
    pose: Tensor,
    v_template: Tensor,
    shapedirs: Tensor,
    posedirs: Tensor,
    J_regressor: Tensor,
    parents: Tensor,
    lbs_weights: Tensor,
    leaf_indices: Tensor,
    pose2rot: bool = True,
):

    batch_size = max(betas.shape[0], pose.shape[0])
    # device, dtype = betas.device, betas.dtype

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])

    else:
        rot_mats = pose.view(batch_size, -1, 3, 3)

    leaf_jts = v_shaped[:, leaf_indices]
    J_concat = torch.cat([J, leaf_jts], dim=1)
    twist_angle = get_twist(rot_mats, J_concat.clone(), parents)

    return twist_angle


def get_twist(rot_mats, joints, parents):
    joints = torch.unsqueeze(joints, dim=-1)
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]].clone()

    # modified by xuchao
    childs = -torch.ones((parents.shape[0]), dtype=parents.dtype, device=parents.device)
    for i in range(1, parents.shape[0]):
        childs[parents[i]] = i

    dtype = rot_mats.dtype
    batch_size = rot_mats.shape[0]
    device = rot_mats.device

    angle_twist = []
    # error = False
    for i in range(1, rot_mats.shape[1]):
        # modified by xuchao
        if childs[i] < 0:
            angle_twist.append(torch.zeros((batch_size, 1), dtype=rot_mats.dtype, device=rot_mats.device))
            continue

        u = rel_joints[:, childs[i]]
        rot = rot_mats[:, i]

        v = torch.matmul(rot, u)

        u_norm = torch.norm(u, dim=1, keepdim=True)
        v_norm = torch.norm(v, dim=1, keepdim=True)

        axis = torch.cross(u, v, dim=1)
        axis_norm = torch.norm(axis, dim=1, keepdim=True)

        # (B, 1, 1)
        cos = torch.sum(u * v, dim=1, keepdim=True) / (u_norm * v_norm + 1e-8)
        sin = axis_norm / (u_norm * v_norm + 1e-8)

        # (B, 3, 1)
        axis = axis / (axis_norm + 1e-8)

        # Convert location revolve to rot_mat by rodrigues
        # (B, 1, 1)
        rx, ry, rz = torch.split(axis, 1, dim=1)
        zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)

        K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
            .view((batch_size, 3, 3))
        ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
        rot_mat_pivot = ident + sin * K + (1 - cos) * torch.bmm(K, K)

        rot_mat_twist = torch.matmul(rot_mat_pivot.transpose(1, 2), rot)
        # _, axis, angle = rotmat_to_aa(rot_mat_twist)
        rot_aa_twist = matrix_to_axis_angle(rot_mat_twist)
        angle = torch.norm(rot_aa_twist, dim=1, keepdim=True)
        axis = rot_aa_twist / angle

        axis = axis / torch.norm(axis, dim=1, keepdim=True)
        spin_axis = u / u_norm
        spin_axis = spin_axis.squeeze(-1)

        pos = torch.norm(spin_axis - axis, dim=1)
        neg = torch.norm(spin_axis + axis, dim=1)

        if float(neg) < float(pos):
            if float(pos) > 1.8:
                angle_twist.append(-1 * angle)
            else:
                angle_twist.append(torch.ones_like(angle) * -999)
                print('error', float(pos), i)
        else:
            if float(neg) > 1.8:
                angle_twist.append(angle)
            else:
                angle_twist.append(torch.ones_like(angle) * -999)
                print('error', float(neg), i)

    angle_twist = torch.stack(angle_twist, dim=1)

    return angle_twist


def rotmat_to_aa(rotmat):
    batch_size = rotmat.shape[0]

    r11 = rotmat[:, 0, 0]
    r22 = rotmat[:, 1, 1]
    r33 = rotmat[:, 2, 2]

    r12 = rotmat[:, 0, 1]
    r21 = rotmat[:, 1, 0]
    r13 = rotmat[:, 0, 2]
    r31 = rotmat[:, 2, 0]
    r23 = rotmat[:, 1, 2]
    r32 = rotmat[:, 2, 1]

    angle_zero_mask = (-1e-6 < (r11 + r22 + r33 - 1) / 2 + 1) & ((r11 + r22 + r33 - 1) / 2 + 1 < 1e-6)

    tmp = torch.clamp((r11 + r22 + r33 - 1) / 2, min=-1.0, max=1.0)
    angle1 = torch.acos(-torch.ones_like(r11))
    angle = torch.acos(tmp)

    if torch.isnan(angle).any() or torch.isinf(angle).any():
        print(angle, (r11 + r22 + r33 - 1) / 2)

    angle[angle_zero_mask] = angle1[angle_zero_mask]

    angle = angle.unsqueeze(dim=1)

    axis = torch.zeros((batch_size, 3), device=rotmat.device)
    axis[:, 0] = r32 - r23
    axis[:, 1] = r13 - r31
    axis[:, 2] = r21 - r12
    # axis = axis / (2 * torch.sin(angle) + 1e-8)
    axis = axis / torch.sqrt(torch.sum(axis**2, dim=-1) + 1e-10).unsqueeze(-1)

    aa = axis * angle
    return aa, axis, angle


def vertices2joints(J_regressor: Tensor, vertices: Tensor) -> Tensor:
    ''' Calculates the 3D joint locations from the vertices
    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices
    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    '''

    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


def blend_shapes(betas: Tensor, shape_disps: Tensor) -> Tensor:
    ''' Calculates the per vertex displacement due to the blend shapes
    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes
    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    '''

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape


def batch_rodrigues(
    rot_vecs: Tensor,
    epsilon: float = 1e-8,
) -> Tensor:
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
    device, dtype = rot_vecs.device, rot_vecs.dtype

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


def transform_mat(R: Tensor, t: Tensor) -> Tensor:
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(
    rot_mats: Tensor,
    joints: Tensor,
    parents: Tensor,
    dtype=torch.float32
) -> Tensor:
    """
    Applies a batch of rigid transformations to the joints
    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32
    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms


def mat2quat(rotation_matrix):
    assert rotation_matrix.shape[1:] == (3, 3)
    rot_mat = rotation_matrix.reshape(-1, 3, 3)
    hom = torch.tensor([0, 0, 1], dtype=torch.float32,
                       device=rotation_matrix.device)
    hom = hom.reshape(1, 3, 1).expand(rot_mat.shape[0], -1, -1)
    rotation_matrix = torch.cat([rot_mat, hom], dim=-1)

    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia
    Convert 3x4 rotation matrix to 4d quaternion vector
    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201
    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.
    Return:
        Tensor: the rotation in quaternion
    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`
    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q
