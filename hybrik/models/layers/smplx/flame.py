import os
import os.path as osp
import pickle
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .body_models import SMPL
from .lbs import find_dynamic_lmk_idx_and_bcoords, lbs, vertices2landmarks
from .utils import (FLAMEOutput, Struct, Tensor, find_joint_kin_chain, to_np,
                    to_tensor)


class FLAME(SMPL):
    NUM_JOINTS = 5
    SHAPE_SPACE_DIM = 300
    EXPRESSION_SPACE_DIM = 100
    NECK_IDX = 0

    def __init__(
        self,
        model_path: str,
        data_struct=None,
        num_expression_coeffs=10,
        create_expression: bool = True,
        expression: Optional[Tensor] = None,
        create_neck_pose: bool = True,
        neck_pose: Optional[Tensor] = None,
        create_jaw_pose: bool = True,
        jaw_pose: Optional[Tensor] = None,
        create_leye_pose: bool = True,
        leye_pose: Optional[Tensor] = None,
        create_reye_pose=True,
        reye_pose: Optional[Tensor] = None,
        use_face_contour=False,
        batch_size: int = 1,
        gender: str = 'neutral',
        dtype: torch.dtype = torch.float32,
        ext='pkl',
        **kwargs
    ) -> None:
        ''' FLAME model constructor
            Parameters
            ----------
            model_path: str
                The path to the folder or to the file where the model
                parameters are stored
            num_expression_coeffs: int, optional
                Number of expression components to use
                (default = 10).
            create_expression: bool, optional
                Flag for creating a member variable for the expression space
                (default = True).
            expression: torch.tensor, optional, Bx10
                The default value for the expression member variable.
                (default = None)
            create_neck_pose: bool, optional
                Flag for creating a member variable for the neck pose.
                (default = False)
            neck_pose: torch.tensor, optional, Bx3
                The default value for the neck pose variable.
                (default = None)
            create_jaw_pose: bool, optional
                Flag for creating a member variable for the jaw pose.
                (default = False)
            jaw_pose: torch.tensor, optional, Bx3
                The default value for the jaw pose variable.
                (default = None)
            create_leye_pose: bool, optional
                Flag for creating a member variable for the left eye pose.
                (default = False)
            leye_pose: torch.tensor, optional, Bx10
                The default value for the left eye pose variable.
                (default = None)
            create_reye_pose: bool, optional
                Flag for creating a member variable for the right eye pose.
                (default = False)
            reye_pose: torch.tensor, optional, Bx10
                The default value for the right eye pose variable.
                (default = None)
            use_face_contour: bool, optional
                Whether to compute the keypoints that form the facial contour
            batch_size: int, optional
                The batch size used for creating the member variables
            gender: str, optional
                Which gender to load
            dtype: torch.dtype
                The data type for the created variables
        '''
        model_fn = f'FLAME_{gender.upper()}.{ext}'
        flame_path = os.path.join(model_path, model_fn)
        assert osp.exists(flame_path), 'Path {} does not exist!'.format(
            flame_path)
        if ext == 'npz':
            file_data = np.load(flame_path, allow_pickle=True)
        elif ext == 'pkl':
            with open(flame_path, 'rb') as smpl_file:
                file_data = pickle.load(smpl_file, encoding='latin1')
        else:
            raise ValueError('Unknown extension: {}'.format(ext))
        data_struct = Struct(**file_data)

        super(FLAME, self).__init__(
            model_path=model_path,
            data_struct=data_struct,
            dtype=dtype,
            batch_size=batch_size,
            gender=gender,
            ext=ext,
            **kwargs)

        self.use_face_contour = use_face_contour

        self.vertex_joint_selector.extra_joints_idxs = to_tensor(
            [], dtype=torch.long)

        if create_neck_pose:
            if neck_pose is None:
                default_neck_pose = torch.zeros([batch_size, 3], dtype=dtype)
            else:
                default_neck_pose = torch.tensor(neck_pose, dtype=dtype)
            neck_pose_param = nn.Parameter(
                default_neck_pose, requires_grad=True)
            self.register_parameter('neck_pose', neck_pose_param)

        if create_jaw_pose:
            if jaw_pose is None:
                default_jaw_pose = torch.zeros([batch_size, 3], dtype=dtype)
            else:
                default_jaw_pose = torch.tensor(jaw_pose, dtype=dtype)
            jaw_pose_param = nn.Parameter(default_jaw_pose,
                                          requires_grad=True)
            self.register_parameter('jaw_pose', jaw_pose_param)

        if create_leye_pose:
            if leye_pose is None:
                default_leye_pose = torch.zeros([batch_size, 3], dtype=dtype)
            else:
                default_leye_pose = torch.tensor(leye_pose, dtype=dtype)
            leye_pose_param = nn.Parameter(default_leye_pose,
                                           requires_grad=True)
            self.register_parameter('leye_pose', leye_pose_param)

        if create_reye_pose:
            if reye_pose is None:
                default_reye_pose = torch.zeros([batch_size, 3], dtype=dtype)
            else:
                default_reye_pose = torch.tensor(reye_pose, dtype=dtype)
            reye_pose_param = nn.Parameter(default_reye_pose,
                                           requires_grad=True)
            self.register_parameter('reye_pose', reye_pose_param)

        shapedirs = data_struct.shapedirs
        if len(shapedirs.shape) < 3:
            shapedirs = shapedirs[:, :, None]
        if (shapedirs.shape[-1] < self.SHAPE_SPACE_DIM +
                self.EXPRESSION_SPACE_DIM):
            print(f'WARNING: You are using a {self.name()} model, with only'
                  ' 10 shape and 10 expression coefficients.')
            expr_start_idx = 10
            expr_end_idx = 20
            num_expression_coeffs = min(num_expression_coeffs, 10)
        else:
            expr_start_idx = self.SHAPE_SPACE_DIM
            expr_end_idx = self.SHAPE_SPACE_DIM + num_expression_coeffs
            num_expression_coeffs = min(
                num_expression_coeffs, self.EXPRESSION_SPACE_DIM)

        self._num_expression_coeffs = num_expression_coeffs

        expr_dirs = shapedirs[:, :, expr_start_idx:expr_end_idx]
        self.register_buffer(
            'expr_dirs', to_tensor(to_np(expr_dirs), dtype=dtype))

        if create_expression:
            if expression is None:
                default_expression = torch.zeros(
                    [batch_size, self.num_expression_coeffs], dtype=dtype)
            else:
                default_expression = torch.tensor(expression, dtype=dtype)
            expression_param = nn.Parameter(default_expression,
                                            requires_grad=True)
            self.register_parameter('expression', expression_param)

        # The pickle file that contains the barycentric coordinates for
        # regressing the landmarks
        landmark_bcoord_filename = osp.join(
            model_path, 'flame_static_embedding.pkl')

        with open(landmark_bcoord_filename, 'rb') as fp:
            landmarks_data = pickle.load(fp, encoding='latin1')

        lmk_faces_idx = landmarks_data['lmk_face_idx'].astype(np.int64)
        self.register_buffer('lmk_faces_idx',
                             torch.tensor(lmk_faces_idx, dtype=torch.long))
        lmk_bary_coords = landmarks_data['lmk_b_coords']
        self.register_buffer('lmk_bary_coords',
                             torch.tensor(lmk_bary_coords, dtype=dtype))
        if self.use_face_contour:
            face_contour_path = os.path.join(
                model_path, 'flame_dynamic_embedding.npy')
            contour_embeddings = np.load(face_contour_path,
                                         allow_pickle=True,
                                         encoding='latin1')[()]

            dynamic_lmk_faces_idx = np.array(
                contour_embeddings['lmk_face_idx'], dtype=np.int64)
            dynamic_lmk_faces_idx = torch.tensor(
                dynamic_lmk_faces_idx,
                dtype=torch.long)
            self.register_buffer('dynamic_lmk_faces_idx',
                                 dynamic_lmk_faces_idx)

            dynamic_lmk_b_coords = torch.tensor(
                contour_embeddings['lmk_b_coords'], dtype=dtype)
            self.register_buffer(
                'dynamic_lmk_bary_coords', dynamic_lmk_b_coords)

            neck_kin_chain = find_joint_kin_chain(self.NECK_IDX, self.parents)
            self.register_buffer(
                'neck_kin_chain',
                torch.tensor(neck_kin_chain, dtype=torch.long))

    @property
    def num_expression_coeffs(self):
        return self._num_expression_coeffs

    def name(self) -> str:
        return 'FLAME'

    def extra_repr(self):
        msg = [
            super(FLAME, self).extra_repr(),
            f'Number of Expression Coefficients: {self.num_expression_coeffs}',
            f'Use face contour: {self.use_face_contour}',
        ]
        return '\n'.join(msg)

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        neck_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        expression: Optional[Tensor] = None,
        jaw_pose: Optional[Tensor] = None,
        leye_pose: Optional[Tensor] = None,
        reye_pose: Optional[Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        **kwargs
    ) -> FLAMEOutput:
        '''
        Forward pass for the SMPLX model
            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3
                If given, ignore the member variable and use it as the global
                rotation of the body. Useful if someone wishes to predicts this
                with an external model. (default=None)
            betas: torch.tensor, optional, shape Bx10
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            expression: torch.tensor, optional, shape Bx10
                If given, ignore the member variable `expression` and use it
                instead. For example, it can used if expression parameters
                `expression` are predicted from some external model.
            jaw_pose: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `jaw_pose` and
                use this instead. It should either joint rotations in
                axis-angle format.
            jaw_pose: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `jaw_pose` and
                use this instead. It should either joint rotations in
                axis-angle format.
            transl: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `transl` and use it
                instead. For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full axis-angle pose vector (default=False)
            Returns
            -------
                output: ModelOutput
                A named tuple of type `ModelOutput`
        '''

        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (global_orient if global_orient is not None else
                         self.global_orient)
        jaw_pose = jaw_pose if jaw_pose is not None else self.jaw_pose
        neck_pose = neck_pose if neck_pose is not None else self.neck_pose

        leye_pose = leye_pose if leye_pose is not None else self.leye_pose
        reye_pose = reye_pose if reye_pose is not None else self.reye_pose

        betas = betas if betas is not None else self.betas
        expression = expression if expression is not None else self.expression

        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None:
            if hasattr(self, 'transl'):
                transl = self.transl

        full_pose = torch.cat(
            [global_orient, neck_pose, jaw_pose, leye_pose, reye_pose], dim=1)

        batch_size = max(betas.shape[0], global_orient.shape[0],
                         jaw_pose.shape[0])
        # Concatenate the shape and expression coefficients
        scale = int(batch_size / betas.shape[0])
        if scale > 1:
            betas = betas.expand(scale, -1)
        shape_components = torch.cat([betas, expression], dim=-1)
        shapedirs = torch.cat([self.shapedirs, self.expr_dirs], dim=-1)

        vertices, joints = lbs(shape_components, full_pose, self.v_template,
                               shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=pose2rot,
                               )

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(
            dim=0).expand(batch_size, -1).contiguous()
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(
            batch_size, 1, 1)
        if self.use_face_contour:
            lmk_idx_and_bcoords = find_dynamic_lmk_idx_and_bcoords(
                vertices, full_pose, self.dynamic_lmk_faces_idx,
                self.dynamic_lmk_bary_coords,
                self.neck_kin_chain,
                pose2rot=True,
            )
            dyn_lmk_faces_idx, dyn_lmk_bary_coords = lmk_idx_and_bcoords
            lmk_faces_idx = torch.cat([lmk_faces_idx,
                                       dyn_lmk_faces_idx], 1)
            lmk_bary_coords = torch.cat(
                [lmk_bary_coords.expand(batch_size, -1, -1),
                 dyn_lmk_bary_coords], 1)

        landmarks = vertices2landmarks(vertices, self.faces_tensor,
                                       lmk_faces_idx,
                                       lmk_bary_coords)

        # Add any extra joints that might be needed
        joints = self.vertex_joint_selector(vertices, joints)
        # Add the landmarks to the joints
        joints = torch.cat([joints, landmarks], dim=1)

        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints=joints, vertices=vertices)

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        output = FLAMEOutput(vertices=vertices if return_verts else None,
                             joints=joints,
                             betas=betas,
                             expression=expression,
                             global_orient=global_orient,
                             neck_pose=neck_pose,
                             jaw_pose=jaw_pose,
                             full_pose=full_pose if return_full_pose else None)
        return output


class FLAMELayer(FLAME):
    def __init__(self, *args, **kwargs) -> None:
        ''' FLAME as a layer model constructor '''
        super(FLAMELayer, self).__init__(
            create_betas=False,
            create_expression=False,
            create_global_orient=False,
            create_neck_pose=False,
            create_jaw_pose=False,
            create_leye_pose=False,
            create_reye_pose=False,
            *args,
            **kwargs)

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        neck_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        expression: Optional[Tensor] = None,
        jaw_pose: Optional[Tensor] = None,
        leye_pose: Optional[Tensor] = None,
        reye_pose: Optional[Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        **kwargs
    ) -> FLAMEOutput:
        '''
        Forward pass for the SMPLX model
            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3x3
                Global rotation of the body. Useful if someone wishes to
                predicts this with an external model. It is expected to be in
                rotation matrix format. (default=None)
            betas: torch.tensor, optional, shape BxN_b
                Shape parameters. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            expression: torch.tensor, optional, shape BxN_e
                If given, ignore the member variable `expression` and use it
                instead. For example, it can used if expression parameters
                `expression` are predicted from some external model.
            jaw_pose: torch.tensor, optional, shape Bx3x3
                Jaw pose. It should either joint rotations in
                rotation matrix format.
            transl: torch.tensor, optional, shape Bx3
                Translation vector of the body.
                For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full axis-angle pose vector (default=False)
            Returns
            -------
                output: ModelOutput
                A named tuple of type `ModelOutput`
        '''
        device, dtype = self.shapedirs.device, self.shapedirs.dtype
        if global_orient is None:
            batch_size = 1
            global_orient = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        else:
            batch_size = global_orient.shape[0]
        if neck_pose is None:
            neck_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, 1, -1, -1).contiguous()
        if jaw_pose is None:
            jaw_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        if leye_pose is None:
            leye_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        if reye_pose is None:
            reye_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        if betas is None:
            betas = torch.zeros([batch_size, self.num_betas],
                                dtype=dtype, device=device)
        if expression is None:
            expression = torch.zeros([batch_size, self.num_expression_coeffs],
                                     dtype=dtype, device=device)
        if transl is None:
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)

        full_pose = torch.cat(
            [global_orient, neck_pose, jaw_pose, leye_pose, reye_pose], dim=1)

        shape_components = torch.cat([betas, expression], dim=-1)
        shapedirs = torch.cat([self.shapedirs, self.expr_dirs], dim=-1)

        vertices, joints = lbs(shape_components, full_pose, self.v_template,
                               shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=False,
                               )

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(
            dim=0).expand(batch_size, -1).contiguous()
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(
            batch_size, 1, 1)
        if self.use_face_contour:
            lmk_idx_and_bcoords = find_dynamic_lmk_idx_and_bcoords(
                vertices, full_pose, self.dynamic_lmk_faces_idx,
                self.dynamic_lmk_bary_coords,
                self.neck_kin_chain,
                pose2rot=False,
            )
            dyn_lmk_faces_idx, dyn_lmk_bary_coords = lmk_idx_and_bcoords
            lmk_faces_idx = torch.cat([lmk_faces_idx,
                                       dyn_lmk_faces_idx], 1)
            lmk_bary_coords = torch.cat(
                [lmk_bary_coords.expand(batch_size, -1, -1),
                 dyn_lmk_bary_coords], 1)

        landmarks = vertices2landmarks(vertices, self.faces_tensor,
                                       lmk_faces_idx,
                                       lmk_bary_coords)

        # Add any extra joints that might be needed
        joints = self.vertex_joint_selector(vertices, joints)
        # Add the landmarks to the joints
        joints = torch.cat([joints, landmarks], dim=1)

        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints=joints, vertices=vertices)

        joints += transl.unsqueeze(dim=1)
        vertices += transl.unsqueeze(dim=1)

        output = FLAMEOutput(vertices=vertices if return_verts else None,
                             joints=joints,
                             betas=betas,
                             expression=expression,
                             global_orient=global_orient,
                             neck_pose=neck_pose,
                             jaw_pose=jaw_pose,
                             full_pose=full_pose if return_full_pose else None)
        return output
