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

JOINT_NAMES = [
    'pelvis',           # 0
    'left_hip',         # 1
    'right_hip',        # 2
    'spine1',           # 3
    'left_knee',        # 4
    'right_knee',       # 5
    'spine2',           # 6
    'left_ankle',       # 7
    'right_ankle',      # 8
    'spine3',           # 9, (10 - 1)
    'left_foot',        # 10
    'right_foot',       # 11
    'neck',             # 12
    'left_collar',      # 13
    'right_collar',     # 14, (15 - 1)
    'head',             # 15
    'left_shoulder',    # 16
    'right_shoulder',   # 17
    'left_elbow',       # 18
    'right_elbow',      # 19, (20 - 1)
    'left_wrist',       # 20
    'right_wrist',      # 21
    'jaw',              # 22, (23 - 1)
    'left_eye_smplhf',  # 23
    'right_eye_smplhf',  # 24, (25 - 1)
    'left_index1',      # 25
    'left_index2',      # 26
    'left_index3',      # 27
    'left_middle1',     # 28
    'left_middle2',     # 29
    'left_middle3',     # 30
    'left_pinky1',      # 31
    'left_pinky2',      # 32
    'left_pinky3',      # 33
    'left_ring1',       # 34
    'left_ring2',       # 35
    'left_ring3',       # 36
    'left_thumb1',      # 37
    'left_thumb2',      # 38
    'left_thumb3',      # 39, (40 - 1)
    'right_index1',     # 40
    'right_index2',     # 41
    'right_index3',     # 42
    'right_middle1',    # 43
    'right_middle2',    # 44
    'right_middle3',    # 45
    'right_pinky1',     # 46
    'right_pinky2',     # 47
    'right_pinky3',     # 48
    'right_ring1',      # 49
    'right_ring2',      # 50
    'right_ring3',      # 51
    'right_thumb1',     # 52
    'right_thumb2',     # 53
    'right_thumb3',     # 54, (55 - 1)
    'nose',             # 55
    'right_eye',        # 56
    'left_eye',         # 57
    'right_ear',        # 58
    'left_ear',         # 59
    'left_big_toe',     # 60
    'left_small_toe',   # 61
    'left_heel',        # 62
    'right_big_toe',    # 63
    'right_small_toe',  # 64, (65 - 1)
    'right_heel',       # 65
    'left_thumb',       # 66
    'left_index',       # 67
    'left_middle',      # 68
    'left_ring',        # 69, (70 - 1)
    'left_pinky',       # 70
    'right_thumb',      # 71
    'right_index',      # 72
    'right_middle',     # 73
    'right_ring',       # 74, (75 - 1)
    'right_pinky',      # 75, (76 - 1)
    # evaluated face jts (76 - 127)
    'right_eye_brow1',  # 76
    'right_eye_brow2',
    'right_eye_brow3',
    'right_eye_brow4',
    'right_eye_brow5',
    'left_eye_brow5',
    'left_eye_brow4',
    'left_eye_brow3',
    'left_eye_brow2',
    'left_eye_brow1',
    'nose1',
    'nose2',
    'nose3',
    'nose4',
    'right_nose_2',
    'right_nose_1',
    'nose_middle',
    'left_nose_1',
    'left_nose_2',
    'right_eye1',
    'right_eye2',
    'right_eye3',
    'right_eye4',
    'right_eye5',
    'right_eye6',
    'left_eye4',
    'left_eye3',
    'left_eye2',
    'left_eye1',
    'left_eye6',
    'left_eye5',
    'right_mouth_1',
    'right_mouth_2',
    'right_mouth_3',
    'mouth_top',
    'left_mouth_3',
    'left_mouth_2',
    'left_mouth_1',
    'left_mouth_5',  # 59 in OpenPose output
    'left_mouth_4',  # 58 in OpenPose output
    'mouth_bottom',  # 116 => 116 - 76 = 40 => the 40-index item in lmk_faces_idx
    'right_mouth_4',
    'right_mouth_5',
    'right_lip_1',
    'right_lip_2',
    'lip_top',
    'left_lip_2',
    'left_lip_1',
    'left_lip_3',
    'lip_bottom',
    'right_lip_3',
    # Face contour
    'right_contour_1',
    'right_contour_2',
    'right_contour_3',
    'right_contour_4',
    'right_contour_5',
    'right_contour_6',
    'right_contour_7',
    'right_contour_8',
    'contour_middle',
    'left_contour_8',
    'left_contour_7',
    'left_contour_6',
    'left_contour_5',
    'left_contour_4',
    'left_contour_3',
    'left_contour_2',
    'left_contour_1',
]


SMPLH_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',  # 10
    'right_foot',  # 11
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',   # 20
    'right_wrist',  # 21
    'left_index1',
    'left_index2',
    'left_index3',
    'left_middle1',  # 25
    'left_middle2',
    'left_middle3',
    'left_pinky1',
    'left_pinky2',
    'left_pinky3',  # 30
    'left_ring1',
    'left_ring2',
    'left_ring3',
    'left_thumb1',
    'left_thumb2',
    'left_thumb3',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',  # 41
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3',
    'nose',
    'right_eye',
    'left_eye',
    'right_ear',
    'left_ear',
    'left_big_toe',
    'left_small_toe',
    'left_heel',
    'right_big_toe',
    'right_small_toe',
    'right_heel',
    'left_thumb',
    'left_index',
    'left_middle',
    'left_ring',
    'left_pinky',
    'right_thumb',
    'right_index',
    'right_middle',
    'right_ring',
    'right_pinky',
]
