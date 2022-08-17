# HybrIK

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href='https://colab.research.google.com/drive/1n41l7I2NxWseuruVQEU8he2XqzSXhu2f?usp=sharing' style='padding-left: 0.5rem;'><img src='https://colab.research.google.com/assets/colab-badge.svg' alt='Google Colab'></a>
<a href='https://paperswithcode.com/sota/3d-human-pose-estimation-on-3dpw?p=hybrik-a-hybrid-analytical-neural-inverse' style='padding-left: 0.5rem;'><img src='https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hybrik-a-hybrid-analytical-neural-inverse/3d-human-pose-estimation-on-3dpw'></a>


<div align="center">
<img src="assets/taiji.gif" width="260" height="160"> <img src="assets/dancer3.gif" width="260" height="160">
</div>


This repo contains the code of our paper:

**HybrIK: A Hybrid Analytical-Neural Inverse Kinematics Solution for 3D Human Pose and Shape Estimation**

[Jiefeng Li](http://jeffli.site/HybrIK/), [Chao Xu](https://www.isdas.cn/), [Zhicun Chen](https://github.com/chenzhicun), [Siyuan Bian](https://github.com/biansy000), [Lixin Yang](https://lixiny.github.io/), [Cewu Lu](http://mvig.org/)

[[`Paper`](https://openaccess.thecvf.com/content/CVPR2021/html/Li_HybrIK_A_Hybrid_Analytical-Neural_Inverse_Kinematics_Solution_for_3D_Human_CVPR_2021_paper.html)]
[[`Supplementary Material`](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Li_HybrIK_A_Hybrid_CVPR_2021_supplemental.zip)]
[[`arXiv`](https://arxiv.org/abs/2011.14672)]
[[`Project Page`](https://jeffli.site/HybrIK/)]

In CVPR 2021


![hybrik](assets/hybrik.png)


<div align="center">
    <img src="assets/decompose.gif", width="600" alt><br>
    Twist-and-Swing Decomposition
</div>

## News :triangular_flag_on_post:
[2022/08/16] [Pretrained model](https://drive.google.com/file/d/1C-jRnay38mJG-0O4_um82o1t7unC1zeT/view?usp=sharing) with HRNet-W48 backbone is available.

[2022/07/31] Training code with predicted camera is released.

[2022/07/25] [HybrIK](https://github.com/Jeff-sjtu/HybrIK) is now supported in [Alphapose](https://github.com/MVIG-SJTU/AlphaPose)! Multi-person demo with pose-tracking is available.

[2022/04/27] <a href='https://colab.research.google.com/drive/1n41l7I2NxWseuruVQEU8he2XqzSXhu2f?usp=sharing' style='padding-left: 0.5rem;'><img src='https://colab.research.google.com/assets/colab-badge.svg' alt='Google Colab'></a> is ready to use.

[2022/04/26] Achieve SOTA results by adding the 3DPW dataset for training.

[2022/04/25] The demo code is released!


## TODO
- [x] Provide pretrained model
- [x] Provide parsed data annotations

## Installation instructions

``` bash
# 1. Create a conda virtual environment.
conda create -n hybrik python=3.7 -y
conda activate hybrik

# 2. Install PyTorch
conda install pytorch==1.9.1 torchvision==0.10.1 -c pytorch

# 3. Install PyTorch3D (Optional, only for visualization)
conda install pytorch3d

# 4. Pull our code
git clone https://github.com/Jeff-sjtu/HybrIK.git
cd HybrIK

# 5. Install
python setup.py develop
```

## Download models
* Download the SMPL model `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from [here](https://smpl.is.tue.mpg.de/) at `common/utils/smplpytorch/smplpytorch/native/models`.
* Download our pretrained model (paper version) from [ [Google Drive](https://drive.google.com/file/d/1SoVJ3dniVpBi2NkYfa2S8XEv0TGIK26l/view?usp=sharing) | [Baidu](https://pan.baidu.com/s/13rPFHO6FWoy7DK066XY1Fw) (code: `qre2`) ].
* Download our pretrained model (with predicted camera) from [ [Google Drive](https://drive.google.com/file/d/16Y_MGUynFeEzV8GVtKTE5AtkHSi3xsF9/view?usp=sharing) | [Baidu](https://pan.baidu.com/s/1kHTKQEKiPnrAKAUzOD-Xww) (code: `4qyv`) ].

## Demo
First make sure you download the pretrained model (with predicted camera) and place it in the `${ROOT}` directory, i.e., `./pretrained_hrnet.pth`.

* Visualize HybrIK on **videos** (run in single frame):

``` bash
python scripts/demo_video.py --video-name examples/dance.mp4 --out-dir res_dance
```


* Visualize HybrIK on **images**:

``` bash
python scripts/demo_image.py --img-dir examples --out-dir res
```


## Fetch data
Download *Human3.6M*, *MPI-INF-3DHP*, *3DPW* and *MSCOCO* datasets. You need to follow directory structure of the `data` as below. Thanks to the great job done by Moon *et al.*, we use the Human3.6M images provided in [PoseNet](https://github.com/mks0601/3DMPPE_POSENET_RELEASE).
```
|-- data
`-- |-- h36m
    `-- |-- annotations
        `-- images
`-- |-- pw3d
    `-- |-- json
        `-- imageFiles
`-- |-- 3dhp
    `-- |-- annotation_mpi_inf_3dhp_train.json
        |-- annotation_mpi_inf_3dhp_test.json
        |-- mpi_inf_3dhp_train_set
        `-- mpi_inf_3dhp_test_set
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- train2017
        `-- val2017
```
* Download Human3.6M parsed annotations. [ [Google](https://drive.google.com/drive/folders/1tLA_XeZ_32Qk86lR06WJhJJXDYrlBJ9r?usp=sharing) | [Baidu](https://pan.baidu.com/s/1bqfVOlQWX0Rfc0Yl1a5VRA) ]
* Download 3DPW parsed annotations. [ [Google](https://drive.google.com/drive/folders/1f7DyxyvlC9z6SFT37eS6TTQiUOXVR9rK?usp=sharing) | [Baidu](https://pan.baidu.com/s/1d42QyQmMONJgCJvHIU2nsA) ]
* Download MPI-INF-3DHP parsed annotations. [ [Google](https://drive.google.com/drive/folders/1Ms3s7nZ5Nrux3spLxmMMAQWc5aAIecmv?usp=sharing) | [Baidu](https://pan.baidu.com/s/1aVBDudbDRT1w_ZxQc9zicA) ]


## Train from scratch

``` bash
./scripts/train_smpl_cam.sh test_3dpw configs/256x192_adam_lr1e-3-res34_smpl_3d_cam_2x_mix_w_pw3d.yaml
```

## Evaluation
Download the pretrained model ([ResNet-34](https://drive.google.com/file/d/16Y_MGUynFeEzV8GVtKTE5AtkHSi3xsF9/view?usp=sharing) or [HRNet-W48](https://drive.google.com/file/d/1C-jRnay38mJG-0O4_um82o1t7unC1zeT/view?usp=sharing)).
``` bash
./scripts/validate_smpl_cam.sh ./configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml ./pretrained_hrnet.pth
```


## Results

<center>

| Method | 3DPW | Human3.6M |
|:-------|:-----:|:-------:|
| SPIN | 59.2 | 41.1 |
| VIBE | 56.5 | 41.5 |
| VIBE *w. 3DPW* | 51.9 | 41.4 |
| PARE | 49.3 | - |
| PARE *w. 3DPW* | 46.4 | - |
| **HybrIK (ResNet-34)** | **48.8** | **34.5** |
| **HybrIK (ResNet-34)** *w. 3DPW* | **45.3** | **36.3** |

</center>


## Citing
If our code helps your research, please consider citing the following paper:

    @inproceedings{li2021hybrik,
        title={Hybrik: A hybrid analytical-neural inverse kinematics solution for 3d human pose and shape estimation},
        author={Li, Jiefeng and Xu, Chao and Chen, Zhicun and Bian, Siyuan and Yang, Lixin and Lu, Cewu},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        pages={3383--3393},
        year={2021}
    }