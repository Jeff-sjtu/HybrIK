# HybrIK


This repo contains the code of our paper:

[HybrIK: A Hybrid Analytical-Neural Inverse Kinematics Solution for 3D Human Pose and Shape Estimation](https://arxiv.org/abs/2011.14672)
> Jiefeng Li, Chao Xu, Zhicun Chen, Siyuan Bian, Cewu Lu    
> CVPR 2021

![hybrik](hybrik.png)

## TODO
- [ ] Provide pretrained model
- [ ] Provide parsed data annotations

## Installation instructions

``` bash
# 1. Create a conda virtual environment.
conda create -n hybrik python=3.6 -y
conda activate hybrik

# 2. Install PyTorch
conda install pytorch==1.1.0 torchvision==0.3.0

# 3. Pull our code
git clone https://github.com/Jeff-sjtu/HybrIK.git
cd HybrIK

# 4. Install
python setup.py develop
```

## Fetch data
Download *Human3.6M*, *MPI-INF-3DHP*, *3DPW* and *MSCOCO* datasets. You need to follow directory structure of the `data` as below.
```
|-- data
`-- |-- h36m
`-- |-- pw3d
`-- |-- 3dhp
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- train2017
        `-- val2017
```
* Download Human3.6M parsed data. *(WIP)*
* Download 3DPW parsed data. *(WIP)*
* Download MPI-INF-3DHP parsed data. *(WIP)*


## Train from scratch

``` bash
./scripts/train_smpl.sh train_res34 ./configs/256x192_adam_lr1e-3-res34_smpl_3d_base_2x_mix.yaml
```

## Evaluation
``` bash
./scripts/validate_smpl.sh ./configs/256x192_adam_lr1e-3-res34_smpl_3d_base_2x_mix.yaml ${CKPT}
```


## Citing
If our code helps your research, please consider citing the following paper:

    @inproceedings{li2020hybrik,
        title={HybrIK: A Hybrid Analytical-Neural Inverse Kinematics Solution for 3D Human Pose and Shape Estimation},
        author={Li, Jiefeng and Xu, Chao and Chen, Zhicun and Bian, Siyuan and Yang, Lixin and Lu, Cewu},
        booktitle={CVPR},
        year={2021}
    }