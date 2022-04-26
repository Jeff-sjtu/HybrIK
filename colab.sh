# conda environment setup
cd /content/HybrIK
conda env create -n hybrik python=3.7
conda init bash
source ~/.bashrc
source activate hybrik
conda install pytorch==1.6.0 torchvision==0.7.0 -c pytorch
# conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
pip install opendr
python setup.py develop
