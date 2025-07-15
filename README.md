# GS-I³：: Gaussian Splatting for Surface Reconstruction from Illumination-Inconsistent Images
[Tengfei Wang](https://github.com/TFwang-9527), Hongmao Hou, Zhaoning Zhang, Yiwei Xu, Zongqian Zhan and Xin Wang*.
### [Project Page](https://tfwang-9527.github.io/GS-3I/) | [arXiv](https://arxiv.org/abs/2503.12335)
<img width="1959" height="623" alt="overflow" src="https://github.com/user-attachments/assets/3e7fa572-006a-44ee-b84f-bbb815d71af0" />




## Data
Dark in the Gaussian dataset can be downloaded from [DK-Gaussian](https://pan.baidu.com/s/1xmZqYEJ5ZMkdldPS9_MgiQ?pwd=jf48) .

The Modified DTU dataset can be downloaded from [Modified DTU](https://drive.google.com/drive/folders/1AGYQzrvNnr1Lfa_RY5tzqDSL61uFTZK8?usp=sharing) .

## Installation

The repository contains submodules, thus please check it out with 
```shell
# SSH
git@github.com:TFwang-9527/GS-3I.git
cd GS-3I

conda create -n GS-3I python=3.8
conda activate GS-3I

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 #replace your cuda version
pip install -r requirements.txt
pip install submodules/diff-plane-rasterization
pip install submodules/simple-knn
```
## Preprocessing
First, we need to obtain the pretrain normal map from StableNormal or other methods. Then, place the normals folder and the images folder under the same directory.

## Training
python train.py -s data_path -m out_path --max_abs_split_points 0 --opacity_cull_threshold 0.05

## Rendering and Extract Mesh
python render.py -m out_path --max_depth 10.0 --voxel_size 0.01

## Acknowledgements
This project is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [PGSR](https://github.com/zju3dv/PGSR), [StableNormal](https://github.com/Stable-X/StableNormal), and  [Gaussian in the Dark](https://github.com/yec22/Gaussian-DK). respectively. We thank all the authors for their great work and repos.
