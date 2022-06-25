<<<<<<< HEAD
# EPT-Net
=======
# Edge-oriented Point-cloud Transformer for 3D Intracranial Aneurysm Segmentation
by [Yifan Liu](https://github.com/yifliu3)


## 1.Introduction
This repository is for our MICCAI 2022 paper "Edge-oriented Point cloud Transformer for 3D Intracranial Aneurysm Segmentation"  

## 2.Data Preparation
Download `fileSplit`, `geo.zip` and `IntrA.zip` from [IntrA repository](https://github.com/intra3d2019/IntrA)  

Unzip `geo.zip` and `IntrA.zip` into `geo` and `IntrA` foler  

Move the unzipped `geo` folder into `IntrA/annoated/geo`  

Move the `fileSplit` into `IntrA/split`
  
Create one foler data in the code respository and add one symbolic link  

`mkdir data && ln -s Yourpath/IntrA data/IntrA`

## 3. Installation
### Requirements
- python 3.7
- pytorch 1.7
- h5py
- pyyaml
- tensorboardx

### Step-by-step installation
```bash
# create python environment
conda create -n ept python=3.7
conda activate ept

# install dependencies
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch
conda install -c anaconda h5py pyyaml -y
pip install tensorboardx

# clone this repository in your own workspace
git clone https://github.com/CityU-AIM-Group/EPT.git
cd EPT
mkdir data && ln -s Yourpath/IntrA data/IntrA

# compile cuda operations
cd point_transformer_lib
python3 setup.py build_exit install

```

## 4. Train/test the Model 
To separately train and test you can use the commands below (take 512 sampling as an example):  
Train:   
`python -m tool.train --config config/IntrA/IntrA_pointtransformer_seg_repro sample_points 512`  
Test:  
`python -m tool.test --config config/IntrA/IntrA_pointtransformer_seg_repro sample_points 512`  


Or you can use the bash scipt to run train.py and test.py sequentially:  
`sh tool/ept.sh IntrA pointtransformer_seg_repro`  

The trained models are provided in [Google Drive]()

## 5. Citation
If you find this work useful for your research, please cite our paper:


## 6. Acknowledgement
This work is based on [point-transformer](https://github.com/POSTECH-CVLab/point-transformer).

>>>>>>> fc96c5d... initial commit
