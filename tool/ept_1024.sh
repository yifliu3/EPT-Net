#!/bin/sh

#SBATCH -J pt4pc
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH -w node6
#SBATCH -c 4
#SBATCH --nodes=1

source /home/yifliu3/.bashrc
nvidia-smi
nvcc -V

export PYTHONPATH=./

conda activate pt4pc

TRAIN_CODE=train.py
TEST_CODE=test.py

dataset=$1
exp_name=$2
exp_dir=exp_1024/${dataset}/${exp_name}
config=config/${dataset}/${dataset}_${exp_name}.yaml

mkdir -p ${exp_dir}
cp tool/train.sh tool/${TRAIN_CODE} tool/${TEST_CODE} ${config} ${exp_dir}

python ${exp_dir}/${TRAIN_CODE} \
  --config=${config} \
  sample_points 1024 \
  save_path ${exp_dir} \
  num_edge_neighbor 6 \
  2>&1 | tee ${exp_dir}/train-$now.log

python ${exp_dir}/${TEST_CODE} \
  --config=${config} \
  save_path ${exp_dir} \
  num_edge_neighbor 6 \
  test_points 1024 \
  2>&1 | tee ${exp_dir}/test-$now.log
