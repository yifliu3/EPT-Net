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
exp_dir=exp_2048/${dataset}/${exp_name}
config=config/${dataset}/${dataset}_${exp_name}.yaml

mkdir -p ${exp_dir}
cp tool/train.sh tool/${TRAIN_CODE} tool/${TEST_CODE} ${config} ${exp_dir}

python ${exp_dir}/${TRAIN_CODE} \
  --config=${config} \
  sample_points 2048 \
  num_edge_neighbor 8 \
  save_path ${exp_dir} \
  2>&1 | tee ${exp_dir}/train-$now.log

python ${exp_dir}/${TEST_CODE} \
  --config=${config} \
  num_edge_neighbor 8 \
  test_points 2048
  save_path ${exp_dir} \
  2>&1 | tee ${exp_dir}/test-$now.log
