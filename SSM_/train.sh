CUDA_VISIBLE_DEVICES=2 python train.py --model SSM8 --s_dim 64 --ss_dim 1024 --comment v1
# CUDA_VISIBLE_DEVICES=2 python train.py --model SSM7 --s_dim 64 --ss_dim 128 --comment v1
# CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=16 python train.py --model SSM5 --s_dim 1024 --comment v1
