CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM12 --s_dim 256 512 --comment ens
# CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=16 python train.py --model SimpleSSM --s_dim 1024
# CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=16 python train.py --model SimpleSSM --s_dim 64
# CUDA_VISIBLE_DEVICES=2 python train.py --model SSM10 --s_dim 64 --ss_dim 1024 --comment v1
# CUDA_VISIBLE_DEVICES=2 python train.py --model SSM9 --s_dim 64 --ss_dim 1024 --comment v1
# CUDA_VISIBLE_DEVICES=2 python train.py --model SSM7 --s_dim 64 --ss_dim 128 --comment v1
# CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=16 python train.py --model SSM5 --s_dim 1024 --comment v1