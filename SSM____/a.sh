CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --s_dim 64 --comment towel --epochs 1500
# CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=16 python train.py --model SimpleSSM --s_dim 64 --comment base_kl01_5_no-std_no-share
# CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=16 python train.py --model SimpleSSM --s_dim 32 --comment base
# CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM15 --s_dim 64 128 --comment trans_share-min_stddev_a-s_trans
# CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM15 --s_dim 64 512 --comment trans_share-min_stddev_a-s_trans
# CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SimpleSSM --s_dim 1024 --comment trans_share-min_stddev
# CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM13 --s_dim 256 1024 --comment hie
# CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=16 python train.py --model SSM14 --s_dim 256 512 --comment res
# CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM13 --s_dim 256 512 --comment hie
# CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM12 --s_dim 256 512 --comment ens
# CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=16 python train.py --model SimpleSSM --s_dim 1024
# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 python train.py --model SimpleSSM --s_dim 512
# CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SimpleSSM --s_dim 64 --comment debug
# CUDA_VISIBLE_DEVICES=2 python train.py --model SSM10 --s_dim 64 --ss_dim 1024 --comment v1
# CUDA_VISIBLE_DEVICES=2 python train.py --model SSM9 --s_dim 64 --ss_dim 1024 --comment v1
# CUDA_VISIBLE_DEVICES=2 python train.py --model SSM7 --s_dim 64 --ss_dim 128 --comment v1
# CUDA_VISIBLE_DEVICES=1 python train.py --model SSM5 --s_dim 1024 --comment v1
