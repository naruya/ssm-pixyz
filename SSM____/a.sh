CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM16 --s_dim 64 512 1024 --separate
# CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM16 --s_dim 64 512 --separate
# CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM16 --s_dim 64 512 1024 --min_stddev 1e-7
# CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=16 python train.py --model SSM16 --s_dim 64 1024 1024 1024 --comment beta1_std_kl01 --min_stddev 1e-7
# CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=16 python train.py --model SSM16 --s_dim 64 1024 --resume --resume_name Jan19_19-06-23_SSM16_s64-1024_9bafb17_base --resume_time Jan21_01-17-10 --resume_itr 379912 --resume_epoch 282
# CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=16 python train.py --model SSM16 --s_dim 64 1024 --resume --resume_name Jan19_19-06-23_SSM16_s64-1024_9bafb17_base --resume_time Jan20_15-00-28 --resume_itr 270400 --resume_epoch 201
# CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM16 --s_dim 64 --comment base
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
