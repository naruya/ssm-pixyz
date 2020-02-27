# ①ベースラインのベース。
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM --s_dim 64  #64

# ②ベースライン。ただ単に次元を大きくすると悪化することを確認する。発散とかしたら、理由も調べる。
# 4は発散しそうだけど、特にもし2も3も悪化しなかった場合には、ある程度学習が進むように工夫して学習させる。
# その場合、以降の1024次元でも同じ処置を加える。
# 逆に改善してしまった場合は実験を考えなおす。
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM --s_dim 128  #128
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM --s_dim 256  #256
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM --s_dim 512  #512
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM --s_dim 1024  #1024

# ③提案手法。②で一番最初に悪化した次元をa、次の次元をbとして、64+aを実験する。
# ただのaより良くなることを示す。
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM --s_dim 64 128  #64+128
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM --s_dim 64 256  #64+256
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM --s_dim 64 512  #64+512
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM --s_dim 64 1024  #64+1024

# ④提案手法。64+a+bを実験する。階層を増やすと良いことを示す。
# ただのbより良くなることと、64+aよりも良くなることを示す。
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM --s_dim 64 128 256  #64+128+256
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM --s_dim 64 256 512  #64+256+512
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM --s_dim 64 512 1024  #64+512+1024

# ----------------------------------------------------------------

# 以下モデルを大きくしたバージョン

# ①ベースラインのベース。
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM --s_dim 64  #64

# ②ベースライン。ただ単に次元を大きくすると悪化することを確認する。発散とかしたら、理由も調べる。
# 4は発散しそうだけど、特にもし2も3も悪化しなかった場合には、ある程度学習が進むように工夫して学習させる。
# その場合、以降の1024次元でも同じ処置を加える。
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM --s_dim 128  #128
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM --s_dim 256  #256
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM --s_dim 512  #512
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM --s_dim 1024  #1024

# ③提案手法。②で一番最初に悪化した次元をa、次の次元をbとして、64+aを実験する。
# ただのaより良くなることを示す。
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM --s_dim 64 128  #64+128
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM --s_dim 64 256  #64+256
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM --s_dim 64 512  #64+512
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM --s_dim 64 1024  #64+1024

# ④提案手法。64+a+bを実験する。階層を増やすと良いことを示す。
# ただのbより良くなることと、64+aよりも良くなることを示す。
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM --s_dim 64 128 256  #64+128+256
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM --s_dim 64 256 512  #64+256+512
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM --s_dim 64 512 1024  #64+512+1024

# ----------------------------------------------------------------

# CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=16 python train.py --model SSM --s_dim 1024 --B 256
# CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=16 python train.py --model SSM --s_dim 1024
# CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=16 python train.py --model SSM --s_dim 64
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