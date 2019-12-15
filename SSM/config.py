import argparse

# B, T, C, H, W
def get_args(jupyter=False):
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('--device_ids', type=int, nargs='+', \
                        help='list of CUDA devices (default: [0])', default=[0])
    parser.add_argument('--B', type=int, default=32) # 32
    parser.add_argument('--T', type=int, default=10) # 30
    parser.add_argument('--s_dim', type=int, default=64) # 64
    parser.add_argument('--h_dim', type=int, default=256) # ?
    parser.add_argument('--a_dim', type=int, default=4) # 4
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--data_dir', type=str, default="~/tensorflow_datasets/")
    
    if not jupyter:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args=[])
    return args
