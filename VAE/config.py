# B, T, C, H, W

import argparse

def get_args(jupyter=False):
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('--device_ids', type=int, nargs='+', \
                        help='list of CUDA devices (default: [0])', default=[0])
    parser.add_argument('--B', type=int, default=128)
    parser.add_argument('--T', type=int, default=30)
    parser.add_argument('--z_dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--path', type=str, default="~/tensorflow_datasets/")
    parser.add_argument('--comment', type=str, default="")
    if not jupyter:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args=[])
    return args
