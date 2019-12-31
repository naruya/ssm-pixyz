import argparse
import os
from datetime import datetime

# B, T, C, H, W
def get_args(jupyter=False):
    parser = argparse.ArgumentParser(description="description")
    parser.add_argument(
        "--device_ids",
        type=int,
        nargs="+",
        help="list of CUDA devices (default: [0])",
        default=[0],
    )
    parser.add_argument("--model", type=str, default="SSM4")
    parser.add_argument("--comment", type=str, default="debug")
    parser.add_argument("--B", type=int, default=32)  # 32
    parser.add_argument("--T", type=int, default=10)  # 30
    parser.add_argument("--s_dim", type=int, default=64)  # 64
    parser.add_argument("--h_dim", type=int, default=256)  # ?
    parser.add_argument("--a_dim", type=int, default=4)  # 4
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--data_dir", type=str, default="~/tensorflow_datasets/")
    parser.add_argument("--runs_dir", type=str, default="../runs/")
    parser.add_argument("--seed", type=int, default=0)

    if not jupyter:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args=[])

    log_dir = os.path.join(
        args.runs_dir,
        datetime.now().strftime("%b%d_%H-%M-%S")
        + "_"
        + args.model
        + "_"
        + args.comment,
    )
    args.log_dir = log_dir
    return args