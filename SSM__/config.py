import argparse
import os
from datetime import datetime


def get_args(jupyter=False, args=None):
    parser = argparse.ArgumentParser(description="description")
    parser.add_argument("--device_ids", type=int, nargs="+", default=[0])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--comment", type=str, default=None)
    parser.add_argument("--B", type=int, default=32)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--s_dim", type=int, nargs="+", default=None)
    parser.add_argument("--h_dim", type=int, default=1024)
    parser.add_argument("--a_dim", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--data_dir", type=str, default="~/tensorflow_datasets/")
    parser.add_argument("--runs_dir", type=str, default="../runs/")
    parser.add_argument("--seed", type=int, default=0)

    if not jupyter:
        args = parser.parse_args()
    else:
        if args:
            args = parser.parse_args(args=args)
        else:
            args = parser.parse_args([])

    s_dim = ""
    for i, d in enumerate(args.s_dim):
        s_dim += "s" * (i+1) + str(d)

    log_dir = os.path.join(
        args.runs_dir,
        datetime.now().strftime("%b%d_%H-%M-%S")
        + "_" + args.model + "_" + s_dim)

    if args.comment:
        log_dir += "_" + args.comment

    args.log_dir = log_dir
    return args