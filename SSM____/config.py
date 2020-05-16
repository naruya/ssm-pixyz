import argparse
import os
from datetime import datetime
import subprocess


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
    parser.add_argument('--gamma', type=float, default=1e-5)
    parser.add_argument('--min_stddev', type=float, default=0.)
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

    s_dim = "s" + str(args.s_dim[0])
    if len(args.s_dim) > 1:
        for i, d in enumerate(args.s_dim[1:]):
            s_dim += "-" + str(d)

    cmd = "git rev-parse --short HEAD"
    args.ghash = subprocess.check_output(cmd.split()).strip().decode('utf-8')
    args.timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join(
        args.runs_dir,
        args.timestamp
        + "_" + args.model + "_" + s_dim
        + "_" + args.ghash
    )

    if args.comment:
        log_dir += "_" + args.comment

    args.log_dir = log_dir
    return args
