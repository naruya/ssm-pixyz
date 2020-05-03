import argparse
import os
import subprocess
from datetime import datetime
import sys


# TODO: MLFlow
def get_args(jupyter=False, args=None):
    parser = argparse.ArgumentParser(description="description")
    parser.add_argument("--device_ids", type=int, nargs="+", default=[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loglevel", type=int, default=20)
    parser.add_argument("--comment", type=str, default=None)

    parser.add_argument("--model", type=str, default="SSM")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument('--resnet', action='store_true')
    parser.add_argument("--B", type=int, default=256)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--s_dims", type=int, nargs="+", default=[])
    parser.add_argument("--h_dim", type=int, default=1024)
    parser.add_argument("--a_dim", type=int, default=4)
    parser.add_argument('--gamma', type=float, default=1e-5)
    parser.add_argument('--min_stddev', type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--static_hierarchy", type=int, nargs="+", default=[])

    parser.add_argument("--data_dir", type=str, default="~/tensorflow_datasets/")
    parser.add_argument("--runs_dir", type=str, default="./runs/")
    parser.add_argument("--models_dir", type=str, default="./models/")
    parser.add_argument("--logzero_dir", type=str, default="./logzero/")

    parser.add_argument("--load_name", type=str, default=None)
    parser.add_argument("--load_epoch", type=int, default=None)
    parser.add_argument('--resume', action='store_true')

    if not jupyter:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args=args)

    ghash = subprocess.check_output(
        "git rev-parse --short HEAD".split()).strip().decode('utf-8')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.ghash = ghash
    args.timestamp = timestamp

    args.load = True if args.load_name or args.load_epoch else False
    args.name = timestamp if not args.resume else args.load_name

    args.log_dir = os.path.join(args.runs_dir, args.name)  # for tensorboard
    args.save_dir = os.path.join(args.models_dir, args.name)  # for save_model()
    args.logfile = os.path.join(args.logzero_dir, args.name + ".txt")

    if args.load or args.resume:
        args.load_dir = os.path.join(args.models_dir, args.load_name)
    if args.resume:
        args.resume_epoch = args.load_epoch + 1

    args.device = args.device_ids[0]
    args.debug = True if args.loglevel <= 10 else False

    with open(".hist.txt", mode='a') as f:
        f.write("{} {} {}\n".format(timestamp, ghash, sys.argv, args))

    return args