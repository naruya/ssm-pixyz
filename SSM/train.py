from config import *
from model import *
from utils import *
from data_loader import PushDataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import sys
import logzero
from logzero import logger
import mlflow


def data_loop(args, epoch, loader, model, writer, interval, train=True):
    summ = None
    for i, batch in enumerate(loader):
        print("==== ({}) Epoch: {} {}/{} ====".format(
            ["Test", "Train"][train], epoch, i+1, interval))

        x, a, itr = batch
        _B = x.size(0)
        x = x.to(args.device).transpose(0, 1)  # T,B,3,28,28
        a = a.to(args.device).transpose(0, 1)  # T,B,1
        feed_dict = {"x0": x[0].clone(), "x": x, "a": a}

        if train:
            loss, info = model.train_(feed_dict)
        else:
            loss, info = model.test_(feed_dict)

        logger.debug(info)
        summ = update_summ(summ, info, _B)

        if itr % interval == 0:
            summ = mean_summ(summ, loader.N)
            logger.info("({}) Epoch: {} {}".format(
                ["Test", "Train"][train], epoch, summ))
            break

#     if writer:
#         video = model.sample_x(feed_dict) if epoch % 10 == 0 else None
#         # foo = model.sample_foo(feed_dict)
#         write_summ(summ, video, writer, loader.N, epoch, train)

    for k, v in summ.items():
        mlflow.log_metric(["Test", "Train"][train] + "/" + k, v)

def main():
    args = get_args()
#     TRAIN_INTERVAL = int(256 / args.B)
    TRAIN_INTERVAL = int(43264 / args.B)

    TEST_INTERVAL = int(256 / args.B)

    logzero.loglevel(args.loglevel)
    logzero.logfile(args.logfile, loglevel=args.loglevel)
    logger.info(args)
    slack("Start! " + str(sys.argv))

    mlflow.start_run()

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.autograd.set_detect_anomaly(True)

    writer = None  # SummaryWriter(log_dir=args.log_dir)
    model = globals()[args.model](args)
    train_loader = PushDataLoader(args, split="train", shuffle=True)
    test_loader = PushDataLoader(args, split="test", shuffle=False)

    if args.load or args.resume:
        load_model(model, args.load_dir, args.load_epoch, load_optim=args.resume)

    resume_epoch = 1 if not args.resume else args.resume_epoch

    for epoch in range(resume_epoch, args.epochs + 1):
        data_loop(args, epoch, train_loader, model, writer, TRAIN_INTERVAL, train=True)
        data_loop(args, epoch, test_loader, model, writer, TEST_INTERVAL, train=False)
        if epoch % 10 == 1:
            save_model(model, args.save_dir, epoch)
            load_model(model, args.save_dir, epoch)

    save_model(model, args.save_dir, epoch)
    logger.info(args)
    slack("Finish! " + str(sys.argv))


if __name__ == "__main__":
    main()