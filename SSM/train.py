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


def data_loop(args, epoch, loader, model, interval, train=True):
    summ = None
    prefix = ["Test", "Train"][train]

    for i, batch in enumerate(loader):
        print("==== ({}) Epoch: {} {}/{} ====".format(
            prefix, epoch, i+1, interval))

        x_0, x, a, itr = batch
        loss, info = model.forward_(x_0, x, a, train)

        logger.debug(info)
        summ = update_summ(summ, info, _B=x.size(0))

        if itr % interval == 0:
            summ = mean_summ(summ, loader.N)
            logger.info("({}) Epoch: {} {}".format(
                prefix, epoch, summ))
            break

    summ = rename_summ(summ, prefix=prefix + "/")
    if not args.debug: mlflow.log_metrics(summ, epoch)

    return summ

def main():
    args = get_args()

    # TRAIN_INTERVAL = int(256 / args.B)
    TRAIN_INTERVAL = int(43264 / args.B)
    TEST_INTERVAL = int(256 / args.B)

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.autograd.set_detect_anomaly(True)

    if not args.debug:
        logzero.logfile(args.logfile, loglevel=args.loglevel)
        slack("Start! " + str(sys.argv))
        mlflow.start_run(run_name=args.timestamp)
        mlflow.log_params(vars(args))

    logzero.loglevel(args.loglevel)
    logger.info("git hash: " + args.ghash)
    logger.info("command: " + str(sys.argv))
    logger.info(args)

    model = globals()[args.model](args)
    train_loader = PushDataLoader(args, split="train", shuffle=True)
    test_loader = PushDataLoader(args, split="test", shuffle=False)

    if args.load or args.resume:
        load_model(model, args.load_dir, args.load_epoch,
                   load_optim=args.resume)

    resume_epoch = 1 if not args.resume else args.resume_epoch

    for epoch in range(resume_epoch, args.epochs + 1):
        _summ = data_loop(
            args, epoch, train_loader, model, TRAIN_INTERVAL, True)
        summ = data_loop(
            args, epoch, test_loader, model, TEST_INTERVAL, False)
        # if epoch % 1 == 0:
        if epoch % 10 == 0:
            save_model(model, args.save_dir, epoch)
            # load_model(model, args.save_dir, epoch)
            slack("Epoch: {} {} {}".format(epoch, str(sys.argv), str(summ)))
        if args.debug:
            slack("Epoch: {} {} {}".format(epoch, str(sys.argv), str(_summ)))
            slack("Epoch: {} {} {}".format(epoch, str(sys.argv), str(summ)))

    save_model(model, args.save_dir, epoch)
    slack("Finish! {} {}".format(sys.argv, summ))

    if not args.debug:
        mlflow.end_run()


if __name__ == "__main__":
    main()
