import os
import sys

import numpy as np
import sklearn.metrics
import torch
import torchvision
import argparse

import wandb

from datasets.echonet_dynamic import EchoNet
from utils import get_optimizer, get_lr_scheduler, get_model, get_mean_and_sd, run_train, run_test, set_seed

def main():
    my_parser = argparse.ArgumentParser(description='Run script')

    my_parser.add_argument('--exp_no',
                       type=str,
                       default="",
                       help='Experiment number for wandb')

    my_parser.add_argument('--exp_name',
                       type=str,
                       default="",
                       help='Experiment name for wandb')

    my_parser.add_argument('--data_dir',
                       type=str,
                       default='./data',
                       help='Path to data directory')

    my_parser.add_argument('--output',
                       type=str,
                       default=None,
                       help='Path to output directory')

    my_parser.add_argument('--model_name',
                       choices=['r2plus1d_18', 'mc3_18', 'r3d_18', 'uniformer_small', 'uniformer_base'],
                       default='uniformer_small',
                       help='Model name')

    my_parser.add_argument('--pretrained',
                       type=bool,
                       default=True,
                       help='Whether to use pretrained model or not')

    my_parser.add_argument('--weights',
                       type=str,
                       default=None,
                       help='Path to checkpoint to load')

    my_parser.add_argument('--epochs',
                       type=int,
                       default=45,
                       help='Number of epochs to train')

    my_parser.add_argument('--optimizer_name',
                       type=str,
                       default='adamW',
                       help='Optimizer name')

    my_parser.add_argument('--lr_scheduler',
                       choices=['step', 'LWCA'],
                       default='step',
                       help='Learning rate scheduler to use')

    my_parser.add_argument('--lr',
                       type=float,
                       default=1e-4,
                       help='Learning rate')

    my_parser.add_argument('--weight_decay',
                       type=float,
                       default=1e-4,
                       help='Weight decay')

    my_parser.add_argument('--lr_step_period',
                       type=int,
                       default=15,
                       help='Learning rate decay period')

    my_parser.add_argument('--frames',
                       type=int,
                       default=32,
                       help='Number of frames to select')

    my_parser.add_argument('--frequency',
                       type=int,
                       default=2,
                       help='Period between frames')

    my_parser.add_argument('--num_workers',
                       type=int,
                       default=4,
                       help='Number of workers')

    my_parser.add_argument('--batch_size',
                       type=int,
                       default=16,
                       help='Batch size')

    my_parser.add_argument('--device',
                       type=str,
                       default=None,
                       help='Device to use')


    args = my_parser.parse_args()

    print("Exp Name: ", args.exp_name)
    print("Exp No.: ", args.exp_no)
    print("Model Name: ", args.model_name)
    print("Pretrained: ", args.pretrained)
    print("Epochs: ", args.epochs)

    print(args)

    wandb.init(
        name=args.exp_name,
        project="EchoCoTr"
    )
    wandb.config.update({
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "weights": args.weights,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lr_step_period": args.lr_step_period,
        "frames": args.frames,
        "frequency": args.frequency,
        "num_workers": args.num_workers,
        "device": args.device,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer_name,
        "lr_scheduler": args.lr_scheduler
    })

    set_seed(0)

    if args.output is None:
        output = os.path.join("output_" + str(args.exp_no), "video", "{}_{}_{}_{}".format(args.model_name, args.frames, args.frequency, "pretrained" if args.pretrained else "random"))
    else:
        output = args.output
    os.makedirs(output, exist_ok=True)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(args.model_name, args)

    wandb.watch(model)

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    optimizer = get_optimizer(model, args)

    lr_scheduler = get_lr_scheduler(optimizer, args)

    args.mean, args.std = get_mean_and_sd(EchoNet(root=args.data_dir, split="train"))

    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
        try:
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['opt_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
            print("Epochs to resume: ", epoch_resume)
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        # Run Training Step
        if epoch_resume < args.epochs:
            run_train(output, device, model, optimizer, lr_scheduler, bestLoss, epoch_resume, wandb, f, args)

        print(model)

        # Run Testing Step
        run_test(output, device, model, wandb, f, args)

if __name__ == "__main__":
    main()
