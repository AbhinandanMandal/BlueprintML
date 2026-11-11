"""Train a ViT from scratch on CIFAR-10 (ImageFolder layout).

Example:
    python main.py --train-dir /path/to/cifar10/train --test-dir /path/to/cifar10/test
"""
import argparse

import torch

from config import config
from data import prepare_data
from model import ViT4Classification
from trainer import Trainer
from utils import load_experiment, plot_metrics, visualize_attention, visualize_images


def parse_args():
    p = argparse.ArgumentParser(description="ViT from scratch")
    p.add_argument("--train-dir", required=True, help="ImageFolder-style train directory")
    p.add_argument("--test-dir", required=True, help="ImageFolder-style test directory")
    p.add_argument("--exp-name", default="experiment_1")
    p.add_argument("--base-dir", default="./experiments")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-every", type=int, default=10, help="checkpoint every N epochs (0 = off)")
    p.add_argument("--no-viz", action="store_true", help="skip plots after training")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    trainloader, testloader, _ = prepare_data(
        args.train_dir, args.test_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    model = ViT4Classification(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    trainer = Trainer(model, optimizer, loss_fn, args.exp_name, device, config, base_dir=args.base_dir)
    trainer.train(trainloader, testloader, args.epochs, args.save_every)

    if not args.no_viz:
        visualize_images(args.train_dir)
        cfg, model, train_losses, test_losses, accuracies = load_experiment(
            args.exp_name, base_dir=args.base_dir
        )
        plot_metrics(train_losses, test_losses, accuracies, "metrics.png")
        visualize_attention(model, args.test_dir, "attention.png", device=str(device))


if __name__ == "__main__":
    main()
