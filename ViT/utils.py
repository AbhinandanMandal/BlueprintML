"""Saving/loading experiments and visualisation helpers."""
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from config import CLASSES
from model import ViT4Classification


# Adapted from https://kunal-kumar-sahoo.github.io/ViT-from-Scratch/main.html
def save_experiments(experiment_name, config, model, train_losses, test_losses,
                     accuracies, base_dir="./experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(config, f, sort_keys=True, indent=4)

    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(
            {"train_losses": train_losses,
                "test_losses": test_losses, "accuracies": accuracies},
            f, sort_keys=True, indent=4,
        )

    save_checkpoint(experiment_name, model, "final", base_dir=base_dir)


def save_checkpoint(experiment_name, model, epoch, base_dir="./experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(outdir, f"model_{epoch}.pt"))


def load_experiment(experiment_name, checkpoint_name="model_final.pt", base_dir="./experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    with open(os.path.join(outdir, "config.json"), "r") as f:
        config = json.load(f)
    with open(os.path.join(outdir, "metrics.json"), "r") as f:
        data = json.load(f)

    model = ViT4Classification(config)
    model.load_state_dict(torch.load(os.path.join(
        outdir, checkpoint_name), map_location="cpu"))
    return config, model, data["train_losses"], data["test_losses"], data["accuracies"]


def visualize_images(dataset_dir):
    """Plot 30 random images from an ImageFolder directory."""
    dataset = torchvision.datasets.ImageFolder(root=dataset_dir)
    indices = torch.randperm(len(dataset))[:30]
    images = [np.asarray(dataset[i][0]) for i in indices]
    labels = [dataset[i][1] for i in indices]
    fig = plt.figure(figsize=(10, 10))
    for i in range(30):
        ax = fig.add_subplot(6, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(images[i])
        ax.set_title(CLASSES[labels[i]])
    plt.show()


@torch.no_grad()
def visualize_attention(model, test_dir, output=None, device="cuda"):
    """Overlay the [CLS] attention (averaged over blocks and heads) on test images."""
    model.eval()
    num_images = 30
    testset = torchvision.datasets.ImageFolder(root=test_dir)
    indices = torch.randperm(len(testset))[:num_images]
    raw_images = [np.asarray(testset[i][0]) for i in indices]
    labels = [testset[i][1] for i in indices]

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    images = torch.stack([test_transform(image)
                         for image in raw_images]).to(device)
    model = model.to(device)

    logits, attention_maps = model(images, output_attentions=True)
    predictions = torch.argmax(logits, dim=1)

    # Concatenate attention maps from all blocks: (B, num_blocks * heads, N, N)
    attention_maps = torch.cat(attention_maps, dim=1)
    # Attention of the [CLS] token to the patches, averaged over blocks/heads
    attention_maps = attention_maps[:, :, 0, 1:].mean(dim=1)
    size = int(math.sqrt(attention_maps.size(-1)))
    attention_maps = attention_maps.view(-1, size, size).unsqueeze(1)
    attention_maps = torch.nn.functional.interpolate(
        attention_maps, size=(32, 32), mode="bilinear", align_corners=False
    ).squeeze(1)

    fig = plt.figure(figsize=(20, 10))
    mask = np.concatenate([np.ones((32, 32)), np.zeros((32, 32))], axis=1)
    for i in range(num_images):
        ax = fig.add_subplot(6, 5, i + 1, xticks=[], yticks=[])
        img = np.concatenate([raw_images[i], raw_images[i]], axis=1)
        ax.imshow(img)
        extended = np.concatenate(
            (np.zeros((32, 32)), attention_maps[i].cpu()), axis=1)
        extended = np.ma.masked_where(mask == 1, extended)
        ax.imshow(extended, alpha=0.5, cmap="jet")
        gt, pred = CLASSES[labels[i]], CLASSES[predictions[i]]
        ax.set_title(f"gt: {gt} / pred: {pred}",
                     color=("green" if gt == pred else "red"))
    if output is not None:
        plt.savefig(output)
    plt.show()


def plot_metrics(train_losses, test_losses, accuracies, output="metrics.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(train_losses, label="Train loss")
    ax1.plot(test_losses, label="Test loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax2.plot(accuracies)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    plt.savefig(output)
    plt.show()
