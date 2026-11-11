"""Dataset preparation (CIFAR-10 in ImageFolder layout)."""
import torch
import torchvision

from config import CLASSES


def prepare_data(train_dir, test_dir, batch_size=4, num_workers=4,
                 train_sample_size=None, test_sample_size=None):
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=train_transform)
    if train_sample_size is not None:
        indices = torch.randperm(len(trainset))[:train_sample_size]
        trainset = torch.utils.data.Subset(trainset, indices)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.3333)),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    testset = torchvision.datasets.ImageFolder(root=test_dir, transform=test_transform)
    if test_sample_size is not None:
        indices = torch.randperm(len(testset))[:test_sample_size]
        testset = torch.utils.data.Subset(testset, indices)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    return trainloader, testloader, CLASSES
