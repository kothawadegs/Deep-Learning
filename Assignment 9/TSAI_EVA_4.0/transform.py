from torchvision import datasets, transforms
import torch
import torchvision
import torchvision.transforms as transforms
import cudas

# Train Phase transformations
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

# dataloader arguments - something you'll fetch these from cmdprmt
dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cudas.cuda else dict(shuffle=True, batch_size=64)
dataloader_args1 = dict(shuffle=True, batch_size=4, num_workers=4, pin_memory=True) if cudas.cuda else dict(shuffle=True, batch_size=64)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)
trainloader1 = torch.utils.data.DataLoader(trainset, **dataloader_args1)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, **dataloader_args)
testloader1 = torch.utils.data.DataLoader(testset, **dataloader_args1)