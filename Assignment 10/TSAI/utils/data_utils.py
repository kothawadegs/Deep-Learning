import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np

# Check that GPU is avaiable
def get_device():
    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)
    device = torch.device("cuda:0" if cuda else "cpu")
    print(device)
    return device
	
def calculate_dataset_mean_std():
    trainset = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
    testset = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())

    data = np.concatenate([trainset.data, testset.data], axis=0)
    data = data.astype(np.float32)/255.

    print("\nTotal dataset(train+test) shape: ", data.shape)

    means = []
    stdevs = []
    for i in range(3): # 3 channels
        pixels = data[:,:,:,i].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    return (means[0], means[1], means[2]), (stdevs[0], stdevs[1], stdevs[2])

def get_dataset(train_transforms, test_transforms):
    trainset = datasets.CIFAR10('./data', train=True, download=True, transform=train_transforms)
    testset = datasets.CIFAR10('./data', train=False, download=True, transform=test_transforms)
    return trainset, testset

# fucntion that define data transform as per image processing needs for the solution
def get_data_transform(means, stds):
    # Train Phase transformations
    normalize = transforms.Normalize(means, stds)
    #normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # mean and std
    #normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # mean and std

    # resize = transforms.Resize((28, 28))

    train_transforms = transforms.Compose([
                                          #resize,
                                          #transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                          #transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
                                          #transforms.RandomCrop(32, padding=4),
                                          #transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize, 
                                          transforms.RandomErasing(scale=(0.02, 0.20), ratio=(0.8, 1.2))
                                          ])

    # Test Phase transformations
    test_transforms = transforms.Compose([
                                          #resize,
                                          transforms.ToTensor(),
                                          normalize
                                          ])
    
    return train_transforms, test_transforms

def get_dataloader(train_transforms, test_transforms, batch_size, num_workers):

    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True) if cuda else dict(shuffle=True, batch_size=8)

    trainset, testset = get_dataset(train_transforms, test_transforms)

    # train dataloader
    train_loader = torch.utils.data.DataLoader(trainset, **dataloader_args)

    # test dataloader
    test_loader = torch.utils.data.DataLoader(testset, **dataloader_args)

    return train_loader, test_loader
