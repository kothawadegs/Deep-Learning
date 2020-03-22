import torch
import torchvision


from albumentations import *
from albumentations.pytorch import ToTensor
import numpy as np

class TrainAlbumentation():
  def __init__(self):
    self.train_transform = Compose([
      HorizontalFlip(),
      Rotate((-30.0, 30.0)),
      RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50, p=0.5),
      Cutout(num_holes=4),
      Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225],
      ),
      ToTensor()
    ])

  def __call__(self, img):
    img = np.array(img)
    img = self.train_transform(image = img)['image']
    return img
    


class TestAlbumentation():
  def __init__(self):
    self.test_transform = Compose([
      Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225],
      ),
      ToTensor()
    ])

  def __call__(self, img):
    img = np.array(img)
    img = self.test_transform(image = img)['image']
    return img

SEED = 1

class Data():

  def __init__(self):
    self.train_album = TrainAlbumentation()
    self.test_album = TestAlbumentation()

  def getTrainDataSet(self, train=True):
    dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=self.train_album)
    return dataset

  def getTestDataSet(self, train=False):
    dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=self.test_album)
    return dataset

  def getDataLoader(self, dataset, batches):
    # checking CUDA
    self.cuda = torch.cuda.is_available()
    # For reproducibility
    torch.manual_seed(SEED)
    if self.cuda:
      torch.cuda.manual_seed(SEED)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=batches, num_workers=4, pin_memory=True) if self.cuda else dict(shuffle=True, batch_size=64)

    # train dataloader
    self.dataset_loader = torch.utils.data.DataLoader(dataset, **dataloader_args)

    return self.dataset_loader

    
  def getGradCamDataLoader(self, dataset):
  # checking CUDA
    self.cuda = torch.cuda.is_available()
    # For reproducibility
    torch.manual_seed(SEED)
    if self.cuda:
      torch.cuda.manual_seed(SEED)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=1, num_workers=4, pin_memory=True) if self.cuda else dict(shuffle=True, batch_size=1)

    # train dataloader
    self.dataset_loader = torch.utils.data.DataLoader(dataset, **dataloader_args)

    return self.dataset_loader