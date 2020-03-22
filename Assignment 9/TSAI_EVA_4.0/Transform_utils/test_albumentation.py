from albumentations import *
from albumentations.pytorch import ToTensor
import numpy as np

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