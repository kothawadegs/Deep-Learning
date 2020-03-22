import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils import data
from torchvision.models import vgg19
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2
from IPython.display import Image, display

class Res18(nn.Module):
    def __init__(self, net):
        super(Res18, self).__init__()
        
        self.res18 = net

        # disect the network to access its last convolutional layer
        self.features_conv = nn.Sequential(self.res18.conv1,
                                           self.res18.bn1,
                                           self.res18.layer1,
                                           self.res18.layer2,
                                           self.res18.layer3,
                                           self.res18.layer4
                                           )#list(self.resx.children())[:-5]  
        
        self.linear = self.res18.linear

        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)

def getheatmap(pred, class_pred, netx, img):
  # get the gradient of the output with respect to the parameters of the model
  pred[:, class_pred].backward()
  # pull the gradients out of the model
  gradients = netx.get_activations_gradient()

  # pool the gradients across the channels
  pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

  # get the activations of the last convolutional layer
  activations = netx.get_activations(img.cuda()).detach()
  
  # weight the channels by corresponding gradients
  for i in range(512):
    activations[:, i, :, :] *= pooled_gradients[i]
    
  # average the channels of the activations
  heatmap = torch.mean(activations, dim=1).squeeze()

  # relu on top of the heatmap
  # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
  heatmap = np.maximum(heatmap.cpu(), 0)

  # normalize the heatmap
  heatmap /= torch.max(heatmap)
 # heatmap = None
  return heatmap

def imshow(img, ax):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))

def superposeimage(heatmap, img):
  heat1 = np.array(heatmap)
  heatmap1 = cv2.resize(heat1, (img.shape[1], img.shape[0]))
  heatmap1 = np.uint8(255 * heatmap1)
  heatmap1 = cv2.applyColorMap(heatmap1, cv2.COLORMAP_JET)
  superimposed_img = heatmap1 * 0.4 + img
  cv2.imwrite('./map.jpg', superimposed_img)

def gradcamof(net, img, classes):
  netx = Res18(net)
  netx.eval()
  
  fig, axes = plt.subplots(nrows=1, ncols=3)
  # get the most likely prediction of the model
  pred = netx(img.cuda())
  from torchvision.utils import save_image
  imx = img[0]
  save_image(imx, 'img1.png')
  class_pred = int(np.array(pred.cpu().argmax(dim=1)))
  imshow(torchvision.utils.make_grid(img),axes[0])
  print(classes[class_pred])

  # draw the heatmap
  heatmap = getheatmap(pred, class_pred, netx, img)
  axes[1].matshow(heatmap.squeeze())

  imx = cv2.imread("./img1.png")
  imx = cv2.cvtColor(imx, cv2.COLOR_BGR2RGB)
 # plt.imshow(imx, cmap='gray', interpolation='bicubic')
  superposeimage(heatmap, imx)

  imx = cv2.imread('./map.jpg')
  imx = cv2.cvtColor(imx, cv2.COLOR_BGR2RGB)
  axes[2].imshow(imx, cmap='gray', interpolation='bicubic')