import torch
import torchvision
import torch.nn as nn
from torchsummary import summary

import numpy as np

import utils.regularization  as regularization # L1 loss fxn
import utils.model_history as model_history

# conv2d->BN->ReLU->Dropout
class Conv2d_BasicBlock(nn.Module):
    def __init__(self, inC, outC, ksize, padding=0, dilation=1, drop_val=0):
        super(Conv2d_BasicBlock,self).__init__()

        self.drop_val = drop_val

        self.conv = nn.Sequential(           
            nn.Conv2d(in_channels=inC, out_channels=outC, kernel_size=ksize, padding=padding, dilation=dilation, bias=False), 
            nn.BatchNorm2d(outC),
            nn.ReLU()
        )

        self.dropout = nn.Sequential(           
            nn.Dropout(self.drop_val)
        )

    def forward(self, x):
        x = self.conv(x)
        if(self.drop_val!=0):
          x = self.dropout(x)
        return x

# depthwise seperable conv followed by pointwise convolution
class Conv2d_Seperable(nn.Module):
    def __init__(self, inC, outC, ksize, padding=0, dilation=1):
        super(Conv2d_Seperable,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inC, out_channels=inC, groups=inC, kernel_size=ksize, padding=padding,  dilation=dilation, bias=False), # depth convolution
            nn.Conv2d(in_channels=inC, out_channels=outC, kernel_size=(1,1), bias=False) # Pointwise convolution
        )

    def forward(self, x):
        return self.conv(x)

# depthwise seperable conv->BN->ReLU->Dropout
class Conv2d_Seperable_BasicBlock(nn.Module):
    def __init__(self, inC, outC, ksize, padding=0, dilation=1, drop_val=0):
        super(Conv2d_Seperable_BasicBlock,self).__init__()

        self.drop_val = drop_val

        self.conv = nn.Sequential(           
            Conv2d_Seperable(inC, outC, ksize, padding, dilation=dilation),   
            nn.BatchNorm2d(outC),
            nn.ReLU()
        )

        self.dropout = nn.Sequential(           
            nn.Dropout(self.drop_val)
        )

    def forward(self, x):
        x = self.conv(x)
        if(self.drop_val!=0):
          x = self.dropout(x)
        return x

# maxpooling followed by pointwise conv
class Conv2d_TransistionBlock(nn.Module):
    def __init__(self, inC, outC):
        super(Conv2d_TransistionBlock,self).__init__()

        self.conv = nn.Sequential(           
            nn.MaxPool2d(2, 2),                                                           #Output: 16X16X16, Jin=1, GRF: 8X8
            nn.Conv2d(in_channels=inC, out_channels=outC, kernel_size=(1, 1), bias=False), #Output: 8X13X13 , Jin=2, GRF: 8X8 (combining channels)
        )

    def forward(self, x):
        return self.conv(x)

"""# Common functions for train, test and build model"""

from tqdm import tqdm

# function to train the model on training dataset
def train(model, device, train_loader, criterion, optimizer, lr_scheduler, L1_loss_enable=False):
    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
      # get samples
      data, target = data.to(device), target.to(device)

      # Init
      optimizer.zero_grad()
      # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
      # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

      # Predict
      y_pred = model(data)

      # Calculate loss
      loss = criterion(y_pred, target)
      
      if(L1_loss_enable == True):
        regloss = regularization.L1_Loss_calc(model, 0.0005)
        regloss /= len(data) # by batch size
        loss += regloss

      train_loss += loss.item()

      # Backpropagation
      loss.backward()
      optimizer.step()

      if(lr_scheduler != None): # this is for batchwise lr update
        lr_scheduler.step()

      # Update pbar-tqdm
      
      pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()
      processed += len(data)

      #pbar.set_description(desc= f'Loss={loss.item():0.6f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
      pbar.set_description(desc= f'Loss={train_loss/(batch_idx+1):0.6f} Batch_id={batch_idx+1} Accuracy={100*correct/processed:0.2f}')

    train_loss /= len(train_loader)
    acc = 100. * correct/len(train_loader.dataset) #processed # 
    return np.round(acc,2), np.round(train_loss,6)

# function to test the model on testing dataset
def test(model, device, test_loader, criterion, L1_loss_enable=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            #test_loss += criterion(output, target, reduction='sum').item()   # sum up batch loss # criterion = F.nll_loss
            test_loss += criterion(output, target).item()                     # sum up batch loss # criterion = nn.CrossEntropyLoss()
            
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        #test_loss /= len(test_loader.dataset)  # criterion = F.nll_loss
        test_loss /= len(test_loader)           # criterion = nn.CrossEntropyLoss()

        if(L1_loss_enable == True):
          regloss = regularization.L1_Loss_calc(model, 0.0005)
          regloss /= len(test_loader.dataset) # by batch size which is here total test dataset size
          test_loss += regloss

    print('\nTest set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    acc = 100. * correct / len(test_loader.dataset)
    return np.round(acc,2), test_loss

# build and train the model. epoch based result is store in Modelhistory and returned.
def build_model(model, device, trainloader, testloader, epochs, criterion, optimizer, 
				lr_scheduler=None, reduceLr_scheduler=None, L1_loss_enable=False):

    # object to store model hsitory such as training and test accuracy and losses, epoch wise lr values etc
    history = model_history.ModelHistory(epochs)

    for epoch in range(1,epochs+1):

      # read the current epoch Lr values
      cur_lr = optimizer.state_dict()["param_groups"][0]["lr"]
      print("EPOCH-{}: learning rate is: {}".format(epoch, cur_lr))
 
      #for param_group in optimizer.param_groups:
        #print("EPOCH-{}: learning rate is: {}".format(epoch, param_group['lr']))
  
      train_acc, train_loss = train(model, device, trainloader, criterion, optimizer, lr_scheduler=None, L1_loss_enable=L1_loss_enable)

      if(lr_scheduler != None):
        lr_scheduler.step()
        
      test_acc, test_loss = test(model, device, testloader, criterion, L1_loss_enable=L1_loss_enable)

      if(reduceLr_scheduler != None):
        reduceLr_scheduler.step(test_loss)

      # store the epoch train and test results
      history.append_epoch_result(train_acc, train_loss, test_acc, test_loss, cur_lr)
    
    return history
  
# to get model summary
def model_summary(model, device, input_size):
    model = model.to(device)
    summary(model, input_size=input_size)
    return

# to get test accuracy
def get_test_accuracy(model, device, testloader):
    #model.eval()
    correct = 0
    for data, target in testloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    acc = 100. * correct / len(testloader.dataset)
    print('\nAccuracy of the network on the {} test images: {:.2f}%\n'.format(len(testloader.dataset), acc))

    return

def get_test_accuracy_cifar10(model, device, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = 100. * correct / total
    print('Accuracy of the network on the %d test images: %0.2f %%' % (total, acc))
    return

# to get test accuracy for each classes on test dataset
def class_based_accuracy(model, device, classes, testloader):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    return