import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
        ) # output_size = 32, RF 3
        self.convblockx = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1, 1), bias=False),
        ) # output_size = 32, RF 3
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
        ) # output_size = 32, RF 5
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convblockx1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), bias=False),
         ) # output_size = 32, RF 3
         # output_size = 16, RF 10
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, dilation = 1,  bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        ) # output_size = 16, RF 18
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, dilation=1,  bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        ) # output_size = 16, RF 26
        self.convblock5_ = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, dilation=1,  bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        ) # output_size = 16, RF 26
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convblockx2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), bias=False),
        ) # output_size = 32, RF 3
        # output_size = 8, RF 32
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, dilation=1,  bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
        ) # output_size = 8, RF 56
        self.convblock7_ = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, dilation=1,  bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
        ) # output_size = 8, RF 56
        self.convblock7__ = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, dilation=1,  bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
        ) # output_size = 8, RF 56
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        ) # output_size = 1
        self.fc1 = nn.Linear(128, 10)
        #self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x_ = self.convblockx(x) #3
        x = self.convblock1(x) #3
        x = x + x_
        x1 = self.convblock2(x) #5
        x = x + x1
        x = self.pool(x) #10
        x_ = self.convblockx1(x) #3
        x = self.convblock4(x) #18
        x = x + x_
        x1 = self.convblock5(x) #26
        x = x + x1
        x1 = self.convblock5_(x) #26
        x = x + x1
        x = self.pool1(x) #28
        x_ = self.convblockx2(x) #3
        x = self.convblock7(x) #52
        x = x + x_
        x1 = self.convblock7_(x) #44
        x = x + x1
        x1 = self.convblock7__(x) #44
        x = x + x1
        x = self.gap(x)
        x = x.view(-1, 128)
        x = (self.fc1(x))
        #x = (self.fc2(x))
        return x