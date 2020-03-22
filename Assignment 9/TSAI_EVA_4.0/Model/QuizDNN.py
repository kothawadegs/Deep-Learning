import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.x1_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(args.dropout),  
        )
        self.x2_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(args.dropout), 
        )
        self.x3_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # nn.Dropout(args.dropout),
        )
        self.x4_pool = nn.MaxPool2d(2, 2) 
        self.x5_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(args.dropout), 
        )
        self.x6_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(args.dropout),  
        )
        self.x7_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # nn.Dropout(args.dropout), 
        )
        self.x8_pool = nn.MaxPool2d(2, 2) 
        self.x9_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(args.dropout), 
        )
        self.x10_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(args.dropout),  
        )
        self.x11_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # nn.Dropout(args.dropout), 
        )
        self.x12_gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.x13_fc = nn.Sequential(
            nn.Linear(in_features=64, out_features=10),
            # nn.ReLU() NEVER!
        )

    def forward(self, x):
        x1 = self.x1_block(x)
        x2 = self.x2_block(x1)
        x3 = self.x3_block(x1+x2)
        x4 = self.x4_pool(x1+x2+x3)
        x5 = self.x5_block(x4)
        x6 = self.x6_block(x4+x5)
        x7 = self.x7_block(x4+x5+x6)
        x8 = self.x8_pool(x5+x6+x7)
        x9 = self.x9_block(x8)
        x10 = self.x10_block(x8+x9)
        x11 = self.x11_block(x8+x9+x10)
        x12 = self.x12_gap(x11)
        x12 = x12.view(-1, 64)
        x13 = self.x13_fc(x12)
        return x13