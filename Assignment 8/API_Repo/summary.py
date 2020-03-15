import torch
from torchsummary import summary             # Import summary with pytorch

def model_summary(model):
    summary(model, input_size=(3, 32, 32))
