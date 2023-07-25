import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from unet import UNet
    
if __name__ == '__main__':
    net = UNet(3,1)
    print(net)
