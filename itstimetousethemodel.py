from datatrain import resnet18
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = ("cuda"
          if torch.cuda.is_available()
          else "cpu"
)
print(f"{device}를 리용합네다")

model = resnet18()