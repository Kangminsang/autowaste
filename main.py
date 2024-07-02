import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms # 이미지 변환(전처리) 기능을 제공
from torchvision.models import resnet18, ResNet18_Weights
from torch.autograd import Variable
from torch import optim # 경사하강법을 이용하여 가중치를 구하기 위한 옵티마이저
import os # 파일 경로에 대한 함수들을 제공
from PIL import Image
from tqdm import tqdm_notebook as tqdm # 진행 상황 표현
import random
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchsummary import summary

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=6):
        super(ResNet, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(block, 512, num_blocks[4], stride=2)
        self.linear1 = nn.Linear(512 * block.expansion, 512)
        self.linear2 = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = F.relu(self.linear1(out))
        out = self.linear2(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2, 2])

# Example of instantiating and using the model
model = ResNet18().to(device)
print(model)
summary(model, input_size=(3, 139, 186))
# To use the model, you would typically define a loss function and an optimizer
# Here is an example of how you might do that:

# import torch.optim as optim
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Then, in your training loop, you would use these as follows:
# for inputs, labels in dataloader:
#     optimizer.zero_grad()
#     outputs = model(inputs)
#     loss = criterion(outputs, labels)
#     loss.backward()
#     optimizer.step()
