import torch.nn as nn


# Define Residual Block
class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride=1):
    super(ResidualBlock, self).__init__()

    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLu(inplace=True)

    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channels)

    # Shortcut connection to handle different input/output dimensions
    self.shortcut = nn.Sequential()
    
    if stride != 1 or in_channels != out_channels:
      self.shortcut = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
          nn.BatchNorm2d(out_channels)
      )

  def forward(self, x):
    residual = x
    
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    out += self.shortcut(residual)
    out = self.relu(out)

    return out
  
# Define ResNet101v2 model
class ResNet101v2(nn.Module):
  def __init__(self, num_classes=1000):
    super(ResNet101v2, self).__init__()

    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias = False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace = True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.layer1 = self._make_layer(64, 64, blocks=3, stride=1)
    self.layer2 = self._make_layer(64, 128, block=4, stride=2)
    self.layer3 = self._make_layer(128, 256, blocks=23, stride=2)
    self.layer4 = self._make_layer(256, 512, blocks=3, stride=2)

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512, num_classes)

  def _make_layer(self, in_channels, out_channels, blocks, stride):
    layers = [ResidualBlock(in_channels, out_channels, stride)]
    for _ in range(1, blocks):
      layers.append(ResidualBlock(out_channels, out_channels, stride=1))
    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x