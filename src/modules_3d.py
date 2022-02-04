import torch
from torch import nn

class Conv3dDownsample(nn.Module):
  def __init__(self, input_channels):
    super().__init__()
    self.input_channels = input_channels
    self.output_channels = input_channels * 2

    self.mod = nn.Sequential(
        nn.Conv3d(self.input_channels, self.output_channels, stride = (1, 2, 2), kernel_size = (3,3,3), padding = (1,1,1)),
        nn.BatchNorm3d(self.output_channels),
        nn.ReLU(),
    )

  def forward(self, x):
    return self.mod(x)

class ConvTranspose3dUpsample(nn.Module):
  def __init__(self, input_channels, kernel_size = 3):
    super().__init__()

    self.input_channels = input_channels
    self.output_channels = input_channels // 2

    self.mod = nn.Sequential(
        nn.ConvTranspose3d(self.input_channels, self.output_channels, stride = (1,2,2), kernel_size = (3,kernel_size,kernel_size), padding = (1,1,1), output_padding = (0,1,1)),
        nn.BatchNorm3d(self.output_channels),
        nn.ReLU(),
    )

  def forward(self, x):
    return self.mod(x)

class Conv3dBlock(nn.Module):
  def __init__(self, input_channels, output_channels = None):
    super().__init__()
    self.input_channels = input_channels
    if output_channels is None:
      self.output_channels = self.input_channels
    else:
      self.output_channels = output_channels

    self.mod = nn.Sequential(
        nn.Conv3d(self.input_channels, self.output_channels, kernel_size = (3,3,3), padding = (1,1,1)),
        nn.BatchNorm3d(self.output_channels),
        nn.ReLU(),
    )

  def forward(self, x):
    return self.mod(x)

class ConvTranspose3dBlock(nn.Module):
  def __init__(self, input_channels, output_channels = None):
    super().__init__()
    self.input_channels = input_channels    
    self.output_channels = self.input_channels
    if output_channels is None:
      self.output_channels = self.input_channels
    else:
      self.output_channels = output_channels

    self.mod = nn.Sequential(
        nn.ConvTranspose3d(self.input_channels, self.output_channels, kernel_size = (3,3,3), padding = (1,1,1)),
        nn.BatchNorm3d(self.output_channels),
        nn.ReLU(),
    )

  def forward(self, x):
    return self.mod(x)

