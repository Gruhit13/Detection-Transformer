import torch
from torch import nn
import torch.nn.functional as F

# A ResNet Basic block that forms the basic block of ResNetModel backbone
class BasicBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride = 1, expansion = 1, downsample = None):
	super(BasicBlock, self).__init__()
	self.expansion = expansion
	self.downsample = downsample

	self.conv1 = nn.Conv2d(
		in_channels,
		out_channels,
		kernel_size=3,
		stride=stride,
		padding=1,
		bias=False
	)
	self.bn1 = nn.BatchNorm2d(out_channels)
	self.relu = nn.ReLU(inplace=True)
	self.conv2 = nn.Conv2d(
		out_channels,
		out_channels*self.expansion,
		kernel_size=3,
		padding=1,
		bias=False
	)
	self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)

  def forward(self, x):
	identity = x

	out = self.conv1(x)
	out = self.bn1(out)
	out = self.relu(out)

	out = self.conv2(out)
	out = self.bn2(out)

	if self.downsample is not None:
	  identity = self.downsample(x)

	out += identity
	out = self.relu(out)

	return out


# Actual Resnet Model that will serve as a backbone for our Object Detection Model
class ResNet(nn.Module):
  """This class contains Resnet18 implementation till 4th layer not including the avgpool and succeeding layers"""
  def __init__(self, img_channels, channles, layers, expansion, num_classes):
	super(ResNet, self).__init__()

	self.layers = layers
	self.expansion = expansion
	self.num_channels = 64

	self.conv1 = nn.Conv2d(
		in_channels=img_channels,
		out_channels=self.num_channels,
		kernel_size=7,
		stride=2,
		padding=3,
		bias=False
	)

	self.bn1 = nn.BatchNorm2d(self.num_channels)
	self.relu = nn.ReLU(inplace=True)
	self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

	self.layer1 = self._make_layer(channles[0], layers[0])
	self.layer2 = self._make_layer(channles[1], layers[1], stride=2)
	self.layer3 = self._make_layer(channles[2], layers[2], stride=2)
	self.layer4 = self._make_layer(channles[3], layers[3], stride=2)

  def _make_layer(self, out_channels, blocks, stride=1):
	downsample = None

	if stride != 1:
	  downsample = nn.Sequential(
		  nn.Conv2d(
			  self.num_channels,
			  out_channels*self.expansion,
			  kernel_size=1,
			  stride=stride,
			  bias=False
		  ),
		  nn.BatchNorm2d(out_channels * self.expansion)
	  )

	layers = []
	layers.append(BasicBlock(self.num_channels, out_channels, stride, self.expansion, downsample))
	self.num_channels = out_channels * self.expansion

	for i in range(1, blocks):
	  layers.append(BasicBlock(self.num_channels, out_channels, expansion=self.expansion))

	return nn.Sequential(*layers)

  def forward(self, x, mask=None):
	x = self.conv1(x)
	x = self.bn1(x)
	x = self.relu(x)
	x = self.maxpool(x)

	x = self.layer1(x)
	x = self.layer2(x)
	x = self.layer3(x)
	x = self.layer4(x)

	if mask is not None:
	  mask = F.interpolate(mask[None].float(), size=x.shape[-2:]).to(torch.bool)[0]

	return x, mask