#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 4 11:47:02 2020

@author: s140452
"""

# Load functions
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn import Linear, Conv2d, Dropout, MaxPool2d, BatchNorm1d, BatchNorm2d, Flatten
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax
import torch.nn.functional as F


use_cuda = torch.cuda.is_available()
print("Running GPU.") if use_cuda else print("No GPU available.")


def get_variable(x):
	""" Converts tensors to cuda, if available. """
	if use_cuda:
		return x.cuda()
	return x


def get_numpy(x):
	""" Get numpy array for both cuda and not. """
	if use_cuda:
		return x.cpu().data.numpy()
	return x.data.numpy()


channels = 3
conv_out_channels = 32
conv2_out_channels = 32
conv3_out_channels = 64
kernel_size =   5	 # <-- Kernel size
conv_stride =   1	 # <-- Stride
conv_pad	=   2	 # <-- Padding
p = 0.5
  
class Net_256(nn.Module):	

		
	def __init__(self):
		super(Net_256, self).__init__()
		
		
		
		self.conv_model = nn.Sequential(
			nn.Conv2d(in_channels=channels,
					  out_channels=conv_out_channels, 
					  kernel_size=kernel_size, 
					  padding=conv_pad,
					  stride=conv_stride),
			nn.MaxPool2d(kernel_size=2,
						 stride=2),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(conv_out_channels),
			nn.Dropout(p=p),

			nn.Conv2d(in_channels=conv_out_channels,
					  out_channels=conv_out_channels, 
					  kernel_size=kernel_size, 
					  padding=conv_pad,
					  stride=conv_stride),
			nn.MaxPool2d(kernel_size=2,
						 stride=2),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(conv_out_channels),
			nn.Dropout(p=p),		  
						   
			nn.Conv2d(in_channels=conv_out_channels, 
					  out_channels=conv2_out_channels, 
					  kernel_size=kernel_size, 
					  padding=conv_pad,
					  stride=conv_stride),
			nn.MaxPool2d(kernel_size=2,
						 stride=2),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(conv2_out_channels),
			nn.Dropout(p=p),
		
		
			nn.Conv2d(in_channels=conv2_out_channels, 
					  out_channels=conv2_out_channels, 
					  kernel_size=kernel_size, 
					  padding=conv_pad,
					  stride=conv_stride),
			nn.MaxPool2d(kernel_size=2,
						 stride=2),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(conv2_out_channels),
			nn.Dropout(p=p),

			nn.Conv2d(in_channels=conv2_out_channels, 
					  out_channels=conv3_out_channels, 
					  kernel_size=kernel_size, 
					  padding=conv_pad,
					  stride=conv_stride),
			nn.MaxPool2d(kernel_size=2,
						 stride=2),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(conv3_out_channels),
			nn.Dropout(p=p),
			
			nn.Conv2d(in_channels=conv3_out_channels, 
					  out_channels=conv3_out_channels, 
					  kernel_size=kernel_size, 
					  padding=conv_pad,
					  stride=conv_stride),
			nn.MaxPool2d(kernel_size=2,
						 stride=2),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(conv3_out_channels),
			nn.Dropout(p=p))
		
		self.lin_model = nn.Sequential(
		Flatten(),
		Linear(in_features = conv3_out_channels * (256//64) * (256//64),
			   out_features = 100),
		nn.ReLU(inplace=True),
		nn.Dropout(p=p),
		Linear(in_features = 100,
			   out_features = 1)
				)
				
	
	def forward(self, x):
		x = self.conv_model(x)
		x = self.lin_model(x)
		return x

class Net_1024(nn.Module):	

		
	def __init__(self):
		super(Net_1024, self).__init__()
		
		
		
		self.conv_model = nn.Sequential(
		nn.Conv2d(in_channels=channels,
				  out_channels=conv_out_channels, 
				  kernel_size=kernel_size, 
				  padding=conv_pad,
				  stride=conv_stride),
		nn.MaxPool2d(kernel_size=2,
					 stride=2),
		nn.ReLU(inplace=True),
		nn.BatchNorm2d(conv_out_channels),
		nn.Dropout(p=p),

		nn.Conv2d(in_channels=conv_out_channels,
				  out_channels=conv_out_channels, 
				  kernel_size=kernel_size, 
				  padding=conv_pad,
				  stride=conv_stride),
		nn.MaxPool2d(kernel_size=2,
					 stride=2),
		nn.ReLU(inplace=True),
		nn.BatchNorm2d(conv_out_channels),
		nn.Dropout(p=p),		  
					   
		nn.Conv2d(in_channels=conv_out_channels, 
				  out_channels=conv_out_channels, 
				  kernel_size=kernel_size, 
				  padding=conv_pad,
				  stride=conv_stride),
		nn.MaxPool2d(kernel_size=2,
					 stride=2),
		nn.ReLU(inplace=True),
		nn.BatchNorm2d(conv2_out_channels),
		nn.Dropout(p=p),
		
		nn.Conv2d(in_channels=conv_out_channels, 
				  out_channels=conv2_out_channels, 
				  kernel_size=kernel_size, 
				  padding=conv_pad,
				  stride=conv_stride),
		nn.MaxPool2d(kernel_size=2,
					 stride=2),
		nn.ReLU(inplace=True),
		nn.BatchNorm2d(conv2_out_channels),
		nn.Dropout(p=p),

		nn.Conv2d(in_channels=conv2_out_channels, 
				  out_channels=conv2_out_channels, 
				  kernel_size=kernel_size, 
				  padding=conv_pad,
				  stride=conv_stride),
		nn.MaxPool2d(kernel_size=2,
					 stride=2),
		nn.ReLU(inplace=True),
		nn.BatchNorm2d(conv3_out_channels),
		nn.Dropout(p=p),
		
		nn.Conv2d(in_channels=conv2_out_channels, 
				  out_channels=conv2_out_channels, 
				  kernel_size=kernel_size, 
				  padding=conv_pad,
				  stride=conv_stride),
		nn.MaxPool2d(kernel_size=2,
					 stride=2),
		nn.ReLU(inplace=True),
		nn.BatchNorm2d(conv3_out_channels),
		nn.Dropout(p=p),
		
		nn.Conv2d(in_channels=conv2_out_channels, 
				  out_channels=conv3_out_channels, 
				  kernel_size=kernel_size, 
				  padding=conv_pad,
				  stride=conv_stride),
		nn.MaxPool2d(kernel_size=2,
					 stride=2),
		nn.ReLU(inplace=True),
		nn.BatchNorm2d(conv3_out_channels),
		nn.Dropout(p=p),
		
		nn.Conv2d(in_channels=conv3_out_channels, 
				  out_channels=conv3_out_channels, 
				  kernel_size=kernel_size, 
				  padding=conv_pad,
				  stride=conv_stride),
		nn.MaxPool2d(kernel_size=2,
					 stride=2),
		nn.ReLU(inplace=True),
		nn.BatchNorm2d(conv3_out_channels),
		nn.Dropout(p=p))
		
		self.lin_model = nn.Sequential(
		Flatten(),
		Linear(in_features = conv3_out_channels * (1024//256) * (1024//256),
			   out_features = 100),
		nn.ReLU(inplace=True),
		nn.Dropout(p=p),
		Linear(in_features = 100,
			   out_features = 1)
				)
				
	
	def forward(self, x):
		x = self.conv_model(x)
		x = self.lin_model(x)
		return x

class Net_256mp(nn.Module):	

	def __init__(self):
		super(Net_256mp, self).__init__()		
		
		
		self.conv_model1 = nn.Sequential(
			nn.Conv2d(in_channels=channels,
					  out_channels=conv_out_channels, 
					  kernel_size=kernel_size, 
					  padding=conv_pad,
					  stride=conv_stride),
			nn.MaxPool2d(kernel_size=2,
						 stride=2),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(conv_out_channels),
			nn.Dropout(p=p)).to('cuda:0')

		self.conv_model2 = nn.Sequential(
			nn.Conv2d(in_channels=conv_out_channels,
					  out_channels=conv_out_channels, 
					  kernel_size=kernel_size, 
					  padding=conv_pad,
					  stride=conv_stride),
			nn.MaxPool2d(kernel_size=2,
						 stride=2),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(conv_out_channels),
			nn.Dropout(p=p),

			nn.Conv2d(in_channels=conv_out_channels, 
					  out_channels=conv2_out_channels, 
					  kernel_size=kernel_size, 
					  padding=conv_pad,
					  stride=conv_stride),
			nn.MaxPool2d(kernel_size=2,
						 stride=2),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(conv2_out_channels),
			nn.Dropout(p=p),

			nn.Conv2d(in_channels=conv2_out_channels, 
					  out_channels=conv2_out_channels, 
					  kernel_size=kernel_size, 
					  padding=conv_pad,
					  stride=conv_stride),
			nn.MaxPool2d(kernel_size=2,
						 stride=2),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(conv2_out_channels),
			nn.Dropout(p=p),
		
			nn.Conv2d(in_channels=conv2_out_channels, 
					  out_channels=conv3_out_channels, 
					  kernel_size=kernel_size, 
					  padding=conv_pad,
					  stride=conv_stride),
			nn.MaxPool2d(kernel_size=2,
						 stride=2),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(conv3_out_channels),
			nn.Dropout(p=p),
			
			nn.Conv2d(in_channels=conv3_out_channels, 
					  out_channels=conv3_out_channels, 
					  kernel_size=kernel_size, 
					  padding=conv_pad,
					  stride=conv_stride),
			nn.MaxPool2d(kernel_size=2,
						 stride=2),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(conv3_out_channels),
			nn.Dropout(p=p)).to('cuda:1')
		
		self.lin_model = nn.Sequential(
		Flatten(),
		Linear(in_features = conv3_out_channels * (256//64) * (256//64),
			   out_features = 100),
		nn.ReLU(inplace=True),
		nn.Dropout(p=p),
		Linear(in_features = 100,
			   out_features = 1)
				).to('cuda:1')
				

	def forward(self, x):
		x = self.conv_model2(self.conv_model1(x).to('cuda:1'))
		return self.lin_model(x.view(x.size(0), -1))

class Net_256pp(Net_256mp):	

		
	def __init__(self, split_size=12):
		super(Net_256pp, self).__init__()
		self.split_size = split_size
				   
	def forward(self, x):
		splits = iter(x.split(self.split_size, dim=0))
		s_next = next(splits)
		s_prev1 = self.conv_model1(s_next).to('cuda:1')
		ret = []

		for s_next in splits:
			# A. s_prev runs on cuda:1
			s_prev1 = self.conv_model2(s_prev1)
			ret.append(self.lin_model(s_prev1.view(s_prev1.size(0), -1)))

			# B. s_next runs on cuda:0, which can run concurrently with A
			s_prev1 = self.conv_model1(s_next).to('cuda:1')

		s_prev1 = self.conv_model2(s_prev1)
		ret.append(self.lin_model(s_prev1.view(s_prev1.size(0), -1)))

		return torch.cat(ret)
