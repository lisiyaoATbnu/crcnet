"""
Created at the midnight of Jan 18th, 2018. 
@author: Li Si-Yao

"""

import torch
import torch.cuda as cuda
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

dtype = cuda.FloatTensor

kernel_size = 7

class IRUnit(nn.Module):
	def __init__(self, input_dim, kernel_size=kernel_size, feature_dim=None):
		super(IRUnit, self).__init__()
		if feature_dim is None:
			feature_dim = input_dim
		self.conv = nn.Conv2d(input_dim, feature_dim, kernel_size)
		self.tconv = nn.ConvTranspose2d(feature_dim, input_dim, kernel_size)
		self.nonlinear = nn.PReLU()
		
	def forward(self, x):
		HTx = self.nonlinear(self.conv(x))
		HHTx = self.tconv(HTx)

		return x - HHTx

class IntegrateUnit(nn.Module):
	def __init__(self, input_dim=101, output_dim=1):
		super(IntegrateUnit, self).__init__()
	
		self.conv1 = nn.Conv2d(input_dim, 50, 7, padding=3)
		self.prelu1 = nn.PReLU()
		self.conv2 = nn.Conv2d(50, 25, 7, padding=3)
		self.prelu2 = nn.PReLU()
		self.conv3 = nn.Conv2d(25, 12, 7, padding=3)
		self.prelu3 = nn.PReLU()
		self.conv4 = nn.Conv2d(12, 1, 1)
	def forward(self, x1):
		x = self.prelu1(self.conv1(x1))
		x = self.prelu2(self.conv2(x))
		x = self.prelu3(self.conv3(x))
		x = self.conv4(x)
		return x

class CRCNet(nn.Module):
	# 10-layer
	def __init__(self):
		super(CRCNet, self).__init__()
		self.up = nn.Conv2d(1, 10, 1)
		self.A1 = IRUnit(10, )
		self.A2 = IRUnit(10, )
		self.A3 = IRUnit(10, )
		self.A4 = IRUnit(10, )
		self.A5 = IRUnit(10, )
		self.A6 = IRUnit(10, )
		self.A7 = IRUnit(10, )
		self.A8 = IRUnit(10, )
		self.A9 = IRUnit(10, )
		self.A10 = IRUnit(10, )
		
		self.S = IntegrateUnit(101, 1)

	def forward(self, x, explore_mode=False):
	
		current = self.up(x)

		inter1 = self.A1(current)
		inter2 = self.A2(inter1)
		inter3 = self.A3(inter2)
		inter4 = self.A4(inter3)
		inter5 = self.A5(inter4)
		inter6 = self.A6(inter5)
		inter7 = self.A7(inter6)
		inter8 = self.A8(inter7)
		inter9 = self.A9(inter8)
		inter10 = self.A10(inter9)
	
		if explore_mode:
			return inter1, inter2, inter3, inter4, inter5, inter6, inter7, inter8, inter9, inter10
			
		s = torch.cat([x, inter1, inter2, inter3, inter4, inter5, inter6, inter7, inter8, inter9, inter10], dim=1) 
			
		return self.S(s)




	







