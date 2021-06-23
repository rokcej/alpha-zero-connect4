import torch
import torch.nn as nn
import torch.nn.functional as F

# Connect 4 version

# N.B. The forward pass returns the logarithm of probabilities!
class AlphaZeroNet(nn.Module):
	def __init__(self):
		super(AlphaZeroNet, self).__init__()

		self.conv_block = ConvBlock()
		self.res_blocks = nn.ModuleList([ ResBlock() for i in range(5) ]) # Originally 19
		self.out_block = OutBlock()

	def initialize_parameters(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.xavier_normal_(m.weight)
	
	def forward(self, s):
		out = self.conv_block(s)
		for i in range(len(self.res_blocks)):
			out = self.res_blocks[i](out)
		p_log, v = self.out_block(out)

		return p_log, v

	def predict_detach(self, s):
		p_log, v = self.forward(s)

		p = torch.exp(p_log)
		p = p.squeeze(0).detach().cpu().numpy()
		v = v.squeeze(0).item()

		return p, v


class ConvBlock(nn.Module):
	def __init__(self):
		super(ConvBlock, self).__init__()

		self.conv = nn.Conv2d(2, 256, 3, stride=1, padding=1, bias=False) # Originally 119
		self.bn = nn.BatchNorm2d(256)

	def forward(self, s):
		out = self.conv(s)
		out = self.bn(out)
		out = F.relu(out)

		return out

class ResBlock(nn.Module):
	def __init__(self):
		super(ResBlock, self).__init__()

		self.conv1 = nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(256)

		self.conv2 = nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(256)

	def forward(self, s):
		out = self.conv1(s)
		out = self.bn1(out)
		out = F.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		
		out += s
		out = F.relu(out)

		return out

class OutBlock(nn.Module):
	def __init__(self):
		super(OutBlock, self).__init__()

		# Policy head
		self.conv1_p = nn.Conv2d(256, 256, 1, stride=1, bias=False)
		self.bn_p = nn.BatchNorm2d(256)
		# self.conv2_p = nn.Conv2d(256, 7, 1, stride=1, bias=False) # Originally 73
		self.fc1_p = nn.Linear(256 * 6 * 7, 7)

		# Value head
		self.conv_v = nn.Conv2d(256, 1, 1, stride=1, bias=False)
		self.bn_v = nn.BatchNorm2d(1)
		self.fc1_v = nn.Linear(1 * 6 * 7, 256)
		self.fc2_v = nn.Linear(256, 1)

	def forward(self, s):
		# Policy head
		p = self.conv1_p(s)
		p = self.bn_p(p)
		p = F.relu(p)
		# p = self.conv2_p(p)
		# p = p.reshape(p.shape[0], -1) # p.reshape(batch_size, -1)
		p = torch.flatten(p, start_dim=1)
		p = self.fc1_p(p)
		p_log = F.log_softmax(p, dim=1) # Use log_softmax for numerical stability
		# p = F.softmax(p, dim=1)

		# Value head
		v = self.conv_v(s)
		v = self.bn_v(v)
		v = F.relu(v)
		v = torch.flatten(v, start_dim=1)
		v = self.fc1_v(v)
		v = F.relu(v)
		v = self.fc2_v(v)
		v = torch.tanh(v)
		v = v.reshape(-1)

		return p_log, v
