import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class SelfPlayDataset(Dataset):
	def __init__(self, data):
		self.data = data # List of lists [s, pi, z]

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx][0], self.data[idx][1], self.data[idx][2]

class AlphaZeroLoss(nn.Module):
	def __init__(self):
		super(AlphaZeroLoss, self).__init__()

	def forward(self, p_log, pi, v, z):
		loss_v = ((z - v) ** 2)
		loss_p = -torch.sum(pi * p_log, 1)

		return torch.mean(loss_v.view(-1) + loss_p)

def train(net, train_data, num_epochs, batch_size, learning_rate, weight_decay):
	train_set = SelfPlayDataset(train_data)
	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
	
	criterion = AlphaZeroLoss()
	optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

	avg_avg_loss = 0.0

	with tqdm(total=num_epochs, desc="Training", unit="epoch") as prog_bar:
		for epoch in range(num_epochs):
			avg_loss = 0.0
			for i, data in enumerate(train_loader, 0):
				s, pi, z = data
				s = s.cuda()
				pi = pi.cuda()
				z = z.cuda()

				optimizer.zero_grad()

				p_log, v = net(s)
				loss = criterion(p_log, pi, v, z)
				loss.backward()
				avg_loss += loss.item()

				optimizer.step()
			
			avg_loss /= len(train_loader)
			avg_avg_loss += avg_loss

			prog_bar.set_postfix_str(f"Avg loss = {avg_loss}")
			prog_bar.update(1)

	avg_avg_loss /= num_epochs
	
	return avg_avg_loss
