from network import AlphaZeroNet
from self_play import self_play
from train import train

import os
import torch
import pickle
import matplotlib.pyplot as plt

SAVE_DIR  = "data/models"

NUM_STEPS = 100
# Self-play
NUM_GAMES = 100
NUM_SIMULATIONS = 25 # Originally 800
# Training
NUM_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.1
LR_DECAY = 0.1
LR_SCHEDULE = [ 50, 100, 150 ]

if __name__ == "__main__":
	net = AlphaZeroNet()
	net.cuda()

	save_file = os.path.join(SAVE_DIR, "model.pt")

	step_start = 0
	avg_losses = []
	if os.path.isfile(save_file):
		print("Loading old network...")
		load_checkpoint = torch.load(save_file)
		net.load_state_dict(load_checkpoint["state_dict"])
		step_start = load_checkpoint["step"]
		avg_losses = load_checkpoint["avg_losses"]
	else:
		print("Initializing new network...")
		if not os.path.exists(SAVE_DIR):
			os.makedirs(SAVE_DIR)
		net.initialize_parameters()


	for step in range(step_start, step_start + NUM_STEPS):
		print(f"Step {step + 1}")

		# Generate training data
		net.eval()
		with torch.no_grad():
			train_data = self_play(net, NUM_GAMES, NUM_SIMULATIONS)

		# Train network
		net.train()
		learning_rate = LEARNING_RATE
		for milestone in LR_SCHEDULE:
			if step >= milestone: learning_rate *= LR_DECAY 
		avg_loss = train(net, train_data, NUM_EPOCHS, BATCH_SIZE, learning_rate)
		avg_losses.append(avg_loss)

		# Save network
		save_checkpoint = { 
			"state_dict": net.state_dict(),
			"step": step + 1,
			"avg_losses": avg_losses
		}
		torch.save(save_checkpoint, save_file + ".bak") # Backup
		torch.save(save_checkpoint, save_file)
		if (step + 1) % 20 == 0:
			torch.save(save_checkpoint, save_file + f".step{step + 1}") # Milestone

		# torch.cuda.empty_cache()

	plt.plot(avg_losses)
	plt.xlabel("Step")
	plt.ylabel("Loss")
	plt.savefig(os.path.join(SAVE_DIR, f"loss_{step_start}-{step_start + NUM_STEPS - 1}.png"), dpi=150)

