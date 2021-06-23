from network import AlphaZeroNet
from self_play import self_play
from train import train

import os
import torch
import pickle

MODEL_DIR  = "data/models"
MODEL_NAME = "model.pt"
SAVE_FILE  = os.path.join(MODEL_DIR, MODEL_NAME)

NUM_STEPS = 30
NUM_GAMES = 10
MAX_MOVES = 300 # Originally 512
NUM_SIMULATIONS = 200 # Originally 800
NUM_EPOCHS = 20
BATCH_SIZE = 128

if __name__ == "__main__":
	net = AlphaZeroNet()
	net.cuda()

	step_start = 0
	if os.path.isfile(SAVE_FILE):
		print("Loading old network...")
		load_checkpoint = torch.load(SAVE_FILE)
		net.load_state_dict(load_checkpoint["state_dict"])
		step_start = load_checkpoint["step"]
	else:
		if not os.path.exists(MODEL_DIR):
			os.mkdir(MODEL_DIR)
		print("Initializing new network...")
		net.initialize_parameters()

	for step in range(step_start, step_start + NUM_STEPS):
		print(f"Step {step + 1}")

		# Generate training data
		net.eval()
		with torch.no_grad():
			train_data = self_play(net, NUM_GAMES, MAX_MOVES, NUM_SIMULATIONS)

		# Save training data
		with open(os.path.join(MODEL_DIR, f"train_data_{step}.pckl"), "wb") as f:
			pickle.dump(train_data, f)

		# Train network
		net.train()
		train(net, train_data, NUM_EPOCHS, BATCH_SIZE)

		# Save network
		print("Saving... ", end="")
		save_checkpoint = { 
			"state_dict": net.state_dict(),
			"step": step + 1
		}
		torch.save(save_checkpoint, SAVE_FILE + ".bak") # Backup
		torch.save(save_checkpoint, SAVE_FILE)
		if (step + 1) % 10 == 0:
			torch.save(save_checkpoint, SAVE_FILE + f".step{step + 1}") # Milestone

		torch.cuda.empty_cache()

		print("Done!")

