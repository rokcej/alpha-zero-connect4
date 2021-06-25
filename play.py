from gui import GUI
from game import Game
from mcts import mcts
from network import AlphaZeroNet

import time
import torch
import random
import time
import sys
from tqdm import tqdm

def play_move_player(game: Game, gui: GUI):
	gui.handle_events()

	while True:
		if gui.clicked is not None:
			ix, iy = gui.clicked
			a = ix
			if a in game.get_actions():
				game.apply(a)
				break
		gui.draw()
		gui.handle_events()

def play_move_ai_mcts(game: Game, net: AlphaZeroNet):
	pi, a, root = mcts(net, game, 25)
	game.apply(a)


def play_move_random(game: Game):
	a = random.choice(game.get_actions())
	game.apply(a)


def play(net: AlphaZeroNet):
	game = Game()

	gui = GUI(game.board)
	gui.draw()
	
	while not game.is_over():
		if game.to_play() == 1: # Yellow
			play_move_player(game, gui)
			# play_move_ai_mcts(game, net)
			# play_move_random(game)
		else: # Red
			# play_move_player(game, gui)
			play_move_ai_mcts(game, net)
			# play_move_random(game)

		gui.draw()
		gui.handle_events()

	print(f"Outcome: {game.outcome()}")

	while gui.running:
		gui.draw()
		gui.handle_events()


def test(net: AlphaZeroNet, num_games):
	results = { +1: 0, -1: 0, 0: 0 }

	net2 = AlphaZeroNet()
	net2.cuda()
	#net2.initialize_parameters()
	net2.load_state_dict(torch.load("data/model.pt")["state_dict"])
	net2.eval()
	
	with tqdm(total=num_games, desc="Playing games", unit="game") as prog_bar:
		for i_game in range(num_games):
			game = Game()

			while not game.is_over():
				if game.to_play() == 1: # Yellow
					play_move_ai_mcts(game, net)
					# play_move_random(game)
				else: # Red
					play_move_ai_mcts(game, net2)
					# play_move_random(game)
					
			
			results[game.outcome()] += 1

			prog_bar.set_postfix_str(f"Yellow = {results[1]} | Red = {results[-1]} | Draw = {results[0]}")
			prog_bar.update(1)

	print()
	print(f"Yellow:\t{100 * results[1] / num_games}%")
	print(f"Red:\t{100 * results[-1] / num_games}%")
	print(f"Draw:\t{100 * results[0] / num_games}%")
	print()


if __name__ == "__main__":
	net = AlphaZeroNet()
	net.cuda()

	net.load_state_dict(torch.load("data/model199.pt")["state_dict"])

	net.eval()

	with torch.no_grad():
		if len(sys.argv) > 1 and sys.argv[1] == "test":
			num_games = int(sys.argv[2]) if len(sys.argv) > 2 else 100
			test(net, num_games)
		else:
			play(net)
