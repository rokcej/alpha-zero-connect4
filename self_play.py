from tqdm import tqdm

from game import Game
from mcts import mcts

def self_play(net, num_games, num_simulations):
	results = { +1: 0, -1: 0, 0: 0 }
	self_play_data = []

	with tqdm(total=num_games, desc="Self play", unit="game") as prog_bar:
		for i_game in range(num_games):
			game = Game()
			game_data = []

			while not game.is_over():
				pi, action = mcts(net, game, num_simulations) # Reuse MCTS results

				for s in game.get_state_symmetries():
					game_data.append([s, pi, game.to_play()])
				game.apply(action)

			z = game.outcome()
			results[z] += 1

			for i in range(len(game_data)):
				game_data[i][2] *= z
			
			self_play_data.extend(game_data)

			prog_bar.set_postfix_str(f"Yellow={results[+1]} | Red={results[-1]} | Draw={results[0]}")
			prog_bar.update(1)
	
	return self_play_data
