from game import Game
from mcts import mcts
from tqdm import tqdm

def results2str(results):
	return f"Yellow={results[+1]} | Red={results[-1]} | Draw={results[0]}"

def self_play(net, num_games, max_moves, num_simulations):
	results = { +1: 0, 0: 0, -1: 0 }
	self_play_data = []

	with tqdm(total=num_games, desc="Self play", unit="game") as prog_bar:
		for i_game in range(num_games):
			game = Game()
			game_data = []

			root = None

			while not game.is_over():
				pi, action, root = mcts(net, game, num_simulations, root) # Reuse MCTS results

				game_data.append([game.get_state(), pi, game.to_play()])
				game.apply(action)
				root = root.children[action]

			z = game.outcome()
			results[z] += 1

			for i in range(len(game_data)):
				game_data[i][2] *= z
			
			self_play_data.extend(game_data)

			prog_bar.set_postfix_str(results2str(results))
			prog_bar.update(1)
	
	return self_play_data

