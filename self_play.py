from game import Game
from mcts import mcts
from tqdm import tqdm

def results2str(results):
	return f"White={results[+1]} | Black={results[-1]} | Draw={results[0]} | None={results[None]}"

def self_play(net, num_games, max_moves, num_simulations):
	results = { +1: 0, 0: 0, -1: 0, None: 0 }
	self_play_data = []
	with tqdm(total=num_games, desc="Self play", unit="game") as prog_bar:
		for i_game in range(num_games):
			game = Game()
			game_data = []

			root = None

			while not game.is_over() and game.num_moves() < max_moves:
				prog_bar.set_postfix_str(f"Move {game.num_moves() + 1}, {results2str(results)}")
				
				pi, action, root = mcts(net, game, num_simulations) #, root) # Reuse MCTS results

				game_data.append([game.get_state(), pi, game.to_play()])
				game.apply(action)
				root = root.children[action]

			z = 0
			if game.is_over():
				z = game.outcome()
				results[z] += 1
			else:
				results[None] += 1

			for i in range(len(game_data)):
				game_data[i][2] *= z
			
			self_play_data.extend(game_data)

			prog_bar.update(1)
		
		prog_bar.set_postfix_str(results2str(results))
	
	return self_play_data

