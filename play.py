from gui import GUI
from game import Game
from mcts import mcts
from network import AlphaZeroNet

import time
import torch
import random
import time

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


# def play_move_ai_without_mcts(game: Game, net: AlphaZeroNet):
# 	s = game.get_state().unsqueeze(0).cuda()
# 	p, v = net(s)
# 	p = p.squeeze(0).detach().cpu().numpy()
# 	v = v.squeeze(0).item()

# 	actions = game.get_actions()
# 	probs = p[actions]

# 	_, a = max((prob, action) for prob, action in zip(probs, actions))
# 	game.apply(a)
	
# 	time.sleep(0.5)

def play_move_ai_mcts(game: Game, net: AlphaZeroNet):
	pi, a, root = mcts(net, game, 50)

	# actions = game.get_actions()
	# for prob, move, action in zip(pi[actions], [endec.decode_action(action, game.board) for action in actions], actions):
	# 	print(prob, move, root.children[action].P, root.children[action].N, root.children[action].W, root.children[action].Q)

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

if __name__ == "__main__":
	net = AlphaZeroNet()
	net.cuda()

	net.initialize_parameters()
	# net.load_state_dict(torch.load("data/models/model.pt")["state_dict"])

	net.eval()

	with torch.no_grad():
		play(net)
