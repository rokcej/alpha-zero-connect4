from gui import GUI
from game import Game
from mcts import mcts
from network import AlphaZeroNet
import encoder_decoder as endec

import chess
import time
import torch
import random
import time

def play_move_player(game: Game, gui: GUI):
	gui.clicked = None

	while True:
		if gui.clicked is not None:
			f, r = gui.clicked
			if gui.selected is None:
				gui.selected = gui.clicked
				for move in game.board.legal_moves:
					if move.from_square == chess.square(f, r):
						f2 = chess.square_file(move.to_square)
						r2 = chess.square_rank(move.to_square)
						gui.highlighted.add((f2, r2))
			else:
				if (f, r) in gui.highlighted:
					f0, r0 = gui.selected
					move = chess.Move(chess.square(f0, r0), chess.square(f, r))
					a = endec.encode_action(move, game.board)
					game.apply(a)
					break
				else:
					gui.selected = None
					gui.highlighted.clear()

		gui.draw()
		gui.handle_events()
	
	gui.selected = None
	gui.highlighted.clear()


def play_move_ai_without_mcts(game: Game, net: AlphaZeroNet):
	s = game.get_state().unsqueeze(0).cuda()
	p, v = net(s)
	p = p.squeeze(0).detach().cpu().numpy()
	v = v.squeeze(0).item()

	actions = game.get_actions()
	probs = p[actions]

	_, a = max((prob, action) for prob, action in zip(probs, actions))
	game.apply(a)
	
	time.sleep(0.5)

def play_move_ai_mcts(game: Game, net: AlphaZeroNet):
	pi, a, root = mcts(net, game, 200)

	# actions = game.get_actions()
	# for prob, move, action in zip(pi[actions], [endec.decode_action(action, game.board) for action in actions], actions):
	# 	print(prob, move, root.children[action].P, root.children[action].N, root.children[action].W, root.children[action].Q)

	game.apply(a)


def play_move_random(game: Game):
	a = random.choice(game.get_actions())
	game.apply(a)


def play(net: AlphaZeroNet):
	game = Game()

	# # Fool's mate
	# game.apply(endec.encode_action(chess.Move.from_uci("f2f3"), game.board))
	# game.apply(endec.encode_action(chess.Move.from_uci("e7e6"), game.board))
	# game.apply(endec.encode_action(chess.Move.from_uci("g2g4"), game.board))

	# # Scholar's mate
	# game.apply(endec.encode_action(chess.Move.from_uci("e2e4"), game.board))
	# game.apply(endec.encode_action(chess.Move.from_uci("e7e5"), game.board))
	# game.apply(endec.encode_action(chess.Move.from_uci("f1c4"), game.board))
	# game.apply(endec.encode_action(chess.Move.from_uci("b8c6"), game.board))
	# game.apply(endec.encode_action(chess.Move.from_uci("d1h5"), game.board))
	# game.apply(endec.encode_action(chess.Move.from_uci("d7d6"), game.board))


	net2 = AlphaZeroNet()
	net2.cuda()
	net2.initialize_parameters()
	net2.eval()

	gui = GUI(game.board)
	gui.draw()
	
	while not game.is_over():
		if game.to_play() == 1: # White
			# play_move_player(game, gui)
			play_move_ai_mcts(game, net)
			# play_move_random(game)
		else: # Black
			# play_move_player(game, gui)
			play_move_ai_mcts(game, net2)
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

	# net.initialize_parameters()
	net.load_state_dict(torch.load("data/models/model.pt")["state_dict"])

	net.eval()

	with torch.no_grad():
		play(net)
