# import torch
import numpy as np
import encoder_decoder as endec

class Game():
	def __init__(self, board=None):
		self.board = board or chess.Board()

	def get_actions(self):
		return [ endec.encode_action(move, self.board) for move in self.board.legal_moves ]

	def get_state(self):
		s = endec.encode_board(self.board)
		return torch.from_numpy(s).float()

	def apply(self, a: int):
		self.board.push(endec.decode_action(a, self.board))
		return self

	def clone(self):
		return Game(self.board.copy())

	def is_over(self):
		return self.board.is_game_over()

	def outcome(self):
		outcome = self.board.outcome()
		if outcome != None:
			if outcome.winner == chess.WHITE:
				return +1
			elif outcome.winner == chess.BLACK:
				return -1
			else:
				return 0
		return None

	def num_moves(self):
		return len(self.board.move_stack)
	
	def to_play(self):
		if (len(self.board.move_stack) % 2) == 0:
			return +1 # White
		else:
			return -1 # Black


