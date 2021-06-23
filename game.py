import torch
import numpy as np

class Game():
	def __init__(self, board=None):
		self.action_space = 7

		if board is not None:
			self.board = board
			self.move_count = np.count_nonzero(self.board)
			self.turn = 1 if self.move_count % 2 == 0 else -1
		else:
			self.board = np.zeros((6, 7), dtype=np.int)
			self.move_count = 0
			self.turn = 1

		self.result = None

	def get_actions(self):
		moves = []
		for x in range(7):
			if self.board[5, x] == 0:
				moves.append(x)

		return moves

	def get_state(self):
		return torch.from_numpy(self.encode_board()).float()

	def get_state_symmetries(self):
		s = self.get_state()
		s_mirror = s.flip(2)

		symmetries = [s]
		if not s.equal(s_mirror):
			symmetries.append(s_mirror)

		return symmetries

	def apply(self, a: int):
		if self.result is not None:
			raise Exception("Tried to play a move in a finished game")

		x_move = a
		y_move = None
		for y in range(6):
			if self.board[y, x_move] == 0:
				y_move = y
				break
		if y_move is not None:
			self.board[y_move, a] = self.turn
		else:
			raise Exception("Tried to add piece to a full column")

		# Check if win
		vert = 1
		hori = 1
		diag1 = 1
		diag2 = 1

		x_max = min(x_move + 3, 6)
		x_min = max(x_move - 3, 0)
		y_max = min(y_move + 3, 5)
		y_min = max(y_move - 3, 0)
		
		for x in range(x_move + 1, x_max + 1): # Right
			if self.board[y_move, x] == self.turn: hori += 1
			else: break
		for x in reversed(range(x_min, x_move)): # Left
			if self.board[y_move, x] == self.turn: hori += 1
			else: break

		for y in range(y_move + 1, y_max + 1): # Up
			if self.board[y, x_move] == self.turn: vert += 1
			else: break
		for y in reversed(range(y_min, y_move)): # Down
			if self.board[y,x_move] == self.turn: vert += 1
			else: break

		for dd1 in range(1, min(x_max-x_move, y_max-y_move) + 1): # NE
			if self.board[y_move+dd1, x_move+dd1] == self.turn: diag1 += 1
			else: break
		for dd1 in range(1, max(x_move-x_min, y_move-y_min) + 1): # SW
			if self.board[y_move-dd1, x_move-dd1] == self.turn: diag1 += 1
			else: break

		for dd2 in range(1, min(x_max-x_move, y_move-y_min) + 1): # SE
			if self.board[y_move-dd2, x_move+dd2] == self.turn: diag2 += 1
			else: break
		for dd2 in range(1, min(x_move-x_min, y_max-y_move) + 1): # NW
			if self.board[y_move+dd2, x_move-dd2] == self.turn: diag2 += 1
			else: break

		if vert >= 4 or hori >= 4 or diag1 >= 4 or diag2 >= 4:
			self.result = self.turn

		# Check if draw
		if self.result is None:
			draw = True
			for x in range(7):
				if self.board[5, x] == 0:
					draw = False
					break
			if draw:
				self.result = 0

		self.turn = -self.turn

		return self

	def clone(self):
		return Game(np.copy(self.board))

	def is_over(self):
		return self.result is not None

	def outcome(self):
		return self.result

	def num_moves(self):
		return self.move_count
	
	def to_play(self):
		return self.turn

	def encode_board(self):
		M = np.zeros((2, 6, 7), dtype=np.float)
		for y in range(6):
			for x in range(7):
				if self.board[y, x] == self.turn:
					M[0, y, x] = 1.0
				elif self.board[y, x] == -self.turn:
					M[1, y, x] = 1.0
		return M


