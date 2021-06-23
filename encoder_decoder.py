import chess
import numpy as np

OUT_SHAPE = (73, 8, 8)

# { key: val } -> { val: key }
def invert_dict(d):
	return { v: k for k, v in d.items() }

encode_underpromotion = {
	( 0, chess.KNIGHT): 0, ( 0, chess.BISHOP): 1, ( 0, chess.ROOK): 2,
	( 1, chess.KNIGHT): 3, ( 1, chess.BISHOP): 4, ( 1, chess.ROOK): 5,
	(-1, chess.KNIGHT): 6, (-1, chess.BISHOP): 7, (-1, chess.ROOK): 8
}
decode_underpromotion = invert_dict(encode_underpromotion)

encode_knight_move = {
	( 1,  2): 0, ( 2,  1): 1,
	( 2, -1): 2, ( 1, -2): 3,
	(-1, -2): 4, (-2, -1): 5,
	(-2,  1): 6, (-1,  2): 7
}
decode_knight_move = invert_dict(encode_knight_move)


def encode_action(move: chess.Move, board: chess.Board):
	from_file = chess.square_file(move.from_square)
	from_rank = chess.square_rank(move.from_square)

	to_file = chess.square_file(move.to_square)
	to_rank = chess.square_rank(move.to_square)

	# Orient board to player's perspective
	if board.turn == chess.BLACK:
		from_rank = 7 - from_rank
		to_rank = 7 - to_rank

	df = to_file - from_file # Delta file
	dr = to_rank - from_rank # Delta rank

	plane = None

	if move.promotion != None and move.promotion != chess.QUEEN: # Underpromotion
		plane = 64 + encode_underpromotion[(df, move.promotion)]
	elif abs(df) == 2 and abs(dr) == 1 or abs(df) == 1 and abs(dr) == 2: # Knight move
		plane = 56 + encode_knight_move[(df, dr)]
	else: # Any other move
		dist = max(abs(df), abs(dr)) - 1 # [0, 6]
		dir = None # [0, 7]
		if df == 0:
			dir = 0 if dr > 0 else 4
		elif dr == 0:
			dir = 2 if df > 0 else 6
		else:
			if df > 0:
				dir = 1 if dr > 0 else 3
			else:
				dir = 7 if dr > 0 else 5
		plane = 7 * dir + dist

	encoded = (plane * 8 + from_rank) * 8 + from_file
	return encoded


def decode_action(encoded: int, board: chess.Board):
	from_file = encoded % 8
	tmp = encoded // 8
	from_rank = tmp % 8
	plane = tmp // 8

	df = None # Delta file
	dr = None # Delta rank

	promotion = None

	if plane >= 64: # Underpromotion
		df, promotion = decode_underpromotion[plane - 64]
		dr = 1 # if from_rank == 6 else -1
	elif plane >= 56: # Knight move
		df, dr = decode_knight_move[plane - 56]
	else: # Any other move
		dist = (plane % 7) + 1 # [1, 7]
		dir = plane // 7 # [0, 7]
		if   dir == 0: df, dr =     0,  dist
		elif dir == 1: df, dr =  dist,  dist
		elif dir == 2: df, dr =  dist,     0
		elif dir == 3: df, dr =  dist, -dist
		elif dir == 4: df, dr =     0, -dist
		elif dir == 5: df, dr = -dist, -dist
		elif dir == 6: df, dr = -dist,     0
		elif dir == 7: df, dr = -dist,  dist

	to_file = from_file + df
	to_rank = from_rank + dr

	# De-orient board from player's perspective
	if board.turn == chess.BLACK:
		from_rank = 7 - from_rank
		to_rank = 7 - to_rank

	from_square = chess.square(from_file, from_rank)
	to_square = chess.square(to_file, to_rank)

	# Queen promotion
	if promotion is None and board.piece_type_at(from_square) == chess.PAWN:
		if to_rank == 0 or to_rank == 7:
			promotion = chess.QUEEN

	return chess.Move(from_square, to_square, promotion)


def encode_board(board: chess.Board):
	# M = np.zeros((14, 8, 8), dtype=np.float) # History planes
	# L = np.zeros((6, 8, 8), dtype=np.float) # Auxiliary planes
	ML = np.zeros((20, 8, 8), dtype=np.float)

	# Orient board to player's perspective
	mirror = board.turn == chess.BLACK
	P1 = board.turn
	P2 = not P1

	# P1 and P2 pieces
	for rank in range(8):
		for file in range(8):
			piece = board.piece_at(chess.square(file, rank))
			if piece != None:
				color_off = 0 if piece.color == P1 else 1
				type_off = 0 # chess.PAWN
				if   piece.piece_type == chess.KNIGHT: type_off = 1
				elif piece.piece_type == chess.BISHOP: type_off = 2
				elif piece.piece_type == chess.ROOK:   type_off = 3
				elif piece.piece_type == chess.QUEEN:  type_off = 4
				elif piece.piece_type == chess.KING:   type_off = 5
				# Flip board if black's turn
				rank_canon = 7 - rank if mirror else rank
				ML[color_off * 6 + type_off, rank_canon, file] = 1
	# Repetitions
	if board.is_repetition(2):
		ML[12, :, :] = 1
		if board.is_repetition(3):
			ML[13, :, :] = 1

	# # Color
	# if board.turn == chess.BLACK: L[0, :, :] = 1
	# # Total moves
	# L[1, :, :] = board.ply()

	# P1 and P2 castling
	if board.has_kingside_castling_rights(P1):  ML[14, :, :] = 1
	if board.has_queenside_castling_rights(P1): ML[15, :, :] = 1
	if board.has_kingside_castling_rights(P2):  ML[16, :, :] = 1
	if board.has_queenside_castling_rights(P2): ML[17, :, :] = 1
	# En passant
	if board.ep_square != None:
		rank = chess.square_rank(board.ep_square)
		file = chess.square_file(board.ep_square)
		# Flip board if black's turn
		rank_canon = 7 - rank if mirror else rank
		ML[18, rank_canon, file] = 1
	# No-progress count
	ML[19, :, :] = board.halfmove_clock

	# return M, L
	return ML

