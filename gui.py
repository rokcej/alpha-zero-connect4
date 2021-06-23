import pygame
import os
import numpy as np

WIDTH = 700
HEIGHT = 600
DX = WIDTH / 7
DY = HEIGHT / 6

LIGHT_SQUARE = (64, 72, 189)
DARK_SQUARE = (52, 59, 158)
YELLOW = (180, 180, 50)
RED = (200, 70, 70)
PIECE_COLORS = [ YELLOW, RED ]

SPRITE_DIR = "data/sprites/"

class GUI():
	def __init__(self, board):
		self.board = board

		pygame.init()
		pygame.display.set_caption("AlphaZero Connect 4")

		self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
		self.running = True

		self.clicked = None

	def handle_events(self):
		self.clicked = None
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				self.running = False
			elif event.type == pygame.MOUSEBUTTONUP:
				x, y = pygame.mouse.get_pos()
				ix, iy = self.get_square(x, y)
				self.clicked = (ix, iy)
		
	def draw_board(self):
		for y in range(8):
			for x in range(8):
				color = LIGHT_SQUARE if (x + y) % 2 == 0 else DARK_SQUARE
				self.screen.fill(color, pygame.Rect(x * DX, y * DY, (x + 1) * DX, (y + 1) * DY))

	def draw_pieces(self):
		for x in range(7):
			for y in range(6):
				piece = self.board[y][x]
				if piece != 0:
					cx = (x + 0.5) * DX
					cy = (5.5 - y) * DY
					r = 0.4 * min(DX, DY)
					color = YELLOW if piece == 1 else RED
					pygame.draw.circle(self.screen, color, (cx, cy), r)

	def get_rect(self, file, rank):
		return pygame.Rect(file * DX, (5 - rank) * DY, (file + 1) * DX, (6 - rank) * DY)

	def get_square(self, x, y):
		ix = (x * 7) // WIDTH
		iy = 5 - (y * 6) // HEIGHT
		return ix, iy

	def draw(self):
		self.draw_board()
		self.draw_pieces()
		pygame.display.flip()



if __name__ == "__main__":
	board = [[0 for x in range(7)] for y in range(6)]
	board[2][4] = 1
	board[5][6] = -1
	gui = GUI(board)
	while gui.running:
		gui.draw()
		gui.handle_events()

