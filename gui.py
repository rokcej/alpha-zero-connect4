import pygame
import os
import numpy as np

WIDTH = 700
HEIGHT = 600
DX = WIDTH / 7
DY = HEIGHT / 6

LIGHT_BLUE = (64, 72, 189)
DARK_BLUE = (52, 59, 158)
YELLOW = (180, 180, 50)
RED = (200, 70, 70)
PIECE_COLORS = [ YELLOW, RED ]

SPRITE_DIR = "data/sprites/"

class GUI():
	def __init__(self, board):
		self.board = board

		pygame.init()
		pygame.display.set_caption("Connect 4")

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
		self.screen.fill(LIGHT_BLUE)
		for y in range(6):
			for x in range(7):
				cx, cy = self.get_center(x, y)
				r = 0.45 * min(DX, DY)
				pygame.draw.circle(self.screen, DARK_BLUE, (cx, cy), r)

	def draw_pieces(self):
		for x in range(7):
			for y in range(6):
				piece = self.board[y, x]
				if piece != 0:
					cx, cy = self.get_center(x, y)
					r = 0.4 * min(DX, DY)
					color = YELLOW if piece == 1 else RED
					pygame.draw.circle(self.screen, color, (cx, cy), r)

	def get_center(self, col, row):
		return (col + 0.5) * DX, (5.5 - row) * DY

	def get_square(self, x, y):
		ix = (x * 7) // WIDTH
		iy = 5 - (y * 6) // HEIGHT
		return ix, iy

	def draw(self):
		self.draw_board()
		self.draw_pieces()
		pygame.display.flip()



if __name__ == "__main__":
	board = np.zeros((6, 7), dtype=np.int)
	board[2, 4] = 1
	board[5, 6] = -1
	gui = GUI(board)
	while gui.running:
		gui.draw()
		gui.handle_events()

