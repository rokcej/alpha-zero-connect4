from game import Game
import math
import numpy as np

class Node():
	def __init__(self, P: float, to_play: int):
		self.P = P
		self.N = 0
		self.W = 0.0
		self.Q = 0.0

		# self.parent = None # MEMORY LEAK???
		self.children = {}
		self.to_play = to_play

# Upper confidence bound
def ucb(parent, child):
	c_base = 19652
	c_init = 1.25
	C = math.log((1 + parent.N + c_base) / c_base) + c_init
	U = C * child.P * math.sqrt(parent.N) / (1 + child.N)

	return -child.Q + U # Minus cause child has opposite sign

# Monte Carlo tree search
def mcts(net, game: Game, num_simulations, root=None):
	if root is None:
		root = Node(0, game.to_play())

	if len(root.children) == 0:
		expand(root, game, net)
		# Dirichlet noise
		actions = root.children.keys()
		noise = np.random.gamma(1.0, 1, len(actions)) # 0.3 for chess
		frac = 0.25
		for a, n in zip(actions, noise):
			root.children[a].P = root.children[a].P * (1.0 - frac) + n * frac

	for simulation in range(num_simulations):
		node = root
		path = [node]
		game_sim = game.clone()

		while len(node.children) > 0:
			# Select action
			action = None
			max_score = None
			for _action, _child in node.children.items():
				score = ucb(node, _child)
				if max_score is None or score > max_score:
					max_score = score
					action = _action

			node = node.children[action]
			path.append(node)
			game_sim.apply(action)
		
		value = expand(node, game_sim, net)

		for _node in path:
			_node.N += 1
			_node.W += value * _node.to_play
			_node.Q = _node.W / _node.N

	# Get policy
	temp = 1
	pi = np.zeros(game.action_space)
	N_sum = sum(child.N for child in root.children.values())
	for action, child in root.children.items():
		pi[action] = (child.N ** (1.0 / temp)) / N_sum

	# Get best action
	# _, best_action = max([(c.N, a) for a, c in root.children.items()])
	actions = list(root.children.keys())
	probs = pi[actions]
	if game_sim.num_moves() < 10: # Softmax sample
		probs = softmax(probs) # Is this necessary?
		best_action = np.random.choice(actions, p=pi[actions])
	else: # Max probability
		best_action = actions[np.argmax(probs)]

	return pi, best_action, root

# Expand leaf node
# Return absolute outcome
def expand(node: Node, game: Game, net):
	if game.is_over():
		return game.outcome()

	####################################################
	# This part takes up >95% of MCTS computation time #
	s = game.get_state().cuda().unsqueeze(0)
	p, v = net.predict_detach(s)
	####################################################

	actions = game.get_actions()
	p_sum = p[actions].sum().item()

	for action in actions:
		prior = p[action].item()
		if p_sum > 0.0:
			prior /= p_sum
		else:
			prior = 1.0 / len(actions)
			global WARNINGS_LEFT
			if WARNINGS_LEFT > 0:
				print(f"Warning: policy sum is zero ({WARNINGS_LEFT} warnings left)")
				WARNINGS_LEFT -= 1

		node.children[action] = Node(prior, -node.to_play)

	return v * node.to_play
	
# https://github.com/DylanSnyder31/AlphaZero-Chess/blob/master/Reinforcement_Learning/Monte_Carlo_Search_Tree/MCTS_main.py
def softmax(x):
	y = np.exp(x - np.max(x))
	y /= np.sum(y)
	return y
