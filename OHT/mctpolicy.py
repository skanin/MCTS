import random
import yaml
import numpy as np
from nn import NeuralNetwork
import tensorflow as tf
from mcts import MCTS, Node
import time
from copy import deepcopy

class Policy:
    def __init__(self, model, name, player, cfg, timeout):
        self.cfg = cfg
        self.mct = MCTS(None, player, None, self.cfg['training']['c'])
        self.name = name
        self.player = player
        self.random_move_prob = 1
        self.timeout = timeout
    
    def get_move(self, game):
        
        s_init = game.to_string_representation()
        root = Node(None, None, self.player, s_init)
        self.mct.root = root
        self.mct.root_state = deepcopy(game)
        
        start_time = time.time()

        while time.time() - start_time < self.timeout:
            leaf_node, leaf_state, path, actions = self.mct.select()
            turn = leaf_state.player
            outcome = self.mct.rollout(leaf_state, path)
            # print(f'{outcome} {outcome == self.mct.player}')
            self.mct.backprop(leaf_node, turn, outcome, path, actions)

        dist = self.mct.get_action_distribution()
        return game.LEGAL_MOVES[dist.index(max(dist))]

class RandomPolicy:
    def __init__(self, name, player):
        self.name = name
        self.player = player
    
    def get_move(self, game):
        return random.choice(game.get_legal_moves())