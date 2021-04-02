# from game import Game
from copy import deepcopy
from collections import defaultdict
import numpy as np
import math
class Node:
    def __init__(self, state=None, game_state=None, move = None, parent = None):
        self.move = move
        self.parent = parent
        self.children = {}
        self.game_state_string = state
        self.game_state = game_state # Game.game_from_string_representation(self.game_state_string)
        self.backprop = 0
        self.nsa = defaultdict(lambda: defaultdict(int))
        self.num_times_action_taken = defaultdict(int)
        self.num_visits = 0
        self.value = 0
        self.c = 1    
    
    def __repr__(self):
        return self.game_state_string

    def add_child(self, child, move):
        if move not in self.children:
            self.children[move] = child
        child.parent = self

    def add_action_taken(self, action):
        self.num_times_action_taken[action] += 1
        # self.num_visits += 1

    def get_q_value(self, action):
        return 0 if self.num_times_action_taken[action] == 0 else self.value / self.num_times_action_taken[action]

    def get_usa(self, action):
        return float('inf') if self.num_visits == 0 else self.c * math.sqrt(math.log(self.num_visits)/(1 + self.num_times_action_taken[action]))

    def get_q_usa_comb(self, action):
        return self.get_q_value(action) + self.get_usa(action)

    def generate_child_states(self):
        legal_moves = self.game_state.get_legal_moves()
        game_copy = deepcopy(self.game_state)
        for move in legal_moves:
            game_copy.make_move(move)
            if move not in self.children:
                self.children[move] = Node(state=game_copy.to_string_representation(), move=move, parent=self, game_state=game_copy)
            game_copy = deepcopy(self.game_state)

    def get_action_distribution(self):
        if len(self.children.keys()) == len(self.game_state.LEGAL_MOVES):
            return list(map(lambda x: x[1].num_visits, list(self.children.items())))
        
        moves = list(map(lambda x: (x[0][1] - 1, x[1].num_visits), list(self.children.items())))
        dist = [0 for _ in range(len(self.game_state.LEGAL_MOVES))]
        for action in moves:
            dist[action[0]] = action[1]
        
        return dist

