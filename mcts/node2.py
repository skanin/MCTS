# from game import Game
from copy import deepcopy
from collections import defaultdict
import numpy as np
import math
class Node2:
    def __init__(self, move = None, parent = None, player = None):
        self.move = move
        self.parent = parent
        self.children = {}
        self.nsa = defaultdict(int)
        self.num_visits = 0
        self.value = 0
        self.c = 2
        self.player = player    

    def add_child(self, child, move):
        if move not in self.children:
            self.children[move] = child
        child.parent = self

    def get_q_value(self):
        return 0 if self.parent.nsa[self.move] == 0 else  self.value / self.parent.nsa[self.move]
        # return 0 if self.num_visits == 0 else self.value / self.num_visits

    def get_usa(self):
        # return float('inf') if self.num_visits == 0 else self.c * math.sqrt(math.log(self.num_visits)/(1 + self.num_times_action_taken[action]))
        val = float('inf') if self.num_visits == 0 else math.sqrt(self.c * math.log(self.num_visits)/(self.parent.nsa[self.move]))
        # val = float('inf') if self.num_visits == 0 else self.c * math.sqrt(math.log(self.num_visits)/(1+self.parent.nsa[self.move]))
        # val = float('inf') if self.num_visits == 0 else self.c * math.sqrt(math.log(self.num_visits)/(self.parent.nsa[self.move]))
        return val

    def get_exploration_constant(self, node):
        return node.get_usa()

    def get_exploitation_constant(self, node):
        return node.get_q_value()

    def get_utc(self, node):
        return self.get_exploitation_constant(node) + self.get_exploration_constant(node)
    
    def get_utc_neg(self, node):
        return self.get_exploitation_constant(node) - self.get_exploration_constant(node)

    def get_action_distribution(self):
        if len(self.children.keys()) == len(self.game_state.LEGAL_MOVES):
            l = list(map(lambda x: x[1].num_visits, list(self.children.items())))
            if sum(l) == 0:
                return l
            return [float(i)/sum(l) for i in l]
            # return l

        game = self.game_state.cfg['game']
        if game == 'nim':
            moves = list(map(lambda x: (x[0] - 1, x[1].num_visits), list(self.children.items())))
            # print(moves)
            dist = [0 for _ in range(len(self.game_state.LEGAL_MOVES))]
            
            for action in moves:
                dist[action[0]] = action[1]
        else:
            # [(0,0), (0,1), (0,2), (0,3), (0,4), (1,0), (1,1), (1,2), (1,3), (1,4), ]
            # pos = ind0 * sqrt(len) + ind1
            moves = list(map(lambda x: (x[0], x[1].num_visits), list(self.children.items())))
            # print(moves)
            dist = [0 for _ in range(len(self.game_state.LEGAL_MOVES))]
            
            inc = len(dist)**(.5)
            for action in moves:
                dist[int(action[0][0]*inc + action[0][1])] = action[1]

        if sum(dist) == 0:
            return dist
        # return dist
        return [float(i)/sum(dist) for i in dist]

    def is_fully_expanded(self):
        if not len(self.children):
            return False

        for child in self.children.values():
            if child.num_visits == 0:
                return False
        return True