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
        self.nsa = defaultdict(int)
        self.num_times_action_taken = defaultdict(int)
        self.num_visits = 0
        self.value = 0
        self.c = 2    
    
    def __repr__(self):
        return self.game_state_string

    def add_child(self, child, move):
        if move not in self.children:
            self.children[move] = child
        child.parent = self

    def add_action_taken(self, action):
        self.num_times_action_taken[action] += 1
        # self.num_visits += 1

    def get_q_value(self):
        return 0 if self.parent.nsa[self.move] == 0 else  self.value / self.parent.nsa[self.move]
        # return 0 if self.num_visits == 0 else self.value / self.num_visits

    def get_usa(self):
        # return float('inf') if self.num_visits == 0 else self.c * math.sqrt(math.log(self.num_visits)/(1 + self.num_times_action_taken[action]))
        val = float('inf') if self.num_visits == 0 else self.c * math.sqrt(math.log(self.parent.nsa[self.move])/(self.num_visits))
        # val = float('inf') if self.num_visits == 0 else self.c * math.sqrt(math.log(self.num_visits)/(1+self.parent.num_visits))
        return val

    @property
    def get_best_utc(self):
        if self.game_state.player == self.game_state.starting_player:
            return max(node.children.items(), key=lambda x: self.get_utc(x[1]), default=node)
        return min(node.children.items(), key=lambda x: self.get_utc_neg(x[1]), default=node)

    def get_exploration_constant(self, node):
        return node.get_usa()

    def get_exploitation_constant(self, node):
        return node.get_q_value()

    def get_utc(self, node):
        return self.get_exploitation_constant(node) + self.get_exploration_constant(node)
    
    def get_utc_neg(self, node):
        return self.get_exploitation_constant(node) - self.get_exploration_constant(node)

    def get_q_usa_comb(self, action):
        return self.get_q_value(action) + self.get_usa(action)

    def generate_child_states(self):
        legal_moves = self.game_state.get_legal_moves()
        # print(legal_moves)
        # print(self._get_legal_moves())
        game_copy = deepcopy(self.game_state) # self.game_state.game_from_game(self.game_state_string, self.game_state)
        expanded = False
        for move in legal_moves:
            game_copy.make_move(move)
            if move not in self.children:
                expanded = True
                self.children[move] = Node(state=game_copy.to_string_representation(), move=move, parent=self, game_state=game_copy)
            game_copy = deepcopy(self.game_state) # self.game_state.game_from_game(self.game_state_string, self.game_state)
        return expanded
        
    # def _get_legal_moves(self):
    #     state = list(self.game_state_string)[1:]
    #     n = int(len(state)**.5)
    #     state = [state[i:i+n] for i in range(0, len(state), n)]
    #     # print(state)
    #     moves = []
    #     for i, vals in enumerate(state):
    #         for j, val in enumerate(vals):
    #             if val != '0':
    #                 continue
    #             moves.append((i, j))
    #     return moves

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