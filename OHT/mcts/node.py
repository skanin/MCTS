# from game import Game
from collections import defaultdict
class Node:
    def __init__(self, move = None, parent = None, player = None, state=None):
        self.move = move
        self.parent = parent
        self.num_visits = 0
        self.value = 0
        self.player = player   
        self.state = state 
        self.children = {}
        self.parents = []
        self.nsa = defaultdict(int)
        self.moves = dict()
        self.Q = defaultdict(int)
        self.add_parent(parent)
        self.add_move(parent, move)

    def __repr__(self):
        return str(self.move)

    def add_parent(self, parent):
        if parent not in self.parents:
            self.parents.append(parent)
            if parent is not None:
                parent.add_child(self)
    
    def add_child(self, node):
        self.children[node.state] = node

    def add_move(self, parent, move):
        self.nsa[(parent, move)] += 1
        self.moves[parent] = move  
    
    def get_move(self, parent):
        return self.moves[parent]

    def update_Q(self, move):
        child = self.get_child_by_move(move)
        self.Q[move] = child.value / child.num_visits

    def get_child_by_move(self, move):
        return list(filter(lambda x: x.moves[self] == move, self.children.values()))[0]