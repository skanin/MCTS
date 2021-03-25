from game import Game
from copy import deepcopy

class Node:
    def __init__(self, state=None, move = None, parent = None):
        self.move = move
        self.parent = parent
        self.children = {}
        self.game_state_string = state
        self.game_state = Game.game_from_string_representation(self.game_state_string)
        self.backprop = 0
    
    def __str__(self):
        return self.game_state.to_string_representation()

    def add_child(self, child, move):
        self.children[move] = child
        child.parent = self

    
    def generate_child_states(self):
        legal_moves = self.game_state.get_legal_moves()
        game_copy = deepcopy(self.game_state)
        for move in legal_moves:
            game_copy.make_move(move)
            if move not in self.children:
                self.children[move] = Node(game_copy.to_string_representation(), move, self)
            game_copy = deepcopy(self.game_state)