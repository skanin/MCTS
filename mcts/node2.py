# from game import Game
class Node2:
    def __init__(self, move = None, parent = None, player = None, state=None):
        self.move = move
        self.parent = parent
        self.num_visits = 0
        self.value = 0
        self.player = player   
        self.state = state 

    def __repr__(self):
        return str(self.move)