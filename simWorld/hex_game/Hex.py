from simWorld.sim_world import SimWorld
from .board import Board

class Hex(SimWorld):
    def __init__(self, board_size, starting_player):
        super(Hex, self).__init__(self)
        self.board = Board(board_size)
        self.player = starting_player

    
    def to_string_representation(self):
        return str(self.player) + self.board.to_string_representation()
    
    def is_win(self):
        pass

    def make_move(self, move):
        pass