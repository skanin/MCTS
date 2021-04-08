import yaml
from simWorld.sim_world import SimWorld

class Nim(SimWorld):
    def __init__(self, num_stones, max_removal, starting_player):
        self.player = 1
        self.starting_player = starting_player
        self.num_stones = num_stones
        self.max_removal = max_removal
        self.cfg = yaml.safe_load(open('config.yaml', 'r'))
        self.LEGAL_MOVES = [i for i in range(1, self.max_removal + 1)]
        self.winner = -1

    
    def get_legal_moves(self):
        return [i for i in self.LEGAL_MOVES if i <= self.num_stones]

    def to_string_representation(self):
        return f'{self.player if not self.is_win() else self.opposite_player()}{self.num_stones if self.num_stones >= 10 else "0" + str(self.num_stones)}'

    def opposite_player(self):
        return 1 if self.player == 2 else 2

    def is_win(self):
        return self.num_stones == 0
    
    def get_winner(self):
        if not self.is_win():
            return False
        return self.player

    def is_legal_move(self, move):
        return move in self.get_legal_moves()

    def change_player(self):
        self.player = 1 if self.player == 2 else 2

    def make_move(self, move):
        if not self.is_legal_move(move):
            raise('Not legal move!')
        self.num_stones -= move
        if self.is_win():
            self.winner = self.player
            return self.to_string_representation(), True, self.player, self.get_legal_moves()
        self.change_player()
        return self.to_string_representation(), False, self.player, self.get_legal_moves()

    def game_from_string_representation(self, st):
        player = int(st[0])
        num_stones = int(st[1:])
        max_removal = self.cfg['nim']['max_removal']
        nim = Nim(num_stones, max_removal)
        nim.player = player
        return nim