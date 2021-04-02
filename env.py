import yaml

from simWorld.hex_game.Hex import Hex
from simWorld.nim.Nim import Nim

cfg = ''

with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)


class Env():
    def __init__(self):
        self.reset()

    def reset(self):
        if cfg['game'].lower() == 'nim':
            self.game = Nim(cfg['nim']['num_stones'], cfg['nim']['max_removal'])
        return self.game, self.game.to_string_representation(), self.game.is_win(), 
    
    def step(self, move):
        return self.game.make_move(move)