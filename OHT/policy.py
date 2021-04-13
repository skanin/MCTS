import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import yaml
import numpy as np
from nn import NeuralNetwork
class Policy:
    def __init__(self, model, name, player, cfg):
        self.game_name = cfg['game']

        if self.game_name == 'nim':
            inp_size = len(str(cfg['nim']['num_stones'])) + 1
        else:
            inp_size = cfg['hex']['board_size']**2 + 1

        self.model = NeuralNetwork(cfg[self.game_name]['actor']['learning_rate'], 
                                    inp_size, cfg[self.game_name]['actor']['layers'],
                                    cfg[self.game_name]['actor']['loss_fn'], 
                                    cfg[self.game_name]['actor']['activation_fn'], 
                                    cfg[self.game_name]['actor']['output_activation'], 
                                    cfg[self.game_name]['actor']['optimizer'])
        
        self.model.load_state_dict(torch.load(model))
        self.model.eval()
        # self.model = self.load_model(model)
        self.name = name
        self.player = player

    def load_model(self, model):
        # model = torch.load(open(model, 'rb'))
        # model.eval()
        # return model
        model = self.ANET.model.load_state_dict(torch.load(model))
        model.eval()
        return model

    
    def string_state_to_tensor(self, st):
        state = torch.Tensor([int(i) for i in st])
        state = state.to(self.model.device, non_blocking=True)
        return state

    def get_move(self, game):
        # print(game.to_string_representation())
        distribution = self.model(self.string_state_to_tensor(game.to_string_representation()))# .detach().numpy().tolist()
        
        if 'cuda' in distribution.device.type:
            distribution = distribution.cpu().detach().numpy().tolist()
        else:
            distribution = distribution.detach().numpy().tolist()
        for i, move in enumerate(game.LEGAL_MOVES):
                if move not in game.get_legal_moves():
                    distribution[i] = 0

        distribution = [i/sum(distribution) for i in distribution]

        random_move_prob = 1
        if random_move_prob < random.uniform(0,1):
            ind = distribution.index(random.choices(population=distribution, weights=distribution)[0])
        else:
            ind = distribution.index(max(distribution))

        return game.LEGAL_MOVES[ind]

class RandomPolicy:
    def __init__(self, name, player):
        self.name = name
        self.player = player
    
    def get_move(self, game):
        return random.choice(game.get_legal_moves())