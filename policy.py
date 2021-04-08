import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import yaml
import numpy as np

class Policy:
    def __init__(self, model, name, player):
        self.model = self.load_model(model)
        self.name = name
        self.player = player

    def load_model(self, model):
        model = torch.load(open(model, 'rb'))
        model.eval()
        return model

    
    def string_state_to_tensor(self, st):
        state = torch.Tensor([int(i) for i in st])
        state = state.to(self.model.device, non_blocking=True)
        return state

    def get_move(self, game):
        distribution = self.model.forward(self.string_state_to_tensor(game.to_string_representation()))# .detach().numpy().tolist()
        
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

