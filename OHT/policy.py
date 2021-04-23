import random
import yaml
import numpy as np
from nn import NeuralNetwork
import tensorflow as tf

class Policy:
    def __init__(self, model, name, player, cfg):
        self.game_name = cfg['game']
        
        if self.game_name == 'nim':
            self.inp_size = len(str(cfg['nim']['num_stones']))
        else:
            self.inp_size = cfg['hex']['board_size']**2 + 1

        self.model = self.load_model(model)
        self.name = name
        self.player = player
        self.random_move_prob = 1

    def load_model(self, model):
        return tf.keras.models.load_model(model, custom_objects={"deepnet_cross_entropy": NeuralNetwork.deepnet_cross_entropy})

    
    def string_state_to_tensor(self, st):
       return np.array([float(i) for i in st]).reshape(1, self.inp_size)

    def string_state_to_numpy(self, st):
        if len(st) == 2:
            state = np.array([float(st[0]), float(st[1:])]).reshape((1,2))
        else:
            player = np.zeros((1,6))
            player.fill(st[0])

            state = np.append(player, np.array([float(i) for i in st[1:]]).reshape((6, 6))).reshape(1,7,6,1)

            # state = np.array([float(i) for i in st]).reshape((1, len(st)))
        return state

    def string_state_to_numpy_dense(self, st):
        if len(st) == 2:
            state = np.array([float(st[0]), float(st[1:])]).reshape((1,2))
        else:
            state = np.array([float(i) for i in st]).reshape((1, len(st)))
        return state
    
    def get_move(self, game):
        # distribution = self.model(game.to_numpy()).numpy().flatten().tolist()
        distribution = self.model(game.to_numpy()).numpy().flatten().tolist()
        for i, move in enumerate(game.LEGAL_MOVES):
                if move not in game.get_legal_moves():
                    distribution[i] = 0

        distribution = [i/sum(distribution) for i in distribution]

        random_move_prob = self.random_move_prob
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