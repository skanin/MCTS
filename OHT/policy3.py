from copy import deepcopy
from mcts import MCTS, Node
import time
import tensorflow as tf
from nn import NeuralNetwork

class Policy3:
    def __init__(self, model, name, player, cfg):
        self.game_name = cfg['game']

        if self.game_name == 'nim':
            inp_size = len(str(cfg['nim']['num_stones'])) + 1
        else:
            inp_size = cfg['hex']['board_size']**2 + 1

        self.ANET = self.load_model(model) #  NeuralNetwork(cfg[self.game_name]['actor']['learning_rate'], 
                                    # inp_size, cfg[self.game_name]['actor']['layers'],
                                    # cfg[self.game_name]['actor']['loss_fn'], 
                                    # cfg[self.game_name]['actor']['activation_fn'], 
                                    # cfg[self.game_name]['actor']['output_activation'], 
                                    # cfg[self.game_name]['actor']['optimizer'])
        self.name = name
        self.player = player
        self.epsilon = cfg['training']['epsilon']
        self.epsilon_decay = cfg['training']['epsilon_decay']
        self.target_epsilon = cfg['training']['target_epsilon']
        self.c = cfg['training']['c']

    def load_model(self, model):
        return tf.keras.models.load_model(model, custom_objects={'deepnet_cross_entropy': NeuralNetwork.deepnet_cross_entropy})

    def get_move(self, game):
        start = time.time()
        s_init = game.to_string_representation()
        root = Node(None, None, game.starting_player, s_init)
        mct = MCTS(root, self.ANET, self.player, self.epsilon, self.epsilon_decay, self.target_epsilon, deepcopy(game), self.c)

        start_time = time.time()
        while time.time() - start_time < 5:
            leaf_node, leaf_state, path, actions = mct.select()
            turn = leaf_state.player
            outcome = mct.rollout(leaf_state, path)
            mct.backprop(leaf_node, turn, outcome, path, actions)

        dist = mct.get_action_distribution()

        print(f'{time.time() - start} seconds!!')
        return game.LEGAL_MOVES[dist.index(max(dist))]
