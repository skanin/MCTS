import matplotlib.pyplot as plt
from games.hex_game import Hex, Board, Graph
from games.nim import Nim
from nn import Actor, Rbuf, ConvActor
from mcts import MCTS, Node
import time
import pickle
from copy import deepcopy
import sys
sys.setrecursionlimit(100000)

class ReinfocementLearner:
    def __init__(self, cfg):
        self.cfg = cfg

        # Training and MCTS configs
        # -------------------------
        self.num_games = cfg['training']['num_games']
        self.alternate_players = cfg['training']['alternate_players']
        self.num_search_games_limit = cfg['training']['num_search_games_limit']
        self.time_limit = cfg['training']['rollout_time_limit']
        self.save_anet = cfg['training']['save']
        self.epsilon = cfg['training']['epsilon']
        self.epsilon_decay = cfg['training']['epsilon_decay']
        self.target_epsilon = cfg['training']['target_epsilon']
        self.c = cfg['training']['c']
        self.starting_player = cfg['training']['starting_player']

        # Game specific configs
        # ---------------------
        self.game_name = cfg['game'].lower()

        if self.game_name.lower() == 'nim':
            self.num_stones = cfg[self.game_name]['num_stones']
            self.max_removal = cfg[self.game_name]['max_removal']
            self.init_game = self._init_nim
        elif self.game_name.lower() == 'hex':
            self.board_size = cfg[self.game_name]['board_size']
            self.display = cfg[self.game_name]['display']
            self.init_game = self._init_hex
        else:
            raise ValueError(f'Game type {self.game_name} not permitted!')

        # ANET specific configs
        # ---------------------
        if self.game_name == 'nim':
            inp_size = len(str(self.num_stones))
        else:
            inp_size = self.board_size**2 + 1

        self.ANET = ConvActor(cfg[self.game_name]['actor']['learning_rate'], 
                                    inp_size, cfg[self.game_name]['actor']['layers'],
                                    cfg[self.game_name]['actor']['loss_fn'], 
                                    cfg[self.game_name]['actor']['activation_fn'], 
                                    cfg[self.game_name]['actor']['output_activation'], 
                                    cfg[self.game_name]['actor']['optimizer'])
        
        self.anet_save_interval = self.num_games//4
        self.RBUF = Rbuf(cfg['training']['rbuf_cap'])
        self.minibatch_size = cfg['training']['minibatch_size']

        self.player1 = [0 for _ in range(self.num_games)]
        self.player2 = [0 for _ in range(self.num_games)]

        self.player1wins = 0
        self.player2wins = 0
        self.losses = []

    def train(self):
        for episode in range(self.num_games):
            print(f'Episode {episode + 1}/{self.num_games}')
            
            game = self.init_game()
            s_init = game.to_string_representation()
            root = Node(None, None, game.starting_player, s_init)
            mct = MCTS(root, self.ANET, 1, self.epsilon, self.epsilon_decay, self.target_epsilon, game, self.c)

            done = False
            
            if self.save_anet and episode == 0:
                self.ANET.save(self.game_name, game, 0, self.num_games)

            while not done:
                start_time = time.time()
                i = 0
                while time.time() - start_time < self.time_limit and i < self.num_search_games_limit:
                    i += 1 
                    leaf_node, leaf_state, path, actions = mct.select()
                    turn = leaf_state.player
                    outcome = mct.rollout(leaf_state, path)
                    mct.backprop(leaf_node, turn, outcome, path, actions)


                dist = mct.get_action_distribution()
               
                self.RBUF.add((game.to_numpy().reshape(self.board_size, self.board_size, 2), dist.copy()))

                move = game.LEGAL_MOVES[dist.index(max(dist))]

                if self.game_name == 'nim':
                    _, done, player, _ = game.make_move(move)
                else:
                    if game.display:
                        _, player, done, _, _ = game.make_move(move)
                    else:
                        _, done, player, _ = game.make_move(move)

                if done:
                    break

                
                new_root = mct.nodes[game.to_string_representation()]
                # time.sleep(.5)
                # mct.show()
                mct.prune(new_root, game)
                root = new_root

            if self.alternate_players:
                self._change_starting_player()

            # time.sleep(.5)
            # mct.show()
            print(f'Player {player} wins!!')

            self._anet_train(episode)
            self._add_player_wins(player, episode)

            if (episode + 1) % self.anet_save_interval == 0 and self.save_anet:
                self.ANET.save(self.game_name, game, episode+1, self.num_games)
    
    def _init_hex(self):
        if self.game_name.lower() != 'hex':
            raise ValueError('Game is not Hex. Cannot call init_hex!')

        return Hex(self.board_size, self.display, self.starting_player, self.cfg)
    
    def _init_nim(self):
        if self.game_name.lower() != 'nim':
            raise ValueError('Game is not nim. Cannot call init_nim!')

        return Nim(self.num_stones, self.max_removal, self.starting_player)

    def _change_starting_player(self):
        self.starting_player = 1 if self.starting_player == 2 else 2

    def _anet_train(self, episode):
        loss = self.ANET.trainOnRBUF(self.RBUF.sample(self.minibatch_size))
        self.losses.append((episode + 1, loss))

    def _add_player_wins(self, player, episode):
        if player == 1:
            self.player1wins += 1
        else:
            self.player2wins += 1
        
        self.player1[episode] = self.player1wins
        self.player2[episode] = self.player2wins

    def plot_loss(self):
        loss_y_values = [x[1] for x in self.losses]
        loss_x_values = [x[0] for x in self.losses]

        plt.cla()
        plt.clf()
        plt.plot(loss_x_values, loss_y_values, color='tab:orange', label='loss')
        plt.legend()
        plt.show(block=True)

    def plot_winnings(self):
        X = [i+1 for i in range(self.num_games)]

        plt.cla()
        plt.clf()

        plt.plot(X, self.player1, color='r', label='Player 1')
        plt.plot(X, self.player2, color='g', label='Player 2')
        plt.legend()
        plt.show(block=True)
