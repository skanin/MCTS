import os
import itertools
import yaml
import seaborn as sns; sns.set_theme()
import numpy as np
import matplotlib.pyplot as plt

from games.hex_game import Hex
from games.nim import Nim
from collections import defaultdict
from policy import Policy, RandomPolicy
from pprint import pprint

class Topp:
    def __init__(self, cfg):
        self.cfg = cfg
        self.M = self.cfg['topp']['M']
        self.num_games = self.cfg['topp']['num_games']
        self.num_series = int((self.M + (self.M-1))/2)
        self.statistics = defaultdict(dict)
        self.stats = np.zeros((self.M, self.M))

    def play_one_game(self, player1, player2, starting_player, game_no):
        game = self.init_game(starting_player, game_no)
        
        while game.winner == -1:
            if game.player == 1:
                move = player1.get_move(game)
            else:
                move = player2.get_move(game)
            
            game.make_move(move)
        
        return game.winner

    def play_tournament(self):
        folder = self.cfg['topp']['folder']
        players_folder = os.listdir(folder)[0:self.M]
        player_episodes_trained = [int(player.split('of')[0][-1-self.cfg['topp']['players']['episode_num_len']:-1]) for player in players_folder]
        lineups = itertools.combinations(players_folder, 2)

        for player1, player2 in lineups:
            vs_string = f'{player1.upper()} VS {player2.upper()}!!'
            print('-'*len(vs_string) + '-'*8)
            print('|   ' + vs_string + '   |')
            print('-'*len(vs_string) + '-'*8)
            
            p1, p2 = Policy(f'{folder}/{player1}', f'{player1}', 1, self.cfg), Policy(f'{folder}/{player2}', f'{player2}', 2, self.cfg)
            player1Wins, player2Wins = 0, 0
            players = [p1, p2]
            starting_player = 1
            for game in range(self.num_games):
                winner = self.play_one_game(p1, p2, starting_player, game)
                player1Wins += winner == p1.player 
                player2Wins += winner == p2.player

                print(f'{list(filter(lambda x: x.player == winner, players))[0].name} wins game {game+1}!')

                if self.cfg['topp']['alternate_players']:
                    starting_player = 1 if starting_player == 2 else 2
                
            self.statistics[p1.name][p2.name] = player1Wins
            self.statistics[p2.name][p1.name] = player2Wins
            
            p1_ind = players_folder.index(p1.name)
            p2_ind = players_folder.index(p2.name)

            self.stats[(p1_ind, p2_ind)] = player1Wins
            self.stats[(p2_ind, p1_ind)] = player2Wins
        
        pprint(self.statistics)
        self.plot_winnings(player_episodes_trained)

    def init_game(self, starting_player, game_no):
        if self.cfg['topp']['game']['display'] and game_no >= self.num_games - self.cfg['topp']['num_games_to_watch']:
            game = Hex(board_size = self.cfg['topp']['game']['board_size'], display = True, starting_player = starting_player, cfg=self.cfg)
        else:
            game = Hex(board_size = self.cfg['topp']['game']['board_size'], display = False, starting_player = starting_player, cfg=self.cfg)
        game.player = starting_player
        return game

    def plot_winnings(self, players_folder):
        plt.cla()
        plt.clf()
        ax = sns.heatmap(self.stats, vmin=0, vmax=self.num_games, annot=True, cmap=sns.cm.rocket_r, xticklabels=players_folder, yticklabels=players_folder)
        plt.show(block=True)