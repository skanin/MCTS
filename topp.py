import os
import itertools
import yaml
from simWorld.hex_game.Hex import Hex
from collections import defaultdict
from policy import Policy
from pprint import pprint

class Topp:
    def __init__(self, cfg):
        self.cfg = cfg['topp']
        self.M = self.cfg['M']
        self.num_games = self.cfg['num_games']
        self.num_series = int((self.M + (self.M-1))/2)
        self.statistics = defaultdict(dict)

    def play_one_game(self, player1, player2, starting_player):
        game = self.init_game(starting_player)
        
        while game.winner == -1:
            if game.player == 1:
                move = player1.get_move(game)
            else:
                move = player2.get_move(game)
            
            game.make_move(move)
        
        return game.winner

    def play_tournament(self):
        folder = f'TrainedNetworks/Hex/{self.cfg["game"]["board_size"]}'
        players = os.listdir(folder)
        lineups = itertools.combinations(players, 2)

        for player1, player2 in lineups:
            vs_string = f'{player1.upper()} VS {player2.upper()}!!'
            print('-'*len(vs_string) + '-'*8)
            print('|   ' + vs_string + '   |')
            print('-'*len(vs_string) + '-'*8)
            # print(f'{player1.upper()} VS {player2.upper()}!!')
            
            p1, p2 = Policy(f'{folder}/{player1}', f'{folder}/{player1}', 1), Policy(f'{folder}/{player2}', f'{folder}/{player2}', 2)
            player1Wins, player2Wins = 0, 0
            players = [p1, p2]
            starting_player = 1
            for game in range(self.num_games):
                winner = self.play_one_game(p1, p2, starting_player)
                
                player1Wins += winner == p1.player # p1w if game % 2 == 0 else p2w
                player2Wins += winner == p2.player # p2w if game % 2 != 0 else p1w

                # if winner == 1:
                print(f'{list(filter(lambda x: x.player == winner, players))[0].name} wins game {game+1}!')
                # else:
                #     print(f'Player 2 wins game {game+1}!')

                # starting_player = 1 if starting_player == 2 else 2

                # p1.player = 1 if p1.player == 2 else 2
                # p2.player = 1 if p2.player == 2 else 2
                
            self.statistics[player1][player2] = player1Wins
            self.statistics[player2][player1] = player2Wins
        
        pprint(self.statistics)

    def init_game(self, starting_player):
        game = Hex(board_size = self.cfg['game']['board_size'], display = self.cfg['game']['display'], starting_player = starting_player)
        game.player = starting_player
        return game

if __name__ == '__main__':
    topp = Topp(yaml.safe_load(open('config.yaml', 'r')))
    topp.play_tournament()