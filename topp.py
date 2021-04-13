import os
import itertools
import yaml
from games.hex_game import Hex
from games.nim import Nim
from collections import defaultdict
from policy import Policy, RandomPolicy
from pprint import pprint

class ToppHex:
    def __init__(self, cfg):
        self.cfg = cfg
        self.M = self.cfg['topp']['M']
        self.num_games = self.cfg['topp']['num_games']
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

    def play_vs_random(self, starting_player, p1, p2):
        game = self.init_game(starting_player)

        while game.winner == -1:
            if game.player == 1:
                move = p1.get_move(game)
            else:
                move = p2.get_move(game)
            game.make_move(move)
        
        return game.winner

    
    def tournament_vs_random(self, alternate):
        starting_player = 1
        player1Wins, player2Wins = 0, 0
        folder = f'TrainedNetworks/Hex/{self.cfg["topp"]["game"]["board_size"]}'
        p1 = Policy(f'{folder}/hex-6x6-at-1000-of-1000-episodes-with-200-simulations-10-04-2021-11-42', f'{folder}/hex-6x6-at-1000-of-1000-episodes-with-200-simulations-10-04-2021-11-42', 1)
        p2 = RandomPolicy("Random", 2)

        for game in range(self.num_games):
            winner = self.play_one_game(p1, p2, starting_player)

            player1Wins += winner == p1.player
            player2Wins += winner == p2.player

            if alternate:
                starting_player = 1 if starting_player == 2 else 2

        self.statistics[p1.name][p2.name] = player1Wins
        self.statistics[p2.name][p1.name] = player2Wins
        pprint(self.statistics)

    def play_tournament(self):
        folder = f'TrainedNetworks/Hex/{self.cfg["topp"]["game"]["board_size"]}'
        players = os.listdir(folder)
        print(sorted(players, reverse=True))
        lineups = itertools.combinations(players, 2)

        for player1, player2 in lineups:
            vs_string = f'{player1.upper()} VS {player2.upper()}!!'
            print('-'*len(vs_string) + '-'*8)
            print('|   ' + vs_string + '   |')
            print('-'*len(vs_string) + '-'*8)
            # print(f'{player1.upper()} VS {player2.upper()}!!')
            
            p1, p2 = Policy(f'{folder}/{player1}', f'{player1}', 1, self.cfg), Policy(f'{folder}/{player2}', f'{player2}', 2, self.cfg)
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

                starting_player = 1 if starting_player == 2 else 2

                # p1.player = 1 if p1.player == 2 else 2
                # p2.player = 1 if p2.player == 2 else 2
                
            self.statistics[p1.name][p2.name] = player1Wins
            self.statistics[p2.name][p1.name] = player2Wins
        
        pprint(self.statistics)

    def init_game(self, starting_player):
        game = Hex(board_size = self.cfg['topp']['game']['board_size'], display = self.cfg['topp']['game']['display'], starting_player = starting_player, cfg=self.cfg['topp'])
        game.player = starting_player
        return game


class ToppNim:
    def __init__(self, cfg):
        self.cfg = cfg
        self.M = self.cfg['topp']['M']
        self.num_games = self.cfg['topp']['num_games']
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
        folder = f'TrainedNetworks/Nim'
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
        game = Nim(self.cfg['nim']['num_stones'], self.cfg['nim']['max_removal'], starting_player)
        return game

if __name__ == '__main__':
    toppHex = ToppHex(yaml.safe_load(open('config.yaml', 'r')))
    toppHex.play_tournament()
    # toppHex.tournament_vs_random(True)
    # toppNim = ToppNim(yaml.safe_load(open('config.yaml', 'r')))
    # toppNim.play_tournament()