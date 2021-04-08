from env import Env
from simWorld.nim.Nim import Nim
from simWorld.hex_game.Hex import Hex
from simWorld.hex_game.Hex2 import Hex2
from simWorld.hex_game.board import Board
from simWorld.hex_game.graph import Graph
import random
import mcts
import nn
import yaml
from copy import deepcopy
import matplotlib.pyplot as plt 
import math
import time
import os
import datetime
import logging
from threading import Thread
import time
from OHT.BasicClientActor import BasicClientActor

def do_games(num_games, num_search_games):
    ANET = nn.Actor(cfg)
    RBUF = []

    anet_save_interval = num_games//4

    player1 = [0 for _ in range(cfg['training']['num_games'])]
    player2 = [0 for _ in range(cfg['training']['num_games'])]

    player1wins = 0
    player2wins = 0
    losses = []
    starting_player = cfg['starting_player']
    for episode in range(num_games):
        print(f'Episode {episode + 1}/{num_games}')
        game = Nim(cfg['nim']['num_stones'], cfg['nim']['max_removal'], starting_player)

        s_init = game.to_string_representation()
        root = mcts.Node2(None, None, game.starting_player)
        root_copy = root
        mct = mcts.MCTS2(root, ANET, game.starting_player, cfg['epsilon'], game)

        legal_moves = game.get_legal_moves()

        done = False
        
        
        while not done:
            # mctGame = game.game_from_string_representation(root.game_state_string)
            leaf_nodes = []
            for search_game in range(num_search_games):
                
                # path = mct.select(root)
                # leaf_node = path[-1]
                #print(leaf_node.game_state.is_win())
                leaf_node, leaf_state = mct.select()
                leaf_nodes.append(leaf_node)
                # mct.expand(leaf_node)
                turn = leaf_state.player
                # final_state, reward = mct.rollout(mctGame, leaf_node)
                outcome = mct.rollout(leaf_state)
                mct.backprop(leaf_node, turn, outcome)
                # print(final_state)
                ## mct.backprop(leaf_node, reward)
                # mctGame = game.game_from_string_representation(root.game_state_string)
            dist = mct.get_action_distribution() # list(map(lambda x: x[1].num_visits, list(root.children.items())))
            # print(dist)
            RBUF.append((game.to_string_representation(), dist))
            # print(dist)
            # print(root.children.values())
            # print(list(map(lambda x: (x[0], x[1].num_visits), root.children.items())))
            move = dist.index(max(dist))+1
            # print(root.game_state_string)
            #  print(move)
            string_state, done, player, legal_moves = game.make_move(move)
            # time.sleep(1)
            if done:
                break
            
            # print(root)
            # print(root.children)

            # new_root = list(filter(lambda x: x.game_state_string == string_state, list(root.children.values())))[0]
            new_root = root.children[move]
            # time.sleep(.5)
            # mct.show()
            mct.prune(new_root, game)
            root = new_root
            

        # mct.prune(root_copy)
        # time.sleep(.5)
        # mct.show()
        starting_player = 1 if starting_player == 2 else 2
        print(f'Player {player} wins!!')
        # print(len(RBUF))
        loss = ANET.trainOnRBUF(RBUF, 128 if len(RBUF) >= 128 else len(RBUF))
        losses.append((episode + 1, loss))
        if player == 1:
            player1wins += 1
        else:
            player2wins += 1
        
        player1[episode] = player1wins
        player2[episode] = player2wins

        if (episode + 1) % anet_save_interval == 0 and cfg['training']['save']:
            ANET.save('Nim', f"nim-{cfg['nim']['num_stones']}stones-{cfg['nim']['max_removal']}removal-at-{episode + 1}-episodes-{datetime.datetime.strftime(datetime.datetime.now(), '%d-%m-%Y-%H-%M')}")
    
    loss_y_values = [x[1] for x in losses]
    loss_x_values = [x[0] for x in losses]

    plt.legend()
    plt.plot(loss_x_values, loss_y_values, color='tab:orange', label='loss')
    plt.show()

    X = [i+1 for i in range(cfg['training']['num_games'])]
    plt.plot(X, player1, color='r', label='Player 1')
    plt.plot(X, player2, color='g', label='Player 2')
    plt.legend()
    plt.show()


def test_hex(cfg):

    num_games = cfg['training']['num_games']
    num_search_games = cfg['training']['num_search_games']

    player1 = [0 for _ in range(cfg['training']['num_games']+1)]
    player2 = [0 for _ in range(cfg['training']['num_games']+1)]

    player1wins = 0
    player2wins = 0

    ANET = nn.Actor(cfg)
    RBUF = []
    if cfg['training']['save']:
        ANET.save(f'Hex/{cfg["hex"]["board_size"]}', f"hex-{cfg['hex']['board_size']}x{cfg['hex']['board_size']}-at-0-of-{num_games}-episodes-with-{num_search_games}-simulations-{datetime.datetime.strftime(datetime.datetime.now(), '%d-%m-%Y-%H-%M')}")
    anet_save_interval = num_games // (cfg['training']['num_saves'] - 1)
    losses = []
    starting_player = cfg['starting_player']

    for episode in range(num_games):
        print(f'Episode {episode + 1}/{num_games}')
        
        # board = Board(cfg['hex']['board_size'])
        game = Hex(cfg['hex']['board_size'], cfg['hex']['display'], starting_player)
        # graph = Graph(board, True, 0.1)
        # s_init = game.to_string_representation()
        # root = mcts.Node(state=s_init, game_state=game)
        root = mcts.Node2(None, None, game.starting_player)
        # root_copy = root
        mct = mcts.MCTS2(root, ANET, starting_player, cfg['epsilon'], game)

        # legal_moves = game.get_legal_moves()

        done = False
        while not done:
            # mctGame = deepcopy(game) # game.game_from_game(root.game_state_string, root.game_state)
            
            # leaf_nodes = []
            for search_game in range(num_search_games):
                # path = mct.select(mctGame, root)
                # leaf_node = path[-1]
                # leaf_nodes.append(leaf_node)
                # mct.expand(leaf_node)
                # final_state, reward = mct.rollout(mctGame, leaf_node)
                # mct.backprop(leaf_node, reward)
                # mctGame = deepcopy(game) # game.game_from_game(root.game_state_string, root.game_state)
                leaf_node, leaf_state = mct.select()
                turn = leaf_state.player
                outcome = mct.rollout(leaf_state)
                mct.backprop(leaf_node, turn, outcome)
            # dist = root.get_action_distribution() # list(map(lambda x: x[1].num_visits, list(root.children.items())))
            dist = mct.get_action_distribution()
            # print(dist)
            # RBUF.append((root.game_state_string, dist))
            RBUF.append((game.to_string_representation(), dist))
            max_ind = dist.index(max(dist))
            game_legal_moves = game.LEGAL_MOVES.copy()
            move = game_legal_moves[max_ind]
            
            if game.display:
                string_state, player, done, winning_path, legal_moves = game.make_move(move)
            else:
                string_state, done, player, legal_moves = game.make_move(move)

            if done:
                break

            # new_root = list(filter(lambda x: x.game_state_string == string_state, list(root.children.values())))[0]
            new_root = root.children[move]
            # time.sleep(.5)
            # mct.show()
            mct.prune(new_root, deepcopy(game))
            root = new_root

        loss = ANET.trainOnRBUF(RBUF, 128 if len(RBUF) >= 128 else len(RBUF))
        losses.append((episode + 1, loss))
        if player == 1:
            player1wins += 1
        else:
            player2wins += 1

        starting_player = 1 if starting_player == 2 else 2
        
        player1[episode+1] = player1wins
        player2[episode+1] = player2wins
        print(f'Player {player} wins!')
        if (episode + 1) % anet_save_interval == 0 and cfg['training']['save']:
            ANET.save(f'Hex/{cfg["hex"]["board_size"]}', f"hex-{cfg['hex']['board_size']}x{cfg['hex']['board_size']}-at-{episode + 1}-of-{num_games}-episodes-with-{num_search_games}-simulations-{datetime.datetime.strftime(datetime.datetime.now(), '%d-%m-%Y-%H-%M')}")


    loss_y_values = [x[1] for x in losses]
    loss_x_values = [x[0] for x in losses]

    plt.legend()
    plt.plot(loss_x_values, loss_y_values, color='tab:orange', label='loss')
    plt.show()

    X = [i for i in range(cfg['training']['num_games']+1)]
    plt.plot(X, player1, color='r', label='Player 1')
    plt.plot(X, player2, color='g', label='Player 2')
    plt.legend()
    plt.show()

def me_against_nim():
    
    ANET = nn.Actor(cfg)
    modes_folder = 'TrainedNetworks/Nim'
    diff = int(input(f'0: Easy, 1: Medium, 2: Hard, 3: Extra hard: '))
    modes = os.listdir(modes_folder)
    print(modes)
    ANET.load(f'{modes[diff]}')
    
    starting_player = input('Who starts, you (me), computer (comp) or random (rand)? ')

    if starting_player.lower() == "me":
        me = 1
        comp = 2
        print('You are starting, get ready!')
    elif starting_player.lower() == 'comp':
        me = 2
        comp = 1
        print('Computer is starting, get ready!')
    else:
        players = [1,2]
        me = random.choice(players)
        players.remove(me)
        comp = players[0]
        if me == 1:
            print('You are starting, get ready!')
        else:
            print('Computer is starting, get ready!')
    
    time.sleep(1)
    print()
    done = False
    game = Nim(15, 3)
    while not done:
        
        print(f'There are {game.num_stones} left in the pile')
        print()
        if game.player == me:
            move = int(input('How many stones do you want to remove? '))
            while not game.is_legal_move(move):
                print('That is not a legal move!!')
                move = int(input('How many stones do you want to remove? '))
            game.make_move(move)
        else:
            print('Computer is making a move ....')
            # dist = ANET.predict_val(game.to_string_representation()).detach().numpy().tolist()
            # print(dist)
            # move = dist.index(max(dist))+1
            move = actor_move(ANET, game)
            time.sleep(1)
            print(f'Computer chose to remove {move} stones!')
            game.make_move(move)
        print('-------------------------')

        if game.winner != -1:      
            if game.winner == me:
                print('Gratz, you won!')
            else:
                print('Computer won :(')
            done = input('Do you want to play again? y/n ').lower()
            if done == 'n':
                done = True
            else:
                game = Nim(15, 3)
                done = False

def actor_move(ANET, game):
    distribution = ANET.predict_val(game.to_string_representation()).detach().numpy().tolist()
    for i, move in enumerate(game.LEGAL_MOVES):
            if move not in game.get_legal_moves():
                distribution[i] = 0

    distribution = [i/sum(distribution) for i in distribution]
    random_move_prob = 1 
    print(distribution)
    if random_move_prob < random.uniform(0,1):
        print('rand')
        ind = distribution.index(random.choices(population=distribution, weights=distribution)[0])
    else:
        ind = distribution.index(max(distribution))

    return game.LEGAL_MOVES[ind]

if __name__ == '__main__':
    # do_games(cfg['training']['num_games'], cfg['training']['num_search_games'])

    hex5conf = yaml.safe_load(open('hex5config.yaml', 'r'))
    hex4conf = yaml.safe_load(open('config.yaml', 'r'))

    # test_hex(cfg['training']['num_games'], cfg['training']['num_search_games'])
    # test_hex(hex5conf)