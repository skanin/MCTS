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
import time
from games.nim import Nim
from games.hex_game import Hex, Hex2, Board, Graph
from policy import Policy

def test_nim(cfg):
    num_games = cfg['training']['num_games']
    num_search_games = cfg['training']['num_search_games']

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
        root = mcts.Node2(None, None, game.starting_player, s_init)
        mct = mcts.MCTS2(root, ANET, game.starting_player, cfg['epsilon'], game)

        done = False
        
        while not done:
            for search_game in range(num_search_games):
                leaf_node, leaf_state, path, actions = mct.select()
                turn = leaf_state.player
                outcome = mct.rollout(leaf_state, path)
                mct.backprop(leaf_node, turn, outcome, path, actions)

            dist = mct.get_action_distribution()
            RBUF.append((game.to_string_representation(), dist))
            move = dist.index(max(dist))+1
            string_state, done, player, legal_moves = game.make_move(move)
            if done:
                break

            tmp = mct.graph[hash(root.state)]
            new_root = list(filter(lambda x: x.move == move, mct.graph[hash(root.state)]))[0]

            # time.sleep(.5)
            # mct.show()
            mct.prune(new_root, game)
            root = new_root
            
        starting_player = 1 if starting_player == 2 else 2

        print(f'Player {player} wins!!')
        loss = ANET.trainOnRBUF(RBUF, 64 if len(RBUF) >= 64 else len(RBUF))
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
        game = Hex(cfg['hex']['board_size'], cfg['hex']['display'], starting_player)
        s_init = game.to_string_representation()
        root = mcts.Node2(None, None, game.starting_player, s_init)
        mct = mcts.MCTS2(root, ANET, starting_player, cfg['epsilon'], game)

        done = False
        while not done:
            for search_game in range(num_search_games):
                leaf_node, leaf_state, path, actions = mct.select()
                turn = leaf_state.player
                outcome = mct.rollout(leaf_state, path)
                mct.backprop(leaf_node, turn, outcome, path, actions)

            dist = mct.get_action_distribution()
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

            new_root = list(filter(lambda x: x.move == move, mct.graph[hash(root.state)]))[0]
            # time.sleep(.5)
            # mct.show()
            mct.prune(new_root, game)
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

def me_against_nim(cfg):
    
    # ANET = nn.Actor(cfg)
    modes_folder = 'TrainedNetworks/Nim'
    # diff = int(input(f'0: Easy, 1: Medium, 2: Hard, 3: Extra hard: '))
    modes = os.listdir(modes_folder)
    print(modes)
    
    starting_player = int(input('Who starts, you (1), computer (2) or random (3)? '))
    # if starting_player.lower() == "me":
    #     me = 1
    #     comp = 2
    #     print('You are starting, get ready!')
    # elif starting_player.lower() == 'comp':
    #     me = 2
    #     comp = 1
    #     print('Computer is starting, get ready!')
    if starting_player == 3:
        players = [1,2]
        starting_player = random.choice(players)
        # me = random.choice(players)
        # players.remove(me)
        # comp = players[0]
        # if me == 1:
        #     print('You are starting, get ready!')
        # else:
        #     print('Computer is starting, get ready!')
    if starting_player == 1:
        print('You are starting, get ready!')
    else:
        print('Computer is starting, get ready!')

    me = 1
    comp = 2
    time.sleep(1)
    print()
    done = False
    game = Nim(15, 3, starting_player)
    actor = Policy('TrainedNetworks/nim/nim-2stones-3removal-at-1000-of-1000-episodes-12-04-2021-18-25', 'nim-2stones-3removal-at-1000-of-1000-episodes-12-04-2021-18-25', 2, cfg)
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
            move = actor.get_move(game) # actor_move(ANET, game)
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
                game = Nim(15, 3, starting_player)
                done = False

def actor_move(ANET, game):
    print(game.to_string_representation())
    distribution = ANET.predict_val(game.to_string_representation()).detach().numpy().tolist()
    print(distribution)
    for i, move in enumerate(game.LEGAL_MOVES):
            if move not in game.get_legal_moves():
                distribution[i] = 0

    distribution = [i/sum(distribution) for i in distribution]
    random_move_prob = 1
    print(distribution)
    if random_move_prob < random.uniform(0,1):
        ind = distribution.index(random.choices(population=distribution, weights=distribution)[0])
    else:
        ind = distribution.index(max(distribution))

    return game.LEGAL_MOVES[ind]

if __name__ == '__main__':
    # do_games(cfg['training']['num_games'], cfg['training']['num_search_games'])
    # test_nim(yaml.safe_load(open('config.yaml', 'r')))
    hex5conf = yaml.safe_load(open('hex5config.yaml', 'r'))
    # hex4conf = yaml.safe_load(open('config.yaml', 'r'))
    me_against_nim(yaml.safe_load(open('config.yaml', 'r')))
    # test_hex(cfg['training']['num_games'], cfg['training']['num_search_games'])
    # test_hex(hex5conf)