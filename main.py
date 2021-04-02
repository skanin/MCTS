# import game
# import random
# import mcts

# from copy import deepcopy

# for episode in range(5):
#     print(episode)
#     board = game.Board(5)
#     g = game.Game(board)
#     graph = game.Graph(board, True, .01)
#     graph.show_board()
#     legal_moves = g.get_legal_moves()

#     winning_player = -1
#     winning_path = []
#     s_init = g.to_string_representation()

#     root = mcts.Node(state=s_init)
#     tree = mcts.MCTS(root)

#     while len(legal_moves):
#         root.generate_child_states()
#         # print(root.children)
#         for _ in range(3):
#             G_mc = game.Game.game_from_string_representation(root.game_state_string)
#             leaf_node = tree.leaf_search(G_mc)
#             F = tree.rollout(G_mc, leaf_node)
#             tree.backprop(F)
        
#         # print(len(game.Game.generate_child_states(g)))
#         g.make_move(random.choice(legal_moves))
#         winning_player, win, winning_path = g.is_win()
#         graph.show_board()
#         if win:
#             break
#         legal_moves = g.get_legal_moves()


#     winning_path = list(map(lambda x: x.get_coords(), winning_path))
#     print(f'PLAYER {winning_player} WINS WITH THE PATH {winning_path} !!!!!!!')
#     graph.update_freq = 5
#     graph.show_board(winning_path)
#     graph.update_freq = .25



from env import Env
from simWorld.nim.Nim import Nim
import random
import mcts
import nn
import yaml
from copy import deepcopy
# hex = Hex(5, 1)

cfg = yaml.safe_load(open('config.yaml', 'r'))

def do_games(num_games, num_search_games):
    ANET = nn.Actor(cfg)
    RBUF = []

    for episode in range(num_games):
        game = Nim(cfg['nim']['num_stones'], 3)

        s_init = game.to_string_representation()
        root = mcts.Node(state=s_init, game_state=game)
        root_copy = root
        mct = mcts.MCTS(root, ANET)

        legal_moves = game.get_legal_moves()

        done = False

        while not done:
            mctGame = game.game_from_string_representation(root.game_state_string)
            leaf_nodes = []
            for search_game in range(num_search_games):
                path = mct.select(mctGame, root)
                leaf_node = path[-1]
                leaf_nodes.append(leaf_node)
                mct.expand(leaf_node)
                final_state, reward = mct.rollout(mctGame, leaf_node)
                mct.backprop(path, reward)
                mctGame = game.game_from_string_representation(root.game_state_string)
            dist = root.get_action_distribution() # list(map(lambda x: x[1].num_visits, list(root.children.items())))
            # print(dist)
            RBUF.append((root.game_state_string, dist))
            move = legal_moves[dist.index(max(dist))]
            string_state, done, player, legal_moves = game.make_move(move)
            if done:
                break

            new_root = list(filter(lambda x: x.game_state_string == string_state, list(root.children.values())))[0]
            mct.prune(new_root)
            root = new_root

        # mct.prune(root_copy)
        # mct.show()
        print(f'Player {player} wins!!')
        ANET.trainOnRBUF(RBUF, 2)

if __name__ == '__main__':
    do_games(cfg['training']['num_games'], cfg['training']['num_search_games'])