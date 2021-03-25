import game
import random
import mcts

from copy import deepcopy

for episode in range(5):
    print(episode)
    board = game.Board(5)
    g = game.Game(board)
    graph = game.Graph(board, True, .01)
    graph.show_board()
    legal_moves = g.get_legal_moves()

    winning_player = -1
    winning_path = []
    s_init = g.to_string_representation()

    root = mcts.Node(state=s_init)
    tree = mcts.MCTS(root)

    while len(legal_moves):
        root.generate_child_states()
        # print(root.children)
        for _ in range(3):
            G_mc = game.Game.game_from_string_representation(root.game_state_string)
            leaf_node = tree.leaf_search(G_mc)
            F = tree.rollout(G_mc, leaf_node)
            tree.backprop(F)
        
        # print(len(game.Game.generate_child_states(g)))
        g.make_move(random.choice(legal_moves))
        winning_player, win, winning_path = g.is_win()
        graph.show_board()
        if win:
            break
        legal_moves = g.get_legal_moves()


    winning_path = list(map(lambda x: x.get_coords(), winning_path))
    print(f'PLAYER {winning_player} WINS WITH THE PATH {winning_path} !!!!!!!')
    graph.update_freq = 5
    graph.show_board(winning_path)
    graph.update_freq = .25

# g = game.Game.game_from_string_representation('12011011112212210221212210')
# board = g.board
# graph = game.Graph(board, True, .1)
# graph.show_board()
# legal_moves = g.get_legal_moves()

# while len(legal_moves):
#     g.make_move(random.choice(legal_moves))
#     winning_player, win, winning_path = g.is_win()
#     graph.show_board()
#     if win:
#         break
#     legal_moves = g.get_legal_moves()

# if not win:
#     print('Heyheyhey')
#     winning_player, win, winning_path = g.is_win()
#     print(g.to_string_representation())
#     print(win)
# winning_path = list(map(lambda x: x.get_coords(), winning_path))
# print(f'PLayer {winning_player} wins with the path {winning_path}')
# graph.pause = False
# graph.show_board(winning_path)