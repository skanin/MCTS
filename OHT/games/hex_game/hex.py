import math
import itertools
from copy import deepcopy, copy
from .board import Board
from .graph import Graph
import yaml
import random
import numpy as np

class Hex:
    def __init__(self, board_size, display, starting_player, cfg):
        self.board = Board(board_size)
        self.player = starting_player
        self.starting_player = starting_player
        self.display = display
        self.cfg = cfg
        self.LEGAL_MOVES = self.board.LEGAL_MOVES
        self.winner = -1

        if self.display:
            self.graph = Graph(self.board, self.cfg['graph']['pause'], self.cfg['graph']['update_freq'])

    
    def get_legal_moves(self):
        return self.board.get_legal_moves()

    def is_legal_move(self, move):
        return move in self.get_legal_moves()

    def make_move(self, move, mcts=False):
        if not self.is_legal_move(move):
            raise Exception('Illegal move')

        self.board.content[move[0]][move[1]] = self.player

        win = self.is_win()

        if self.player == 1 and not win[1]:
            self.player = 2
        elif self.player == 2 and not win[1]:
            self.player = 1

        self.winner = win[0]

        if not self.display:
            return self.to_string_representation(), win[1], self.player, self.get_legal_moves()

        if win[1] and not mcts:
<<<<<<< HEAD
            self.graph.update_freq = 5
=======
            self.graph.update_freq = 2
>>>>>>> c9388ebca9b1b337c54cb8f8c20f15125e4f692d
            self.graph.show_board(win[2])
            self.graph.update_freq = self.cfg['graph']['update_freq']
        elif not mcts and not win[1]:
            self.graph.show_board()

        return self.to_string_representation(), win[0], win[1], win[2], self.get_legal_moves()


    def euclidean(self, x, y):
        return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))

    def traverse(self, start, goal, player=0):
        open_list = []
        closed = []
        get_player = lambda x: self.board[x]
        start = Space(start, get_player(start))

        goal = Space(goal, get_player(goal))

        if start.player == -1 or start.player != player or goal.player != start.player:
            return []

        open_list.append(start)

        while len(open_list) > 0:
            space = min(open_list, key = lambda x: x.f)
            open_list.remove(space)
            if not len(open_list) and space.player == 0:
                return []

            while space.player == 0:
                space = min(open_list, key = lambda x: x.f)
                open_list.remove(space)
                if not len(open_list):
                    return []

            if space.player == 0 or space.player != player:
                continue

            closed.append(space.coords)

            if space.coords == goal.coords:
                path = []
                current = space
                while current is not None:
                    coords = current.coords
                    if coords in path:
                        return path[::-1]
                    if coords not in path:
                        path.append(coords)
                    
                    current = current.parent
                return path[::-1]
            
            for neighbor in filter(lambda x: get_player(x) == space.player and x not in closed, self.board.get_neighbors(space.coords)):
                g = space.f + 1
                h = self.euclidean(goal.coords, neighbor)
                f = h + g
                if neighbor in list(map(lambda x: x.coords, open_list)):
                    tmp = list(filter(lambda x: x.coords == neighbor, open_list))[0]
                    if tmp.g > g:
                        continue
                
                n = Space(neighbor, get_player(neighbor))
                n.parent = space if space.coords != neighbor else space.parent
                n.g = g
                n.h = h
                n.f = f
                open_list.append(n)
        return []
        

    def is_win(self):
        start_states = [
            [(0, i) for i in range(self.board.board_size)],
            [(i, 0) for i in range(self.board.board_size)]
        ]

        win_states = [ 
            [(self.board.board_size - 1, i) for i in range(self.board.board_size)],
            [(i, self.board.board_size - 1) for i in range(self.board.board_size)]
        ]

        paths = []
        if len(list(filter(lambda x: x==1, itertools.chain(*self.board.content)))) < 4:
            return -1, False, []

        for (player, start) in enumerate(start_states):
            player = player + 1
            for start_coord in start:
                for goal in win_states[player-1]:
                    if self.board[goal] != player:
                        continue
                    path = self.traverse(start_coord, goal, player=player)
                    if len(path) > 0:
                        paths.append((player, True, path))
        if len(paths) == 0:
            return -1, False, []
        
        return sorted(paths, key=lambda x: len(x[-1]))[0]


    def is_win_without_path(self):
        pass

    def to_string_representation_new(self):
        st = '-1' if self.player == 2 else '1'
        for elem in self.board.to_string_representation():
            st += '-1' if elem == '2' else elem
        return st
    
    def to_string_representation(self):
        return str(self.player) + self.board.to_string_representation()
    
    def to_list_representation_old(self):
        return list(map(float, ''.join(self.to_string_representation_old().split())))
    
    def to_list_representation(self):
        st = [-1] if self.player == 2 else [1]
        for elem in self.board.to_string_representation():
            st.append(-1) if elem == '2' else st.append(int(elem))
        return st


    def to_numpy(self, *args):
        st = args[0] if len(args) else self.to_list_representation_old()

        player = st[0]
        board_size = int((len(st) - 1)**.5)
        state = np.zeros((board_size, board_size, 2))
        state[..., 1] = player
        state[..., 0] = np.array(st[1:]).reshape(board_size, board_size)

        return state.reshape(1, board_size, board_size, 2)


    def game_from_game(self, st, old_game):
        curr_player = int(st[0])
        board = Board.board_from_string_representation(st)
        game = Hex(old_game.board.board_size, display=old_game.display, starting_player=old_game.starting_player, cfg=self.cfg)
        game.player = curr_player
        game.starting_player = old_game.starting_player
        game.board = board
        return game

    
    def get_winner(self):
        win = self.is_win()
        if not win[1]:
            return False
        return win[0]

class Space:
    def __init__(self, coords, player):
        self.coords = coords
        self.player = player
        self.f = 0
        self.h = 0
        self.g = 0
        self.parent = None

if __name__ == '__main__':
    game = Hex(6, True, 1, yaml.safe_load(open('config.yaml', 'r')))

    st = (1, 2, 0, 2, 0, 0, 1, 0, 2, 0, 0, 1, 2, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 2, 0, 2, 2, 1, 2, 2, 2, 0, 0, 0, 0)
    
    # 2, 0, 2, 0, 0, 1, 
    # 0, 2, 0, 0, 1, 2, 
    # 1, 1, 1, 1, 0, 0, 
    # 0, 1, 0, 1, 0, 0, 
    # 2, 0, 2, 2, 1, 2, 
    # 2, 2, 0, 0, 0, 0
    
    # 2, 1, 2, 1, 1, 1,
    # 1, 2, 2, 1, 1, 2,
    # 1, 1, 1, 1, 1, 2,
    # 0, 1, 0, 1, 0, 0,
    # 2, 2, 2, 2, 1, 2,
    # 2, 2, 2, 0, 2, 2

    # 2, 1, 2, 1, 1, 1,
    # 1, 2, 2, 1, 1, 2,
    # 1, 1, 1, 1, 1, 2,
    # 1, 1, 0, 1, 2, 0,
    # 2, 2, 2, 2, 1, 2,
    # 2, 2, 2, 0, 2, 2
    
    state = ''
    for elem in st:
        state += str(elem)
    print(state)
    game = game.game_from_game(state, game)
    game.graph.pause = False
    print(game.is_win())
    game.graph.show_board()
    # while game.winner == -1:
    #     _, _, _, _, moves = game.make_move(random.choice(game.get_legal_moves()))
    # game = Hex(6, True, 1, yaml.safe_load(open('config.yaml', 'r')))
    
