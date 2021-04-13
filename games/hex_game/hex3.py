import math
import itertools
# from .space import Space
from copy import deepcopy, copy
from .board2 import Board2
# from .UnionFind import UnionFind
from .graph2 import Graph
import yaml
import random

class Hex3:
    def __init__(self, board_size, display, starting_player, cfg):
        self.board = Board2(board_size)
        self.player = starting_player
        self.starting_player = starting_player
        self.display = display
        self.cfg = cfg # yaml.safe_load(open('config.yaml', 'r'))
        self.LEGAL_MOVES = self.board.LEGAL_MOVES
        self.winner = -1

        if self.display:
            self.graph = Graph(self.board, self.cfg['graph']['pause'], self.cfg['graph']['update_freq']) #board, pause, update_freq 

    
    def get_legal_moves(self):
        # return list(map(lambda x: (self.player, x), self.board.get_legal_moves()))
        return self.board.get_legal_moves()

    def is_legal_move(self, move):
        return move in self.get_legal_moves()

    def make_move(self, move):
        if not self.is_legal_move(move):
            raise Exception('Illegal move')

        self.board.content[move[0]][move[1]] = self.player
        # space = self.board.get_space_from_coord(move)
        # space.set_piece(True, self.player)

        win = self.is_win()

        if self.player == 1 and not win[1]:
            self.player = 2
        elif self.player == 2 and not win[1]:
            self.player = 1

        self.winner = win[0]
        # self.player = int(not self.player) if not win[1] else self.player
        if not self.display:
            return self.to_string_representation(), win[1], self.player, self.get_legal_moves()

        if win[1]:
            # self.graph.pause = self.cfg['graph']['pause_on_win']
            self.graph.update_freq = 2
            self.graph.show_board(win[2])
            self.graph.update_freq = self.cfg['graph']['update_freq']
        else:
            
            self.graph.show_board()

        return self.to_string_representation(), win[0], win[1], win[2], self.get_legal_moves()


    def euclidean(self, x, y):
        return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))

    def traverse(self, start, goal, player=0):
        open_list = []
        closed = []
        get_player = lambda x: self.board[x]
        start = Space2(start, get_player(start)) # self.board.get_space_from_coord(start)

        goal = Space2(goal, get_player(goal))

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

            # space = copy(space) # deepcopy(space) if self.display else copy(space)

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
                # if not neighbor.has_piece() or neighbor.get_player() != player or neighbor.get_coords() in closed:
                #     print('Hey')
                #     continue
                
                g = space.f + 1
                h = self.euclidean(goal.coords, neighbor)
                f = h + g
                if neighbor in list(map(lambda x: x.coords, open_list)):
                    tmp = list(filter(lambda x: x.coords == neighbor, open_list))[0]
                    if tmp.g > g:
                        continue
                
                n = Space2(neighbor, get_player(neighbor))
                n.parent = space if space.coords != neighbor else space.parent
                n.g = g
                n.h = h
                n.f = f
                open_list.append(n)
        return []
        

    def is_win(self):
        start_states = [
            [(i, 0) for i in range(self.board.board_size)],
            [(0, i) for i in range(self.board.board_size)]
        ]

        win_states = [
            [(i, self.board.board_size - 1) for i in range(self.board.board_size)], 
            [(self.board.board_size - 1, i) for i in range(self.board.board_size)]
        ]

        paths = []
        if len(list(filter(lambda x: x==1, itertools.chain(*self.board.content)))) < 4:
            return -1, False, []

        for (player, start) in enumerate(start_states):
            player = player + 1
            # end_spaces = list(filter(lambda x: x != 0 and x == player, map(lambda y: self.board[y], win_states[player-1])))
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

    def to_string_representation(self):
        st = str(self.player) + self.board.to_string_representation()
        return st

    def game_from_string_representation(self, st):
        curr_player = int(st[0])
        board = Board.board_from_string_representation(st)
        game = Hex(board)
        game.player = curr_player
        return game

    def game_from_game(self, st, old_game):
        curr_player = int(st[0])
        board = Board.board_from_string_representation(st)
        game = Hex(old_game.board.board_size, display=old_game.display, starting_player=old_game.starting_player, cfg=self.cfg)
        game.player = curr_player
        game.starting_player = old_game.starting_player
        game.board = board
        # game.starting_player = old_game.starting_player
        return game

    # @staticmethod
    # def generate_child_states(game):
    #     legal_moves = game.get_legal_moves()
    #     game_copy = copy(game)
    #     child_states = []
    #     for move in legal_moves:
    #         game_copy.make_move(move)
    #         child_states.append((move, game_copy.to_string_representation()))
    #         game_copy = deepcopy(game)
    #     return child_states
    
    def get_winner(self):
        win = self.is_win()
        if not win[1]:
            return False
        return win[0]

class Space2:
    def __init__(self, coords, player):
        self.coords = coords
        self.player = player
        self.f = 0
        self.h = 0
        self.g = 0
        self.parent = None

if __name__ == '__main__':
    
    
    for _ in range(5):
        game = Hex3(15, True, 1, yaml.safe_load(open('config.yaml', 'r')))
        done = False
        while not done:
            move = random.choice(game.get_legal_moves())
            _, winner, done, _, _ = game.make_move(move)
        print(f'Player {winner} wins!')