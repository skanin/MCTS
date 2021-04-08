import math
import itertools
from .space import Space
from copy import deepcopy, copy
from .board import Board
from .UnionFind import UnionFind
from .graph import Graph
import yaml
import random

class Hex2:
    def __init__(self, board_size, display, starting_player):
        self.board = Board(board_size)
        self.player = 1
        self.starting_player = starting_player
        self.display = display
        self.cfg = yaml.safe_load(open('config.yaml', 'r'))
        self.LEGAL_MOVES = list(map(lambda x: x.get_coords(), itertools.chain(*self.board)))
        # self.winner = -1
        self.player_1_played = 0
        self.player_2_played = 0
        self.player_1_groups = UnionFind()
        self.player_2_groups = UnionFind()
        self.player_1_groups.set_ignored_elements([1, 2])
        self.player_2_groups.set_ignored_elements([3, 4])

        if self.display:
            self.graph = Graph(self.board, self.cfg['graph']['pause'], self.cfg['graph']['update_freq']) #board, pause, update_freq 

    
    def get_legal_moves(self):
        # return list(map(lambda x: (self.player, x), self.board.get_legal_moves()))
        return self.board.get_legal_moves()

    def is_legal_move(self, move):
        return move in self.get_legal_moves()

    # def make_move(self, move):
    #     if not self.is_legal_move(move):
    #         raise Exception('Illegal move')

    #     space = self.board.get_space_from_coord(move)
    #     space.set_piece(True, self.player)

    #     win = self.is_win()

    #     if self.player == 1 and not win[1]:
    #         self.player = 2
    #     elif self.player == 2 and not win[1]:
    #         self.player = 1

    #     self.winner = win[0]
    #     # self.player = int(not self.player) if not win[1] else self.player
    #     if not self.display:
    #         return self.to_string_representation(), win[1], self.player, self.get_legal_moves()

    #     if win[1]:
    #         # self.graph.pause = self.cfg['graph']['pause_on_win']
    #         self.graph.update_freq = 2
    #         self.graph.show_board(win[2])
    #         self.graph.update_freq = self.cfg['graph']['update_freq']
    #     else:
            
    #         self.graph.show_board()

    #     return self.to_string_representation(), win[0], win[1], win[2], self.get_legal_moves()


    def change_player(self):
        self.player = 1 if self.player == 2 else 2

    def make_move(self, move):
        if self.player == 1:
            self.place_player1(move)
        elif self.player == 2:
            self.place_player2(move)
        
        if self.winner == -1:
            self.change_player()

        if self.winner != -1:
            self.graph.pause = False

        self.graph.show_board()

    def place_player1(self, move):
        if move in self.get_legal_moves():
            self.board.content[move[0]][move[1]].set_piece(True, 1)
            self.player_1_played += 1
        else:
            raise ValueError(f"{move} is an illegal move!")
        # if the placed cell touches a white edge connect it appropriately
        if move[0] == 0:
            self.player_1_groups.join(1, move)
        if move[0] == self.board.board_size - 1:
            self.player_1_groups.join(2, move)
        # join any groups connected by the new white stone
        for n in self.board.get_space_from_coord(move).neighbors:
            if n.player == 1:
                self.player_1_groups.join(n.get_coords(), move)


    def place_player2(self, move):
        if move in self.get_legal_moves():
            self.board.content[move[0]][move[1]].set_piece(True, 2)
            self.player_2_played += 1
        else:
            raise ValueError(f"{move} is an illegal move!")
        # if the placed cell touches a white edge connect it appropriately
        if move[0] == 0:
            self.player_2_groups.join(3, move)
        if move[0] == self.board.board_size - 1:
            self.player_2_groups.join(4, move)
        # join any groups connected by the new white stone
        for n in self.board.get_space_from_coord(move).neighbors:
            if n.player == 2:
                self.player_2_groups.join(n.get_coords(), move)
    
    @property
    def winner(self) -> int:
        """
        Return a number corresponding to the winning player,
        or none if the game is not over.
        """
        if self.player_1_groups.connected(1, 2):
            return 1
        elif self.player_2_groups.connected(1, 2):
            return 2
        else:
            return -1

    def euclidean(self, x, y):
        return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))

    def traverse(self, start, goal, player=0):
        open_list = []
        closed = []

        start = self.board.get_space_from_coord(start)

        open_list.append(start)

        while len(open_list) > 0:
            space = min(open_list, key = lambda x: x.f)
            open_list.remove(space)
            
            # space = deepcopy(space) if self.display else copy(space)

            if not space.has_piece() or space.get_player() != player:
                continue

            closed.append(space.get_coords())

            if space.get_coords() == goal.get_coords():
                path = []
                current = space
                while current is not None:
                    path.append(current.get_coords())
                    current = current.parent
                return path[::-1]
            
            for neighbor in space.get_neighbors():
                if not neighbor.has_piece() or neighbor.get_player() != player or neighbor.get_coords() in closed:
                    continue
                
                g = space.f + 1
                h = self.euclidean(goal.get_coords(), neighbor.get_coords())
                f = h + g
                if neighbor.get_coords() in list(map(lambda x: x.get_coords(), open_list)):
                    if neighbor.g > g:
                        continue

                neighbor.parent = space
                neighbor.g = g
                neighbor.h = h
                neighbor.f = f
                open_list.append(neighbor)
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
        if len(list(filter(lambda x: x.player==1, itertools.chain(*self.board.content)))) < 4:
            return -1, False, []

        for (player, start) in enumerate(start_states):
            player = player + 1
            end_spaces = list(filter(lambda x: x.has_piece() and x.get_player() == player, map(self.board.get_space_from_coord, win_states[player-1])))
            for start_coord in start:
                for goal in end_spaces:
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
        print('To string')
        print(st)
        return st

    def game_from_string_representation(self, st):
        curr_player = int(st[0])
        board = Board.board_from_string_representation(st)
        game = Hex(board)
        game.player = curr_player
        return game

    def game_from_game(self, st, old_game):
        curr_player = int(st[0])
        # board = Board.board_from_string_representation(st)
        game = Hex(old_game.board.board_size, display=old_game.display, starting_player=old_game.starting_player)
        game.player = curr_player
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

class TreeNode:
    def __init__(self, coord):
        self.coord = coord
        self.player = -1
        self.neighbors = []
        self.parent = None
        self.g = 0
        self.h = 0
        self.f = 0
# if __name__ == '__main__':
    