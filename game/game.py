import math
import itertools
from .space import Space
from copy import deepcopy, copy
from .board import Board

class Game:
    def __init__(self, board):
        self.board = board
        self.player = 0

    
    def get_legal_moves(self):
        return list(map(lambda x: (self.player, x), self.board.get_legal_moves()))

    def is_legal_move(self, move):
        return move in self.get_legal_moves()

    def make_move(self, move):
        if not self.is_legal_move(move):
            raise Exception('Illegal move')

        space = self.board.get_space_from_coord(move[1])
        space.set_piece(True, move[0])

        self.player = int(not self.player)


    def euclidean(self, x, y):
        return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))

    def traverse2(self, start, goal, player=0):
        open_list = []
        closed = []

        start = self.board.get_space_from_coord(start)

        open_list.append(start)

        while len(open_list) > 0:
            space = min(open_list, key = lambda x: x.f)
            open_list.remove(space)
            
            space = deepcopy(space)

            if not space.has_piece() or space.get_player() != player:
                continue

            closed.append(space.get_coords())

            if space.get_coords() == goal.get_coords():
                path = []
                current = space
                while current is not None:
                    path.append(current)
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
        

    def is_win(self, ):
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
            end_spaces = list(filter(lambda x: x.has_piece() and x.get_player() == player, map(self.board.get_space_from_coord, win_states[player])))
            for start_coord in start:
                for goal in end_spaces:
                    path = self.traverse2(start_coord, goal, player=player)
                    if len(path) > 0:
                        # start = False
                        # end = False
                        # for space in path:
                        #     if space.get_coords() in start_states[player]:
                        #         start = True
                        #     elif space.get_coords() in win_states[player]:
                        #         end = True
                        #     if start and end:
                        paths.append((player + 1, True, path))
        if len(paths) == 0:
            return -1, False, []
        
        return sorted(paths, key=lambda x: len(x[-1]))[0]

        # for (player, stack, goal) in zip([0, 1], start_states, win_states):
        #     # print(f'Player: {player}, Stack: {stack}, goal: {goal}')
        #     seen = set(stack)

        #     while stack:
        #         (x, y) = stack.pop()
        #         space = self.board.content[x][y]
        #         if (x, y) in goal and space.has_piece():
        #             print('HEYHEYHEYHEYHEYHEYHEYHEYHEYHEY')
        #             return player
                
        #         for n in space.get_neighbors():
        #             if n is None:
        #                 continue
        #             if n.get_player() != player:
        #                 continue
        #             if n not in seen:
        #                 print('Heyooo')
        #                 seen.add(n)
        #                 stack.append(n.get_coords())
        #         print(stack)
                
                
    def to_string_representation(self):
        st = str(self.player + 1)
        for space in itertools.chain(*self.board):
            st += str(space.player + 1) if space.has_piece() else '0'
        
        return st

    @staticmethod
    def game_from_string_representation(st):
        curr_player = int(st[0]) - 1
        board = Board.board_from_string_representation(st)
        game = Game(board)
        game.player = curr_player
        return game

    @staticmethod
    def generate_child_states(game):
        legal_moves = game.get_legal_moves()
        game_copy = copy(game)
        child_states = []
        for move in legal_moves:
            game_copy.make_move(move)
            child_states.append((move, game_copy.to_string_representation()))
            game_copy = deepcopy(game)
        return child_states