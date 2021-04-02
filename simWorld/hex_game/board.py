import itertools

from .space import Space
from .boarditerator import BoardIterator

class Board:
    def __init__(self, board_size):
        self.check_board_size(board_size)
        self.board_size = board_size
        self.content = self.init_board()


    def check_correct_variable_type(self, var, name, expected):
        if not isinstance(var, expected):
            raise Exception(f'Exception in creating board. {name} should be {expected}, got {type(var)}')

    def check_board_size(self, board_size):
        self.check_correct_variable_type(board_size, "Board_size", int)
        if board_size < 1:
            raise Exception(f'Board size {board_size} is illegal. Must be 1 or more.')

    def generate_board(self, size):
        return [[Space(False, (j, i)) for i in range(size)] for j in range(size)]
    

    def inside_board(self, coord):
        return 0 <= coord[0] < self.board_size and 0 <= coord[1] < self.board_size

    def add_neighbors(self, content):
        for r, row in enumerate(content):
            for c, space in enumerate(row):
                for (x, y) in [(r-1, c), (r+1, c), (r, c-1), (r, c+1), (r-1, c+1), (r+1, c-1)]:
                    if self.inside_board((x,y)):
                        space.add_neighbor(content[x][y])
                    # else:
                    #     space.add_neighbor(None)

    def init_board(self):
        content = self.generate_board(self.board_size)
        self.add_neighbors(content)

        return content
    
    def get_space_from_coord(self, coord):
        return self.content[coord[0]][coord[1]]

    def get_legal_moves(self):
        return list(map(lambda x: x.get_coords(), filter(lambda x: not x.has_piece(), itertools.chain(*self))))

    @staticmethod
    def board_from_string_representation(st):
        st = st[1:]
        board_size = int(len(st)**(1/2))
        board = Board(board_size)
        for i, space in enumerate(itertools.chain(*board)):
            player = -1 if st[i] == '0' else int(st[i]) - 1
            if player != -1:
                space.set_piece(True, player)
        return board

    def __repr__(self):
        return str(self.content)

    def __iter__(self):
        return BoardIterator(self)

    def to_string_representation(self):
        st = ''
        for space in self.content:
            if space.has_piece():
                st += str(space.player)
            else:
                st += '0'
        return st