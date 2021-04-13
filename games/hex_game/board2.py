import itertools

# from .space import Space
# from .boarditerator import BoardIterator

class Board2:
    def __init__(self, board_size):
        self.check_board_size(board_size)
        self.board_size = board_size
        self.content = self.init_board()
        self.LEGAL_MOVES = list(itertools.chain(*[[(j, i) for i in range(self.board_size)] for j in range(self.board_size)]))
       

    def check_correct_variable_type(self, var, name, expected):
        if not isinstance(var, expected):
            raise Exception(f'Exception in creating board. {name} should be {expected}, got {type(var)}')

    def check_board_size(self, board_size):
        self.check_correct_variable_type(board_size, "Board_size", int)
        if board_size < 1:
            raise Exception(f'Board size {board_size} is illegal. Must be 1 or more.')

    def generate_board(self, size):
        return [[0 for i in range(size)] for j in range(size)]
    

    def inside_board(self, coord):
        return 0 <= coord[0] < self.board_size and 0 <= coord[1] < self.board_size

    def get_neighbors(self, coord):
        r = coord[0]
        c = coord[1]
        n = []
        for (x,y) in [(r-1, c), (r+1, c), (r, c-1), (r, c+1), (r-1, c+1), (r+1, c-1)]:
            if self.inside_board((x,y)):
                n.append((x,y))
        return n

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
        # self.add_neighbors(content)

        return content
    
    def get_space_from_coord(self, coord):
        return self.content[coord[0]][coord[1]]

    def get_legal_moves(self):
        moves = []
        for r, row in enumerate(self.content):
            for c, space in enumerate(row):
                if space == 0:
                    moves.append((r, c))
        return moves
        # return list(map(lambda x: x.get_coords(), filter(lambda x: not x.has_piece(), itertools.chain(*self))))

    @staticmethod
    def board_from_string_representation(st):
        st = st[1:]
        board_size = int(len(st)**(1/2))
        board = Board(board_size)
        for i, space in enumerate(itertools.chain(*board)):
            player = -1 if st[i] == '0' else int(st[i])
            if player != -1:
                space.set_piece(True, player)
        return board

    def __repr__(self):
        return str(self.content)

    # def __iter__(self):
    #     return BoardIterator(self)

    def to_string_representation(self):
        return ''.join(map(str, itertools.chain(*self.content)))

    def __getitem__(self, coord):
        return self.content[coord[0]][coord[1]]

if __name__ == '__main__':
    board = Board2(5)
    print(board.get_neighbors((0,1)))
    print(board[(0,0)])
    print(board.to_string_representation())