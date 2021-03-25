from copy import deepcopy

class Space:
    def __init__(self, piece=False, coord=(-1, -1)):
        self.piece = piece
        self.coord = coord
        self.player = -1
        self.neighbors = []
        self.parent = None
        self.g = 0
        self.h = 0
        self.f = 0
    
    def has_piece(self):
        return self.piece

    def get_player(self):
        return self.player

    def set_player(self, player):
        self.player = player

    def add_neighbor(self, space):
        self.neighbors.append(space)
    
    def get_neighbors(self):
        return sorted(self.neighbors, key=lambda x: x.get_coords()[1])

    def set_piece(self, piece, player):
        self.piece = piece
        self.player = player
    
    def get_coords(self):
        return self.coord
    
    def get_state(self):
        if not self.piece:
            return (0,0)
        if self.player == 0:
            return (1, 0)
        return (0, 1)

    @staticmethod
    def space_from_space(space):
        s = Space()
        s.piece = space.piece
        s.coord = space.coord
        s.player = space.player
        s.neighbors = []

        s.parent = Space.space_from_space(space.parent) if space.parent is not None else None
        s.g = space.g
        s.h = space.h
        s.f = space.f
        return s
    # def __deepcopy__():
    #     s = Space(self.piece, (self.coord[0], self.coord[1]))
    #     s.neighbors = self.neighbors.copy()
    #     s.player = self.player
    #     s.parent = deepcopy(self.parent) if self.parent is not None else None
    #     s.f = self.f
    #     s.g = self.g
    #     s.h = self.h
    #     return s

    # def __repr__(self):
    #     # st = "Coord: " + str(self.coord) + ', State: ' + str(self.get_state())
    #     # return st
    #     return str(self.coord)

