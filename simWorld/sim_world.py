class SimWorld:
    def __init__(self, game):
        self.game = game
    

    def get_game_state_as_string(self):
        return self.game.to_string_representation()

    def is_winning_state(self):
        return self.game.is_win()

    def make_move(self, move):
        return self.game.make_move(move)    