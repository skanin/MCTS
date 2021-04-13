import math
import yaml
from BasicClientActorAbs import BasicClientActorAbs
from simWorld.hex_game.Hex import Hex
from policy import Policy

class BasicClientActor(BasicClientActorAbs):

    def __init__(self, IP_address=None, verbose=True):
        self.cfg = yaml.safe_load(open('../config.yaml', 'r'))
        self.series_id = -1
        BasicClientActorAbs.__init__(self, IP_address, verbose=verbose)
        self.game = None
        self.actor = None
        self.game_size = 0

    def handle_get_action(self, state):
        """
        Here you will use the neural net that you trained using MCTS to select a move for your actor on the current board.
        Remember to use the correct player_number for YOUR actor! The default action is to select a random empty cell
        on the board. This should be modified.
        :param state: The current board in the form (1 or 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), where
        1 or 2 indicates the number of the current player.  If you are player 2 in the current series, for example,
        then you will see a 2 here throughout the entire series, whereas player 1 will see a 1.
        :return: Your actor's selected action as a tuple (row, column)
        """

        # This is an example player who picks random moves. REMOVE THIS WHEN YOU ADD YOUR OWN CODE !!
        self.game = self.game.game_from_game(''.join(map(str, state)), self.game) 
        next_move = self.actor.get_move(self.game) # tuple(self.pick_random_free_cell(
            # state, size=int(math.sqrt(len(state)-1))))
        #############################
        #
        #
        # YOUR CODE HERE
        #
        # next_move = ???
        ##############################
        return next_move

    def handle_series_start(self, unique_id, series_id, player_map, num_games, game_params):
        """
        Set the player_number of our actor, so that we can tell our MCTS which actor we are.
        :param unique_id - integer identifier for the player within the whole tournament database
        :param series_id - (1 or 2) indicating which player this will be for the ENTIRE series
        :param player_map - a list of tuples: (unique-id series-id) for all players in a series
        :param num_games - number of games to be played in the series
        :param game_params - important game parameters.  For Hex = list with one item = board size (e.g. 5)
        :return

        """
        self.series_id = series_id
        self.game_size = game_params[0]
        print(game_params)
        if self.game_size == 4:
            self.actor = Policy('hex-4x4-at-1000-episodes', 'hex-4x4-at-1000-episodes', self.series_id)
        elif self.game_size == 5:
            self.actor = Policy('hex-5x5-at-1000-of-1000-episodes-with-100-simulations-04-04-2021-20-00', 'hex-5x5-at-1000-of-1000-episodes-with-100-simulations-04-04-2021-20-00', self.series_id)
        elif self.game_size == 6:
            
            self.actor = Policy('Hex-6x6-at-750-of-1000-episodes-13-04-2021-07-11', 'Hex-6x6-at-750-of-1000-episodes-13-04-2021-07-11', self.series_id, self.cfg)
        #############################
        #
        #
        # YOUR CODE (if you have anything else) HERE
        #
        #
        ##############################

    def handle_game_start(self, start_player):
        """
        :param start_player: The starting player number (1 or 2) for this particular game.
        :return
        """
        self.starting_player = start_player
        self.game = Hex(board_size = self.game_size, display = False, starting_player = start_player)
        
        #############################
        #
        #
        # YOUR CODE (if you have anything else) HERE
        #
        #
        ##############################

    def handle_game_over(self, winner, end_state):
        """
        Here you can decide how to handle what happens when a game finishes. The default action is to print the winner and
        the end state.
        :param winner: Winner ID (1 or 2)
        :param end_state: Final state of the board.
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        ##############################
        print("Game over, these are the stats:")
        print('Winner: ' + str(winner))
        print('End state: ' + str(end_state))

    def handle_series_over(self, stats):
        """
        Here you can handle the series end in any way you want; the initial handling just prints the stats.
        :param stats: The actor statistics for a series = list of tuples [(unique_id, series_id, wins, losses)...]
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("Series ended, these are the stats:")
        print(str(stats))

    def handle_tournament_over(self, score):
        """
        Here you can decide to do something when a tournament ends. The default action is to print the received score.
        :param score: The actor score for the tournament
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("Tournament over. Your score was: " + str(score))

    def handle_illegal_action(self, state, illegal_action):
        """
        Here you can handle what happens if you get an illegal action message. The default is to print the state and the
        illegal action.
        :param state: The state
        :param action: The illegal action
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("An illegal action was attempted:")
        print('State: ' + str(state))
        print('Action: ' + str(illegal_action))


if __name__ == '__main__':
    bsa = BasicClientActor(verbose=True)
    bsa.connect_to_server()
