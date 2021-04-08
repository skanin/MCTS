import networkx as nx
import itertools
import matplotlib.pyplot as plt

class Graph():
    """
    A class to visualize a board of Peg solitaire.
    ...
    Attributes
    ----------
    board: Board
        Instance of a game board.
    pause: boolean
        Wether or not the graph should live update or not.
    update_freq: float
        The update frequency of the live update graph.
    """

    def __init__(self, board, pause, update_freq):
        """
        Initializes an instance of a graph to visualize a board of Peg solitaire.
        ...
        Parameters
        ----------
        board: Board
            Instance of a game board.
        pause: boolean
            Wether or not the graph should live update or not.
        update_freq: float
            The update frequency of the live update graph.
        """

        self.board = board # Set the game board

        self.G = nx.Graph() # Initialize graph
        self.edges = [] # Initialize edges to be empty

        for space in itertools.chain(*self.board): # Loop through all spaces in the game board
            for n in space.get_neighbors(): # Loop through this space's neighbors
                if n is not None:
                    self.edges.append((space, n)) # Add the (space, neighbor) relationship to edges
        
        self.pause = pause # Set pause
        self.init_graph() # Initialize graph
        self.positions = self.generate_positions() # Positions the nodes should have in the graph

        self.update_freq = update_freq # Set the update frequency

    def init_graph(self):
        if self.pause: # If the figure are to live update
            plt.ion() # ... set interactive mode on
        self.G.clear() # Clear the graph
        self.G.add_nodes_from(itertools.chain(*self.board)) # Add nodes to the graph
        self.G.add_edges_from(self.edges) # Add edges to the graph (egdes calculated at instantiation)
    
    def generate_positions(self):
        pos = {} # Init empty position dictionary to fill with positions
        for node in self.G: # Loop through the nodes i G
            (x, y) = node.coord # Get the node's coordinate
            # Set position of the node based on it's coordinate and board type (want triangle shape for triangle and diamond shape for diamond).
            # To be honest, theese values for position are trial and error. I just tweaked them until they fit.
            pos[node] = (100 + (-x)*10 + y*10, 100 + (-y)*10 + -x*10)
        return pos

    def show_board(self, winning_path=[]):
        """
        Shows the board as a graph
        """
        plt.clf() # Clear the firgure

        sizes = [300 for _ in range(len(self.G.nodes))]
        weights = [1 for _ in range(len(self.G.edges))]
        edge_color = ['black' for _ in range(len(self.G.edges))]

        if len(winning_path):
            for node in winning_path:
                sizes[list(map(lambda x: x.get_coords(), self.G.nodes)).index(node)] = 600
            
            for i in range(0, len(winning_path)-1):
                try:
                    index = list(map(lambda x: (x[0].get_coords(), x[1].get_coords()), self.G.edges)).index((winning_path[i], winning_path[i+1]))
                except:
                    index = list(map(lambda x: (x[0].get_coords(), x[1].get_coords()), self.G.edges)).index((winning_path[i+1], winning_path[i]))
                weights[index] = 5
                edge_color[index] = 'lightgreen'

        nx.draw(self.G, edge_color=edge_color, with_labels=False, pos=self.positions, node_size=sizes, node_color=self.node_color(), width=weights) # Draw the graph with different colors based on their empty status
        if self.pause: # If there should be a live update of the figure
            plt.pause(self.update_freq) # Pause for update_freq seconds
        else: # If not ...
            plt.show(block=True) # ... Stop the figure from closing

    
    def node_color(self):
        """
        Returns what colors different nodes should have.
        Empty nodes get black color and non-empty nodes get blue.
        """

        colors = [] # Empty colors list
        for node in itertools.chain(*self.board): # Iterate over the board
            color = ''
            if node.has_piece(): # If the space has a piece
                color = 'black' if node.get_player() == 1 else 'red'
            else:
                color = 'grey' # Else, set node color black 
            colors.append(color) # Append the color
        return colors # Return the colors.
