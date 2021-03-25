import random
from .node import Node
from game import Graph
from graphviz import Digraph

class MCTS:
    def __init__(self, root):
        self.root = root

    
    def leaf_search(self, game, node=None):
        if node is None:
            node = self.root

        if not node.children:
            return node
        
        action, next_node = self.tree_policy(list(node.children.items()))
        # print(action)
        game.make_move(action)
        return self.leaf_search(game, next_node)

        
        
    def tree_policy(self, nodes):
        return random.choice(nodes)

    
    def rollout(self, game, node):
        # print('Rollout')
        # graph = Graph(game.board, True, .5)
        # graph.show_board()
        legal_moves = game.get_legal_moves()
        # print(legal_moves)
        curr_node = node
        while len(legal_moves):
            move = random.choice(legal_moves)
            game.make_move(move)
            # graph.show_board()
            child_node = Node(game.to_string_representation(), move, curr_node)
            # curr_node.add_child(child_node, move)
            curr_node = child_node

            winning_player, win, winning_path = game.is_win()
            if win:
                break
            legal_moves = game.get_legal_moves()
        # print('Rollout end')
        return curr_node
    
    def backprop(self, node):
        node.backprop += 1
        if node == self.root:
            return self.root
        return self.backprop(node.parent)



    def generate_nodes_and_edges(self, node, graph):
        graph.node(node.game_state_string, node.game_state_string)
        for key, val in node.children.items():
            graph.node(val.game_state_string, val.game_state_string)
            if val.parent is not None:
                graph.edge(val.parent.game_state_string, val.game_state_string)
        
            self.generate_nodes_and_edges(val, graph)


    def show(self):
        graph = Digraph(format='pdf')
        self.generate_nodes_and_edges(self.root, graph)
        graph.render(f'mct.gv', view=True)
