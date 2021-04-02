import random
from .node import Node
# from game import Graph
from graphviz import Digraph
from copy import deepcopy
from collections import defaultdict
class MCTS:
    def __init__(self, root, ANET):
        self.root = root
        self.ANET = ANET
        self.children = dict()
        self.N = defaultdict(int)
        self.Q = defaultdict(int)

    
    def leaf_search(self, game, node=None):
        if node is None:
            node = self.root

        # if node not in self.children:
        #     return node.find_random_child()
        
        if not node.children:
            return node


        if (self.is_leaf(node) or not node.children) and node != self.root:
            if not node.children:
                return node

            action, next_node = self.tree_policy(list(node.children.items()), game.player)
            self.make_move(action, game, next_node)
            print(next_node.num_times_action_taken)
            return next_node

        action, next_node = self.tree_policy(list(node.children.items()), game.player)
        # print(action)
        self.make_move(action, game, next_node)
        return self.leaf_search(game, next_node)

    
    def select(self, game, node):
        path = []
        while True:
            path.append(node)
            self.make_move(node.move, game, node)
            if not node.children:
                # node is either unexplored or terminal
                return path
            unexplored = list(filter(lambda x: x.num_visits == 0, node.children.values()))
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                self.make_move(n.move, game, n)
                return path
            node = self.tree_policy(list(node.children.items()), game.player)[1]  # descend a layer deeper

    def expand(self, node):
        node.generate_child_states()

    def is_leaf(self, node):
        return len(list(filter(lambda x: x.num_visits == 0, node.children.values())))

    def tree_policy(self, nodes, player):
        values = list(map(lambda x: [x[0], x[1].get_q_value(x[0])], nodes))
        # print(values)
        for i, val in enumerate(values):
            values[i][1] = val[1] + nodes[i][1].get_usa(val[0]) * -1 if player == 2 else 1
        # print(values)
        if player == 1:
            ind = values.index(max(values, key=lambda x: x[1]))
        else:
            ind = values.index(min(values, key=lambda x: x[1]))
        # print(ind)
        return nodes[ind]
        # return random.choice(nodes)

    def make_move(self, action, game, node):
        if node.move in game.get_legal_moves():
            node.add_action_taken(action)
            game.make_move(action)

    def get_move_default_policy(self, state, legal_moves):
        distribution = self.ANET.predict_val(state)
        best_action = None
        best_val = float('-inf')
        for i, val in enumerate(distribution.detach().numpy()):
            if best_val > val:
                continue
            if i+1 in list(map(lambda x: x[1], legal_moves)):
                best_val = val
                best_action = list(filter(lambda x: x[1] == i+1, legal_moves))[0]
        return best_action

    def rollout(self, game, node):
        # print('Rollout')
        # graph = Graph(game.board, True, .5)
        # graph.show_board()
        legal_moves = game.get_legal_moves()
        # print(legal_moves)
        curr_node = node
        curr_player = game.player

        while len(legal_moves):
            move = self.get_move_default_policy(game.to_string_representation(), legal_moves)
            _, done, _, legal_moves = game.make_move(move)
            # graph.show_board()
            child_node = Node(state=game.to_string_representation(), move=move, parent=curr_node, game_state=game)
            curr_node.add_child(child_node, move)
            # child_node.num_visits += 1
            # child_node.nsa[id(curr_node)][id(child_node)] += 1
            curr_node = child_node
            # print('Node')
            # print(node.children)
            # print('Curr node')
            # print(curr_node.children)

            if done:
                break
            legal_moves = game.get_legal_moves()
        # print('Rollout end')

        player = game.get_winner()
        if player == curr_player:
            reward = 1
        else:
            reward = -1

        return curr_node, reward
    
    # def backprop(self, node, reward):
    #     node.backprop += 1
    #     node.value += reward
    #     if node.parent == None:
    #         return node
    #     return self.backprop(node.parent, reward)

    def backprop(self, path, reward):
        for node in reversed(path):
            node.value += reward
            node.num_visits += 1


    def generate_nodes_and_edges(self, node, graph):
        graph.node(str(id(node)), node.game_state_string)
        for key, val in node.children.items():
            graph.node(str(id(val)), val.game_state_string)
            graph.edge(str(id(val.parent)), str(id(val)))
        
            self.generate_nodes_and_edges(val, graph)


    def show(self):
        graph = Digraph(format='pdf')
        self.generate_nodes_and_edges(self.root, graph)
        graph.render(f'mct.gv', view=True)

    def prune(self, node):
        self.root = node