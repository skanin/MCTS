import random
from .node import Node
# from game import Graph
from graphviz import Digraph
from copy import deepcopy
from collections import defaultdict
class MCTS:
    def __init__(self, root, ANET, player):
        self.root = root
        self.ANET = ANET
        self.children = dict()
        self.N = defaultdict(int)
        self.Q = defaultdict(int)
        self.player = player
        self.epsilon = .5

    
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

    
    def select1(self, game, node):
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

    def select2(self, game, node) -> tuple:
        """
        Select a node in the tree to preform a single simulation from.
        """
        # stop if we find reach a leaf node
        path = []
        while len(node.children) != 0:
            path.append(node)
            # descend to the maximum value node, break ties at random
            children = node.children.values()
            max_value = max(children, key=lambda n: n.get_value).value
            max_nodes = [n for n in node.children.values()
                         if n.get_value == max_value]
            node = random.choice(max_nodes)
            self.make_move(node.move, game, node)

            # if some child node has not been explored select it before expanding
            # other children
            if node.num_visits == 0:
                path.append(node)
                return path

        # if we reach a leaf node generate its children and return one of them
        # if the node is terminal, just return the terminal node
        if self.expand(node):
            node = random.choice(list(node.children.values()))
            self.make_move(node.move, game, node)
            path.append(node)

        return path

    def select(self, game, node):
        path = []
        path.append(node)
        while node.is_fully_expanded():
            self.make_move(node.move, game, node)
            node = self.tree_policy(node)[1]
            path.append(node)

        node = self.pick_unvisited_child(node) or node
        path.append(node)
        self.make_move(node.move, game, node)
        # self.make_move(node.move, game, node)
        # node = self.tree_policy(list(node.children.items()), game)[1]
        # path.append(node)
        # self.make_move(node.move, game, node)
        return path

    def pick_unvisited_child(self, node):
        children = list(filter(lambda x: x.num_visits == 0, list(node.children.values())))
        return random.choice(children) if children else False


    def expand(self, node):
        return node.generate_child_states()

    def is_leaf(self, node):
        return len(list(filter(lambda x: x.num_visits == 0, node.children.values())))

    def tree_policy(self, node):
        # values = list(map(lambda x: [x[0], x[1].get_q_value(x[0])], nodes))
        # # print(values)
        # for i, val in enumerate(values):
        #     values[i][1] = val[1] + (nodes[i][1].get_usa(val[0]) * -1 if game.player != game.starting_player else 1)
        # # print(values)
        # if game.player == game.starting_player:
        #     ind = values.index(max(values, key=lambda x: x[1]))
        # else:
        #     # print('Values')
        #     # print(values)
        #     ind = values.index(min(values, key=lambda x: x[1]))
        #     # print('Nodes')
        #     # print(nodes[ind])
        # # print(ind)
        # return nodes[ind]
        # return random.choice(nodes)
        val = self.get_best_utc(node)
        print(val)
        return val

    def get_best_utc(self, node):
        if node.game_state.player == self.player:
            return max(node.children.items(), key=lambda x: self.get_utc(x[1]), default=node)
        return min(node.children.items(), key=lambda x: self.get_utc_neg(x[1]), default=node)

    def get_exploration_constant(self, node):
        return node.get_usa()

    def get_exploitation_constant(self, node):
        return node.get_q_value()

    def get_utc(self, node):
        return self.get_exploitation_constant(node) + self.get_exploration_constant(node)
    
    def get_utc_neg(self, node):
        return self.get_exploitation_constant(node) - self.get_exploration_constant(node)

    def make_move(self, action, game, node):
        if node.move in game.get_legal_moves():
            node.add_action_taken(action)
            game.make_move(action)

    def get_move_default_policy(self, state, legal_moves, game_legal_moves):
        distribution = self.ANET.predict_val(state)
        best_action = None
        best_val = float('-inf')
        if 'cuda' in distribution.device.type:
            distribution = distribution.cpu().detach().numpy()
        else:
            distribution = distribution.detach().numpy()
        for i, val in enumerate(distribution):
            move = game_legal_moves[i]
            if best_val > val or move not in legal_moves:
                continue

            # if i+1 in list(map(lambda x: x[1], legal_moves)):
            best_val = val
            best_action = move # list(filter(lambda x: x[1] == i+1, legal_moves))[0]
        return best_action


    def should_random_move(self):
        return self.epsilon < random.uniform(0,1)

    def rollout(self, game, node):
        # print('Rollout')
        # graph = Graph(game.board, True, .5)
        # graph.show_board()
        legal_moves = game.get_legal_moves()
        # print(legal_moves)
        curr_node = node
        curr_player = game.player
        game_type = game.cfg['game']
        
        while len(legal_moves):
            if self.should_random_move():
                move = random.choice(legal_moves)
            else:
                move = self.get_move_default_policy(game.to_string_representation(), legal_moves, game.LEGAL_MOVES)
            
            if game_type == 'hex':
                if game.display:
                    game_state, _, done, _, legal_moves = game.make_move(move)
                else:
                    game_state, done, _, legal_moves = game.make_move(move)
            else:
                game_state, done, _, legal_moves = game.make_move(move)
            # graph.show_board()
            
            # child_node = Node(state=game_state, move=move, parent=curr_node, game_state=deepcopy(game))
            # curr_node.add_child(child_node, move)
            # if curr_node.game_state_string == child_node.game_state_string:
            #     print(curr_node.game_state_string)
            #     print(child_node.move)
            #     print(child_node.game_state_string)
            # child_node.num_visits += 1
            # child_node.nsa[id(curr_node)][id(child_node)] += 1
            
            # curr_node = child_node
            
            # print('Node')
            # print(node.children)
            # print('Curr node')
            # print(curr_node.children)

            if done:
                break
        # print('Rollout end')

        player = game.get_winner()
        # print(player)
        if player == self.player:
            print(game.to_string_representation())
            reward = 1
        else:
            reward = -1

        # print(f'Player: {player}, reward: {reward}')
        return node, reward
    
    # def backprop(self, node, reward):
    #     node.backprop += 1
    #     node.value += reward
    #     if node.parent == None:
    #         return node
    #     return self.backprop(node.parent, reward)

    # def backprop(self, path, reward):
    #     for node in reversed(path):
    #         # if player == 1:
    #         #     node.value += reward if reward > 0 else 0
    #         # else:
    #         #     node.value -= reward if reward < 0 else 0
    #         node.value += reward
    #         # if node.game_state.player == node.game_state.starting_player and reward > 0:
    #         #     print(node.value)
    #         #     node.value += reward
    #         # elif node.game_state.player != node.game_state.starting_player and reward < 0:
                
    #             # node.value += reward
    #         node.num_visits += 1
    #     return True

    def backprop(self, node, reward):
        # print(type(node))
        if node:
            node.value += reward # if (node.game_state.player == self.player and reward > 0) or (node.game_state.player != self.player and reward < 0) else 0
            node.num_visits += 1
            self.backprop(node.parent, reward)
        return           

    def generate_nodes_and_edges(self, node, graph):
        graph.node(str(id(node)), str(node.game_state_string))
        for key, val in node.children.items():
            graph.node(str(id(val)), str(val.game_state_string))
            graph.edge(str(id(val.parent)), str(id(val)), label = str(self.get_utc(val) if val.game_state.player == 1 else self.get_utc_neg(val)))
        
            self.generate_nodes_and_edges(val, graph)


    def show(self):
        graph = Digraph(format='pdf')
        self.generate_nodes_and_edges(self.root, graph)
        graph.render(f'mct.gv', view=True)

    def prune(self, node):
        self.root = node