import random
from collections import defaultdict
from copy import deepcopy, copy
from graphviz import Digraph
from numpy.random import choice
from .node2 import Node2

class MCTS2:
    def __init__(self, root, ANET, player, epsilon, game):
        self.children = defaultdict(dict)
        self.root = root
        self.ANET = ANET
        self.player = player
        self.epsilon = epsilon
        self.root_state = game
        # print(self.player)


    def select(self):
        node = self.root
        state = deepcopy(self.root_state)

        # stop if we find reach a leaf node
        while node.children:
            move, node = self.get_best_utc(node)
            # print(move)
            # print(state.get_legal_moves())
            # print(state)
            state.make_move(move)
            node.parent.nsa[move] += 1

            # if some child node has not been explored select it before expanding
            # other children
            if node.num_visits == 0:
                return node, state

        # if we reach a leaf node generate its children and return one of them
        # if the node is terminal, just return the terminal node
        if self.expand(node, state):
            node = random.choice(list(node.children.values()))
            state.make_move(node.move)
            node.parent.nsa[node.move] += 1

        return node, state

    def expand(self, node, state):
        if state.winner != -1:
            return False

        def get_new_player(state):
            return 1 if state.player == 2 else 2

        for move in state.get_legal_moves():
            node.add_child(Node2(move, node, get_new_player(state)), move)

        return True

        # return node.generate_child_states()

    def should_random_move(self):
        return random.uniform(0,1) < self.epsilon
    
    def rollout(self, state):
        moves = state.get_legal_moves() 
    
        while state.winner == -1:
            # if self.should_random_move():
            #     move = random.choice(moves)
            # else:
            move = self.get_move_default_policy(state.to_string_representation(), moves, state.LEGAL_MOVES)
            if state.cfg['game'] == 'hex' and state.cfg['hex']['display']:
                _, _, _, _, moves = state.make_move(move)
            else:
                _, _, _, moves = state.make_move(move)

        return state.winner

    def backprop(self, node, turn, outcome):
        reward = 1 if outcome == self.player else -1
        # print(outcome)
        while node is not None:
            node.num_visits += 1
            node.value += reward if outcome == node.player else 0
            # if not node.parent:
            #     print(f'Value: {node.value}, Outcome: {outcome}, self.player: {self.player}, reward: {reward}')
            node = node.parent
            # reward = -1 if reward == 1 else 1
        

    def get_move_default_policy(self, state, legal_moves, game_legal_moves):
        distribution = self.ANET.predict_val(state)
        best_action = None
        best_val = float('-inf')
        if 'cuda' in distribution.device.type:
            distribution = distribution.cpu().detach().numpy().tolist()
        else:
            distribution = distribution.detach().numpy().tolist()

        game_legal_moves_copy = game_legal_moves.copy()
        for i, move in enumerate(game_legal_moves_copy):
            if move not in legal_moves:
                distribution[i] = 0


        if sum(distribution) <= 0:
            return random.choice(legal_moves)

        distribution = [i/sum(distribution) for i in distribution]

        if self.should_random_move():
            ind = distribution.index(random.choices(population=distribution, weights=distribution)[0])
        else:
            ind = distribution.index(max(distribution))

        return game_legal_moves[ind]

    def get_best_utc(self, node):
        if node.player == self.player:
            return max(node.children.items(), key=lambda x: self.get_utc(x[1])) # , [self.get_utc(x[1]) for x in node.children.items()]
        return min(node.children.items(), key=lambda x: self.get_utc_neg(x[1]))

    def get_exploration_constant(self, node):
        return node.get_usa()

    def get_exploitation_constant(self, node):
        return node.get_q_value()

    def get_utc(self, node):
        return self.get_exploitation_constant(node) + self.get_exploration_constant(node)
    
    def get_utc_neg(self, node):
        return self.get_exploitation_constant(node) - self.get_exploration_constant(node)

    def prune(self, root, game):
        self.root = root
        self.root_state = deepcopy(game)

    def generate_nodes_and_edges(self, node, graph):
        graph.node(str(id(node)), str(node.num_visits))
        for key, val in node.children.items():
            graph.node(str(id(val)), str(val.num_visits))
            graph.edge(str(id(val.parent)), str(id(val)))
        
            self.generate_nodes_and_edges(val, graph)


    def show(self):
        graph = Digraph(format='pdf')
        self.generate_nodes_and_edges(self.root, graph)
        graph.render(f'mct.gv', view=True)

    def get_action_distribution(self):
        node = self.root
        game_state = self.root_state

        if len(node.children.keys()) == len(game_state.LEGAL_MOVES):
            l = list(map(lambda x: x[1].num_visits, list(node.children.items())))
            if sum(l) == 0:
                return l
            return [float(i)/sum(l) for i in l]
            # return l

        game = game_state.cfg['game']
        if game == 'nim':
            moves = list(map(lambda x: (x[0] - 1, x[1].num_visits), list(node.children.items())))
            # print(moves)
            dist = [0 for _ in range(len(game_state.LEGAL_MOVES))]
            
            for action in moves:
                dist[action[0]] = action[1]
        else:
            # [(0,0), (0,1), (0,2), (0,3), (0,4), (1,0), (1,1), (1,2), (1,3), (1,4), ]
            # pos = ind0 * sqrt(len) + ind1
            moves = list(map(lambda x: (x[0], x[1].parent.nsa[x[1].move]), list(node.children.items())))
            # print(moves)
            dist = [0 for _ in range(len(game_state.LEGAL_MOVES))]
            
            inc = len(dist)**(.5)
            for action in moves:
                dist[int(action[0][0]*inc + action[0][1])] = action[1]

        if sum(dist) == 0:
            return dist
        # return dist
        return [float(i)/sum(dist) for i in dist]