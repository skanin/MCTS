import random
from collections import defaultdict
from copy import deepcopy, copy
from graphviz import Digraph
from numpy.random import choice
from .node import Node
import math
import time

class MCTS:
    def __init__(self, root, ANET, player, epsilon, epsilon_decay, target_epsilon, game, c):
        self.children = defaultdict(dict)
        self.root = root
        self.ANET = ANET
        self.player = player
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.target_epsilon = target_epsilon
        self.graph = defaultdict(list)
        self.states = dict()
        self.nodes = dict()
        self.root_state = game
        self.c = c
        self.Q = defaultdict(int)
        self.nsa = defaultdict(int)


    def add_to_state_dict(self, node, state):
        if state not in self.states:
            self.states[state] = node

    def add_to_nodes(self, node):
        if node.state not in self.nodes:
            self.nodes[node.state] = node 

    def select(self):
        node = self.root
        state = deepcopy(self.root_state)

        parent = node
        path = [node]
        actions = []

        self.add_to_nodes(node)

        while node.children:
            node = self.get_best_utc(node)

            move = node.get_move(parent)
            parent = node
            state.make_move(move, mcts=True)

            path.append(node)
            actions.append(move)
            
            self.add_to_nodes(node)
            
            if node.num_visits == 0:
                return node, state, path, actions

        if self.expand(node, state):
            parent = node
            node = random.choice(list(parent.children.values()))
            move = node.get_move(parent)
            state.make_move(move, mcts=True)
            path.append(node)
            actions.append(move)
            self.add_to_nodes(node)
        return node, state, path, actions

    def get_node_from_state(self, state):
        return self.states[state]

    def expand(self, node, state):
        if state.winner != -1:
            return False

        new_player = 1 if state.player == 2 else 2
        tmp_state = deepcopy(state)
        for move in state.get_legal_moves():
            tmp_state.make_move(move, mcts=True)
            tmp_state_rep = tmp_state.to_string_representation()

            if tmp_state_rep in self.nodes:
                child = self.nodes[tmp_state_rep]
                child.add_parent(node)
                child.add_move(node, move)
            else: 
                child = Node(move, node, new_player, tmp_state_rep)
            
            self.add_to_nodes(child)

            tmp_state = deepcopy(state)
       
        return True

    def should_random_move(self):
        return random.uniform(0,1) < self.epsilon
    
    def rollout(self, state, path):
        moves = state.get_legal_moves() 
    
        while state.winner == -1:
            move = self.get_move_default_policy(state.to_numpy(), moves, state.LEGAL_MOVES)
            if state.cfg['game'] == 'hex' and state.cfg['hex']['display']:
                _, _, _, _, moves = state.make_move(move, mcts=True)
            else:
                _, _, _, moves = state.make_move(move, mcts=True)

        return state.winner

    def backprop(self, node, turn, outcome, path, actions):
        reward =  1 if outcome == self.player else -1 
        for i, node in enumerate(path):
            node.num_visits += 1
            node.value += reward
            if i > 0:
                node.add_move(path[i-1], actions[i-1])
                path[i-1].update_Q(actions[i-1])
            
    def get_node_parent(self, node):
        for parent, children in self.graph.items():
            if node in children:
                return self.states[parent]

        return None

    def get_move_default_policy(self, state, legal_moves, game_legal_moves):
        distribution = self.ANET.predict_val(state).numpy().tolist()
        for i, move in enumerate(game_legal_moves):
            if move not in legal_moves:
                distribution[i] = 0


        if sum(distribution) <= 0:
            return random.choice(legal_moves)

        distribution = [i/sum(distribution) for i in distribution]

        if self.should_random_move():
            self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon > self.target_epsilon else self.epsilon
            ind = distribution.index(random.choices(population=distribution, weights=distribution)[0])
        else:
            ind = distribution.index(max(distribution))
            
        return game_legal_moves[ind]

    def get_node_children(self, node):
        return self.graph[hash(node.state)]

    def get_best_utc(self, node):
        if node.player == self.player:
            return max(list(node.children.values()), key=lambda x: self.get_utc(x, node))
        return min(list(node.children.values()), key=lambda x: self.get_utc_neg(x, node))

    def get_exploration_constant(self, node, parent):
        return float('inf') if node.nsa[(parent, node.moves[parent])] == 0 else self.c * math.sqrt(math.log(parent.num_visits)/node.nsa[(parent, node.moves[parent])])

    def get_exploitation_constant(self, node, parent):
        return 0 if parent.nsa[node.move] == 0 else  node.value / parent.nsa[node.move]

    def get_utc(self, node, parent):
        return parent.Q[node.moves[parent]] + self.get_exploration_constant(node, parent)

    def get_utc_neg(self, node, parent):
        return parent.Q[node.moves[parent]] - self.get_exploration_constant(node, parent)

    def prune(self, root, game):
        self.root = root
        self.root_state = deepcopy(game)
        

    def generate_nodes_and_edges(self, node, graph):
        graph.node(str(hash(node.state)), str(node.state) + ' : ' + str(node.num_visits))
        for child in node.children.values():
            graph.node(str(hash(child.state)), str(node.state) + ' : ' + str(node.num_visits))
            graph.edge(str(hash(node.state)), str(hash(child.state)), label = f'{child.move} : {child.nsa[(node, child.get_move(node))]}')

            self.generate_nodes_and_edges(child, graph)

    def show(self):
        graph = Digraph(format='pdf', strict=True)
        self.generate_nodes_and_edges(self.root, graph)
        graph.render(f'mct.gv', view=True)


    def get_action_distribution(self):
        node = self.root
        game_state = self.root_state

        if len(node.children) == len(game_state.LEGAL_MOVES):
            
            l = list(map(lambda x: x.nsa[(node, x.get_move(node))], list(node.children.values())))
            if sum(l) == 0:
                return l
            return [float(i)/sum(l) for i in l]

        game = game_state.cfg['game']
        if game == 'nim':
            moves = list(map(lambda x: (x.get_move(node) - 1,  x.nsa[(node, x.get_move(node))]), list(node.children.values())))
            dist = [0 for _ in range(len(game_state.LEGAL_MOVES))]
            
            for action in moves:
                dist[action[0]] = action[1]
        else:
            moves = list(map(lambda x: (x.get_move(node), x.nsa[(node, x.get_move(node))]), list(node.children.values())))
            dist = [0 for _ in range(len(game_state.LEGAL_MOVES))]
            
            inc = len(dist)**(.5)
            for action in moves:
                dist[int(action[0][0]*inc + action[0][1])] = action[1]

        if sum(dist) == 0:
            return dist
        return [float(i)/sum(dist) for i in dist]