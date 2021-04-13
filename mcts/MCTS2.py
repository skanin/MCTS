import random
from collections import defaultdict
from copy import deepcopy, copy
from graphviz import Digraph
from numpy.random import choice
from .node2 import Node2
import math
import time

class MCTS2:
    def __init__(self, root, ANET, player, epsilon, epsilon_decay, target_epsilon, game):
        self.children = defaultdict(dict)
        self.root = root
        self.ANET = ANET
        self.player = player
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.target_epsilon = target_epsilon
        self.graph = defaultdict(list)
        self.states = dict()
        self.root_state = game
        self.c = 1
        # print(self.player)
        self.Q = defaultdict(int)
        self.nsa = defaultdict(int)


    def add_to_state_dict(self, node, state):
        if state not in self.states:
            self.states[state] = node

    def select(self):
        node = self.root
        state = deepcopy(self.root_state)

        

        parent = node
        path = [node]
        actions = []

        self.add_to_state_dict(node, hash(state.to_string_representation()))
        
        # stop if we find reach a leaf node
        # while node.children:
        while len(self.graph[hash(node.state)]):
            # best_utc = self.get_best_utc2(node)
            # children = [child for child in node.children.values() if self.get_utc(child) == best_utc and node.player == self.player or self.get_utc_neg(child) == best_utc and child.player != self.player]
            # print(children)
            
            node = self.get_best_utc(node)

            # print(move)
            # print(state.get_legal_moves())
            # print(state)
            state.make_move(node.move)

            path.append(node)
            actions.append(node.move)

            self.add_to_state_dict(node, hash(state.to_string_representation()))

            # if some child node has not been explored select it before expanding
            # other children
            if node.num_visits == 0:
                return node, state, path, actions

        # if we reach a leaf node generate its children and return one of them
        # if the node is terminal, just return the terminal node
        if self.expand(node, state):
            # node = random.choice(list(node.children.values()))
            node = random.choice(self.graph[hash(state.to_string_representation())])
            state.make_move(node.move)
            path.append(node)
            actions.append(node.move)
            self.add_to_state_dict(node, hash(state.to_string_representation()))
        return node, state, path, actions

    def get_node_from_state(self, state):
        return self.states[state]

    def expand(self, node, state):
        if state.winner != -1:
            return False

        new_player = 1 if state.player == 2 else 2
        state_rep = hash(state.to_string_representation())
        tmp_state = deepcopy(state)
        # print('Start-------------------')
        for move in state.get_legal_moves():
            tmp_state.make_move(move)
            tmp_state_rep = tmp_state.to_string_representation()
            tmp_state_rep_hash = hash(tmp_state_rep)

            possible_children = list(filter(lambda x: x.move == move, self.graph[state_rep]))
            # print(state_rep)
            # print(state.to_string_representation())
            # print(self.graph[state_rep])
            if tmp_state_rep_hash in self.states and len(possible_children): # If seen state before, it has a parent
                child = possible_children[0]# self.states[tmp_state_rep] # Get the node from states
            else: # If not 
                child = Node2(move, node, new_player, tmp_state_rep) # Create new node.
            
            # print(f'State before: {state.to_string_representation()}')
            # print(f'Move: {move}')
            # print(f'State after: {tmp_state_rep}')
            # print(f'Child state: {child.state}')

            self.add_to_state_dict(child, tmp_state_rep_hash)
            if child not in self.graph[state_rep]:
                self.graph[state_rep].append(child)

            tmp_state = deepcopy(state)
        # print('End-------------------')
            # node.add_child(Node2(move, node, get_new_player(state)), move)

        return True

        # return node.generate_child_states()

    def should_random_move(self):
        return random.uniform(0,1) < self.epsilon
    
    def rollout(self, state, path):
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

            # path.append(Node2(move, None, state.player, state.to_string_representation()))

        return state.winner

    def backprop(self, node, turn, outcome, path, actions):
        reward =  1 if outcome == self.player else -1 # (1 + math.e**(-len(path)/10) if outcome == self.player else -1 - math.e**(-len(path)/10))
        # print(outcome)
        # while node is not None:
        #     node.num_visits += 1
        #     node.value += reward # if outcome == node.player else 0
        #     # if not node.parent:
        #     #     print(f'Value: {node.value}, Outcome: {outcome}, self.player: {self.player}, reward: {reward}')
            
        #     # node = node.parent
        #     node = self.get_node_parent(node) # if node.parent is not None else None

        for i, node in enumerate(path):
            node.num_visits += 1
            node.value += reward # if outcome == node.player else 0
            if i > 0:
                 self.nsa[(hash(path[i-1].state), node.move)] += 1
                 self.Q[(hash(path[i-1].state), node.move)] = node.value / node.num_visits # self.nsa[(hash(path[i-1].state), node.move)]
            # if i == len(path) - 1:
            #     self.nsa[(hash(path[i-1].state), node.move)] += 1
            #     self.Q[(hash(path[i-1].state), node.move)] = node.value / self.nsa[(hash(node.state), node.move)]
            # else:
            #     self.nsa[(hash(node.state), path[i+1].move)] += 1
            #     self.Q[(hash(node.state), path[i+1].move)] = node.value / self.nsa[(hash(node.state), path[i+1].move)]
            # else:
                # self.nsa[(hash(node.state), node.move)] += 1
                # self.Q[(hash(node.state), None)] = node.value / self.nsa[(hash(node.state), node.move)]

        # for i, node in enumerate(path[::-1]):
        #     node.num_visits += 1
        #     node.value += reward if outcome == node.player else 0
        #     self.nsa[(hash(node.state), node.move)] += 1
        #     self.Q[(hash(node.state), node.move)] = node.value / self.nsa[(hash(node.state), node.move)] 

            # reward = -1 if reward == 1 else 1
        

    def get_node_parent(self, node):
        # parents = list(self.graph.keys())
        # children = list(self.graph.values())
        # child_list = list(filter(lambda x: node in x, children))[0]
        # print(child_list)
        # parent = parents.index(child_list)
        for parent, children in self.graph.items():
            if node in children:
                return self.states[parent]

        return None

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
            self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon > self.target_epsilon else self.epsilon
            ind = distribution.index(random.choices(population=distribution, weights=distribution)[0])
        else:
            ind = distribution.index(max(distribution))
            
        return game_legal_moves[ind]

    # def get_best_utc(self, node):
    #     if node.player == self.player:
    #         return max(node.children.items(), key=lambda x: self.get_utc(x[1])) # , [self.get_utc(x[1]) for x in node.children.items()]
    #     return min(node.children.items(), key=lambda x: self.get_utc_neg(x[1]))

    def get_node_children(self, node):
        # print(self.graph[hash(node.state)])
        return self.graph[hash(node.state)] # self.graph[list(self.states.keys())[list(self.states.values()).index(node)]]


    def get_best_utc(self, node):
        if node.player == self.player:
            return max(self.get_node_children(node), key=lambda x: self.get_utc(x, node)) # , [self.get_utc(x[1]) for x in node.children.items()]
        return min(self.get_node_children(node), key=lambda x: self.get_utc_neg(x, node))

    def get_best_utc2(self, node):
        if node.player == self.player:
            return max(list(map(lambda x: self.get_utc(x[1]), node.children.items())))

        print(list(map(lambda x: self.get_utc_neg(x), list(node.children.values()))))
        return min(list(map(lambda x: self.get_utc_neg(x), node.children.values())))

    # def get_exploration_constant(self, node):
    #     return node.get_usa()

    # def get_exploitation_constant(self, node):
    #     return node.get_q_value()

    def get_exploration_constant(self, node, parent):
        # return float('inf') if self.nsa[(hash(parent.state), node.move)] == 0 else self.c * math.sqrt(math.log(node.num_visits)/(1 + self.nsa[(hash(parent.state), node.move)]))
        # return float('inf') if node.num_visits == 0 else self.c * math.sqrt(math.log(self.nsa[(hash(parent.state), node.move)])/(node.num_visits))
        return float('inf') if self.nsa[(hash(parent.state), node.move)] == 0 else self.c * math.sqrt(math.log(parent.num_visits)/self.nsa[(hash(parent.state), node.move)])

    def get_exploitation_constant(self, node, parent):
        return 0 if parent.nsa[node.move] == 0 else  node.value / parent.nsa[node.move]

    def get_utc(self, node, parent):
        # return self.get_exploitation_constant(node, parent) + self.get_exploration_constant(node, parent)
        return self.Q[(hash(parent.state), node.move)] + self.get_exploration_constant(node, parent)

    def get_utc_neg(self, node, parent):
        # return self.get_exploitation_constant(node, parent) - self.get_exploration_constant(node, parent)
        return self.Q[(hash(parent.state), node.move)] - self.get_exploration_constant(node, parent)

    # def get_utc(self, node):
    #     return self.get_exploitation_constant(node) + self.get_exploration_constant(node)
    
    # def get_utc_neg(self, node):
    #     return self.get_exploitation_constant(node) - self.get_exploration_constant(node)

    def prune(self, root, game):
        self.root = root
        self.root_state = deepcopy(game)
        root_parent = self.get_node_parent(root)

        states = dict()
        graph = defaultdict(list)
        Q = defaultdict(int)
        nsa = defaultdict(int)

        Q[(hash(root_parent.state), root.move)] = self.Q[(hash(root_parent.state), root.move)]
        nsa[(hash(root_parent.state), root.move)] = self.nsa[(hash(root_parent.state), root.move)]

        graph[hash(root.state)] = self.graph[hash(root.state)].copy()
        for child in graph[hash(root.state)]:
            states[hash(child.state)] = child
            Q[(hash(root.state), child.move)] = self.Q[(hash(root.state), child.move)]
            nsa[(hash(root.state), child.move)] = self.nsa[(hash(root.state), child.move)] 
        
        states[hash(root.state)] = root
        
        self.Q = Q
        self.nsa = nsa
        self.graph = graph
        self.states = states

    def generate_nodes_and_edges(self, node, graph):
        # graph.node(str(id(node)), str(node.num_visits))
        # for key, val in node.children.items():
        #     graph.node(str(id(val)), str(val.num_visits))
        #     graph.edge(str(id(val.parent)), str(id(val)), label=str(val.parent.nsa[val.move]))
        
        #     self.generate_nodes_and_edges(val, graph)

        graph.node(str(hash(node.state)), str(node.state) + ' : ' + str(node.num_visits))
        for child in self.graph[hash(node.state)]:
            graph.node(str(hash(child.state)), str(node.state) + ' : ' + str(node.num_visits))
            graph.edge(str(hash(node.state)), str(hash(child.state)), label = f'{child.move} : {self.Q[(hash(node.state), child.move)]}')

            self.generate_nodes_and_edges(child, graph)

    def show(self):
        graph = Digraph(format='pdf', strict=True)
        self.generate_nodes_and_edges(self.root, graph)
        graph.render(f'mct.gv', view=True)

    # def get_action_distribution(self):
    #     node = self.root
    #     game_state = self.root_state

    #     if len(node.children.keys()) == len(game_state.LEGAL_MOVES):
    #         l = list(map(lambda x: x[1].num_visits, list(node.children.items())))
    #         if sum(l) == 0:
    #             return l
    #         return [float(i)/sum(l) for i in l]
    #         # return l

    #     game = game_state.cfg['game']
    #     if game == 'nim':
    #         moves = list(map(lambda x: (x[0] - 1, x[1].num_visits), list(node.children.items())))
    #         # print(moves)
    #         dist = [0 for _ in range(len(game_state.LEGAL_MOVES))]
            
    #         for action in moves:
    #             dist[action[0]] = action[1]
    #     else:
    #         # [(0,0), (0,1), (0,2), (0,3), (0,4), (1,0), (1,1), (1,2), (1,3), (1,4), ]
    #         # pos = ind0 * sqrt(len) + ind1
    #         moves = list(map(lambda x: (x[0], x[1].parent.nsa[x[1].move]), list(node.children.items())))
    #         # print(moves)
    #         dist = [0 for _ in range(len(game_state.LEGAL_MOVES))]
            
    #         inc = len(dist)**(.5)
    #         for action in moves:
    #             dist[int(action[0][0]*inc + action[0][1])] = action[1]

    #     if sum(dist) == 0:
    #         return dist
    #     # return dist
    #     return [float(i)/sum(dist) for i in dist]

    def get_action_distribution(self):
        node = self.root
        game_state = self.root_state
        game_state_rep = game_state.to_string_representation()
        game_state_rep_hash = hash(game_state_rep)
        
        # print('Start-------------')
        # print(node.state)
        # print(game_state_rep)
        # print('End-------------')

        # print('Yo')
        # print(self.graph[game_state_rep_hash])

        if len(self.graph[game_state_rep_hash]) == len(game_state.LEGAL_MOVES):
            
            l = list(map(lambda x: self.nsa[(game_state_rep_hash, x.move)], self.graph[game_state_rep_hash]))
            if sum(l) == 0:
                return l
            return [float(i)/sum(l) for i in l]
            # return l

        game = game_state.cfg['game']
        if game == 'nim':
            # print(self.graph[game_state_rep_hash])
            moves = list(map(lambda x: (x.move - 1, self.nsa[(game_state_rep_hash, x.move)]), self.graph[game_state_rep_hash]))
            #print(moves)
            # print(moves)
            dist = [0 for _ in range(len(game_state.LEGAL_MOVES))]
            
            for action in moves:
                dist[action[0]] = action[1]
        else:
            # [(0,0), (0,1), (0,2), (0,3), (0,4), (1,0), (1,1), (1,2), (1,3), (1,4), ]
            # pos = ind0 * sqrt(len) + ind1
            moves = list(map(lambda x: (x.move, self.nsa[(game_state_rep_hash, x.move)]), self.graph[game_state_rep_hash]))
            # print(moves)
            dist = [0 for _ in range(len(game_state.LEGAL_MOVES))]
            
            inc = len(dist)**(.5)
            for action in moves:
                dist[int(action[0][0]*inc + action[0][1])] = action[1]

        if sum(dist) == 0:
            return dist
        # return dist
        return [float(i)/sum(dist) for i in dist]