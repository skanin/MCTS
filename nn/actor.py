import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

class NeuralNetwork(nn.Module):
    def __init__(self, learning_rate, inp_size, layers):
        super(NeuralNetwork, self).__init__()

        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()

        self.init_model(inp_size, layers)

        self.optimizer = optim.SGD(self.parameters(), learning_rate, 0.9)
        self.head = nn.Linear(layers[-1], layers[-1])

    def init_model(self, inp_size, layers):
        tmp_layers = [nn.Linear(inp_size, layers[0])]

        for i, size in enumerate(layers):
            tmp_layers.append(nn.Linear(size, layers[i+1] if i < len(layers) - 1 else size))

        self.layers = nn.ModuleList(tmp_layers)
    
    def forward(self, state):
        for layer in self.layers:
            state = layer(F.relu(state))
        return self.head(state)
        
    def train_on_rbuf(self, RBUF, minibatchSize):
        minibatch = random.sample(RBUF, minibatchSize)
        for item in minibatch:
            state = item[0]
            actionDistribution = torch.FloatTensor(item[1])

            self.optimizer.zero_grad()
            output = self.forward(state)

            # print(actionDistribution)
            # print(output)
            
            loss = self.loss_fn(actionDistribution, output)

            loss.backward()

            self.optimizer.step()
    

class Actor():
    def __init__(self, cfg):
        game = cfg['game']
        if game == 'nim':
            inp_size = 3
        else:
            inp_size = cfg['hex']['board_size']**2

        self.model = NeuralNetwork(cfg[game]['actor']['learning_rate'], inp_size, cfg[game]['actor']['layers'])


    def predict_val(self, state):
        return self.model(self.string_state_to_tensor(state))

    def string_state_to_tensor(self, st):
        return torch.Tensor([int(i) for i in st])

    def trainOnRBUF(self, RBUF, minibatchSize:int):
        self.model.train_on_rbuf(list(map(lambda x: (self.string_state_to_tensor(x[0]), x[1]), RBUF)), minibatchSize)
        

if __name__ == "__main__":
    state = '180'
    a = Actor(0.9, 3, [15, 20, 30, 3])

    for i, val in enumerate(a.predict_val(state)):
        print(f'i: {i}, val: {val}')

