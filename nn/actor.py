import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import yaml
import numpy as np
import datetime

class NeuralNetwork(nn.Module):
    
    def __init__(self, learning_rate, inp_size, layers, loss_fn, activation_fn, output_activation, optimizer):
        super(NeuralNetwork, self).__init__()

        LOSS_FUNCTIONS = {
            'mse': nn.MSELoss(),
            'kldiv': nn.KLDivLoss(),
            'nllloss': nn.NLLLoss(),
            'cross_entropy': self._custom_cross_entropy
        }

        self.ACTIVATION_FUNCTIONS = {
            'relu': nn.ReLU,
            'linear': nn.Linear,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'softmax': nn.Softmax,
            'logsoftmax': nn.LogSoftmax
        }

        OPTIMIZERS = {
            'adam': optim.Adam,
            'adagrad': optim.Adagrad,
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop
        }

        if loss_fn.lower() not in LOSS_FUNCTIONS:
            raise Exception('Loss function ' + loss_fn + ' not available!') 

        if activation_fn.lower() not in self.ACTIVATION_FUNCTIONS:
            raise Exception('Loss function ' + activation_fn + ' not available!') 

        if output_activation.lower() not in self.ACTIVATION_FUNCTIONS:
            raise Exception('Loss function ' + output_activation + ' not available!') 

        if optimizer.lower() not in OPTIMIZERS:
            raise Exception('Loss function ' + optimizer + ' not available!') 

        self.learning_rate = learning_rate
        # self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.KLDivLoss()
        # self.loss_fn = nn.NLLLoss()
        # self.loss_fn = nn.CrossEntropyLoss()
        
        self.loss_fn = LOSS_FUNCTIONS[loss_fn] # self._custom_cross_entropy
        
        if torch.cuda.is_available():  
            dev = "cuda" 
        else:  
            dev = "cpu"  
        
        self.device = 'cpu' # dev
        self.inp_size = inp_size
        self.layers = self.init_model(inp_size, layers, activation_fn, output_activation)
        # self.optimizer = optim.SGD(self.parameters(), learning_rate, 0.9)
        self.optimizer = OPTIMIZERS[optimizer](self.parameters(), lr=learning_rate) # optim.Adam(self.parameters(), lr=learning_rate)
        self.head = nn.LogSoftmax(dim=0)
        # self.head = nn.Linear(layers[-1], layers[-1]).to(self.device)
        # self = self.to(self.device)


    def _custom_cross_entropy(self, outs, targets):
        loss = torch.mean(-1*torch.sum(targets*self._safe_log(outs)))
        # loss = loss.to(self.device, non_blocking=True)
        return loss
    
    def _safe_log(self, tensor, base=0.0001):
        log = torch.log(torch.max(tensor, torch.full(tensor.size(), base)))# .to(self.device, non_blocking=True)))
        # log = log.to(self.device, non_blocking=True)
        return log

    def init_model(self, inp_size, layers, activation_fn, output_activation):
        tmp_layers = [nn.Linear(inp_size, layers[0])]

        for i, size in enumerate(layers):
            tmp_layers.append(self.ACTIVATION_FUNCTIONS[activation_fn]())
            tmp_layers.append(nn.Linear(size, layers[i+1] if i < len(layers) - 1 else size))

        # tmp_layers.append(nn.Dropout(p=0.2))
        if output_activation.lower() == 'softmax':
            tmp_layers.append(nn.Softmax(dim=0))
        elif output_activation.lower() == 'logsoftmax':
            tmp_layers.append(nn.LogSoftmax(dim=0))
        else:
            tmp_layers.append(self.ACTIVATION_FUNCTIONS[output_activation])

        return nn.ModuleList(tmp_layers)
    
    def forward(self, state):
        for layer in self.layers:
            state = layer(state)
        # return self.head(state)
        # state = state.to(self.device, non_blocking=True)
        return state

    def train_on_rbuf(self, minibatch):
        # minibatch = random.sample(RBUF, minibatchSize)
        losses = []
        for _ in range(1):
            for item in minibatch:
                state = item[0]

                actionDistribution = torch.FloatTensor(item[1])# .to(self.device, non_blocking=True)
                # target = torch.Tensor([item[1].index(max(item[1]))]).long()
                
                self.optimizer.zero_grad()
                
                output = self(state)

                loss = self.loss_fn(output, actionDistribution)

                loss.backward()
                
                losses.append(loss.item())

                self.optimizer.step()
       
        return np.mean(losses)

class Actor():
    def __init__(self, lr, inp_size, layers, loss_fn, activation_fn, output_activation, optimizer):
        self.model = NeuralNetwork(lr, inp_size, layers, loss_fn, activation_fn, output_activation, optimizer)


    def predict_val(self, state):
        return self.model(self.string_state_to_tensor(state))

    def string_state_to_tensor(self, st):
        state_tensor = torch.Tensor([float(i) for i in st])
        # state_tensor = state_tensor.to(self.model.device, non_blocking=True)
        return state_tensor

    def trainOnRBUF(self, RBUF):
        return self.model.train_on_rbuf(map(lambda x: (self.string_state_to_tensor(x[0]), x[1]), RBUF))

    def save(self, game_name, game, episode, num_games):
        if game_name == 'nim':
            filename = f"nim-{self.model.inp_size-1}stones-{game.max_removal}removal-at-{episode}-of-{num_games}-episodes-{datetime.datetime.strftime(datetime.datetime.now(), '%d-%m-%Y-%H-%M')}"
        else:
            filename = f"{game.board.board_size}/Hex-{game.board.board_size}x{game.board.board_size}-at-{episode}-of-{num_games}-episodes-{datetime.datetime.strftime(datetime.datetime.now(), '%d-%m-%Y-%H-%M')}"
        print(f'Saving: {filename}')
        torch.save(self.model.state_dict(), f'TrainedNetworks/{game_name}/{filename}')

    def load(self, game, filename):
        self.model = torch.load(open(f'TrainedNetworks/{game}/{filename}', 'rb'))
        self.model.eval()

if __name__ == "__main__":
    a = Actor(yaml.safe_load(open('config.yaml', 'r')))
    
    # t1 = torch.FloatTensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0])
    # t2 = torch.FloatTensor([0.0426, 0.0311, 0.0368, 0.0395, 0.0444, 0.0502, 0.0364, 0.0417, 0.0316, 0.0447, 0.0454, 0.0324, 0.0410, 0.0417, 0.0312, 0.0342, 0.0473, 0.0444, 0.0363, 0.0494, 0.0429, 0.0324, 0.0486, 0.0396, 0.0344])
    t1 = torch.Tensor([0.25, 0.25, 0.25])
    t2 = torch.Tensor([0.25, 0.25, 0.25])
    print(a.model._custom_cross_entropy(t1, t2))
    # t = torch.Tensor([2.0, 1.0, 5.0])
    # print(a.predict_val('215'))
    # print(a.predict_val('203'))