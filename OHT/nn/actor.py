import os
import datetime

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_CACHE_MAXSIZE'] = "2147483648"
os.environ["TF_CPP_VMODULE"] = "asm_compiler=2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.activations as activations
import tensorflow.keras.losses as loss_fns
import tensorflow.keras.optimizers as optimizers
import yaml
import numpy as np


class NeuralNetwork():
    
    def __init__(self, learning_rate, inp_size, layers, loss_fn, activation_fn, output_activation, optimizer):
        super(NeuralNetwork, self).__init__()

        LOSS_FUNCTIONS = {
            'mse': loss_fns.MSE,
            'kldiv': loss_fns.KLDivergence,
            'cross_entropy': self.deepnet_cross_entropy 
        }

        self.ACTIVATION_FUNCTIONS = {
            'relu': activations.relu,
            'linear': activations.linear,
            'sigmoid': activations.sigmoid,
            'tanh': activations.tanh,
            'softmax': activations.softmax,
        }

        OPTIMIZERS = {
            'adam': optimizers.Adam,
            'adagrad': optimizers.Adagrad,
            'sgd': optimizers.SGD,
            'rmsprop': optimizers.RMSprop
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
        
        self.loss_fn = LOSS_FUNCTIONS[loss_fn]
        

        self.inp_size = inp_size
        self.model = self.init_model(inp_size, layers, activation_fn, output_activation, OPTIMIZERS[optimizer](learning_rate=learning_rate))
        

    def deepnet_cross_entropy(self, targets, outs):
     return tf.reduce_mean(tf.reduce_sum(-1 * targets * self.safelog(outs), axis=[1]))

    def safelog(self, tensor,base=0.0001):
        return tf.math.log(tf.math.maximum(tensor,base))
    
    def init_model(self, inp_size, layers, activation_fn, output_activation, optimizer):
        model = keras.Sequential()
        model.add(keras.layers.Dense(layers[0], input_dim=inp_size, kernel_initializer=keras.initializers.RandomUniform(minval=0., maxval=1.), activation=self.ACTIVATION_FUNCTIONS[activation_fn]))

        for i, size in enumerate(layers[1:]):
            if i < len(layers) - 2:
                model.add(keras.layers.Dense(size, activation=self.ACTIVATION_FUNCTIONS[activation_fn]))
            else:
                model.add(keras.layers.Dense(size, activation=self.ACTIVATION_FUNCTIONS[output_activation]))
        
        model.compile(loss=self.loss_fn, optimizer=optimizer)

        return model
    

    def predict(self, state):
        return self.model(state)[0]


    def train(self, RBUF):
        x = []
        y = []
        for st, target in RBUF:
            if self.inp_size == 2:
                x.append(np.array([float(st[0]), float(st[1:])]))
            else:
                x.append(np.array([float(i) for i in st]))
            y.append(np.array(target))
        return self.model.fit(np.array(x), np.array(y), epochs=50, verbose=0).history['loss']

class Actor():
    def __init__(self, lr, inp_size, layers, loss_fn, activation_fn, output_activation, optimizer):
        self.model = NeuralNetwork(lr, inp_size, layers, loss_fn, activation_fn, output_activation, optimizer)


    def predict_val(self, state):
        return self.model.predict(self.string_state_to_numpy(state))

    def string_state_to_numpy(self, st):
        if self.model.inp_size == 2:
            state = np.array([float(st[0]), float(st[1:])]).reshape((1,2))
        else:
            state = np.array([float(i) for i in st]).reshape((1, len(st)))
        return state

    def trainOnRBUF(self, RBUF):
        return np.mean(self.model.train(RBUF))

    def save(self, game_name, game, episode, num_games):
        if len(str(num_games)) != len(str(episode)):
            episode = ('0' * (len(str(num_games)) - len(str(episode)))) + str(episode)

        if game_name == 'nim':
            filename = f"nim-{self.model.inp_size-1}stones-{game.max_removal}removal-at-{episode}-of-{num_games}-episodes-{datetime.datetime.strftime(datetime.datetime.now(), '%d-%m-%Y-%H-%M')}"
        else:
            filename = f"{game.board.board_size}/Hex-{game.board.board_size}x{game.board.board_size}-at-{episode}-of-{num_games}-episodes-{datetime.datetime.strftime(datetime.datetime.now(), '%d-%m-%Y-%H-%M')}"
        print(f'Saving: {filename}')
        self.model.model.save(f'TrainedNetworks/{game_name}/{filename}')

    def load(self, game, filename):
        self.model = tf.keras.models.load_model(open('TrainedNetworks/{game}/{filename}', 'rb'))