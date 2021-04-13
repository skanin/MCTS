import random

class Rbuf:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def add(self, case):
        if len(self.memory) < self.capacity:
            self.memory.append(case)
        else:
            self.memory[self.position] = case
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size if len(self.memory) > batch_size else len(self.memory))

    def __len__(self):
        return len(self.memory)