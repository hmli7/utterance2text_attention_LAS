import numpy as np

class BasicDecay:
    def __init__(self, initial_rate, anneal_rate, min_rate, every_step, mode='step'):
        self.initial_rate = initial_rate
        self.anneal_rate = anneal_rate
        self.min_rate = min_rate
        self.every_step = every_step
        self.mode = mode
    
    def get_rate_exp(self, step):
        '''np.maximum(self.initial_rate*np.exp(-self.anneal_rate*step), self.min_rate)'''
        return np.maximum(self.initial_rate*np.exp(-self.anneal_rate*step), self.min_rate)
    
    def get_rate_step(self, step):
        '''np.maximum(self.initial_rate*(self.anneal_rate ** (step//self.every_step)), self.min_rate)'''
        return np.maximum(self.initial_rate*(self.anneal_rate ** (step//self.every_step)), self.min_rate)
    
    def get_rate(self, step):
        if self.mode == 'step':
            return self.get_rate_step(step)
        elif self.mode == 'exp':
            return self.get_rate_exp(step)
        else:
            NotImplemented