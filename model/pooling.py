import torch.nn as nn
import torch


class Pooling(nn.Module):
    @classmethod
    def max_pool(cls):
        return cls('max')

    @classmethod
    def avg_pool(cls):
        return cls('avg')

    @classmethod
    def attention_pool(cls, hidden_size):
        cls.hidden_size=hidden_size
        return cls('attention')

    def __init__(self, method):
        super(Pooling, self).__init__()
        self.method = method
        if method=='attention':
            self.query = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, x, dim=0, method='attention'):
        """ 
        method:{'max-pool','avg-pool','attention'}
        """
        if method == 'max-pool':
            return torch.max(x, dim=dim)
        elif method == 'avg-pool':
            return torch.mean(x, dim=dim)
        else:
            A=self.query(x) # (...,1)
            x=torch.sum(A*x,dim=dim)
            return x

