import torch as torch 
from torch.distributions import Categorical
from torch.distributions.bernoulli import Bernoulli

x=torch.tensor([[0.5,0.5],[0.3,0.7],[0.7,0.3]])

print(x)
print(torch.zeros_like(x).scatter(1, x.argmax(1,True), value=1))

x=torch.zeros_like(x).scatter(1, x.argmax(1,True), value=1)
print(Categorical(x).entropy())

print(Bernoulli(torch.tensor([0.],device='cpu')).sample())