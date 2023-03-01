import torch as torch
from torch.distributions import Categorical
from torch.distributions.bernoulli import Bernoulli

'''
eval=False
probs=torch.tensor([[0.5,0.5],[0.1,0.9],[0.9,0.1]])

if eval:

    print(probs)
    probs=torch.zeros_like(probs).scatter(1, probs.argmax(1,True), value=1)
    print(probs)

distrib=Categorical(probs=probs)

random=Categorical(probs=torch.tensor([0,1]))

misstransmission=Bernoulli(probs=torch.tensor([0.5]))

x=distrib.sample()
noise=random.sample((3,))

X=torch.concat([x.reshape(-1,1),noise.reshape(-1,1)],dim=1)

misstransmitted=misstransmission.sample((3,)).to(dtype=torch.long)


print(x.shape)
print(noise.shape)
print(misstransmitted.shape)
print(X)

print(misstransmitted)


print(torch.gather(X,1,misstransmitted).flatten())
'''

probs=torch.tensor([[0.5,0.5],[0.1,0.9],[0.9,0.1]],requires_grad=True)
distrib=Categorical(probs=probs)