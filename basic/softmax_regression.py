import torch
import torch.nn.functional as F

torch.manual_seed(666)

z = torch.FloatTensor([1, 2, 3])
hypothesis = F.softmax(z, dim=0)
print(hypothesis)
print(hypothesis.sum())

z = torch.rand(3, 5, requires_grad=True)
hypothesis = F.softmax(z, dim=1)
print(hypothesis)
print(hypothesis.sum(dim=1))

y = torch.randint(5, (3,)).long()
  git config --global user.email "you@example.com"
  git config --global user.name "Your Name"