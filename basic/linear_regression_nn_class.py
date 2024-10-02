import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(666)

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

num_epochs = 2000
for epoch in range(num_epochs+1):
    prediction = model(x_train)               # x_train*W + b
    cost = F.mse_loss(prediction, y_train)    # torch.mean((hypothesis - y_train) ** 2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch:4d}/{num_epochs}'
        f'W: {list(model.parameters())[0].item():.3f}, '
        f'b: {list(model.parameters())[1].item():.3f}, '
        f'Cost: {cost.item():.6f}')

print( f' W={list(model.parameters())[0]},'
       f' b={list(model.parameters())[1]} ')
print( f' W={list(model.parameters())[0].item():6f},'
       f' b={list(model.parameters())[1].item():6f} ')

# inference (forward)
new_var = torch.FloatTensor([[4.0]])
pred_y = model(new_var)


###### --------------------------------------------------- ######
# Multivariable Linear regression
import torch
import torch.nn as nn
import torch.nn.functional as F

# 데이터
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])


class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=3, out_features=1)

    def forward(self, x):
        return self.linear(x)

model = MultivariateLinearRegressionModel()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
list(model.parameters())

num_epochs = 1000
for epoch in range(num_epochs+1):
    prediction = model(x_train)
    cost = F.mse_loss(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch:4d}/{num_epochs} '
        f'W: {list(model.parameters())[0]}, '
        f'b: {list(model.parameters())[1]}, '
        f'Cost: {cost.item():.6f}')

print( f' W={list(model.parameters())[0]},'
       f' b={list(model.parameters())[1]} ')


new_var =  torch.FloatTensor([[73, 80, 75]])
pred_y = model(new_var)