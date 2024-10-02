import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(666)

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

model = nn.Linear(in_features=1, out_features=1)
# list (model.parameters()) # [W, b] 랜덤 초기화, requires_grad=True
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

num_epochs = 2000
for epoch in range(num_epochs+1):
    hypothesis = model(x_train)               # x_train*W + b
    cost = F.mse_loss(hypothesis, y_train)    # torch.mean((hypothesis - y_train) ** 2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch:4d}/{num_epochs} \n\
        W: {model.weight.item():.3f}, b: {model.bias.item():.3f} Cost: {cost.item():.6f}')

print( f' W={list(model.parameters())[0]} \n b={list(model.parameters())[1]} \n')
print( f' W={model.weight.item()} \n b={model.bias.item()}')


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

model = nn.Linear(in_features=3, out_features=1)
list(model.parameters())
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

num_epochs = 10000
for epoch in range(num_epochs+1):
    prediction = model(x_train)
    cost = F.mse_loss(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch:4d}/{num_epochs} \n\
        W: {model.weight}, b: {model.bias} Cost: {cost.item():.6f}')

new_var =  torch.FloatTensor([[73, 80, 75]])
pred_y = model(new_var)