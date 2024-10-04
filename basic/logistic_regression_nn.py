import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(666)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

x_train.shape  # torch.Size([6, 2])
y_train.shape  # torch.Size([6, 1])

# W = torch.zeros((2, 1), requires_grad=True)
# b = torch.zeros(1, requires_grad=True)

model = nn.Sequential( # 여러 함수들(Wx+b수식과 sigmoid함수)을 연결해주는 역할
    nn.Linear(2, 1),
    nn.Sigmoid()
)


# optimizer = optim.SGD([W, b], lr=1)
optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    # hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(W) + b)))
    # hypothesis = torch.sigmoid(x_train.matmul(W) + b)
    hypothesis = model(x_train)
    cost = F.binary_cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5]) # 예측값이 0.5를 넘으면 True로 간주
        correct_prediction = prediction.float() == y_train # 실제값과 일치하는 경우만 True로 간주
        accuracy = correct_prediction.sum().item() / len(correct_prediction) # 정확도를 계산
        print(f'Epoch {epoch:4d}/{nb_epochs} Cost: { cost.item():.6f} '
              f'Accuracy {accuracy * 100:2.2f}%'
        )


#     if epoch % 10 == 0:
#         print(f'Epoch {epoch:4d}/{nb_epochs} Cost: {cost.item():.6f}')

# print(hypothesis)
# prediction = hypothesis >= torch.FloatTensor([0.5])
# prediction