import matplotlib.pyplot as plt
import torch

torch.manual_seed(666)
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# fig, ax = plt.subplots()
# ax.scatter(x_train, y_train)
# ax.set_title('Linear Regression')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# plt.show()

W = torch.zeros(1, requires_grad=True)  # 가중치 W를 0으로 초기화하고 학습을 통해 값이 변경되는 변수임을 명시
b = torch.zeros(1, requires_grad=True)  # 편향 b도 0으로 초기화하고 학습을 통해 값이 변경되는 변수임을 명시
# hypothesis = x_train*W + b
# cost = torch.mean((hypothesis - y_train) ** 2)
optimizer = torch.optim.SGD([W, b], lr=0.01)

# optimizer.zero_grad()
# cost.backward()
# optimizer.step()

num_epochs = 1000
for epoch in range(num_epochs+1):
    hypothesis = x_train*W + b
    cost = torch.mean((hypothesis - y_train) ** 2)

    optimizer.zero_grad() # gradient 초기화
    cost.backward()
    optimizer.step()      # 인수로 들어갔던 W와 b에서 리턴되는 변수들의 requires_grad가 True인 경우 업데이트
    #print(f"Epoch {epoch}/{num_epochs} W: {W.item()}, b: {b.item()} Cost: {cost.item()}")

    # log every 100 epochs
    if epoch % 100 == 0:
        print(
            f'Epoch {epoch:4d}/{num_epochs} \n\
            W: {W.item():.3f}, b: {b.item():.3f} Cost: {cost.item():.6f}'
        )


fig, ax = plt.subplots()
ax.scatter(x_train, y_train)
ax.plot(x_train, x_train*W.item() + b.item(), color='r')
ax.set_title('Linear Regression')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()



###### --------------------------------------------------- ######
# Multivariable Linear regression
# way 1
torch.manual_seed(666)

x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

X_train = torch.stack([x1_train, x2_train, x3_train], dim=1)
w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = torch.optim.SGD([w1, w2, w3, b], lr=1e-5)

num_epochs = 100000
for epoch in range(num_epochs+1):
    hypothesis = X_train[:, 0] * w1 + X_train[:, 1] * w2 + X_train[:, 2] * w3 + b
    cost = torch.mean((hypothesis - y_train) ** 2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(
            f'Epoch {epoch:4d}/{num_epochs} \n\
            w1: {w1.item():.3f}, w2: {w2.item():.3f}, w3: {w3.item():.3f}, b: {b.item():.3f} Cost: {cost.item():.6f}'
        )










# Way 2
x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
X_train = torch.cat([x1_train, x2_train, x3_train], dim=1)
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# w1 = torch.zeros(1, requires_grad=True)
# w2 = torch.zeros(1, requires_grad=True)
# w3 = torch.zeros(1, requires_grad=True)
W = torch.zeros((3,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = torch.optim.SGD([W, b], lr=1e-5)

num_epochs = 100000
for epoch in range(num_epochs+1):
    hypothesis = X_train.matmul(W) + b
    cost = torch.mean((hypothesis - y_train) ** 2)

    optimizer.zero_grad() # 이전 Batch에서 남아있을 수 있는 기울기(gradient) 초기화.
    cost.backward()    # 가파른 경사 찾기. 모델의 모든 가중치에 대한 기울기를 계산.
    optimizer.step()   # 한걸음 나아가기. 계산된 기울기를 사용하여 모델의 가중치를 업데이트
                       # 경사하강법(gradient descent) 알고리즘을 사용

    if epoch % 100 == 0:
        print(
            f'Epoch {epoch:4d}/{num_epochs} \n\
            hypo: {hypothesis.squeeze().detach()}, cost: {cost.item()}'

        )
        # W: {W.item():.3f}, b: {b.item():.3f} Cost: {cost.item():.6f}


# inference
with torch.no_grad():
    X_test = torch.FloatTensor([[73, 80, 75]])
    pred = X_test.matmul(W) + b
    print(f'Prediction: {pred.item()}, input:{X_test.squeeze().tolist()}')