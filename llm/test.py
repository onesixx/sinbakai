# https://wikidocs.net/217157
import pandas as pd
my_sr = pd.Series(
    data=[16000, 17000, 18000, 19000, 20000],
    name='salary',
    index=['2021', '2022', '2023', '2024', '2025'])
print(my_sr)

vals = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
idx = ['one', 'two', 'three']
cols = ['A', 'B', 'C']

my_df = pd.DataFrame(data=vals, index=idx, columns=cols)
print(my_df)

data_dict = {
    'A': [1, 4, 7],
    'B': [2, 5, 8],
    'C': [3, 6, 9]
}
my_df2 = pd.DataFrame(data_dict, index=idx)
print(my_df2)

###### ------ Numpy ------
import numpy as np

vec = np.array([1, 2, 3, 4, 5])
print(vec)

mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(mat)
# my_df2 = pd.DataFrame(data=mat, index=idx, columns=cols)
# my_df2.equals(my_df)

print(f'Num of axis : {vec.ndim}')
vec.shape
print(f'Num of axis : {mat.ndim}')
mat.shape

vec_rng = np.arange(6)
vec_rng_step = np.arange(1, 10, 2)

mat_zero = np.zeros((3, 2))
mat_one  = np.ones((2, 3))
mat_full = np.full((4, 2), 6)
mat_eye  = np.eye(3)
np.random.seed(666)
mat_rand = np.random.rand(3, 2)

mat_reshape = np.arange(12).reshape(3, 4)

###### ------ Pytorch ------
import torch
### torch.autograd
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = x*y + y**2
z.backward()
print(f"x의 기울기: {x.grad}")
print(f"y의 기울기: {y.grad}")

### torch.nn : 간단한 선형 회귀 모델을 학습시키는 과정
import torch.nn as nn
# 확률적 경사 하강법(Stochastic Gradient Descent, SGD)를 중심으로 한
# 파라미터 최적화 알고리즘이 구현
import torch.optim as optim

# y = 2*x
x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_train = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1) # input_dim=1, output_dim=1

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 1000
for epoch in range(num_epochs):
    outputs = model(x_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()    # batch마다 모델 파라미터 기울기 0으로 초기화
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    predicted = model(x_train)
    print(predicted)


# Tensor
scala = torch.tensor(1.0)
print(f'0-d Num of axis : {scala.dim()}, batch_size: {scala.size()},{scala.shape}')
vector = torch.tensor([1, 2, 3, 4, 5])
print(f'1-d Num of axis : {vector.dim()}, batch_size: {vector.size()}',{vector.shape})
matrix = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]])
print(f'2-d Num of axis : {matrix.dim()}, batch_size: {matrix.size()}',{matrix.shape})
cube = torch.tensor([ [[1, 2, 3], [3, 4, 5]], [[5, 6, 7], [7, 8, 9]] ])
print(f'3-d Num of axis : {cube.dim()}, batch_size: {cube.size()}', {cube.shape})

matrix[:, 1]
matrix[0, :]

cube[:,1,:]
cube[:,:,1]

t = torch.FloatTensor([1, 2])
print(t.mean())

t = torch.FloatTensor([[1, 2], [3, 4]])
print(t.mean())
print(t.mean(dim=0))
print(t.mean(dim=1))

print(t.max())
print(t.max(dim=0))
print(t.max(dim=1))
print(t.max(dim=-1))

ft = torch.Tensor([0,1,2])
print(ft.shape)
a = ft.unsqueeze(0)
a
b = a.squeeze(0)
b
b.squeeze(0)

ft.view(1, -1)
ft.view(1, -1).shape

ft_unsqueezed = ft.unsqueeze(0)


###### ------ function vs. class ------
result = 0

def add(num):
    global result
    result += num
    return result

print(add(3))
print(add(4))

class Calculator():
    def __init__(self):
        self.result = 0

    def add(self, num):
        self.result += num
        return self.result

cal1 = Calculator()
cal2 = Calculator()
cal1.add(3)
cal1.add(4)
cal2.add(3)
cal2.add(7)

# Linear Regression and Autograd
import matplotlib.pyplot as plt

torch.manual_seed(1)
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

fig, ax = plt.subplots()
ax.scatter(x_train, y_train)
ax.set_title('Linear Regression')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()

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

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    #print(f"Epoch {epoch}/{num_epochs} W: {W.item()}, b: {b.item()} Cost: {cost.item()}")

    # log every 100 epochs
    if epoch % 100 == 0:
        print(
            f'Epoch {epoch:4d}/{num_epochs} \n \
W: {W.item():.3f}, b: {b.item():.3f} Cost: {cost.item():.6f}'
        )