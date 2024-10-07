import numpy as np
import matplotlib.pyplot as plt

d = np.array(5)
d.ndim   # axis 갯수 , tensor의 차원 , 0D tensor
d.shape  # shape은 0D tensor에는 없음 ()

fig, ax = plt.subplots()
ax.scatter([0], [d], color='r')
ax.text(0, d, f'{d}', fontsize=12, ha='right')
ax.set_title('0D Tensor (Scalar)')
ax.set_xlabel('Dimensionless')
ax.set_ylabel('Value')
ax.grid(True)
plt.show()


d = np.array([1, 2, 3, 4])
d.ndim   # 1D tensor,  4차원 벡터(한축의 원소갯수)이지만, 1차원 텐서(axis 갯수)
d.shape  # (4,)   각 축에 몇개의 원소가 있는지 tuple로 표현

fig, ax = plt.subplots()
ax.plot(d,
    marker='o', linestyle='-', color='b', label='1D Tensor')
ax.set_title('1D Tensor Visualization')
ax.set_xlabel('Index')
ax.set_ylabel('Value')
ax.grid(True)
ax.legend()
plt.show()


d = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
d.ndim   # 2D tensor, matrix
d.shape  # (2, 4)

fig, ax = plt.subplots()
for i, row in enumerate(d):
    ax.plot(row,
        marker='o', linestyle='-', label=f'Row {i+1}')
ax.set_title('2D Tensor (Matrix) Visualization')
ax.set_xlabel('Index')
ax.set_ylabel('Value')
ax.grid(True)
ax.legend()
plt.show()


# 그래프 시각화
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 각 축의 인덱스 생성
x = np.arange(d.shape[1])
y = np.arange(d.shape[0])

# 3D 그래프에 데이터 추가
for i, row in enumerate(d):
    ax.plot3D(x, [i] * len(x), row, marker='o', linestyle='-', label=f'Row {i+1}')

ax.set_title('2D Tensor (Matrix) Visualization in 3D')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Value')
ax.grid(True)
ax.legend()
plt.show()





d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
d.ndim   # 2D tensor
d.shape  # (3, 4)

# 그래프 시각화
fig, ax = plt.subplots()
for i, row in enumerate(d):
    ax.plot(row, marker='o', linestyle='-', label=f'Row {i+1}')
ax.set_title('2D Tensor (Matrix) Visualization')
ax.set_xlabel('Index')
ax.set_ylabel('Value')
ax.grid(True)
ax.legend()
plt.show()



d = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]])
d.ndim   # 3D tensor
d.shape  # (1, 3, 4)

# 그래프 시각화
fig, axes = plt.subplots(1, d.shape[0], figsize=(15, 5))

for i, matrix in enumerate(d):
    ax = axes[i] if d.shape[0] > 1 else axes
    for j, row in enumerate(matrix):
        ax.plot(row, marker='o', linestyle='-', label=f'Row {j+1}')
    ax.set_title(f'2D Matrix {i+1}')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.grid(True)
    ax.legend()

plt.suptitle('3D Tensor Visualization')
plt.show()


# 그래프 시각화
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 각 축의 인덱스 생성
x = np.arange(d.shape[2])
y = np.arange(d.shape[1])
X, Y = np.meshgrid(x, y)

# 3D 그래프에 데이터 추가
for i in range(d.shape[0]):
    Z = d[i]
    ax.plot_surface(X, Y, Z, label=f'Matrix {i+1}', alpha=0.7)

ax.set_title('3D Tensor Visualization')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Value')
plt.show()





import numpy as np
arr = np.arange(0, 32)
# array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
#        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])
v = arr.reshape([4,2,4])

#------------------------------------------------------------
# 1차원 텐서 (벡터)
# 2차원 텐서 (행렬)
# 3차원 텐서
import torch
import torch.nn.functional as F
# 1차원 텐서(벡터)  차원은 하나의 축
# 차원 수(dimensions): 1    1차원 (차원갯수=축의갯수)
# 축(axis): 0      축의갯수는 1개
# 크기(size): 3    3개의 원소   (1st차원의 크기)
# shape: (3,)      표현방식이 다름. 각 차원의 크기

vector = torch.tensor([1, 2, 3])
torch.FloatTensor([1, 2, 3]).ndimension()
torch.randint(5, (3,)).long().ndimension()

print(vector)
print("# of dimensions:", vector.ndimension())
print("axis:", list(range(vector.ndimension())))
print("size:", vector.shape)
print("shape:", tuple(vector.shape))

v = torch.squeeze(torch.FloatTensor(3,1))
F.softmax(v, dim=0)

m = torch.FloatTensor(3, 2)
F.softmax(m, dim=0)
F.softmax(m, dim=1)



# 2차원 텐서 (행렬)
matrix = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])
print("\n2D Tensor (Matrix): \n\t", matrix)
print("# of dimensions:", matrix.ndimension())
print("axis:", list(range(matrix.ndimension())))
print("size:", matrix.shape)
print("shape:", tuple(matrix.shape))

# 3차원 텐서
tensor_3d = torch.tensor([
    [
        [1, 2,3],
        [3, 4,5]],
    [
        [5, 6,7],
        [7, 8, 9]],
    [
        [9, 10, 11],
        [11, 12, 13]],
    [
        [13, 14, 15],
        [15, 16, 17]]
])
print( tensor_3d)
print("# of dimensions:", tensor_3d.ndimension())
print("axis:", list(range(tensor_3d.ndimension())))
print("size:", tensor_3d.size())  # = tensor_3d.shape
print("shape:", tuple(tensor_3d.shape))