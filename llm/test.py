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

