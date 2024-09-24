import torch
import torch.nn as nn

# 단어 사전 정의
vocab = {'hello': 0, 'world': 1, 'goodbye': 2}
vocab_size = len(vocab)  # 3
embedding_dim = 5

# 임베딩 레이어 생성
embedding = nn.Embedding(vocab_size, embedding_dim)

# 단어 인덱스 정의
word_indices = torch.tensor([vocab['hello'], vocab['world']])

# 단어 임베딩 벡터 얻기
word_embeddings = embedding(word_indices)

print("Word Indices:", word_indices)
print("Word Embeddings:\n", word_embeddings)

###### ------ 1. Transformer Input (positional encoding)------ ######
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    '''
    위치 인코딩 : 단어의 순서를 모델에 전달하는 방법'''
    def __init__(self, position, d_model):
        '''
        클래스의 생성자
        position: 각 단어들의 위치(순서)정보, d_model: 임베딩 벡터의 차원
        부모 클래스(nn.Module)의 생성자를 먼저 호출gkdu 부모클래스의 초기화 작업
        에서 미리 포지셔널 인코딩 행렬(위치 인코딩값)을 구해 놓는다. '''
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        '''
        위치 인코딩값 구하기
        i: 각 임베딩 벡터의 차원 인덱스
        '''
        angles = 1 / torch.pow(10000.0, (2*(i//2)) / torch.tensor(d_model, dtype=torch.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position = torch.arange(position, dtype=torch.float32).unsqueeze(1),
            i        = torch.arange(d_model,  dtype=torch.float32).unsqueeze(0),
            d_model  = d_model
        )
        # 트랜스포머 모델이 단어의 순서 정보를 인식할 수 있도록 하기 위함
        # 모델은 단어의 위치 정보를 주기적인 패턴으로 인코딩하여 학습할 수 있음
        # 배열의 짝수 인덱스(2i)에는 사인 함수 적용
        # 배열의 홀수 인덱스(2i+1)에는 코사인 함수 적용
        sines = torch.sin(angle_rads[:, 0::2])
        cosines = torch.cos(angle_rads[:, 1::2])

        angle_rads = np.zeros(angle_rads.shape)
        angle_rads[:, 0::2] = sines.numpy()
        angle_rads[:, 1::2] = cosines.numpy()

        pos_encoding = torch.tensor(angle_rads)
        pos_encoding = pos_encoding.unsqueeze(0)
        print(pos_encoding.shape)

        return pos_encoding.float()

    def forward(self, inputs):
        return inputs + self.pos_encoding[:, :inputs.size(1), :]

# 문장의 길이 50, 임베딩 벡터의 차원 128
sample_pos_encoding = PositionalEncoding(50, 128)

fig, ax = plt.subplots(figsize=(10, 5))
c= ax.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0], cmap='RdBu')
ax.set_xlabel('Depth - Embedding Dimension')
ax.set_xlim((0, 128))
ax.set_ylabel('Position')
fig.colorbar(c, ax=ax) # 빨간색은 양의 값을, 파란색은 음의 값
plt.show()

###### ------ 2. Attention ------ ######
def scaled_dot_product_attention(query, key, value, mask=None):
    # query 크기: (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
    # key   크기: (batch_size, num_heads,   key의 문장 길이, d_model/num_heads)
    # value 크기: (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
    # mask      : (batch_size,  1,   1,   key의 문장 길이)

    # Q와 K의 곱. 어텐션 스코어 행렬.
    matmul_qk = torch.matmul(query, key.transpose(-1, -2))

    # 스케일링
    # dk의 루트값으로 나눠준다.
    depth = query.size(-1)
    logits = matmul_qk / torch.sqrt(torch.tensor(depth).float())

    # 마스킹. 어텐션 스코어 행렬의 마스킹 할 위치에 매우 작은 음수값을 넣는다.
    if mask is not None:
        logits += (mask * -1e9)

    # 소프트맥스 함수는 마지막 차원인 key의 문장 길이 방향으로 수행된다.
    # attention weights : (batch_size, num_heads, query의 문장 길이, key의 문장 길이)
    attention_weights = F.softmax(logits, dim=-1)

    # output : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


# 임의의 Query, Key, Value인 Q, K, V 행렬 생성
temp_k = torch.tensor([[10,0,0],
                       [0,10,0],
                       [0,0,10],
                       [0,0,10]], dtype=torch.float32)  # (4, 3)

temp_v = torch.tensor([[1,0],
                       [10,0],
                       [100,5],
                       [1000,6]], dtype=torch.float32)  # (4, 2)

temp_q = torch.tensor([[0, 10, 0]], dtype=torch.float32)  # (1, 3)


# 함수 실행
temp_out, temp_attn = scaled_dot_product_attention(temp_q, temp_k, temp_v, None)
print(temp_attn) # 어텐션 분포(어텐션 가중치의 나열)
print(temp_out) # 어텐션 값

temp_q = torch.tensor([[0, 0, 10]], dtype=torch.float32)
temp_out, temp_attn = scaled_dot_product_attention(temp_q, temp_k, temp_v, None)
print(temp_attn) # 어텐션 분포(어텐션 가중치의 나열)
print(temp_out) # 어텐션 값

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0, "d_model must be divisible by num_heads"

        self.depth = d_model // self.num_heads

        self.query_dense = nn.Linear(d_model, d_model)
        self.key_dense = nn.Linear(d_model, d_model)
        self.value_dense = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]

        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3)

        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)

        outputs = self.dense(concat_attention)

        return outputs