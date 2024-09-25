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


###### ------ . MultiHead Attention ------ ######
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0, "d_model must be divisible by num_heads"

        self.depth = d_model // self.num_heads

        self.query_dense = nn.Linear(d_model, d_model)
        self.key_dense   = nn.Linear(d_model, d_model)
        self.value_dense = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]

        query = self.query_dense(query)
        key   = self.key_dense(key)
        value = self.value_dense(value)

        query = self.split_heads(query, batch_size)
        key   = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3)

        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)

        outputs = self.dense(concat_attention)

        return outputs

def create_padding_mask(x):
    mask = torch.eq(x, 0).float()
    # (batch_size, 1, 1, key의 문장 길이)
    return mask.unsqueeze(1).unsqueeze(2)

# 예제 실행
x = torch.tensor([[1, 21, 777, 0, 0]])
mask = create_padding_mask(x) #  1의 값을 가진 위치의 열을 어텐션 스코어 행렬에서 마스킹하는 용도로 사용 가능
print(mask)
# [[1, 21, 777, 0, 0]] 벡터를 스케일드 닷 프로덕트 어텐션의 인자로 전달하면,
# 스케일드 닷 프로덕트 어텐션에서는 위 벡터에다가 매우 작은 음수값인 -1e9를 곱하고,
# 이를 행렬에 더해주어 해당 열을 전부 마스킹(1)

###### Position-wise FF NN (Feed Forward Neural Network)

###### ------ Encoder ------ ######
class EncoderLayer(nn.Module):
    def __init__(self, dff, d_model, num_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)

        self.dense1 = nn.Linear(d_model, dff)
        self.dense2 = nn.Linear(dff, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, padding_mask):
        attn_output = self.multi_head_attention(x, x, x, padding_mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.norm1(x + attn_output)

        ffn_output = F.relu(self.dense1(out1))
        ffn_output = self.dense2(ffn_output)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.norm2(out1 + ffn_output)

        return out2

class Encoder(nn.Module):
    def __init__(self, vocab_size, num_layers, dff, d_model, num_heads, dropout):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(vocab_size, d_model)
        self.enc_layers = nn.ModuleList([EncoderLayer(dff, d_model, num_heads, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask):
        seq_len = x.shape[1]

        # Adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = self.pos_encoding(x)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, padding_mask)

        return x  # (batch_size, input_seq_len, d_model)


###### ------ Encoder Layer ------ ######
def create_look_ahead_mask(x):
    seq_len = x.shape[1]
    look_ahead_mask = torch.ones(seq_len, seq_len).triu(diagonal=1)  # 대각선 위의 값을 1로 채움
    padding_mask = create_padding_mask(x)  # 앞서 정의한 패딩 마스크 함수 사용

    # 룩어헤드 마스크와 패딩 마스크의 최댓값을 취함
    # 룩어헤드 마스크와 패딩 마스크는 (batch_size, 1, seq_len, seq_len) 형태로 확장 필요
    look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(0)
    max_mask = torch.max(look_ahead_mask, padding_mask)
    return max_mask

# 예제 실행
x = torch.tensor([[1, 2, 3, 0, 0]])
mask = create_look_ahead_mask(x)
print(mask)
#룩어헤드 마스크(look-ahead mask)와 패딩 마스크(padding mask)를 결합한 결과

###### ------ Decoder  ------ ######
class DecoderLayer(nn.Module):
    def __init__(self, dff, d_model, num_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model),
        )

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        # 셀프 어텐션과 레이어 정규화
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)

        # 디코더-인코더 어텐션과 레이어 정규화
        attn2, _ = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)

        # 피드 포워드 네트워크와 레이어 정규화
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)

        return out3


class Decoder(nn.Module):
    def __init__(self, vocab_size, num_layers, dff, d_model, num_heads, dropout):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(vocab_size, d_model)

        self.dec_layers = nn.ModuleList([
            DecoderLayer(dff, d_model, num_heads, dropout) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        seq_len = x.size(1)

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = self.pos_encoding(x)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)

        return x  # (batch_size, target_seq_len, d_model)

###### ------ Transformer
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_layers, dff, d_model, num_heads, dropout):
        super(Transformer, self).__init__()
        self.encoder = Encoder(vocab_size, num_layers, dff, d_model, num_heads, dropout)
        self.decoder = Decoder(vocab_size, num_layers, dff, d_model, num_heads, dropout)
        self.final_layer = nn.Linear(d_model, vocab_size)

    def forward(self, inp, tar, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, enc_padding_mask)
        dec_output = self.decoder(tar, enc_output, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output

# 하이퍼파라미터 정의
small_transformer = Transformer(
    vocab_size=9000,
    num_layers=4,
    dff=512,
    d_model=128,
    num_heads=4,
    dropout=0.3
)
small_transformer

# loss function 정의
def loss_function(y_true, y_pred, pad_token=0):
    # y_true의 shape을 조정하지 않아도 되며, PyTorch CrossEntropyLoss가 처리합니다.
    # y_pred: (batch_size, seq_len, vocab_size), y_true: (batch_size, seq_len)

    # CrossEntropyLoss는 target이 (N, ) 형태이고 입력이 (N, C)인 경우 자동으로 평탄화(flatten)을 수행합니다.
    # 여기서 N은 배치 크기 * 시퀀스 길이, C는 클래스 수(어휘 크기)입니다.
    # 그러나 여기서는 시퀀스 길이를 직접 관리하므로, 수동으로 평탄화하지 않고,
    # 대신 mask를 적용하여 패딩 토큰을 제외합니다.

    loss_obj = torch.nn.CrossEntropyLoss(reduction='none')  # 각 요소별로 손실 계산
    loss = loss_obj(y_pred.view(-1, y_pred.size(-1)), y_true.view(-1))

    mask = (y_true != pad_token).float()  # 패딩이 아닌 위치는 1, 패딩 위치는 0인 마스크
    loss = loss * mask.view(-1)  # 마스크 적용

    # 마스크를 적용한 후 평균 손실을 계산
    loss = torch.sum(loss) / torch.sum(mask)

    return loss
