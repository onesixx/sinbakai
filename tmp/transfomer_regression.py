'''
PositionalEncoding클래스를 사용하여 시퀀스의 위치 정보를 인코딩합니다.
TransformerModel 클래스는 트랜스포머 인코더를 사용하여 시계열 데이터를 처리합니다.
간단한 사인 함수에 노이즈를 추가한 데이터를 생성하여 모델을 학습시킵니다.
모델을 학습한 후 예측을 수행하고 결과를 시각화합니다.
이 예제는 트랜스포머를 단변량 시계열 회귀에 적용한 것입니다. 실제 응용에서는 데이터의 특성에 따라 모델 구조와 하이퍼파라미터를 조정해야 할 수 있습니다.
'''
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# GPU 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# 데이터 생성
x = np.linspace(0, 100, 1000)
y = np.sin(x) + np.random.normal(0, 0.1, 1000)

fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()

# 데이터 전처리
x = torch.FloatTensor(x).view(-1, 1).to(device)
y = torch.FloatTensor(y).view(-1, 1).to(device)

# 모델 초기화
model = TransformerModel(input_dim=1, d_model=64, nhead=4, num_layers=3).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습
n_epochs = 10
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

# 예측
model.eval()
with torch.no_grad():
    y_pred = model(x)
    y_pred = y_pred.squeeze(1)  # 불필요한 차원 제거

# 결과 시각화
x_cpu = x.cpu().numpy()  # GPU에서 CPU로 이동
y_cpu = y.cpu().numpy()  # GPU에서 CPU로 이동
y_pred_cpu = y_pred.cpu().numpy()  # GPU에서 CPU로 이동

plt.figure(figsize=(10, 6))
plt.scatter(x_cpu, y_cpu, color='blue', label='Actual')
plt.plot(x_cpu, y_pred_cpu, color='red', label='Predicted')
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.scatter(x, y, color='blue', label='Actual')
ax.plot(x, y_pred, color='red', label='Predicted')
ax.legend()
plt.show()