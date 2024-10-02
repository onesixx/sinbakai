import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(666)

from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [[73,  80,  75],
                       [93,  88,  93],
                       [89,  91,  90],
                       [96,  98,  100],
                       [73,  66,  70]]
        self.y_data = [[152], [185], [180], [196], [142]]
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y

dataset = CustomDataset()
# dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1) # 다중 선형 회귀이므로 input_dim=3, output_dim=1.

    def forward(self, x):
        return self.linear(x)

model = MultivariateLinearRegressionModel()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

num_epochs = 1000
for epoch in range(num_epochs+1):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples
        prediction = model(x_train)
        cost = F.mse_loss(prediction, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if num_epochs <= 10 or epoch % 100 == 0:
            print(f'Epoch {epoch:4d}/{num_epochs} | '
                  f'Batch {batch_idx+1}/{len(dataloader)} |'
                  f'Cost: {cost.item():.6f}')