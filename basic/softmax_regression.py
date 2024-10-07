import torch
import torch.nn.functional as F

torch.manual_seed(666)

z = torch.FloatTensor([1, 2, 3])
hypothesis = F.softmax(z, dim=0)
print(hypothesis)
print(hypothesis.sum())

###### ------ 비용 함수를 직접 구현 ------- ######
z = torch.rand(3, 5, requires_grad=True)

# 각 클래스에 대해서 소프트맥스 함수 적용 (dim=0)
hypothesis_dim0 = F.softmax(z, dim=0)
print("\nHypothesis with dim=0 (Softmax probabilities along rows):")
print(hypothesis_dim0)
print("Sum of probabilities for each class (dim=0):")
print(hypothesis_dim0.sum(dim=0))

# 각 샘플에 대해서 소프트맥스 함수 적용 (dim=1)
hypothesis_dim1 = F.softmax(z, dim=1)
print("Hypothesis with dim=1 (Softmax probabilities along columns):")
print(hypothesis_dim1)
print("Sum of probabilities for each sample (dim=1):")
print(hypothesis_dim1.sum(dim=1))



# 각 샘플에 대해서 임의의 레이블 생성
hypothesis = F.softmax(z, dim=1)
print(hypothesis)
print(hypothesis.sum(dim=1))

y = torch.randint(5, (3,)).long()

y_one_hot = torch.zeros_like(hypothesis)
y_one_hot.scatter_(1, y.unsqueeze(1), 1)  # 차원dim  index, src할당값

F.log_softmax(z, dim=1)
F.cross_entropy(z, y)