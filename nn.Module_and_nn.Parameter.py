import torch
from torch import nn
from torch import Tensor

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(
            torch.randn(in_features, out_features))
        # self.weight = Tensor(
        #     torch.randn(in_features, out_features)) # 잘못된 방식 (학습이 되는 값을 만든게 아니라 그냥 값을 만든거임. 즉 미분의 대상이 아님)
        
        self.bias = nn.Parameter(torch.randn(out_features))
    def get_weight_shape(self):
        return self.weight.shape
        
    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias # y_hat = xW + b (@는 행렬 곱셈 키워드)

x = torch.randn(5, 7) # 5개의 데이터, 7개의 피쳐
print("x.shape: ", x.shape)
print("x: ", x)
print()

layer = MyLinear(7, 12) # 7개의 피쳐, 12개의 아웃풋
y_hat = layer(x)
print("layer.weight.shape: ", layer.weight.shape)
print("y_hat.shape: ", y_hat.shape)
print("y_hat: ", y_hat)
print()
for value in layer.parameters():
    print(value, end="\n\n")
    