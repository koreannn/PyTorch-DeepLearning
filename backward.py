import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

"""간단한 예시용 값 생성"""
# y = 2x+1
epochs =  10
x_values = [i for i in range(11)] 
x_trains = np.array(x_values, dtype='float32')
x_trains = x_trains.reshape(-1, 1)
print("x_trains.shape: ", x_trains.shape)
print("x_trains: \n", x_trains)
print()

y_values = [2*i+1 for i in x_values]
y_trains = np.array(y_values, dtype='float32')
y_trains = y_trains.reshape(-1, 1)
print("y_trains.shape: ", y_trains.shape)
print("y_trains: \n", y_trains)
print()

"""예시용 인스턴스 생성"""
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        output = self.linear(x)
        return output
    # Linear같은 Layer를 직접 가져와서 쓰기때문에, 직접적으로 파라미터를 가져와서 쓸 일이 없음(Layer 내부에 이미 다 정의되어있다.)

input_dim = 1
output_dim = 1 # x, y의 dim은 각각 1이므로
lr = 0.01
epochs = 10

model = LinearRegression(input_dim, output_dim)
if torch.cuda.is_available():
    model.cuda()
print("model.parameters(): ", model.parameters()) # 내부에 이미 다 정의되어있음
print()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01) # 각 인자의 의미는, "대상이 되는 파라미터", "학습률". 따라서 "model.parameters()에 해당하는 값을 최적화하겠다" 라는 뜻

# 일반적으로 데이터 로더가 있기때문에 데이터가 잘려서 들어가긴 하는데, 이 예시에서는 데이터 로더 없이 한번에 넣어서 학습한다.
for epoch in range(epochs):
    if torch.cuda.is_available():
        inputs = Variable(torch.from_numpy(x_trains).cuda())
        labels = Variable(torch.from_numpy(y_trains).cuda())
    else:
        inputs = Variable(torch.from_numpy(x_trains))
        labels = Variable(torch.from_numpy(y_trains))
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    print("loss: ", loss)
    loss.backward()
    optimizer.step()
    
    print("epoch: {} loss: {}".format(epoch, loss.item()))
print()

# 학습 후 실제로 예측해보기
with torch.no_grad():
    if torch.cuda.is_available():
        predicted = model(torch.from_numpy(x_trains).cuda()).cpu().numpy()
    else:
        predicted = model(torch.from_numpy(x_trains)).data.numpy()

print("Predicted(y_hat):\n",predicted, "\nReal(y):\n", y_trains)
print()

for p in model.parameters(): # 값들이 어떻게 업데이트되었는지 확인
    print("p: \n", p)
    if p.requires_grad: # requires_grad는 미분의 대상이 되는 값
        print("p.name: \n", p.name, "\np.data: \n", p.data)
        

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LR(nn.Module):
    def __init__(self, dim, lr=torch.scalar_tensor(0.01)):
        super(LR, self).__init__()
        self.weight = torch.zeros(dim, dtype=torch.float32).to(device)
        self.bias = torch.scalar_tensor(0).to(device)
        self.grads = {
            "dw": torch.zeros(dim, 1, dtype=torch.float32).to(device),
            "db": torch.zeros(0).to(device)
        }
        self.lr = lr.to(device)
        
    def forward(self, x):
        z = torch.mm(self.w.T, x)
        a = self.sigmoid(z)
        return a
    
    def sigmoid(self, z):
        return 1/(1+torch.exp(-z))
    
    def backward(self, x, y_hat, y):
        self.grads["dw"] = (1/x.shape[1])*torch.mm(x, (y_hat-y).T)
        self.grads["db"] = (1/x.shape[1])*torch.sum(y_hat-y)
        
    def optimize(self):
        self.weight = self.weight - self.lr*self.grads["dw"]
        self.bias = self.bias - self.lr*self.grads["db"]
        