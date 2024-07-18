import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization
from torch.utils.data import DataLoader, TensorDataset

# Oddiy summator modelini yaratamiz
class SummatorModel(nn.Module):
    def __init__(self):
        super(SummatorModel, self).__init__()
        self.linear1 = nn.Linear(2, 10)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(10, 1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# Modelni yaratamiz
model = SummatorModel()

# Ma'lumotlar to'plami yaratamiz (oddiy qo'shish operatsiyasi uchun)
x_train = torch.tensor([[i, j] for i in range(100) for j in range(100)], dtype=torch.float32)
y_train = torch.tensor([[i + j] for i in range(100) for j in range(100)], dtype=torch.float32)

# Ma'lumotlar to'plamini DataLoader bilan o'ramiz
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# O'qitish uchun yo'qotish funksiyasi va optimizerni belgilaymiz
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Modelni o'qitamiz
num_epochs = 5
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

print("Model o'qitildi.")

# Modelni kvantlash uchun tayyorlaymiz
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

# Modelni kvantlaymiz
torch.quantization.convert(model, inplace=True)

# Kvantlash qilingan modelni saqlash
torch.save(model.state_dict(), 'summator_model.nit')

# Kvantlash qilingan modelni yuklash
model_loaded = SummatorModel()
model_loaded.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model_loaded, inplace=True)
torch.quantization.convert(model_loaded, inplace=True)

state_dict = torch.load('summator_model.nit')
model_loaded.load_state_dict(state_dict)
model_loaded.eval()

print("Kvantlash qilingan model yuklandi va tayyor!")

# Kvantlash qilingan modelni sinash
def test_model(model, a, b):
    with torch.no_grad():
        input_tensor = torch.tensor([[a, b]], dtype=torch.float32)
        output = model(input_tensor)
        return output.item()

# Sinov
a, b = 2, 2
result = test_model(model_loaded, a, b)
print(f"{a} + {b} = {result}")

a, b = 5, 3
result = test_model(model_loaded, a, b)
print(f"{a} + {b} = {result}")
