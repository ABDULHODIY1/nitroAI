import torch
import torch.nn as nn
import torch.quantization

# Modelni yaratamiz
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(10, 10)
        self.linear2 = nn.Linear(10, 10)
        self.linear3 = nn.Linear(10, 1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

# Kvantlash uchun modelni tayyorlaymiz
model_fp32 = SimpleModel()
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_fp32_prepared = torch.quantization.prepare(model_fp32)

# Modelni kvantlash
model_int8 = torch.quantization.convert(model_fp32_prepared)

# Kvantlash qilingan modelni saqlash
torch.save(model_int8.state_dict(), 'model.nit')

# Kvantlash qilingan modelni yuklash
model_int8_loaded = SimpleModel()
model_int8_loaded.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_int8_loaded_prepared = torch.quantization.prepare(model_int8_loaded)
model_int8_loaded = torch.quantization.convert(model_int8_loaded_prepared)

# State dict yuklanishi
state_dict = torch.load('model.nit')
model_int8_loaded.load_state_dict(state_dict)

model_int8_loaded.eval()
print("Model yuklandi va tayyor!")
