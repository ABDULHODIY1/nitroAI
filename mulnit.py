import torch
import torch.nn as nn

class NITModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(NITModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)

def load_nit_model(file_path):
    state_dict = torch.load(file_path, map_location=torch.device('cpu'))
    input_size = state_dict['linear.weight'].size(1)
    output_size = state_dict['linear.weight'].size(0)
    
    model = NITModel(input_size, output_size)
    model.load_state_dict(state_dict)
    
    return model

# Modelni yuklash
model_path = '/Users/abdulhodiy/VPro/gpt2.7b/summator_model.nit'
model = load_nit_model(model_path)

# Model parametrlarini hisoblash
current_params = sum(p.numel() for p in model.parameters())

# 2X oshirish
factor = 2
new_params = current_params * factor

# Yangi modelni tuzish
new_model = nn.Sequential(
    nn.Linear(current_params, new_params),
    nn.ReLU(),
    nn.Linear(new_params, current_params),
    nn.ReLU()
)

# Eski modelning parametrlarini yangi modelga ko'chirish
with torch.no_grad():
    new_model[0].weight.copy_(model.linear.weight.view(current_params, 1))
    new_model[0].bias.copy_(model.linear.bias.view(current_params))
    new_model[2].weight.copy_(model.linear.weight.view(1, current_params))
    new_model[2].bias.copy_(model.linear.bias.view(new_params))

# Yangi modelni saqlash
new_model_path = '/Users/abdulhodiy/VPro/gpt2.7b/new_model.nit'
torch.save(new_model.state_dict(), new_model_path)

print(f"Current parameters: {current_params}, New parameters: {new_params}")
