import torch
import torch.nn as nn

# Modelni e'lon qilish (o'rnatilgan parametrlar bilan)
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output)
        return output, hidden

# Parametrlarni o'rnatish
vocab_size = 21  # Yuklangan model faylidagi vocab_size
embed_size = 100
hidden_size = 256

# Modelni yaratish
model = LanguageModel(vocab_size, embed_size, hidden_size)

# Modelni yuklash
model.load_state_dict(torch.load('language_model.nit'))

# Parametrlar sonini aniqlash
total_params = sum(p.numel() for p in model.parameters())
print(f'Modelning jami parametrlari soni: {total_params}')
