import torch
import json
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Model klassi
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

# Tokenizator va lug'at (o'rgatishda ishlatilgan)
with open('vocab.json', 'r') as f:
    vocab = json.load(f)
    print(vocab)
special_tokens = {'[CLS]': 709, '[SEP]': 710, '[SOS]': 711, '[EOS]': 712, '[PAD]': 713, '[UNK]': 714}

class CustomTokenizer:
    def __init__(self, vocab, special_tokens):
        self.vocab = {**vocab, **special_tokens}
    
    def encode(self, text):
        tokens = text.split()
        ids = [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]
        return ids
    
    def decode(self, ids):
        tokens = [list(self.vocab.keys())[id] for id in ids if id != self.vocab['[PAD]']]
        return ' '.join(tokens)

tokenizer = CustomTokenizer(vocab, special_tokens)

# Saqlangan holat lug'atini yuklash
saved_state_dict = torch.load('language_model.nit', map_location=torch.device('cpu'))

# Parametr hajmlarini aniqlash
vocab_size = len(tokenizer.vocab)
embed_size = saved_state_dict['embedding.weight'].size(1)
hidden_size = saved_state_dict['rnn.weight_ih_l0'].size(0) // 3  # GRU-da 3 ga bo'linadi

# Modelni yaratish
model = LanguageModel(vocab_size, embed_size, hidden_size)
model.load_state_dict(saved_state_dict)
model.eval()

# Sinov matnlari
test_texts = [
    "Salom loiha haqida nima deysiz?",
    "Salom loiha haqida nima deysiz?"
]

# Test dataset va dataloader yaratish
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=10):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer.encode(text)
        padded_sequence = self.pad_sequence(encoded, self.max_length)
        return torch.tensor(padded_sequence, dtype=torch.long)
    
    def pad_sequence(self, sequence, max_length):
        if len(sequence) < max_length:
            sequence = sequence + [self.tokenizer.vocab['[PAD]']] * (max_length - len(sequence))
        else:
            sequence = sequence[:max_length]
        return sequence

test_dataset = TextDataset(test_texts, tokenizer, max_length=10)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Modelni sinov ma'lumotlari bilan ishlatish
for batch in test_dataloader:
    inputs = batch
    outputs, _ = model(inputs)
    
    predicted_ids = torch.argmax(outputs, dim=2)
    predicted_texts = [tokenizer.decode(ids.tolist()) for ids in predicted_ids]
    
    print("Inputs:", tokenizer.decode(inputs[0].tolist()))
    print("Predictions:", predicted_texts[0])
