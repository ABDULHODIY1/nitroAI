import torch
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Sample dataset for language modeling (toy example)
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=50):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer.encode(text)
        padded_sequence = self.pad_sequence(encoded, self.max_length)
        return torch.tensor(padded_sequence)  # Return as a tensor
    
    def pad_sequence(self, sequence, max_length):
        if len(sequence) < max_length:
            sequence = sequence + [self.tokenizer.vocab['[PAD]']] * (max_length - len(sequence))
        else:
            sequence = sequence[:max_length]
        return sequence

# Sample language model class
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

# Initialize tokenizer (assuming you have a tokenizer class or function)
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

# Create tokenizer and vocab (toy example, adjust based on your actual vocab creation method)
with open('vocab.json', 'r') as f:
    vocab = json.load(f)

special_tokens = {'[CLS]': 709, '[SEP]': 710, '[SOS]': 711, '[EOS]': 712, '[PAD]': 713, '[UNK]': 714}
tokenizer = CustomTokenizer(vocab, special_tokens)

# Create dataset and dataloader
texts = [
    "qayerdan san",
]
dataset = TextDataset(texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Model, criterion, and optimizer
vocab_size = len(tokenizer.vocab)
embed_size = 600
hidden_size = 510
learning_rate = 0.001
num_epochs = 200

model = LanguageModel(vocab_size, embed_size, hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        inputs = batch  # No need for conversion, already tensor
        
        targets = inputs[:, 1:]   # Target sequence (shifted by one position)
        inputs = inputs[:, :-1]
        
        outputs, _ = model(inputs)
        outputs = outputs.view(-1, vocab_size)  # Flatten outputs
        targets = targets.contiguous().view(-1)  # Flatten targets
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")

# Save the model with .nit extension
torch.save(model.state_dict(), 'language_model.nit')

print("Model saved with .nit extension.")

# Load the model for testing
model = LanguageModel(vocab_size, embed_size, hidden_size)
model.load_state_dict(torch.load('language_model.nit'))
model.eval()

# Test dataset and dataloader  
test_texts = input(">>>")
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
