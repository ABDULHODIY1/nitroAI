import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Load the vocab from the JSON file
with open('vocab.json', 'r') as f:
    vocab = json.load(f)

# Add special tokens if not already present
special_tokens = ['[PAD]', '[UNK]']
for token in special_tokens:
    if token not in vocab:
        vocab[token] = len(vocab)

# Initialize tokenizer
class CustomTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
    
    def encode(self, text):
        tokens = text.split()
        ids = [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]
        return ids
    
    def decode(self, ids):
        tokens = [list(self.vocab.keys())[id] for id in ids if id != self.vocab['[PAD]'] and id < len(self.vocab)]
        return ' '.join(tokens)

tokenizer = CustomTokenizer(vocab)

# Create dataset and dataloader
texts = [
    "Salom",
]

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

# Define hyperparameters
embed_size = 900
hidden_size = 900
learning_rate = 0.001
num_epochs = 10

# Correctly calculate vocab_size
vocab_size = max(vocab.values()) + 1
model = LanguageModel(vocab_size, embed_size, hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create dataloader
dataloader = DataLoader(TextDataset(texts, tokenizer), batch_size=2, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        inputs = batch  # No need for conversion, already tensor

        if inputs.max() >= vocab_size:
            raise ValueError(f"Input index {inputs.max()} out of range for vocab size {vocab_size}")

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

# Save the model
torch.save(model.state_dict(), 'language_model.pth')

# Test the model with new texts
test_texts = [
    "Salom kim siz qachon keldingiz"
    ]

test_dataset = TextDataset(test_texts, tokenizer, max_length=10)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Use the model with test data
model.eval()
for batch in test_dataloader:
    inputs = batch
    outputs, _ = model(inputs)
    
    predicted_ids = torch.argmax(outputs, dim=2)
    predicted_texts = [tokenizer.decode(ids.tolist()) for ids in predicted_ids]
    
    print("Inputs:", tokenizer.decode(inputs[0].tolist()))
    print("Predictions:", predicted_texts[0])

print("Testing finished.")
