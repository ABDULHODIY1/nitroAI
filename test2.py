import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class CustomTokenizer:
    def __init__(self, vocab_file, special_tokens):
        self.load_vocab(vocab_file)
        self.special_tokens = special_tokens
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text):
        tokens = text.split()
        ids = []
        for token in tokens:
            if token not in self.vocab:
                new_id = len(self.vocab)
                self.vocab[token] = new_id
                self.reverse_vocab[new_id] = token
            ids.append(self.vocab[token])
        return ids
    
    def decode(self, ids):
        tokens = [self.reverse_vocab.get(id, '[UNK]') for id in ids if id in self.reverse_vocab]
        return ' '.join(tokens)
    
    def save_vocab(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.vocab, f)
    
    def load_vocab(self, filepath):
        with open(filepath, 'r') as f:
            self.vocab = json.load(f)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

class TextLabelDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, max_length=20):
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]['text']
        label = self.data[idx]['label']
        text_encoded = self.tokenizer.encode(text)
        label_encoded = self.tokenizer.encode(label)
        text_padded = self.pad_sequence(text_encoded, self.max_length)
        label_padded = self.pad_sequence(label_encoded, self.max_length)
        return torch.tensor(text_padded), torch.tensor(label_padded)
    
    def pad_sequence(self, sequence, max_length):
        if len(sequence) < max_length:
            sequence = sequence + [self.tokenizer.vocab['[PAD]']] * (max_length - len(sequence))
        else:
            sequence = sequence[:max_length]
        return sequence

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Seq2Seq, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.decoder = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.fc.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(src.device)
        
        embedded = self.embedding(src)
        encoder_outputs, hidden = self.encoder(embedded)
        
        input = trg[:, 0]
        for t in range(1, trg_len):
            input = input.unsqueeze(1)
            embedded = self.embedding(input)
            output, hidden = self.decoder(embedded, hidden)
            output = self.fc(output.squeeze(1))
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
        return outputs

# Parameters
embed_size = 3000
hidden_size = 2560

# Load tokenizer and vocab
vocab_file = 'vocab.json'
special_tokens = {'[CLS]': '[CLS]', '[SEP]': '[SEP]', '[SOS]': '[SOS]', '[EOS]': '[EOS]', '[PAD]': '[PAD]', '[UNK]': '[UNK]'}
tokenizer = CustomTokenizer(vocab_file, special_tokens)

# Load data
dataset = TextLabelDataset('data.jsonl', tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Create model
model = Seq2Seq(len(tokenizer.vocab), embed_size, hidden_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab['[PAD]'])

# Training loop
epochs = 1
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for src, trg in dataloader:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        # Expand embedding layer if vocab size has changed
        if len(tokenizer.vocab) != model.embedding.num_embeddings:
            model.embedding = nn.Embedding(len(tokenizer.vocab), embed_size).to(device)
            model.fc = nn.Linear(hidden_size, len(tokenizer.vocab)).to(device)
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader)}')

# Save updated vocab
tokenizer.save_vocab('updated_vocab.json')

# Evaluation
model.eval() 
test_text = "salom sen kimsan?"
with torch.no_grad():
    src = torch.tensor([tokenizer.encode(test_text)]).to(device)
    trg = torch.tensor([[tokenizer.vocab['[SOS]']]]).to(device)
    # Expand embedding layer if vocab size has changed
    if len(tokenizer.vocab) != model.embedding.num_embeddings:
        model.embedding = nn.Embedding(len(tokenizer.vocab), embed_size).to(device)
        model.fc = nn.Linear(hidden_size, len(tokenizer.vocab)).to(device)
    output = model(src, trg, teacher_forcing_ratio=0.0)
    output = output.squeeze(0).argmax(1)
    output_text = tokenizer.decode(output.tolist())
    print("Input:", test_text)
    print("Prediction:", output_text)
