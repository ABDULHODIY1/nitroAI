import json
from collections import Counter

# Sample large text data (replace with your own data)
large_texts = open('data.txt', 'r').readlines()

# Tokenize the text data
def tokenize(text):
    return text.split()

# Count the frequency of each token
counter = Counter()
for text in large_texts:
    tokens = tokenize(text)
    counter.update(tokens)

# Get the most common 5000 tokens
most_common_tokens = counter.most_common(5000)

# Create vocab dictionary
vocab = {token: idx for idx, (token, _) in enumerate(most_common_tokens)}

# Add special tokens
special_tokens = {'[CLS]': len(vocab), '[SEP]': len(vocab)+1, '[SOS]': len(vocab)+2, '[EOS]': len(vocab)+3, '[PAD]': len(vocab)+4, '[UNK]': len(vocab)+5}
vocab.update(special_tokens)

# Save the vocab to a JSON file
with open('vocab.json', 'w') as f:
    json.dump(vocab, f, ensure_ascii=False, indent=4)

print("Vocab saved to vocab.json")
