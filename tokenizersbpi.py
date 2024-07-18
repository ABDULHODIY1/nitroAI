from typing import Dict, List
from base_tokens import BaseTokenizer
import json

class CustomTokenizer(BaseTokenizer):
    def __init__(self, vocab: Dict[str, int], special_tokens: Dict[str, str], max_length: int = 512):
        super().__init__(vocab, special_tokens, max_length)
    
    def tokenize(self, text: str) -> List[str]:
        # Matnni tokenlarga ajratish (misol uchun, Bo'limli tokenizatsiya)
        tokens = text.split()
        return tokens

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        # Tokenlarni indekslarga o'girish
        return [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        # Indekslarni tokenlarga qaytarish
        inv_vocab = {v: k for k, v in self.vocab.items()}
        return [inv_vocab.get(idx, '[UNK]') for idx in ids]

    def encode(self, text: str) -> List[int]:
        # Matnni to'liq tokenizatsiya va indekslash, maxsus tokenlar qo'shish
        tokens = [self.special_tokens['[CLS]']] + self.tokenize(text) + [self.special_tokens['[SEP]']]
        ids = self.convert_tokens_to_ids(tokens)
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
        return ids
    
    def decode(self, ids: List[int]) -> str:
        # Indekslar ro'yxatini asl matnga qaytarish
        tokens = self.convert_ids_to_tokens(ids)
        return ' '.join(tokens).replace(self.special_tokens['[CLS]'], '').replace(self.special_tokens['[SEP]'], '').strip()

# Lug'at va maxsus tokenlarni yaratish
with open('vocab.json', 'r') as f:
    vocab = json.load(f)
    special_tokens = {'[CLS]': '[CLS]', '[SEP]': '[SEP]'}

# CustomTokenizer ni yaratish
tokenizer = CustomTokenizer(vocab, special_tokens)

# Tokenizatorni ishlatish
text = ""
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)

print("Encoded:", encoded)
print("Decoded:", decoded)
