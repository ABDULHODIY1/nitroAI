import json

class BaseTokenizer:
    def __init__(self, vocab: dict[str, int], special_tokens: dict[str, str], max_length: int = 512):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.special_tokens = special_tokens
        self.max_length = max_length
    
    def tokenize(self, text: str) -> list[str]:
        # Matnni tokenlarga ajratish (simple whitespace tokenization as an example)
        tokens = text.split()
        return tokens
    
    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        # Tokenlarni indekslarga aylantirish
        return [self.vocab.get(token, self.vocab.get('[UNK]')) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        # Indekslarni tokenlarga aylantirish
        return [self.inv_vocab.get(i, '[UNK]') for i in ids]
    
    def encode(self, text: str) -> list[int]:
        # Matnni to'liq tokenizatsiya va indekslash
        tokens = self.tokenize(text)
        ids = self.convert_tokens_to_ids(tokens)
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
        return ids
    
    def decode(self, ids: list[int]) -> str:
        # Indekslar ro'yxatini asl matnga qaytarish
        tokens = self.convert_ids_to_tokens(ids)
        return ' '.join(tokens)
    
    def add_special_tokens(self):
        # Maxsus tokenlarni lug'atga qo'shish
        for token in self.special_tokens.values():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                self.inv_vocab[len(self.vocab) - 1] = token
    
    def save_pretrained(self, save_directory: str):
        # Tokenizatorni va lug'atni faylga saqlash
        with open(f'{save_directory}/vocab.json', 'w') as f:
            json.dump(self.vocab, f)
        with open(f'{save_directory}/special_tokens.json', 'w') as f:
            json.dump(self.special_tokens, f)
    
    @classmethod
    def from_pretrained(cls, load_directory: str):
        # Oldindan o'rgatilgan tokenizatorni fayldan yuklash
        with open(f'{load_directory}/vocab.json', 'r') as f:
            vocab = json.load(f)
        with open(f'{load_directory}/special_tokens.json', 'r') as f:
            special_tokens = json.load(f)
        return cls(vocab, special_tokens)