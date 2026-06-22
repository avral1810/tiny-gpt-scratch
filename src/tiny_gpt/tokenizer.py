
class CharTokenizer:
    def __init__(self, text: str):
        self.text = text
        self.vocab = sorted(set(text))
        self.vocab_size = len(self.vocab)
        self.stoi = {}
        self.itos = {}
        for i, char in enumerate(self.vocab):
            self.itos[i] = char
            self.stoi[char] = i
    
    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, token: list[int]) -> str:
        return "".join(self.itos[i] for i in token)
 
