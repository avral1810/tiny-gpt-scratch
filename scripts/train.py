from tiny_gpt import CharTokenizer, get_batch
import torch


tok = CharTokenizer("hello")
ids = tok.encode("hello")
torch.manual_seed(42)
data = torch.randint(0, 64, size=(128,),dtype=torch.long)

print(get_batch(batch_size=4, block_size=4, data=data))
