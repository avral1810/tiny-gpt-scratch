import torch
from path import Path
from tiny_gpt import (
    CharTokenizer,
    get_batch,
    train_val_split,
    BigramLM,
)

BATCH_SIZE = 16
BLOCK_SIZE = 8
# MAX_STEPS = 10
MAX_STEPS = 10_000


def data_reader(file_name: str="data/raw/tiny.txt"):
    with open(file_name, 'r',  encoding="utf-8") as ip_file:
        text = ip_file.read()
        return text


def main():
    raw_data = data_reader().lower()
    tokenizer = CharTokenizer(raw_data)
    encoded = tokenizer.encode(raw_data)
    data = torch.tensor(encoded, dtype=torch.long)
    train_data, val_data = train_val_split(data, val_frac=0.1)
    model = BigramLM(tokenizer.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for step in range(MAX_STEPS):
        x, y = get_batch(train_data, batch_size=BATCH_SIZE, block_size=BLOCK_SIZE)
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if step % 100 == 0:
        #     print(f"At Step {step}, loss is {loss}")
        if step % 500 == 0:
            model.eval()
            with torch.no_grad():
                lloss = 0
                for i in range(20):
                    x, y = get_batch(val_data, batch_size=BATCH_SIZE, block_size=BLOCK_SIZE)
                    logits, loss = model(x, y)
                    lloss += loss
                print(f"@Step: {step} loss val: {(lloss / 20):.4f}")
            model.train()
    model.eval()
    with torch.no_grad():
        context = tokenizer.encode("happy")
        tokens = torch.tensor(context, dtype=torch.long).unsqueeze(0)
        generated = model.generate(tokens, max_new_tokens=200)
        print(tokenizer.decode(generated[0].tolist()))



if __name__ == "__main__":
    main()