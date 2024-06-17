chars = ""
with open(r'D:\ML_Projects\AI_Tech_ChatBot\Data\mbox\Inbox_cleaned.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))

vocab_size = len(chars)
string_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_string = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

# memory map for using small snippets of text from a single file of any size
def get_random_chunk(split):
    filename = r'D:\ML_Projects\AI_Tech_ChatBot\Data\mbox\Inbox_cleaned_train.txt' if split == 'train' else r'D:\ML_Projects\AI_Tech_ChatBot\Data\mbox\Inbox_cleaned_val.txt'
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Determine the file size and a random position to start reading
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size*batch_size)
            # Seek to the random position and read the block of text
            mm.seek(start_pos)
            block = mm.read(block_size*batch_size-1)
            # Decode the block to a string, ignoring any invalid byte sequences
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            # Train and test splits
            data = torch.tensor(encode(decoded_block), dtype=torch.long)
    return data

def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            print(X.shape, ":", Y.shape)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

if __name__ == "__main__":
    model = GPTLanguageModel(vocab_size)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        if iter % eval_iters == 0:
            losses = estimate_loss()
            print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")
            torch.save(model, os.path.join(r'D:\ML_Projects\AI_Tech_ChatBot\Models\model_checkpoints', f'model_temp_{iter}_loss-{losses["val"]:.3f}.pth'))

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model.forward(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print(loss.item())

    torch.save(model, os.path.join(r'D:\ML_Projects\AI_Tech_ChatBot\Models', f'model-07_loss-{loss.item():.3f}.pth'))
