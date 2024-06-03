import torch
import sys

sys.path.append(r'D:\ML_Projects\AI_Tech_ChatBot\Scripts')
from LLM_Finetune import Head, MultiHeadAttention, FeedFoward, Block, GPTLanguageModel, encode, decode, string_to_int, int_to_string

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.load(r'D:\ML_Projects\AI_Tech_ChatBot\Models\model-06.pth')
model.eval()
model.to(device)

while True:
    prompt = input("Prompt (enter 'x' to exit):\n")
    if prompt.lower() == "x":
        break
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated_chars = decode(model.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist())
    print(f'Generated text:\n{generated_chars}')
