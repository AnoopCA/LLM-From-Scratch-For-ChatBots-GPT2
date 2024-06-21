import torch
import sys
import tiktoken

sys.path.append(r'D:\ML_Projects\AI_Tech_ChatBot\Scripts')
from LLM_Finetune_Tiktoken import Head, MultiHeadAttention, FeedFoward, Block, GPTLanguageModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = tiktoken.get_encoding("cl100k_base")

model = torch.load(r'D:\ML_Projects\AI_Tech_ChatBot\Models\model-10_loss-0.254.pth') #model-09_loss-0.770.pth') #model-09_loss-0.944.pth
model.eval()
model.to(device)

while True:
    prompt = input("Prompt (enter 'x' to exit):\n")
    if prompt.lower() == "x":
        break
    prompt = " [SEP] Question: " + str(prompt) + " [SEP] Answer: "
    context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)
    generated_chars = tokenizer.decode(model.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist())
    print(f'Generated text:\n{generated_chars}')
