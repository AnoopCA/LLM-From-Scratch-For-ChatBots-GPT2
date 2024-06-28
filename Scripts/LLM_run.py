
#LLM-From-Scratch-For-ChatBots-GPT2

import torch
import sys
import tiktoken
import re

sys.path.append(r'D:\ML_Projects\LLM-From-Scratch-For-ChatBots-GPT2\Scripts')
from LLM_GPT2_Tiktoken import Head, MultiHeadAttention, FeedFoward, Block, GPTLanguageModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = tiktoken.get_encoding("cl100k_base")

model = torch.load(r'D:\ML_Projects\LLM-From-Scratch-For-ChatBots-GPT2\Models\model-23_loss-0.181.pth')
#model = torch.load(r'D:\ML_Projects\LLM-From-Scratch-For-ChatBots-GPT2\Models\model_checkpoints\model_temp_275_loss-1.209.pth')
model.eval()
model.to(device)

while True:
    prompt = input("Prompt (enter 'x' to exit):\n")
    if prompt.lower() == "x":
        break
    #prompt = " [SEP] Question: " + str(prompt) + " [SEP] Answer: "
    #prompt = " [SEP] " + prompt #+ " [SEP] "
    prompt = prompt + " [SEP] "
    context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)
    generated_chars = tokenizer.decode(model.generate(context.unsqueeze(0), max_new_tokens=10)[0].tolist())

    pattern = r'\[SEP\]\s*(.*)'
    match = re.search(pattern, generated_chars, re.DOTALL)
    if match:
        result = match.group(1)
        result = re.sub(r'!{3,}', '', result)
        print(f'Generated text:\n{result}')
    else:
        print(f'Generated text:\n{generated_chars}')
