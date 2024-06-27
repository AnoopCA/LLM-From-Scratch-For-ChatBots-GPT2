# LLM-From-Scratch-For-ChatBots-GPT2

This project implements a Large Language Model (LLM) from scratch in Python, utilizing the transformer mechanism similar to that of GPT-2. The model is trained on a customer service dataset from an online product-selling company, aiming to assist in generating accurate and contextually relevant responses for customer inquiries. The project leverages `tiktoken` for tokenizing the text data.

## Features

- **Transformer Architecture:** Built using the transformer mechanism, which includes self-attention and positional encoding to effectively capture the context and dependencies in the input text.
- **GPT-2 Like Model:** Implements a GPT-2-like architecture for generating high-quality text responses.
- **Custom Implementation:** Entire model and training pipeline are implemented from scratch in Python, providing flexibility and a deep understanding of the underlying processes.
- **Tokenization:** Uses `tiktoken` for efficient and effective tokenization of the input text data.

## Prerequisites

- Python 3.8 or higher
- PyTorch
- pandas
- tiktoken
- Other dependencies listed in `requirements.txt`

## Datasets

The project uses several datasets for training the model:

- **Customer support data:** https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
- **ChatGPT conversations:** https://www.kaggle.com/datasets/noahpersaud/89k-chatgpt-conversations
- **SQuAD dataset:** https://www.kaggle.com/datasets/stanfordu/stanford-question-answering-dataset
- **QuaC dataset:** https://www.kaggle.com/datasets/jeromeblanchet/quac-question-answering-in-context-dataset
- **Facebook Chat dataset:** https://www.kaggle.com/datasets/atharvjairath/personachat
- **Human Conversations:** https://www.kaggle.com/datasets/projjal1/human-conversation-training-data
- **Conversation dataset:** https://www.kaggle.com/datasets/kreeshrajani/3k-conversations-dataset-for-chatbot

## Installation

Clone the repository:

git clone https://github.com/yourusername/LLM-from-Scratch.git
cd LLM-from-Scratch

## References

- **GPT-2:** https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3752s
- **Tiktoken tokenizer:** https://www.youtube.com/watch?v=zduSFxRajkE&t=5687s