
word2vec_model = KeyedVectors.load_word2vec_format(r'D:\ML_Projects\Vehicle_Insurance_Claims_Prediction\models\GoogleNews-vectors-negative300.bin', binary=True)
vocab_size = word2vec_model.vector_size  # Size of Word2Vec embeddings

def get_word_vector(word):
    try:
        return word2vec_model[word]
    except KeyError:
        return np.zeros(word2vec_model.vector_size)

def pad_or_truncate(sequence, max_length):
    if len(sequence) > max_length:
        return sequence[:max_length]
    elif len(sequence) < max_length:
        padding = [np.zeros(word2vec_model.vector_size)] * (max_length - len(sequence))
        return sequence + padding
    return sequence

def preprocess_data(data):
    questions = []
    answers = []
    for i in range(len(data)):
        try:
            question = data['Question'][i].split()
            answer = data['Answer'][i].split()
        except:
            continue
        question_vectors = [get_word_vector(w) for w in question]
        answer_vectors = [get_word_vector(w) for w in answer]
        questions.append(pad_or_truncate(question_vectors, block_size))
        answers.append(pad_or_truncate(answer_vectors, block_size))
    return np.array(questions), np.array(answers)

# Load the question-answer dataset
qa_data = pd.read_csv(r'D:\ML_Projects\AI_Tech_ChatBot\Data\ChatGPT_chatlogs\GPT_chatlogs_all_filtered_Q50_A50_10K.csv')
questions, answers = preprocess_data(qa_data)

print('questions[0][0]: ', len(questions[0][0]))
print('questions[0]: ', len(questions[0]))
print('questions: ', len(questions))

#def get_batch(split):
#    data_len = len(questions) if split == 'train' else len(answers)
#    
#    ix = torch.randint(data_len, (batch_size,))
#    x = torch.tensor(questions[ix])
#    y = torch.tensor(answers[ix])
#    x, y = x.to(device), y.to(device)
#    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
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
            torch.save(model, os.path.join('path_to_your_model_checkpoints', f'model_temp_{iter}_loss-{losses["val"]:.3f}.pth'))

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model.forward(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print(loss.item())

    torch.save(model, os.path.join('path_to_your_model_checkpoints', f'model-06_loss-{loss.item():.3f}.pth'))
