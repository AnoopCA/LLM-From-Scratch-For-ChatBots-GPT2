import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

word2vec_model = KeyedVectors.load_word2vec_format(r'D:\ML_Projects\Vehicle_Insurance_Claims_Prediction\models\GoogleNews-vectors-negative300.bin', binary=True)

def get_word_vector(word):
    try:
        return word2vec_model[word]
    except KeyError:
        return np.zeros(word2vec_model.vector_size)

gpt_data = pd.read_csv(r'D:\ML_Projects\AI_Tech_ChatBot\Data\ChatGPT_chatlogs\GPT_chatlogs_all.csv')

vec = []
for i in range(len(gpt_data)):
    words = gpt_data['Question'][i].split()
    vec.append([get_word_vector(w) for w in words])

for i in range(len(gpt_data)):
    for j in vec[i]:
        print(word2vec_model.similar_by_vector(j, topn=1)[0][0])
