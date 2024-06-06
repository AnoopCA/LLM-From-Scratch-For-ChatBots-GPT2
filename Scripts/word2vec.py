import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

word2vec_model = KeyedVectors.load_word2vec_format(r'D:\ML_Projects\Vehicle_Insurance_Claims_Prediction\models\GoogleNews-vectors-negative300.bin', binary=True)

def get_word_vector(word):
    try:
        return word2vec_model[word][:5]  # Extract first 5 dimensions of the word vector
    except KeyError:
        return np.zeros(5)  # Return zero vector if word is not found in the Word2Vec model

data = x_train
Cat_list = ['Cat_1-4', 'Cat_5-8', 'Cat_9-12']

for i in Cat_list:
    word_vector_cols = [i + f'_w2v_{n}' for n in range(1, 6)]
    data[word_vector_cols] = pd.DataFrame(data[i].apply(lambda word: get_word_vector(word)).tolist(), index=data.index)
