import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import joblib

word2vec_model = KeyedVectors.load_word2vec_format(r'D:\ML_Projects\Vehicle_Insurance_Claims_Prediction\models\GoogleNews-vectors-negative300.bin', binary=True)

def get_word_vector(word):
    try:
        return word2vec_model[word]
    except KeyError:
        return np.zeros(300)

#vocab = joblib.load(r'D:\ML_Projects\AI_Tech_ChatBot\Data\ChatGPT_chatlogs\GPT_Vocab_all.joblib')
#print(get_word_vector(np.random.randint(0, len(vocab))))

gpt_data = pd.read_csv(r'D:\ML_Projects\AI_Tech_ChatBot\Data\ChatGPT_chatlogs\GPT_chatlogs_all.csv')

vec = []
#print_lim = 0
for i in range(len(gpt_data)):
#    if print_lim == 2:
#        break
    words = gpt_data['Question'][i].split()
    vec.append([get_word_vector(w) for w in words])
    #list(map(lambda w: get_word_vector(w), words))
#    print_lim+=1

for i in range(len(gpt_data)):
    for j in vec[i]:
        print(word2vec_model.similar_by_vector(j, topn=1)[0][0])

#for i in range(len(gpt_data)):
#    for j in range(len(gpt_data['Question'][i].split())):
#        print(word2vec_model.similar_by_vector(vec[i][j], topn=1)[0][0])

#for j in range(len(gpt_data)):
#    print("actual word: ", gpt_data['Question'][j])
#    print("Predicted word: ")
#    for k in get_word_vector(gpt_data['Question'][i].split()[1]):
#        print("k:", k)
        #print(word2vec_model.similar_by_vector(k, topn=1))
