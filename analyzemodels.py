#
# Author: Christopher Minson 
# www.christopherminson.com
# 
#
import tensorflow as tf
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity

import gensim
import numpy as np
import textinput

MODEL_PATH = './MODELS/'

def cos_sim(input_vectors):
    similarity = cosine_similarity(input_vectors)
    return similarity

def get_top_similar(sentence_list, stopped_sentence, stopped_sentence_list, similarity_matrix, topN):

    index = stopped_sentence_list.index(stopped_sentence)
    similarity_row = np.array(similarity_matrix[index, :])
    
    indices = similarity_row.argsort()[-topN:][::-1]
    matches =  [sentence_list[i] for i in indices]
    scores = [similarity_row[i] for i in indices]
    return list(zip(matches, scores))

def get_top_different(sentence_list, stopped_sentence, stopped_sentence_list, similarity_matrix, topN):

    index = stopped_sentence_list.index(stopped_sentence)
    similarity_row = np.array(similarity_matrix[index, :])
    
    indices = similarity_row.argsort()[0:topN]
    return [sentence_list[i] for i in indices]

#
#
#
if __name__ == '__main__':


    textinput.load_bible()

    model = gensim.models.Word2Vec.load("./MODELS/model.words.10")

    result = model.wv.most_similar(positive=['god'], topn=10)
    print('god\n', result)
    result = model.wv.most_similar(positive=['jesus'], topn=10)
    print('jesus\n', result)
    result = model.most_similar(positive=['god'], negative=['jesus'], topn=20)
    print('god - jesus\n', result)
    result = model.most_similar(positive=['jesus'], negative=['god'], topn=20)
    print('jesus - god\n', result)
    result = model.wv.most_similar(positive=['mary'], topn=10)
    print('mary\n', result)
    result = model.wv.most_similar(positive=['noah', 'jesus'], topn=10)
    print('noah + jesus\n', result)
    result = model.wv.most_similar(positive=['forest', 'war'], topn=10)
    print('forest + war\n', result)

    sentence_matrix = np.load('./MODELS/model.sentences.npy')
    result = get_top_similar(textinput.AllSentences, textinput.AllStoppedSentences[0], textinput.AllStoppedSentences, sentence_matrix, 3)
    print(result)

