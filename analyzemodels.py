import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import spacy
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
    return [sentence_list[i] for i in indices]

def get_top_different(sentence_list, stopped_sentence, stopped_sentence_list, similarity_matrix, topN):

    index = stopped_sentence_list.index(stopped_sentence)
    similarity_row = np.array(similarity_matrix[index, :])
    
    #indices = similarity_row.argsort()[topN:][::-1]
    indices = similarity_row.argsort()[0:topN]
    return [sentence_list[i] for i in indices]

#
#
#
if __name__ == '__main__':

    textinput.load_bible()

    model = gensim.models.Word2Vec.load("./MODELS/model.words.10")
    result = model.most_similar(positive=['jesus'])
    print('jesus', result)

    """
    result = model.most_similar(positive=['judas'])
    print('judas',result)
    result = model.most_similar(positive=['god'], negative=['jesus'])
    print('god - jesus', result)
    result = model.most_similar(positive=['jesus'], negative=['god'])
    print('jesus - god', result)
    result = model.most_similar(positive=['holy', 'ghost'])
    print('holy ghost', result)
    result = model.most_similar(positive=['mary'])
    print('mary', result)
    result = model.most_similar(positive=['matthew'])
    print('matthew', result)
    result = model.most_similar(positive=['matthew'], negative=['jesus'])
    print('matthew-jesus', result)

    result = model.most_similar(positive=['children'])
    print(result)
    result = model.most_similar(positive=['god'], negative=['jesus'])
    print(result)
    result = model.most_similar(positive=['jesus'], negative=['god'])
    print(result)
    """

    """
    sentence_matrix = np.load('./MODELS/model.sentences.npy')
    result = get_top_similar(textinput.Sentences, textinput.StoppedSentences[0], textinput.StoppedSentences, sentence_matrix, 3)
    print(result)
    result = get_top_different(textinput.Sentences, textinput.StoppedSentences[0], textinput.StoppedSentences, sentence_matrix, 3)
    print(result)

    """


"""
DIFFERENCES!

matrix = np.load('model.books.npy')
print(matrix)


top_similar = get_top_similar(sentence, sentences_list, similarity_matrix, 3)

# printing the list using loop
for x in range(len(top_similar)):
    print(top_similar[x])


#result = model.most_similar(positive=['jesus'], negative=['matthew'])
#print(result)
#print (list(model.wv.vocab))
"""

"""
print("similarity between god and jesus", model.similarity('god', 'mary'))
print(model.similar_by_vector(model['marriage'], topn=50))
print(model.similar_by_vector(model['god'], topn=50))
print(model.similar_by_vector(model['jesus'], topn=50))
print(model.similar_by_vector(model['magdalene'], topn=50))
print(model.similar_by_vector(model['god'] - model['jesus']))
"""
