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

def process_text(text_path, stop_words_path):

    book_dict = {}

    # get all the stop words
    stop_words = set(stopwords.words('english'))
    with open(stop_words_path, 'r') as fd:
        while True:
            word = fd.readline().strip().lower()
            if not word: break;
            stop_words.add(word)

    # read in data
    with open(text_path, 'r') as fd:
        lines = fd.readlines()


    for line in lines:

        citation, sentence = line.lower().strip().replace("\n", " ").split('\t')
        TextCitation.append(citation)

        sentence = [w.lower() for w in word_tokenize(sentence) if not w in stop_words and len(w) > 2]
        TextWords.append(sentence)
        sentence = ' '.join(sentence)
        TextSentences.append(sentence)

        citation_parts = citation.split(' ')
        del citation_parts[-1]
        book_name = ' '.join(citation_parts)

        if book_name in book_dict:
            book_dict[book_name] += sentence
        else:
            book_dict[book_name] = sentence

    for book, content in book_dict.items():
        print(book, content)
        TextBooks.append(content)

def cos_sim(input_vectors):
    similarity = cosine_similarity(input_vectors)
    return similarity

def get_top_similar(sentence_list, stopped_sentence, stopped_sentence_list, similarity_matrix, topN):

    index = stopped_sentence_list.index(stopped_sentence)
    similarity_row = np.array(similarity_matrix[index, :])

    indices = similarity_row.argsort()[-topN:][::-1]
    return [sentence_list[i] for i in indices]

#
#
#
if __name__ == '__main__':

    textinput.load_bible()

    model = gensim.models.Word2Vec.load("./MODELS/model.words.5")
    result = model.most_similar(positive=['god'])
    print(result)

    sentence_matrix = np.load('./MODELS/model.sentences.npy')
    result = get_top_similar(textinput.Sentences, textinput.StoppedSentences[0], textinput.StoppedSentences, sentence_matrix, 3)
    print(result)



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
