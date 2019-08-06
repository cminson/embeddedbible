import sys
import os
import time
import json
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_similarity

import gensim
import textinput


URL_SENTENCE_ENCODER = "https://tfhub.dev/google/universal-sentence-encoder/2"

MODEL_PATH = './MODELS/'
TEXT_PATH = './TEXT/bible.txt'
STOP_WORDS_PATH = './TEXT/STOPWORDS.txt'

MODEL_WORDS = 'model.words'
MODEL_SENTENCES = 'model.sentences'
MODEL_CHAPTERS = 'model.chapters'
MODEL_BOOKS = 'model.books'

MAX_WORD2VEC_WINDOW = 10
WORD2VEC_SG = 1
WORD2VEC_SIZE = 300
WORD2VEC_MINWORD_COUNT = 5


def build_wordwindow_models(word_list):

    print('building word models')
    for window_value in range(1, MAX_WORD2VEC_WINDOW):

        model = gensim.models.Word2Vec(word_list, 
            min_count = WORD2VEC_MINWORD_COUNT, 
            size = WORD2VEC_SIZE, 
            window = window_value,
            sg = WORD2VEC_SG)

        path = MODEL_PATH + MODEL_WORDS + '.' + str(window_value)
        print(f'Saving word model: {path}')
        model.save(path)

def build_word_model(word_list):

    print('building word models')

    model = gensim.models.Word2Vec(word_list, 
        min_count = WORD2VEC_MINWORD_COUNT, 
        size = WORD2VEC_SIZE, 
        window = MAX_WORD2VEC_WINDOW,
        sg = WORD2VEC_SG)

    path = MODEL_PATH + MODEL_WORDS + '.' + str(MAX_WORD2VEC_WINDOW)
    print(f'Saving word model: {path}')
    model.save(path)

def build_sentence_model(text_content):

    print('computing sentence embeddings')
    print(text_content)
    embed = hub.Module(URL_SENTENCE_ENCODER)
    with tf.compat.v1.Session() as session:

        session.run([tf.compat.v1.global_variables_initializer(),  tf.compat.v1.tables_initializer()])
        embeddings = session.run(embed(text_content))
    print('embedding complete')

    print('computing similarity matrix')
    similarity_matrix = cosine_similarity(np.array(embeddings))

    path = MODEL_PATH + MODEL_SENTENCES 
    print(f'Saving sentence model: {path}')
    np.save(path, similarity_matrix)

def build_book_model(book_content):

    print('computing book embeddings')
    embed = hub.Module(URL_SENTENCE_ENCODER)
    with tf.compat.v1.Session() as session:

        session.run([tf.compat.v1.global_variables_initializer(),  tf.compat.v1.tables_initializer()])
        embeddings = session.run(embed(book_content))
    print('embedding complete', embeddings.shape)

    print('computing similarity matrix')
    similarity_matrix = cosine_similarity(np.array(embeddings))

    path = MODEL_PATH + MODEL_BOOKS 
    print(f'Saving book model: {path}')
    np.save(path, similarity_matrix)



#os.environ["TFHUB_CACHE_DIR"] = ''

if __name__ == '__main__':

    textinput.load_bible()

    build_word_model(textinput.AllWordsWithCitations)
    build_sentence_model(textinput.AllStoppedSentences)
    build_book_model(textinput.BookStoppedSentencesList)

    print('done')
