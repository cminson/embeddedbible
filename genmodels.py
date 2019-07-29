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
WORD2VEC_SIZE = 500
WORD2VEC_MINWORD_COUNT = 5

TextCitation = []
TextWords = []
TextSentences = []
TextBooks = []

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


def build_word_models(text_content):

    print('building word models')
    for window_value in range(1, MAX_WORD2VEC_WINDOW):

        model = gensim.models.Word2Vec(text_content, 
            min_count = WORD2VEC_MINWORD_COUNT, 
            size = WORD2VEC_SIZE, 
            window = window_value,
            sg = WORD2VEC_SG)

        path = MODEL_PATH + MODEL_WORDS + '.' + str(window_value)
        print(f'Saving word model: {path}')
        model.save(path)

def build_sentence_model(text_content):

    print('computing sentence embeddings')
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

def build_book_model(text_content):

    print('computing book embeddings')
    embed = hub.Module(URL_SENTENCE_ENCODER)
    with tf.compat.v1.Session() as session:

        session.run([tf.compat.v1.global_variables_initializer(),  tf.compat.v1.tables_initializer()])
        embeddings = session.run(embed(text_content))
    print('embedding complete')

    print('computing similarity matrix')
    similarity_matrix = cosine_similarity(np.array(embeddings))

    path = MODEL_PATH + MODEL_BOOKS 
    print(f'Saving book model: {path}')
    np.save(path, similarity_matrix)


#os.environ["TFHUB_CACHE_DIR"] = ''

if __name__ == '__main__':

    process_text(TEXT_PATH, STOP_WORDS_PATH)

    build_word_models(TextWords)
    build_sentence_model(TextSentences)
    build_book_model(TextBooks)
    print('done')
