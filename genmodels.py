import sys
import os
import time
import json
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import gensim


BOOKS_OLD_TESTAMENT = [
'Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy', 'Joshua', 'Judges', 'Ruth',
'1 Samuel', '2 Samuel', '1 Kings', '2 Kings', '1 Chronicles', '2 Chronicles', 'Ezra', 'Nehemiah',
'Esther', 'Job', 'Psalms', 'Proverbs', 'Ecclesiastes', 'Song of Solomon', 'Isaiah', 'Jeremiah',
'Lamentations', 'Ezekiel', 'Daniel', 'Hosea', 'Joel', 'Amos', 'Obadiah', 'Jonah',
'Micah', 'Nahum', 'Habakkuk', 'Zephaniah', 'Haggai', 'Zechariah', 'Malachi'
]
BOOKS_NEW_TESTAMENT = [
'Matthew', 'Mark', 'Luke', 'John', 'Acts', 'Romans', '1 Corinthians', '2 Corinthians',
'Galatians', 'Ephesians', 'Philippians', 'Colossians', '1 Thessalonians', '2 Thessalonians', '1 Timothy', '2 Timothy',
'Titus', 'Philemon', 'Hebrews', 'James', '1 Peter', '2 Peter', '1 John', '2 John',
'3 John', 'Jude', 'Revelation'
]
ALL_BOOKS = BOOKS_OLD_TESTAMENT + BOOKS_NEW_TESTAMENT


URL_SENTENCE_ENCODER = "https://tfhub.dev/google/universal-sentence-encoder/2"

BIBLE_TXT = './TEXT/bible.txt'
STOP_WORDS = './TEXT/STOPWORDS.txt'
BIBLE_JSON= './CONFIG/bible.json'

MODEL_PATH = './MODELS/'

MODEL_WORDS = 'model.words'
MODEL_SENTENCES = 'model.sentences'
MODEL_CHAPTERS = 'model.chapters'
MODEL_BOOKS = 'model.books'

MAX_WORD2VEC_WINDOW = 10
WORD2VEC_SG = 1
WORD2VEC_SIZE = 300
WORD2VEC_MINWORD_COUNT = 5

AllBooks = {}
AllStoppedWords = []
AllSentences = []
AllStoppedSentences = []
AllCitations = []

BookStoppedSentencesDict = {}
BookStoppedSentencesList = []

def load_bible():

    print('loading text ...')
    # get all the stop words
    stop_words = set(nltk.corpus.stopwords.words('english'))
    with open(STOP_WORDS, 'r') as fd:
        while True:
            word = fd.readline().strip().lower()
            if not word: break;
            stop_words.add(word)

    # read in data
    with open(BIBLE_TXT, 'r') as fd:
        lines = fd.readlines()

    for line in lines:

        citation, raw_sentence = line.replace('\n', '').split('\t')
        lowered_sentence = raw_sentence.lower().strip()

        table = lowered_sentence.maketrans(dict.fromkeys(string.punctuation))
        lowered_sentence = lowered_sentence.translate(table)
        #lowered_sentence = re.sub(r'[^\w\s]','',lowered_sentence)
        citation_parts = citation.split(' ')
        citation = citation_parts[-1]
        del citation_parts[-1]
        #book = ' '.join(citation_parts).replace(' ','').lower()
        book = ' '.join(citation_parts)

        AllSentences.append(lowered_sentence)

        stopped_words = [w.lower() for w in nltk.tokenize.word_tokenize(lowered_sentence) if not w in stop_words and len(w) > 2]

        stopped_sentence = ' '.join(stopped_words)
        AllStoppedSentences.append(stopped_sentence)
        AllCitations.append(book + ' ' + citation)

        if book in AllBooks:
            AllBooks[book].append((citation, raw_sentence, stopped_sentence))
        else:
            AllBooks[book] = [(citation, raw_sentence, stopped_sentence)]


        citation_parts = citation.split(' ')
        del citation_parts[-1]
        book_name = ' '.join(citation_parts)
        AllStoppedWords.append(stopped_words)

        if book_name in BookStoppedSentencesDict:
            BookStoppedSentencesDict[book_name] += stopped_sentence
            BookStoppedSentencesDict[book_name] += ' '
        else:
            BookStoppedSentencesDict[book_name] = stopped_sentence
            BookStoppedSentencesDict[book_name] += ' '

    for book, stopped_sentences in BookStoppedSentencesDict.items():
        BookStoppedSentencesList.append(stopped_sentences)

    print('text load complete')

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

def build_sentence_model(sentence_list):

    print('computing sentence embeddings')
    print(sentence_list)
    embed = hub.Module(URL_SENTENCE_ENCODER)
    with tf.compat.v1.Session() as session:

        session.run([tf.compat.v1.global_variables_initializer(),  tf.compat.v1.tables_initializer()])
        embeddings = session.run(embed(sentence_list))
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

def get_top_similar(citation_list, stopped_sentence, stopped_sentence_list, similarity_matrix, topN):

    index = stopped_sentence_list.index(stopped_sentence)
    similarity_row = np.array(similarity_matrix[index, :])

    indices = similarity_row.argsort()[-topN:][::-1]
    return [citation_list[i] for i in indices]

def get_top_different(citation_list, stopped_sentence, stopped_sentence_list, similarity_matrix, topN):

    index = stopped_sentence_list.index(stopped_sentence)
    similarity_row = np.array(similarity_matrix[index, :])

    #indices = similarity_row.argsort()[topN:][::-1]
    indices = similarity_row.argsort()[0:topN]
    return [citation_list[i] for i in indices]

def save_bible_json(sentence_matrix):

    with open(BIBLE_JSON, 'w+') as fd:
        fd.write("{\n")

        for book_index, book in enumerate(ALL_BOOKS):

            verses = AllBooks[book]
            fd.write("\"{}\": [\n".format(book))


            count_verses = 0
            for verse_index, verse in enumerate(verses):

                citation = verse[0]
                sentence = verse[1]
                stopped_sentence = verse[2]
                fd.write("{\n")
                fd.write("\t\"c\": \"" + citation + "\",\n")
                fd.write("\t\"t\": \"" + sentence + "\",\n")
                fd.write("\t\"st\": \"" + stopped_sentence + "\",\n")

                list_similar = get_top_similar(AllCitations, stopped_sentence, AllStoppedSentences, sentence_matrix, 11)
                list_similar.pop(0)
                similar_verses = []
                for similar in list_similar:
                    terms = similar.split(' ')
                    citation =  terms.pop(-1)
                    book = ' '.join(terms)
                    similar_book_index = ALL_BOOKS.index(book)
                    similar_verses.append(str(similar_book_index) + " " + citation)
                fd.write("\t\"s\": \"" + ','.join(similar_verses) + "\",\n")

                list_different = get_top_different(AllCitations, stopped_sentence, AllStoppedSentences, sentence_matrix, 10)
                different_verses = []
                for different in list_different:
                    terms = different.split(' ')
                    citation =  terms.pop(-1)
                    book = ' '.join(terms)
                    different_book_index = ALL_BOOKS.index(book)
                    different_verses.append(str(different_book_index) + " " + citation)
                fd.write("\t\"d\": \"" + ','.join(different_verses) + "\"\n")

                if verse_index  != (len(verses) - 1):
                    fd.write("},\n")
                else:
                    fd.write("}\n")

            if book_index  != (len(ALL_BOOKS) - 1):
                fd.write("],\n")
            else:
                fd.write("]\n")

        fd.write("}\n")


#os.environ["TFHUB_CACHE_DIR"] = ''

if __name__ == '__main__':

    load_bible()

    build_word_model(AllStoppedWords)
    #build_sentence_model(AllStoppedSentences)
    #build_book_model(BookStoppedSentencesList)

    sentence_matrix = np.load(MODEL_PATH + 'model.sentences.npy')
    save_bible_json(sentence_matrix)

    with open(BIBLE_JSON,'r') as fd:
        json_data  = json.load(fd)
        print('json validated')

    print('done')
