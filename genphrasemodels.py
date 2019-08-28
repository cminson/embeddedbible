#
# Author: Christopher Minson 
# www.christopherminson.com
# 
#
import sys
import os
import json
import string
import textinput

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity

URL_SENTENCE_ENCODER = "https://tfhub.dev/google/universal-sentence-encoder/2"
BIBLE_TXT = './TEXT/bible.txt'
STOP_WORDS = './TEXT/STOPWORDS.txt'
BIBLE_JSON= './CONFIG/bible.json'
MODEL_PATH = './MODELS/'
MODEL_WORDS = 'model.words'
MODEL_SENTENCES = 'model.sentences'
MODEL_CHAPTERS = 'model.chapters'
MODEL_BOOKS = 'model.books'


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
    print(book_content)
    embed = hub.Module(URL_SENTENCE_ENCODER)
    with tf.compat.v1.Session() as session:

        session.run([tf.compat.v1.global_variables_initializer(),  tf.compat.v1.tables_initializer()])
        embeddings = session.run(embed(book_content))
    print('embedding complete', embeddings.shape)

    print('computing similarity matrix')
    similarity_matrix = cosine_similarity(np.array(embeddings))
    print(similarity_matrix)

    path = MODEL_PATH + MODEL_BOOKS 
    print(f'Saving book model: {path}')
    np.save(path, similarity_matrix)

def get_top_similar(citation_list, stopped_sentence, stopped_sentence_list, similarity_matrix, topN):

    index = stopped_sentence_list.index(stopped_sentence)
    similarity_row = np.array(similarity_matrix[index, :])

    indices = similarity_row.argsort()[-topN:][::-1]
    matches =  [citation_list[i] for i in indices]
    scores = [similarity_row[i] for i in indices]
    return list(zip(matches, scores))

def get_top_different(citation_list, stopped_sentence, stopped_sentence_list, similarity_matrix, topN):

    index = stopped_sentence_list.index(stopped_sentence)
    similarity_row = np.array(similarity_matrix[index, :])

    indices = similarity_row.argsort()[0:topN]
    return [citation_list[i] for i in indices]

def save_bible_json(sentence_matrix):

    with open(BIBLE_JSON, 'w+') as fd:
        fd.write("{\n")

        for book_index, book in enumerate(text_input.ALL_BOOKS):

            verses = textinput.AllBooks[book]
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

                list_similar = get_top_similar(textinput.AllCitations, stopped_sentence, textinput.AllStoppedSentences, sentence_matrix, 11)
                list_similar.pop(0)
                similar_verses = []

                for similar in list_similar:

                    terms = similar[0].split(' ')
                    score = str(int(similar[1]*100))

                    citation =  terms.pop(-1)
                    book = ' '.join(terms)

                    similar_book_index = textinput.ALL_BOOKS.index(book)
                    similar_verses.append(str(similar_book_index) + " " + citation + " " + score)

                fd.write("\t\"s\": \"" + ','.join(similar_verses) + "\"\n")

                if verse_index  != (len(verses) - 1):
                    fd.write("},\n")
                else:
                    fd.write("}\n")

            if book_index  != (len(text_input.ALL_BOOKS) - 1):
                fd.write("],\n")
            else:
                fd.write("]\n")

        fd.write("}\n")


#os.environ["TFHUB_CACHE_DIR"] = ''

if __name__ == '__main__':

    textinput.load_bible()

    build_sentence_model(textinput.AllStoppedSentences)
    build_book_model(textinput.BookSentencesList)

    print('done')
