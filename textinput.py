import sys
import os
import time
import nltk

TEXT_PATH = './TEXT/bible.txt'
STOP_WORDS_PATH = './TEXT/STOPWORDS.txt'

Citations = []
Words = []
Sentences = []
StoppedSentences = []
Chapters = []
Books = []

def load_bible():

    book_dict = {}

    print('loading text ...')
    # get all the stop words
    stop_words = set(nltk.corpus.stopwords.words('english'))
    with open(STOP_WORDS_PATH, 'r') as fd:
        while True:
            word = fd.readline().strip().lower()
            if not word: break;
            stop_words.add(word)

    # read in data
    with open(TEXT_PATH, 'r') as fd:
        lines = fd.readlines()

    for line in lines:

        citation, sentence = line.lower().strip().replace("\n", " ").split('\t')
        Citations.append(citation)

        Sentences.append(sentence)
        sentence = [w.lower() for w in nltk.tokenize.word_tokenize(sentence) if not w in stop_words and len(w) > 2]
        Words.append(sentence)
        sentence = ' '.join(sentence)
        StoppedSentences.append(sentence)

        citation_parts = citation.split(' ')
        del citation_parts[-1]
        book_name = ' '.join(citation_parts)

        if book_name in book_dict:
            book_dict[book_name] += sentence
        else:
            book_dict[book_name] = sentence

    for book, content in book_dict.items():
        #print(book, content)
        Books.append(content)

    print('text load complete')


if __name__ == '__main__':
    
    process_text(TEXT_PATH, STOP_WORDS_PATH)

