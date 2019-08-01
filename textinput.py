import sys
import os
import time
import nltk

TEXT_PATH = './TEXT/bible.txt'
STOP_WORDS_PATH = './TEXT/STOPWORDS.txt'

PRASES = [ 'holy ghost', 'holy spirit']

AllCitations = []
AllWords = []
AllWordsWithCitations = []
AllSentences = []
AllStoppedSentences = []

BookStoppedSentencesDict = {}
BookStoppedSentencesList = []

def load_bible():

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
        AllCitations.append(citation)
        AllSentences.append(sentence)

        stopped_words = [w.lower() for w in nltk.tokenize.word_tokenize(sentence) if not w in stop_words and len(w) > 2]

        stopped_sentence = ' '.join(stopped_words)
        AllStoppedSentences.append(stopped_sentence)

        citation_parts = citation.split(' ')
        del citation_parts[-1]
        book_name = ' '.join(citation_parts)
        #stopped_words.insert(0, book_name)
        AllWordsWithCitations.append(stopped_words)
        AllWords.append(stopped_words)

        if book_name in BookStoppedSentencesDict:
            BookStoppedSentencesDict[book_name] += stopped_sentence
            BookStoppedSentencesDict[book_name] += ' '
        else:
            BookStoppedSentencesDict[book_name] = stopped_sentence
            BookStoppedSentencesDict[book_name] += ' '

    for book, stopped_sentences in BookStoppedSentencesDict.items():
        BookStoppedSentencesList.append(stopped_sentences)

    print('text load complete')


if __name__ == '__main__':
    
    process_text(TEXT_PATH, STOP_WORDS_PATH)

