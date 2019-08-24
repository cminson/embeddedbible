import sys
import os
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

BIBLE_TXT = './TEXT/bible.txt'
STOP_WORDS = './TEXT/STOPWORDS.txt'
MIN_WORD_SIZE = 2

AllBooks = {}
AllStoppedWords = []
AllSentences = []
AllStoppedSentences = []
AllCitations = []
BookSentencesDict = {}
BookSentencesList = []

def load_bible():

    print('loading text ...')

    # get all the stop words
    #stop_words = set(nltk.corpus.stopwords.words('english'))
    stop_words = set(stopwords.words('english'))
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
        citation_parts = citation.split(' ')
        citation = citation_parts[-1]
        del citation_parts[-1]
        book = ' '.join(citation_parts)

        AllSentences.append(lowered_sentence)

        stopped_words = [w.lower() for w in word_tokenize(lowered_sentence) if not w in stop_words and len(w) > MIN_WORD_SIZE]
        AllStoppedWords.append(stopped_words)

        stopped_sentence = ' '.join(stopped_words)
        AllStoppedSentences.append(stopped_sentence)
        AllCitations.append(book + ' ' + citation)

        if book in AllBooks:
            AllBooks[book].append((citation, raw_sentence, stopped_sentence))
        else:
            AllBooks[book] = [(citation, raw_sentence, stopped_sentence)]

        if book in BookSentencesDict:
            BookSentencesDict[book] += lowered_sentence
            BookSentencesDict[book] += ' '
        else:
            BookSentencesDict[book] = lowered_sentence
            BookSentencesDict[book] += ' '

    for book, sentences in BookSentencesDict.items():
        BookSentencesList.append(sentences)


if __name__ == '__main__':

    load_bible()
    print('done')
