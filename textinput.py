import sys
import os
import time
import json
import nltk

CONFIG_PATH = './CONFIGS/'
TEXT_PATH = './TEXT/bible.txt'
STOP_WORDS_PATH = './TEXT/STOPWORDS.txt'


AllBooks = {}
AllWords = []
AllWordsWithCitations = []
AllSentences = []
AllStoppedSentences = []
AllCitations = []

BookStoppedSentencesDict = {}
BookStoppedSentencesList = []


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


def gen_config(config_dict, file_name):

    pass

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

        #citation, sentence = line.lower().strip().replace("\n", " ").split('\t')
        citation, raw_sentence = line.replace('\n', '').split('\t')
        lowered_sentence = raw_sentence.lower().strip()
        citation_parts = citation.split(' ')
        citation = citation_parts[-1]
        del citation_parts[-1]
        book = ' '.join(citation_parts).replace(' ','').lower()

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
    
    load_bible()

    file_path = CONFIG_PATH + 'bible.json'
    with open(file_path, 'w') as fd:
        fd.write("{\n")

        for book_index, book in enumerate(ALL_BOOKS):

            book = book.replace(' ','').lower()
            verses = AllBooks[book]
            fd.write("\"{}\": [\n".format(book))


            count_verses = 0
            for verse_index, verse in enumerate(verses):

                citation = verse[0]
                sentence = verse[1]
                stopped_sentence = verse[2]

                fd.write("{\n")
                fd.write("\t\"citation\": \"" + citation + "\",\n")
                #fd.write("\t\"citation\": \"{}"\",\n".format(citation))
                fd.write("\t\"text\": \"" + sentence + "\",\n")
                fd.write("\t\"stopped_text\": \"" + stopped_sentence + "\"\n")

                if verse_index  != (len(verses) - 1):
                    fd.write("},\n")
                else:
                    fd.write("}\n")

            if book_index  != (len(ALL_BOOKS) - 1):
                fd.write("],\n")
            else:
                fd.write("]\n")

        fd.write("}\n")
        


