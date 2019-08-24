import sys
import os
import textinput
import gensim

MODEL_PATH = './MODELS/'
MODEL_WORDS = 'model.words'

MAX_WORD2VEC_WINDOW = 10
WORD2VEC_SG = 1
WORD2VEC_SIZE = 300
WORD2VEC_MINWORD_COUNT = 5

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


if __name__ == '__main__':

    textinput.load_bible()
    build_word_model(textinput.AllStoppedWords)

    print('done')
