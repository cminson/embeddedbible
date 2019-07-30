import os
import numpy as np
import matplotlib.pyplot 
import sklearn.manifold

RS = 25111993
CHART_PATH = './CHARTS/'
MODEL_PATH = './MODELS/'

BOOKS_OLD_TESTAMENT = [
'Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy', 'Joshua', 'Judges', 'Ruth',
'1 Samuel', '2 Samuel', '1 Kings', '2 Kings', '1 Chronicles', '2 Chronicles', 'Ezra', 'Nehemiah',
'Esther', 'Job', 'Psalms', 'Proverbs', 'Ecclesiastes', 'Song of Solomon', 'Isaiah', 'Jeremiah',
'Lamentations', 'Ezekiel', 'Daniel', 'Hosea', 'Joel', 'Amos', 'Obadiah', 'Jonah',
'Micah', 'Nahum', 'Habakkuk', 'Zephaniah', 'Haggai', 'Zechariah', 'Malachi'
]
BOOKS_NEW_TESTAMENT = [
'Matthew', 'Mark', 'Luke', 'John', 'Acts', 'Romans', '1 Corinthians', '2 Corinthians', 
'Galatians', 'Ephesians', 'Phillippians', 'Colossians', '1 Thessalonians', '2 Thessoalonians', '1 Timothy', '2 Timothy', 
'Titus', 'Philemon', 'Hebrews', 'James', '1 Peter', '2 Peter', '1 John', '2 John', 
'3 John', 'Jude', 'Revelation'
]

ALL_BOOKS = BOOKS_OLD_TESTAMENT + BOOKS_NEW_TESTAMENT


def plot_similar_books(matrix, chart_name):

    coordinates2D = sklearn.manifold.TSNE(random_state=RS).fit_transform(matrix)
    x = []
    y = []
    for coordinate2D in coordinates2D:
        x.append(coordinate2D[0])
        y.append(coordinate2D[1])

    colors = []
    sizes = []

    for book in ALL_BOOKS:
        if book in BOOKS_OLD_TESTAMENT: color = 'red'
        if book in BOOKS_NEW_TESTAMENT: color = 'green'

        colors.append(color)
        sizes.append(1500)

    matplotlib.pyplot.scatter(x, y, s=sizes, alpha=0.6, color=colors)

    for book_index, book_name in enumerate(ALL_BOOKS):
        coordinate2D = coordinates2D[book_index]
        matplotlib.pyplot.text(coordinate2D[0], coordinate2D[1], book_name, fontsize=6, ha='center', va='center')

    matplotlib.pyplot.savefig(chart_name, dpi=120)


if __name__ == '__main__':

    matrix  = np.load(MODEL_PATH + 'model.books.npy')
    plot_similar_books(matrix, CHART_PATH + 'book_similarity.png')
