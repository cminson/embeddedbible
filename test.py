import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

# Importing sklearn and TSNE.
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans

# We'll hack a bit with the t-SNE code in sklearn.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)
#from sklearn.utils.extmath import _ravel

RS = 25111993
BOOKS = np.asarray([
'Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy', 'Joshua', 'Judges', 'Ruth',
'1 Samuel', '2 Samuel', '1 Kings', '2 Kings', '1 Chronicles', '2 Chronicles', 'Ezra', 'Nehemiah',
'Esther', 'Job', 'Psalms', 'Proverbs', 'Ecclesiastes', 'Song of Solomon', 'Isaiah', 'Jeremiah',
'Lamentations', 'Ezekiel', 'Daniel', 'Hosea', 'Joel', 'Amos', 'Obadiah', 'Jonah',
'Micah', 'Nahum', 'Habakkuk', 'Zephaniah', 'Haggai', 'Zechariah', 'Malachi', 'Matthew',
'Mark', 'Luke', 'John', 'Acts', 'Romans', '1 Corinthians', '2 Corinthians', 'Galatians',
'Ephesians', 'Phillippians', 'Colossians', '1 Thessalonians', '2 Thessoalonians', '1 Timothy', '2 Timothy', 'Titus',
'Philemon', 'Hebrews', 'James', '1 Peter', '2 Peter', '1 John', '2 John', '3 John', 'Jude',
'Revelation'])

print(BOOKS)
print('BOOKS: ', len(BOOKS))

# Importing matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

# Importing seaborn to make nice plots.
import seaborn as sns
"""
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
"""

matrix  = np.load('./MODELS/model.books.npy')
coords = TSNE(random_state=RS).fit_transform(matrix)
print(coords)
x = []
y = []
for coord in coords:
    x.append(coord[0])
    y.append(coord[1])

rng = np.random.RandomState(0)
#x = rng.randn(100)
#y = rng.randn(100)
#colors = rng.rand(100)
sizes = 1000 * rng.rand(100)

colors = []
sizes = []
for i in range(66):
    """
    if i < 49: 
        colors.append(0xff0000)
    else: 
        colors.append(0x0000ff)
    """
    colors.append(0x00ff00)
    sizes.append(500)

plt.scatter(x, y, c=colors, s=sizes, alpha=0.3, cmap='viridis')
#plt.colorbar();  # show color scale
plt.savefig('z2.png', dpi=120)

