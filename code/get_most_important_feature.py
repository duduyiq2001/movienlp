import pickle
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)
alist = list(vectorizer.vocabulary_.keys())
mostimportant = [257, 94, 432, 177, 414, 346, 224, 43, 221, 434, 27, 243, 11, 290, 417, 258, 59, 24, 361, 369, 244, 52, 254, 356, 198, 398, 431, 57, 34, 311, 205, 307, 90, 166, 350, 228, 92, 120, 136, 499, 77, 292, 201, 299, 65, 481, 151, 112, 111, 46]
print([alist[k] for k in mostimportant])
data_str = """
data:0.0
...
data:0.99
"""

