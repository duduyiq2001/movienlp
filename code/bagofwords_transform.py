from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from pathlib import Path
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

def preprocess(text):
    text = re.sub('<.*?>', '', text)
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    words = nltk.word_tokenize(text)
    words = [w for w in words if w not in stopwords.words('english')]
    words = [PorterStemmer().stem(w) for w in words]
    text = ' '.join(words)
    return text
counter1 = 0
negdocuments = []
for p in Path('../aclimdb/test/neg').glob('*.txt'):
    print("counter at " , counter1)
    print(f"{p.name}")
    file1 = open('../aclimdb/test/neg/'+p.name, "r")
    try:
        thecomment = preprocess(file1.readlines()[0])
        negdocuments.append(thecomment)
        #print(negdocuments)
        counter1 += 1
    except UnicodeDecodeError:
        print(f"Error: {p.name} cannot be decoded.")
        counter1 += 1

        file1.close()
for p in Path('../aclimdb/test/neg').glob('*.txt'):
    print("counter at " , counter1)
    print(f"{p.name}")
    file1 = open('../aclimdb/test/neg/'+p.name, "r")
    try:
        thecomment = preprocess(file1.readlines()[0])
        negdocuments.append(thecomment)
        #print(negdocuments)
        counter1 += 1
    except UnicodeDecodeError:
        print(f"Error: {p.name} cannot be decoded.")
        counter1 += 1

        file1.close()

print(negdocuments[0:10])
# Create a Vectorizer Object
vectorizer = CountVectorizer(max_features=500)

vectorizer.fit(negdocuments)

# Printing the identified Unique words along with their indices
print("Vocabulary: ", vectorizer.vocabulary_)

# Encode the Document
# Encode the Document
vector = vectorizer.transform(negdocuments)

# Summarizing the Encoded Texts
print("Encoded Document is:")
print(vector.toarray()[0:20])

negtensor = tf.convert_to_tensor(vector.toarray(), dtype=tf.int32)

#tf.io.write_file("negtensortest.txt", tf.io.serialize_tensor(negtensor))
tf.io.write_file("negtensorbagofwordstest.txt", tf.io.serialize_tensor(negtensor))
