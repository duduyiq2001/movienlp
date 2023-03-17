
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from pathlib import Path
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.preprocessing import StandardScaler
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
posdocuments = []
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
for p in Path('../aclimdb/test/pos').glob('*.txt'):
    print("counter at " , counter1)
    print(f"{p.name}")
    file1 = open('../aclimdb/test/pos/'+p.name, "r")
    try:
        thecomment = preprocess(file1.readlines()[0])
        posdocuments.append(thecomment)
        #print(negdocuments)
        counter1 += 1
    except UnicodeDecodeError:
        print(f"Error: {p.name} cannot be decoded.")
        counter1 += 1

        file1.close()

print(negdocuments[0:10])
# Create a Vectorizer Object
'''
vectorizer = CountVectorizer(max_features=500)

vectorizer.fit(negdocuments+posdocuments)
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)
    
'''
with open('negdoctest.pkl', 'wb') as file:
    pickle.dump(negdocuments, file)
with open('posdoctest.pkl', 'wb') as file:
    pickle.dump(posdocuments, file)
# Printing the identified Unique words along with their indices
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

print("Vocabulary: ", vectorizer.vocabulary_)

# Encode the Document
# Encode the Document
vector = vectorizer.transform(negdocuments)

# Summarizing the Encoded Texts
print("Encoded Document is:")
print(vector.toarray()[0:20])
#scale = StandardScaler.fit(vector.toarray())
#negtransformed = scale.transform(vector.toarray())
negtensor = tf.convert_to_tensor(vector.toarray(), dtype=tf.int32)
vector1 = vectorizer.transform(posdocuments)

# Summarizing the Encoded Texts
print("Encoded Document is:")
print(vector1.toarray()[0:20])
#postransformed = scale.transform(vector1.toarray())
postensor = tf.convert_to_tensor(vector1.toarray(), dtype=tf.int32)
tf.io.write_file("postensortestbow.txt", tf.io.serialize_tensor(postensor))
tf.io.write_file("negtensortestbow.txt", tf.io.serialize_tensor(negtensor))
