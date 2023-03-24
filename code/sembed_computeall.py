import tensorflow as tf
from sentence_transformers import SentenceTransformer
from pathlib import Path


# Enable GPU support
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
model = SentenceTransformer('all-MiniLM-L6-v2')




'''
counter1 = 0
for p in Path('./aclimdb/train/neg').glob('*.txt'):
    print("counter at " , counter1)
    print(f"{p.name}")
    file1 = open('./aclimdb/train/neg/'+p.name, "r")
    try:
        negative_embeds.append(list(model.encode(file1.readlines())[0]))
        counter1 += 1
    except UnicodeDecodeError:
        print(f"Error: {p.name} cannot be decoded.")
        counter1 += 1

        file1.close()



#iterate through all files in the pos dir
counter2 = 0
for p in Path('./aclimdb/train/pos').glob('*.txt'):
    try:
        print("counter at ", counter2)
        print(f"{p.name}")
        file1 = open('./aclimdb/train/pos/'+ p.name, "r")
        positive_embeds.append(list(model.encode(file1.readlines())[0]))
        file1.close()
        counter2 += 1
    except UnicodeDecodeError:
        print(f"Error: {p.name} cannot be decoded.")
        counter2 += 1
        file1.close()
        continue

#print(negative_embeds, "neg")
print(len(positive_embeds))
#convert all to tensors
#print(positive_embeds, "pos")
#negtensor = tf.convert_to_tensor(negative_embeds, dtype=tf.float32)
postensor = tf.convert_to_tensor(positive_embeds, dtype=tf.float32)

#tf.io.write_file("negtensor.txt", tf.io.serialize_tensor(negtensor))
tf.io.write_file("postensor.txt", tf.io.serialize_tensor(postensor))




negative_embeds = []
positive_embeds = []
'''
#negative_embeds = []
positive_embeds = []
'''
counter1 = 0
for p in Path('./aclimdb/test/neg').glob('*.txt'):
    print("counter at " , counter1)
    print(f"{p.name}")
    file1 = open('./aclimdb/test/neg/'+p.name, "r")
    try:
        negative_embeds.append(list(model.encode(file1.readlines())[0]))
        counter1 += 1
    except UnicodeDecodeError:
        print(f"Error: {p.name} cannot be decoded.")
        counter1 += 1

        file1.close()
'''


#iterate through all files in the pos dir

counter2 = 0
for p in Path('./aclimdb/test/pos').glob('*.txt'):
    try:
        print("counter at ", counter2)
        print(f"{p.name}")
        file1 = open('./aclimdb/test/pos/'+ p.name, "r")
        positive_embeds.append(list(model.encode(file1.readlines())[0]))
        file1.close()
        counter2 += 1
    except UnicodeDecodeError:
        print(f"Error: {p.name} cannot be decoded.")
        counter2 += 1
        file1.close()
        continue

#print(negative_embeds, "neg")
print(len(positive_embeds))
#convert all to tensors
#print(positive_embeds, "pos")
#negtensor = tf.convert_to_tensor(negative_embeds, dtype=tf.float32)
postensor = tf.convert_to_tensor(positive_embeds, dtype=tf.float32)

#tf.io.write_file("negtensortest.txt", tf.io.serialize_tensor(negtensor))
tf.io.write_file("postensortest.txt", tf.io.serialize_tensor(postensor))
