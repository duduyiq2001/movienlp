import tensorflow as tf
from sentence_transformers import SentenceTransformer
# Enable GPU support
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
model = SentenceTransformer('all-MiniLM-L6-v2')
sentence = ["""Wasn't quite sure what to expect with this one, outside of the uniform positive reviews I've read. Turns out, I could h
            ave never imagined this movie, because it's very close to The Bride with White Hair in being 
            operatic and dealing with the fantastic. This walks a fine line 
            between being a farce, a comedy, and just plain good old fashion ghost story telling. 
            There's nothing scary about it, that's not the theme, it's really mostly a love story dealing 
            with a bumbling guy who encounters a beautiful ghost, who is in a lot of trouble with other ghosts. So the main 
            theme is the guy trying to save the beautiful ghost. This also takes place in ancient China, with wild 
            outlandish Kung Fu exhibitions, and a trip to hell (more or less). Some of the stop-action ghosts are pretty 
            cool, and the visual effects are top rate all the way. I could watch this genre of Chinese 
            movies all day, because they are highly entertaining, great visuals, and pretty much tongue-in-cheek. 
            And I'm looking forward to watching the first sequel of this movie also. Highly recommended. """]
embeddings = model.encode(sentence)
print(embeddings)

#Print the embeddings
'''
for sentence, embedding in zip(sentence, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")
'''