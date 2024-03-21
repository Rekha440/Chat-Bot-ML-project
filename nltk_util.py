import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

nltk.download('punkt')

stemmer = PorterStemmer()

def tokenize(sentence):
    return np.array(nltk.word_tokenize(sentence))

def stem(word):
    return stemmer.stem(word.lower())

# Vectorizes the tokenized_sentence
'''
Example:
tokenized_sentence = ['hello', 'how', 'are', 'you']
all_words = ['hi', 'hello', 'i', 'you', 'bye', 'thank', 'cool']
bag =       [0,     1,       0,   1,     0,     0,       0]
'''
def bag_of_words(tokenized_sentence, all_words):
    for i in range(tokenized_sentence.shape[0]):
        tokenized_sentence[i] = stem(tokenized_sentence[i])

    bag = np.zeros(len(all_words), dtype = np.float32)
    
    for i in range(len(all_words)):
        if all_words[i] in tokenized_sentence:
            bag[i] = 1.0

    return bag



# Testing block for this file.
# Won't run if this file is not executed separately
def main():
    sentence = 'Hello, thanks for visiting!'
    tokens = tokenize(sentence)
    stems = [stem(w) for w in tokens]
    print(stems)

    # sentence = np.array(['hello', 'how', 'are', 'you'])
    # all_words = np.array(['hi', 'hello', 'i', 'you', 'bye', 'thank', 'cool'])
    # print(bag_of_words(sentence, all_words))

if __name__ == '__main__':
    main()