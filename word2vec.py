import numpy as np
import tensorflow as tf
import pandas
import re

#####################################################################
#                               Data                                #
#####################################################################
#read tweets from csv and convert to list of strings
tweets = pandas.read_csv("tweets_training.csv", encoding = "ISO-8859-1", header=0, usecols=[5])
tweets_list = tweets.values.flatten().tolist()
tweets_list = [x.lower() for x in tweets_list]


#remove non-alphanumerica characters and remove words that start with @
tweets_list = [' '.join(word for word in tweet.split(' ') if not word.startswith('@')) for tweet in tweets_list]
tweets_list = [re.sub(r'[^\w\s]','',tweet) for tweet in tweets_list]
tweets_list = [re.sub(' +', ' ', tweet) for tweet in tweets_list]
#print(tweets_list)

#create a word2int and int2word dictionary
words = []
for tweet in tweets_list:
    for word in tweet.split():
        words.append(word)

words = set(words)
word2int = {}
int2word = {}
vocab_size = len(words)

for i,word in enumerate(words):
    word2int[word] = i
    int2word[i] = word

WINDOW_SIZE = 2

# get nearby words
data = []
for tweet in tweets:
    for word_index, word in enumerate(tweet):
        for nearby_word in tweet[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(tweet)) + 1] :
            if nearby_word != word:
                data.append([word, nearby_word])

x_train = [] # input word
y_train = [] # output word

# function to convert numbers to one hot vectors
def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp

for data_word in data:
    x_train.append(to_one_hot(word2int[data_word[0]], vocab_size))
    y_train.append(to_one_hot(word2int[data_word[1]], vocab_size))

# convert them to numpy arrays
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)


#####################################################################
#                               Model                               #
#####################################################################
# placeholders for x_train and y_train
x = tf.placeholder(tf.float32, shape=(None, vocab_size))
y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))

EMBEDDING_DIM = 5
# we multipley matrix x with W1 to turn it into our hidden state
# x = N x D matrix (where N is number of words, D is vocab size)
# W1 = D x E (where D is vocab size and E is EMBEDDING_DIM)
# hidden_representation = N x E
W1 = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal[EMBEDDING_DIM]) # bias
# multiple x with W1 and add on the bias
hidden_representation =tf.add(tf.matmul(x, W1), b1)

# W2 = E x D
# prediction = N x D
W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size]))
b2 = tf.Variable(tf.random_normal([vocab_size])) # bias
prediction = tf.nn.softmax(tf.add(tf.matmul(hidden_representation, W2), b2))



