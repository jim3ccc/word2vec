import numpy as np
import tensorflow as tf
import pandas
import re

#####################################################################
#                              Helpers                              #
#####################################################################
# function to convert numbers to one hot vectors
def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp

def euclidean_dist(v1, v2):
    return np.linalg.norm(v1-v2)

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

words_set = set(words)
word2int = {}
int2word = {}
vocab_size = len(words_set)

for i,word in enumerate(words_set):
    word2int[word] = i
    int2word[i] = word

WINDOW_SIZE = 2

sentences = tweets_list
sentences = [s.split() for s in sentences]

# create list of nearby word pairs 
data = []
for s in sentences:
    for word_index, word in enumerate(s):
        for nearby_word in s[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(s)) + 1] :
            if nearby_word != word:
                data.append([word, nearby_word])
                print([word, nearby_word])

x_train = [] # input word
y_train = [] # output word

for data_word in data:
    x_train.append(to_one_hot(word2int[data_word[0]], vocab_size))
    y_train.append(to_one_hot(word2int[data_word[1]], vocab_size))

# convert them to numpy arrays
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)


#####################################################################
#                               Model                               #
#####################################################################
# placeholders for x_train and y_label
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

#####################################################################
#                             Training                              #
#####################################################################

# Start session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# loss function
# multiply the label with log(prediction), add all of them up and get the mean
cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), axis=[1]))

# train step
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)

n_iters = 10000

# train for n_iters iterations
for _ in range(n_iters):
    sess.run(train_step, feed_dict={x: x_train, y_label: y_train})
    print('loss is : ', sess.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train}))

vectors = sess.run(W1 + b1)



