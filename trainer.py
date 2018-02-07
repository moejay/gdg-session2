import datetime
from os import listdir
from os.path import isfile, join
from random import randint

import numpy as np
import tensorflow as tf

wordsList = np.load('wordsList.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist() #Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('wordVectors.npy')
print ('Loaded the word vectors!')

print(len(wordsList))
print(wordVectors.shape)


numDimensions = 300 #Dimensions for each word vector

maxSeqLength = 250
ids = np.load('idsMatrix.npy')

batchSize = 50
lstmUnits = 64

iterations = 50000
numClasses = 2

def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0):
            num = randint(1,11499)
            labels.append([1,0])
        else:
            num = randint(13499,24999)
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels

def getTestBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(11499,13499)
        if (num <= 12499):
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels

tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses], name='labels')
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength], name='input_data')

with tf.name_scope('word2vec'):
    data = tf.nn.embedding_lookup(wordVectors,input_data, name='word2vec')

with tf.name_scope('sentiment_network'):
    # 1 Let's create one or more LSTMCell
    # https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMCell
    # 2-optional Connect the cells with a MultiRNNCell
    # https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell

    # 3 Create the network ( hint: use tf.nn.dynamic_rnn )
    # and assign it to variable 'value'
    # https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn
    # value, _ = tf.nn.dynamic_rnn

    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]), name='Weights')
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]), name='Biases')
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)

    correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels), name='loss')
    optimizer = tf.train.AdamOptimizer().minimize(loss, name='AdamOptimizer')

tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"


sess = tf.Session()
writer = tf.summary.FileWriter(logdir, sess.graph)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

print("You can open tensorboard to monitor the progress")

for i in range(iterations):
   #Next Batch of reviews
   nextBatch, nextBatchLabels = getTrainBatch()
   sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})
   #Write summary to Tensorboard
   if (i % 50 == 0):
       print("dumping log: %d" % i)
       summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
       writer.add_summary(summary, i)
       writer.flush()

   #Save the network every 10,000 training iterations
   if (i % 10000 == 0 and i != 0):
       dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
       save_path = saver.save(sess, "models/%s-%s-%s/pretrained.ckpt" % (dt, batchSize, lstmUnits) , global_step=i)
       print("saved to %s" % save_path)
writer.close()

iterations = 10
for i in range(iterations):
    nextBatch, nextBatchLabels = getTestBatch()
    print("Accuracy for this batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)

