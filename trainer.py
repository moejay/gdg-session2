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

sess = tf.Session()
positiveFiles = ['positiveReviews/' + f for f in listdir('positiveReviews/') if isfile(join('positiveReviews/', f))]
negativeFiles = ['negativeReviews/' + f for f in listdir('negativeReviews/') if isfile(join('negativeReviews/', f))]

maxSeqLength = 250
ids = np.load('idsMatrix.npy')

batchSize = 50
lstmUnits = 64
numClasses = 2
iterations = 50000

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

with tf.name_scope('input') as scope:
    labels = tf.placeholder(tf.float32, [batchSize, numClasses], name='input_labels')
    input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength], name='input_data')

with tf.name_scope('word2vec'):
    data = tf.nn.embedding_lookup(wordVectors,input_data, name='word2vec')

with tf.name_scope('sentiment_network'):
    c1= tf.nn.rnn_cell.LSTMCell(lstmUnits, state_is_tuple=True)
    c1 = tf.nn.rnn_cell.DropoutWrapper(cell=c1)
    c2 = tf.nn.rnn_cell.LSTMCell(lstmUnits, state_is_tuple=True)
    c2 = tf.nn.rnn_cell.DropoutWrapper(cell=c2)
    lstmCell = tf.nn.rnn_cell.MultiRNNCell([c1, c2], state_is_tuple=True)
    value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

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
writer = tf.summary.FileWriter(logdir, sess.graph)


sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

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

