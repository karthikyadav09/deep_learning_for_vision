import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) #call mnist function

learningRate = 0.0025
trainingIters = 100000
batchSize = 256
displayStep = 10

nInput =28 #we want the input to take the 28 pixels
nSteps =28 #every 28
nHidden =256 #number of neurons for the RNN
nClasses =10 #this is MNIST so you know

x = tf.placeholder('float', [None, nSteps, nInput])
y = tf.placeholder('float', [None, nClasses])

weights = {
        'out': tf.Variable(tf.random_normal([nHidden, nClasses]))
}

biases = {
        'out': tf.Variable(tf.random_normal([nClasses]))
}

def RNN(x, weights, biases):
      	x = tf.transpose(x, [1,0,2])
      	x = tf.reshape(x, [-1, nInput])
      	x = tf.split(x, nSteps, 0) #configuring so you can get it as needed for the 28 pixels

        lstmCell = rnn.BasicLSTMCell(nHidden, forget_bias=1.0)  #find which lstm to use in the documentation
        rnnCell = rnn.BasicRNNCell(nHidden)
        gruCell = rnn.GRUCell(nHidden)
        outputs, states =  rnn.static_rnn(lstmCell, x, dtype=tf.float32)  #for the rnn where to get the output and hidden state 

        return tf.matmul(outputs[-1], weights['out'])+ biases['out']

pred = RNN(x, weights, biases)

#optimization
#create the cost, optimization, evaluation, and accuracy
#for the cost softmax_cross_entropy_with_logits seems really good
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)

correctPred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy =  tf.reduce_mean(tf.cast(correctPred, tf.float32))

init = tf.global_variables_initializer()
train_accuracy = []
train_loss = []
idx = []
with tf.Session() as sess:
	sess.run(init)
	step = 1

	while step* batchSize < trainingIters:
		batchX, batchY =  mnist.train.next_batch(batchSize) #mnist has a way to get the next batch
		batchX = batchX.reshape((batchSize, nSteps, nInput))

		sess.run(optimizer, feed_dict={x: batchX, y: batchY})

		if step % displayStep == 0:
			acc = sess.run(accuracy, feed_dict={x: batchX, y: batchY})
			loss =  sess.run(cost, feed_dict={x: batchX, y: batchY})
			train_accuracy.append(acc)
			train_loss.append(loss)
			print("Iter " + str(step*batchSize) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
		step +=1
	print('Optimization finished')

	testData = mnist.test.images.reshape((-1, nSteps, nInput))
	testLabel = mnist.test.labels
	print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: testData, y: testLabel}))

print "Training Accuracy :"
print train_accuracy
print "Train Loss"
print train_loss

