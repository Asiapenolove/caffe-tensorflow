# Import the converted model's class
import numpy as np
import random
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from vggnet import VGG_FACE_16_layers as MyNet

# load image and labels
X = np.load("../../outfile_x.npy")
y = np.load("../../outfile_y.npy").astype(int)
n_class= len(np.unique(y))
print "It's a {}-class classification problem".format(str(n_class))
print "====The shape of data : X : {} , y : {}====".format(str(X.shape),str(y.shape))


arr = np.arange(len(X))
np.random.shuffle(arr)
X=X[arr]
y=y[arr]
data={
  'X_train': X[:int(len(X)*0.8)],
  'y_train': y[:int(len(X)*0.8)],
  'X_val': X[int(len(X)*0.8):],
  'y_val': y[int(len(X)*0.8):],
}
print "there are "+ str(data['X_train'].shape[0]) + " images in training set"
print "there are "+ str(data['X_val'].shape[0]) + " images in testing set"



def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.0001)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

images_raw = tf.placeholder(tf.float32, shape=[None, 32,32,3])
images = tf.image.resize_images(images_raw,(224,224))
labels_raw = tf.placeholder(tf.int32, shape=[None,],name='truth_y')
labels = tf.one_hot(labels_raw,n_class)

# concate network
net = MyNet({'data': images})

pool5 = net.layers['pool5']
with tf.name_scope('fintune_whimh'):
    with tf.name_scope('fc6'):
        pool5_flat_size = int(np.prod(pool5.shape[1:]))
        pool5_flat = tf.reshape(pool5, [-1, pool5_flat_size])
        W_fc6 = weight_variable([pool5_flat_size, 1024])
        b_fc6 = bias_variable([1024])
        H_fc6=tf.nn.relu(tf.matmul(pool5_flat, W_fc6) + b_fc6) 
    with tf.name_scope('cross_entropy'):
        #calculate_entropy
        W_fc7 = weight_variable([1024, n_class])
        b_fc7 = bias_variable([n_class])
        fc7 = tf.matmul(H_fc6, W_fc7) + b_fc7 
        predctions = tf.argmax(fc7,1, name='predictions')
        ground_truth = tf.argmax(labels,1)

correct_prediction = tf.equal(predctions, ground_truth)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc7 ,labels= labels), 0)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

with tf.Session() as sess:
    # Load the data
    sess.run(tf.global_variables_initializer())
    net.load('vggnet.npy', sess)

    for i in range(1000):
        arr = np.arange(len(data['X_train']))
        np.random.shuffle(arr)
        np_images, np_labels = (data['X_train'][arr[:30]]\
                                ,data['y_train'][arr[:30]])

        feed = {images_raw: np_images, labels_raw: np_labels}
        np_loss,acc, _ = sess.run([loss,accuracy, train_op], feed_dict=feed)
        if i % 30 == 0:
            arr = np.arange(len(data['X_val']))
            np.random.shuffle(arr)
            np_images, np_labels = (data['X_val'][arr[:30]]\
                                    ,data['y_val'][arr[:30]])
            feed = {images_raw: np_images, labels_raw: np_labels}
            np_loss,acc = sess.run([loss,accuracy], feed_dict=feed)
            print('Iteration: ', i, np_loss,acc)
