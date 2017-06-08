# Import the converted model's class
import numpy as np
import random
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from vggnet import VGG_FACE_16_layers as MyNet


images = tf.placeholder(tf.float32, [None, 224, 224, 3])
net = MyNet({'data': images})

prob_op = net.layers['prob']
with tf.Session() as sess:
    # Load the data
    sess.run(tf.global_variables_initializer())
    net.load('vggnet.npy', sess)
    fake_img=np.zeros([1,224,224,3])
    print fake_img.shape
    feed = {images: fake_img}
    prob = sess.run([prob_op], feed_dict=feed)
    print np.sum(prob[0])
    # take a look of graph before exit
    train_writer = tf.summary.FileWriter('/tmp/loser',sess.graph)
