import numpy as np
import tensorflow as tf
from data_loader import dl
from vcnn_train import batch_size, basic_vcnn, model_path

test_filename = 'ModelNet10_binvox_30_test.npz'


x, y, weights, biases, pred = basic_vcnn()

saver = tf.train.Saver()

with tf.Session() as sess:

    print "Restoring model..."
    saver.restore(sess, model_path)
    print "Model restored from file: %s" % model_path

    # Make predictions
    print('Making predictions...')
    dl.prepare_test_data(test_filename)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    curr_batch = 0
    acc = 0
    n_test_batches = np.ceil(float(dl.n_test) / batch_size)
    while curr_batch < n_test_batches:
        curr_batch += 1
        batch_x, batch_y = dl.next_test_batch(batch_size)
        batch_acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        acc += batch_acc * batch_x.shape[0] / dl.n_test

    print("Accuracy= " + "{:.5f}".format(acc))
