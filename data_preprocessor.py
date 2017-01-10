import tensorflow as tf
import numpy as np
from data_loader import dl

batch_size = 100

train_filename = 'ModelNet10_binvox_60_train.npz'
test_filename = 'ModelNet10_binvox_60_test.npz'

input_dim = 60
output_dim = 30
pool_size = int(np.ceil(input_dim / output_dim))


def main():
    print('pool_size: ' + str(pool_size))
    if pool_size == 1:
        print('pool_size is invalid')
        return

    x = tf.placeholder(tf.float32, [None, input_dim, input_dim, input_dim])
    x_in = tf.reshape(x, shape=[-1, input_dim, input_dim, input_dim, 1])
    x_out = tf.nn.avg_pool3d(x_in, ksize=[1, pool_size, pool_size, pool_size, 1], strides=[1, pool_size, pool_size, pool_size, 1], padding='SAME')

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        dl.prepare_train_val_data(train_filename, train_ratio=1.0)
        step = 0
        n_batches = dl.n_train / batch_size
        train_batches = []
        while step <= n_batches:
            batch_x, _ = dl.next_train_batch(batch_size)
            train_batches.append(sess.run(x_out, feed_dict={x: batch_x}))
            step += 1

        train_pooled = tf.reshape(tf.concat(0, train_batches), shape=[-1, output_dim, output_dim, output_dim]).eval()
        print('train resized to:')
        print(train_pooled.shape)
        np.savez_compressed(train_filename, x=train_pooled, y=np.load(train_filename)['y'])

        dl.prepare_test_data(test_filename)
        step = 0
        n_batches = dl.n_test / batch_size
        test_batches = []
        while step <= n_batches:
            batch_x, _ = dl.next_test_batch(batch_size)
            test_batches.append(sess.run(x_out, feed_dict={x: batch_x}))
            step += 1

        test_pooled = tf.reshape(tf.concat(0, test_batches), shape=[-1, output_dim, output_dim, output_dim]).eval()
        print('test resized to:')
        print(test_pooled.shape)
        np.savez_compressed(test_filename, x=test_pooled, y=np.load(test_filename)['y'])


if __name__ == '__main__':
    main()