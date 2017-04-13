import tensorflow as tf
import numpy as np
from data_loader import dl

batch_size = 100

train_filename = 'ModelNet10_binvox_60_train.npz'
test_filename = 'ModelNet10_binvox_60_test.npz'

input_dim = 60
output_dim = 30
pool_size = int(np.ceil(input_dim / output_dim))


x = tf.placeholder(tf.float32, [None, input_dim, input_dim, input_dim])
x_in = tf.reshape(x, shape=[-1, input_dim, input_dim, input_dim, 1])
x_out = tf.nn.avg_pool3d(x_in, ksize=[1, pool_size, pool_size, pool_size, 1],
                         strides=[1, pool_size, pool_size, pool_size, 1], padding='SAME')
init = tf.global_variables_initializer()


def main():
    print('pool_size: ' + str(pool_size))
    if pool_size == 1:
        print('pool_size is invalid')
        return

    dl.prepare_train_val_data(train_filename, train_ratio=1.0)
    n_batches = dl.n_train / batch_size
    resize_and_save(train_filename, dl.next_train_batch, n_batches)

    dl.prepare_test_data(test_filename)
    n_batches = dl.n_test / batch_size
    resize_and_save(test_filename, dl.next_test_batch, n_batches)


def resize_and_save(filename, batch_function, n_batches):
    with tf.Session() as sess:
        sess.run(init)

        step = 0
        batches = []
        while step <= n_batches:
            batch_x, _ = batch_function(batch_size)
            batches.append(sess.run(x_out, feed_dict={x: batch_x}))
            step += 1

        pooled = tf.reshape(tf.concat(0, batches), shape=[-1, output_dim, output_dim, output_dim]).eval()
        print('resized to:')
        print(pooled.shape)
        np.savez_compressed(filename.split(".")[0] + '_resized_' + str(output_dim), x=pooled, y=np.load(filename)['y'])


if __name__ == '__main__':
    main()
