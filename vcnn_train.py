import sys
import tensorflow as tf
import numpy as np
import datetime as dt
from data_loader import dl

learning_rate = 0.001
training_epochs = 50
batch_size = 64

display_step = 50
input_dim = 30


def basic_vcnn(n_labels):
    x = tf.placeholder(tf.float32, [None, input_dim, input_dim, input_dim])
    y = tf.placeholder(tf.float32, [None, n_labels])

    weights = {
        'c1': tf.Variable(tf.random_normal([5, 5, 5, 1, 32]), name='wc1'),
        'c2': tf.Variable(tf.random_normal([3, 3, 3, 32, 64]), name='wc2'),
        'fc1': tf.Variable(tf.random_normal([8 * 8 * 8 * 64, 512]), name='wfc1'),
        'fc2': tf.Variable(tf.random_normal([512, n_labels]), name='wfc2'),
    }

    biases = {
        'c1': tf.Variable(tf.random_normal([weights['c1'].get_shape().as_list()[-1]]), name='bc1'),
        'c2': tf.Variable(tf.random_normal([weights['c2'].get_shape().as_list()[-1]]), name='bc2'),
        'fc1': tf.Variable(tf.random_normal([weights['fc1'].get_shape().as_list()[-1]]), name='bfc1'),
        'fc2': tf.Variable(tf.random_normal([weights['fc2'].get_shape().as_list()[-1]]), name='bfc2'),
    }

    # Reshape input
    x_in = tf.reshape(x, shape=[-1, input_dim, input_dim, input_dim, 1])

    # Convolution Layers
    c1 = tf.nn.conv3d(x_in, weights['c1'], strides=[1, 2, 2, 2, 1], padding='SAME')
    c1 = tf.nn.bias_add(c1, biases['c1'])
    c1 = tf.nn.relu(c1)
    # c1 = tf.nn.dropout(c1, 0.8)

    c2 = tf.nn.conv3d(c1, weights['c2'], strides=[1, 1, 1, 1, 1], padding='SAME')
    c2 = tf.nn.bias_add(c2, biases['c2'])
    c2 = tf.nn.relu(c2)
    c2 = tf.nn.avg_pool3d(c2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
    # c2 = tf.nn.dropout(c2, 0.7)

    # Fully connected layers
    fc1 = tf.reshape(c2, [-1, weights['fc1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['fc1']), biases['fc1'])
    fc1 = tf.nn.relu(fc1)
    # fc1 = tf.nn.dropout(fc1, 0.6)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['fc2']), biases['fc2'])
    return x, y, weights, biases, out


def vgg(n_labels):
    x = tf.placeholder(tf.float32, [None, input_dim, input_dim, input_dim])
    y = tf.placeholder(tf.float32, [None, n_labels])

    weights = {
        'c1': tf.Variable(tf.random_normal([3, 3, 3, 1, 32]), name='wc1'),
        'c2': tf.Variable(tf.random_normal([3, 3, 3, 32, 32]), name='wc2'),

        'c3': tf.Variable(tf.random_normal([3, 3, 3, 32, 48]), name='wc3'),
        'c4': tf.Variable(tf.random_normal([3, 3, 3, 48, 48]), name='wc4'),

        'c5': tf.Variable(tf.random_normal([3, 3, 3, 48, 64]), name='wc5'),
        'c6': tf.Variable(tf.random_normal([3, 3, 3, 64, 64]), name='wc6'),
        'c7': tf.Variable(tf.random_normal([3, 3, 3, 64, 64]), name='wc7'),

        'fc1': tf.Variable(tf.random_normal([4 * 4 * 4 * 64, 512]), name='wfc1'),
        'fc2': tf.Variable(tf.random_normal([512, n_labels]), name='wfc2'),
    }

    biases = {
        'c1': tf.Variable(tf.random_normal([weights['c1'].get_shape().as_list()[-1]]), name='bc1'),
        'c2': tf.Variable(tf.random_normal([weights['c2'].get_shape().as_list()[-1]]), name='bc2'),
        'c3': tf.Variable(tf.random_normal([weights['c3'].get_shape().as_list()[-1]]), name='bc3'),
        'c4': tf.Variable(tf.random_normal([weights['c4'].get_shape().as_list()[-1]]), name='bc4'),
        'c5': tf.Variable(tf.random_normal([weights['c5'].get_shape().as_list()[-1]]), name='bc5'),
        'c6': tf.Variable(tf.random_normal([weights['c6'].get_shape().as_list()[-1]]), name='bc6'),
        'c7': tf.Variable(tf.random_normal([weights['c7'].get_shape().as_list()[-1]]), name='bc7'),
        'fc1': tf.Variable(tf.random_normal([weights['fc1'].get_shape().as_list()[-1]]), name='bfc1'),
        'fc2': tf.Variable(tf.random_normal([weights['fc2'].get_shape().as_list()[-1]]), name='bfc2'),
    }

    # Reshape input
    x_in = tf.reshape(x, shape=[-1, input_dim, input_dim, input_dim, 1])

    # Convolution Layers
    c1 = conv_relu(x_in, weights['c1'], biases['c1'], [1, 1, 1, 1, 1])
    c2 = conv_relu(c1, weights['c2'], biases['c2'], [1, 1, 1, 1, 1])
    c2 = tf.nn.max_pool3d(c2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

    c3 = conv_relu(c2, weights['c3'], biases['c3'], [1, 1, 1, 1, 1])
    c4 = conv_relu(c3, weights['c4'], biases['c4'], [1, 1, 1, 1, 1])
    c4 = tf.nn.max_pool3d(c4, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

    c5 = conv_relu(c4, weights['c5'], biases['c5'], [1, 1, 1, 1, 1])
    c6 = conv_relu(c5, weights['c6'], biases['c6'], [1, 1, 1, 1, 1])
    c7 = conv_relu(c6, weights['c7'], biases['c7'], [1, 1, 1, 1, 1])
    c7 = tf.nn.max_pool3d(c7, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

    # Fully connected layers
    fc1 = tf.reshape(c7, [-1, weights['fc1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['fc1']), biases['fc1'])
    fc1 = tf.nn.relu(fc1)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['fc2']), biases['fc2'])
    return x, y, weights, biases, out


def aniprobing():

    x = tf.placeholder(tf.float32, [None, input_dim, input_dim, input_dim])
    y = tf.placeholder(tf.float32, [None, 10])

    weights = {
        'c1': tf.Variable(tf.random_normal([5, 5, 5, 1, 32]), name='wc1'),
        'c2': tf.Variable(tf.random_normal([3, 3, 3, 32, 32]), name='wc2'),
        'fc1': tf.Variable(tf.random_normal([8 * 8 * 8 * 32, 128]), name='wfc2'),
        'fc2': tf.Variable(tf.random_normal([128, 10]), name='wfc2'),
    }

    biases = {
        'c1': tf.Variable(tf.random_normal([weights['c1'].get_shape().as_list()[4]]), name='bc1'),
        'c2': tf.Variable(tf.random_normal([weights['c2'].get_shape().as_list()[4]]), name='bc2'),
        'fc1': tf.Variable(tf.random_normal([weights['fc1'].get_shape().as_list()[1]]), name='bfc1'),
        'fc2': tf.Variable(tf.random_normal([weights['fc2'].get_shape().as_list()[1]]), name='bfc2'),
    }

    # Reshape input
    x_in = tf.reshape(x, shape=[-1, 60, 60, 60, 1])
    x_in = tf.nn.avg_pool3d(x_in, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

    # Convolution Layers
    c1 = tf.nn.conv3d(x_in, weights['c1'], strides=[1, 2, 2, 2, 1], padding='SAME')
    c1 = tf.nn.bias_add(c1, biases['c1'])
    c1 = tf.nn.relu(c1)
    # c1 = tf.nn.dropout(c1, 0.8)

    c2 = tf.nn.conv3d(c1, weights['c2'], strides=[1, 1, 1, 1, 1], padding='SAME')
    c2 = tf.nn.bias_add(c2, biases['c2'])
    c2 = tf.nn.relu(c2)
    c2 = tf.nn.avg_pool3d(c2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
    # c2 = tf.nn.dropout(c2, 0.7)

    # Fully connected layers
    fc1 = tf.reshape(c2, [-1, weights['fc1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['fc1']), biases['fc1'])
    fc1 = tf.nn.relu(fc1)
    # fc1 = tf.nn.dropout(fc1, 0.6)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['fc2']), biases['fc2'])
    return x, y, weights, biases, out


def conv_relu(x, weight, bias, strides):
    x = tf.nn.conv3d(x, weight, strides=strides, padding='SAME')
    x = tf.nn.bias_add(x, bias)
    x = tf.nn.relu(x)
    return x

def mlpconv3d(x, weight, bias, strides):
    x = tf.nn.conv3d(x, weight, strides=strides, padding='SAME')
    x = tf.nn.bias_add(x, bias)
    x = tf.nn.batch_normalization(x)
    x = tf.nn.relu(x)
    return x


def mlpconv2d(x, weight, bias, strides):
    x = tf.nn.conv2d(x, weight, strides=strides, padding='SAME')
    x = tf.nn.bias_add(x, bias)
    x = tf.nn.batch_normalization(x)
    x = tf.nn.relu(x)
    return x


def main(model, n_labels):

    dl.prepare_train_val_data(train_filename, train_ratio=0.9)

    # Construct model
    x, y, weights, biases, pred = model(n_labels)

    # Define loss and optimizer
    with tf.name_scope('Loss'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

    with tf.name_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    with tf.name_scope('Accuracy'):
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    tf.summary.scalar("loss", cost)
    tf.summary.scalar("accuracy", accuracy)
    merged_summary_op = tf.summary.merge_all()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        train_summary_writer = tf.train.SummaryWriter(train_logs_path, graph=tf.get_default_graph())
        valid_summary_writer = tf.train.SummaryWriter(valid_logs_path, graph=tf.get_default_graph())

        step = 0
        # Keep training until reach max iterations
        batches_per_epoch = dl.n_train / batch_size
        n_batches = batches_per_epoch * training_epochs
        print('Number of batches {0}'.format(n_batches))
        while step < n_batches:
            batch_x, batch_y = dl.next_train_batch(batch_size)
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc, summary = sess.run([cost, accuracy, merged_summary_op],
                                              feed_dict={x: batch_x, y: batch_y})
                train_summary_writer.add_summary(summary, step)

                print("Batch " + str(step) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

                valid_batch = 0
                valid_cost = 0
                valid_acc = 0
                n_valid_batches = np.ceil(float(dl.n_valid) / batch_size)
                while valid_batch < n_valid_batches:
                    valid_batch += 1
                    batch_x, batch_y = dl.next_valid_batch(batch_size)
                    valid_batch_cost, valid_batch_acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
                    valid_cost += valid_batch_cost * batch_x.shape[0] / dl.n_valid
                    valid_acc += valid_batch_acc * batch_x.shape[0] / dl.n_valid

                valid_summary = tf.Summary()
                valid_summary.value.add(tag="accuracy", simple_value=valid_acc)
                valid_summary.value.add(tag="loss", simple_value=valid_cost)
                valid_summary_writer.add_summary(valid_summary, step)
                print("Validation Accuracy= " + "{:.5f}".format(valid_acc) + ', Loss=' + "{:.5f}".format(valid_cost))

            step += 1

        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        print("---Final model saved in file: " + save_path)


model_selector = {
    'basic_vcnn': basic_vcnn,
    'vgg': vgg
}

if __name__ == '__main__':
    model_name = sys.argv[1]
    train_filename = sys.argv[2]
    n_labels = int(sys.argv[3])

    model_name_with_metadata = model_name + '_' + train_filename.split('.')[0] + '_' \
                               + str(dt.datetime.utcnow()).replace(' ', '_').split('.')[0]
    train_logs_path = '/tmp/tensorflow_logs/' + model_name_with_metadata + '_train'
    valid_logs_path = '/tmp/tensorflow_logs/' + model_name_with_metadata + '_valid'
    model_path = model_name_with_metadata + '.ckpt'

    main(model_selector[model_name], n_labels)