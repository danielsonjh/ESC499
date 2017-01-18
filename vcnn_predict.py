import sys
import numpy as np
import tensorflow as tf
from data_loader import dl
from vcnn_train import batch_size, basic_vcnn, model_path


test_filename = sys.argv[1]
model_filename = './' + sys.argv[2]

def main():
    dl.prepare_test_data(test_filename)
    x, y, weights, biases, pred = basic_vcnn()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        print("Restoring model...")
        saver.restore(sess, model_filename)
        print("Model restored from file: %s" % model_filename)

        incorrect_cases_x = []
        incorrect_cases_y = []
        incorrect_cases_pred = []

        # Make predictions
        print('Making predictions...')
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        curr_batch = 0
        acc = 0
        n_test_batches = np.ceil(float(dl.n_test) / batch_size)
        while curr_batch < n_test_batches:
            curr_batch += 1
            batch_x, batch_y = dl.next_test_batch(batch_size)
            batch_acc, batch_pred, batch_correct_pred = sess.run([accuracy, pred, correct_pred], feed_dict={x: batch_x, y: batch_y})
            acc += batch_acc * batch_x.shape[0] / dl.n_test

            incorrect_case_indices = np.argwhere(batch_correct_pred == False).flatten()
            incorrect_cases_x.append(batch_x[incorrect_case_indices])
            incorrect_cases_y.append(batch_y[incorrect_case_indices])
            incorrect_cases_pred.append(batch_pred[incorrect_case_indices])

            # print(tf.nn.softmax(batch_pred[incorrect_case_indices]).eval())

        incorrect_cases_x = np.concatenate(incorrect_cases_x)
        incorrect_cases_y = np.concatenate(incorrect_cases_y)
        incorrect_cases_pred = np.concatenate(incorrect_cases_pred)
        print('Incorrect cases')
        print(incorrect_cases_x.shape)
        print(incorrect_cases_pred.shape)

        save_path = model_filename.split('.ckpt')[0] + '_incorrect_pred.npz'
        np.savez_compressed(save_path, x=incorrect_cases_x, y=incorrect_cases_y, pred=incorrect_cases_pred)
        print('Saved incorrect predictions in :' + save_path)

        print("Accuracy= " + "{:.5f}".format(acc))

if __name__ == '__main__':
    main()