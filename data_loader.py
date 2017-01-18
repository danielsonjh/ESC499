import numpy as np
import time
import csv
from random import shuffle


class DataLoader:

    n_data = None
    n_train = None
    n_valid = None
    n_test = None

    valid_x = []
    valid_y = []
    __valid_batch_counter = 0

    train_x = []
    train_y = []
    __train_batch_counter = 0

    test_x = []
    test_y = []
    __test_batch_counter = 0

    def __init__(self):
        pass

    def load_label_file(self, label_filename):
        labels = []
        with open(label_filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                labels.append(row[1])
        return labels

    def prepare_train_val_data(self, train_filename, train_ratio):
        data = np.load(train_filename)
        x = data['x']
        y = data['y']

        self.n_data = x.shape[0]

        start_time = time.time()

        # Separate data evenly among classes
        n_labels = np.max(y) + 1
        train_indices = []
        valid_indices = []
        for label in range(0, n_labels):
            label_indices = np.squeeze(np.argwhere(y == label))
            n_train_in_label = int(len(label_indices) * train_ratio)
            train_indices.extend(label_indices[:n_train_in_label])
            valid_indices.extend(label_indices[n_train_in_label:])

        # Shuffle training data
        shuffle(train_indices)

        self.train_x = x[train_indices]
        self.train_y = y[train_indices]
        print(np.bincount(self.train_y))
        self.valid_x = x[valid_indices]
        self.valid_y = y[valid_indices]
        print(np.bincount(self.valid_y))

        self.train_y = self.__one_hot_encode_labels(self.train_y)
        self.valid_y = self.__one_hot_encode_labels(self.valid_y)

        self.n_train = self.train_x.shape[0]
        self.n_valid = self.valid_x.shape[0]

        end_time = time.time()

        print('Finished preparing training and validation sets. Took {0}s'.format(end_time - start_time))

        print('train shape')
        print(self.train_x.shape)
        print(self.train_y.shape)
        print('valid shape')
        print(self.valid_x.shape)
        print(self.valid_y.shape)

    def prepare_test_data(self, test_filename):
        data = np.load(test_filename)
        self.test_x = data['x']
        self.test_y = data['y']
        self.test_y = self.__one_hot_encode_labels(self.test_y)
        self.n_test = self.test_x.shape[0]
        print('test shape')
        print(self.test_x.shape)

    def next_train_batch(self, batch_size):
        batch_x, batch_y, new_batch_counter = self.__process_next_batch(self.train_x, self.train_y, self.n_train,
                                                                        self.__train_batch_counter, batch_size)
        self.__train_batch_counter = new_batch_counter
        return batch_x, batch_y

    def next_valid_batch(self, batch_size):
        batch_x, batch_y, new_batch_counter = self.__process_next_batch(self.valid_x, self.valid_y, self.n_valid,
                                                                        self.__valid_batch_counter, batch_size)
        self.__valid_batch_counter = new_batch_counter
        return batch_x, batch_y

    def next_test_batch(self, batch_size):
        batch_x, batch_y, new_batch_counter = self.__process_next_batch(self.test_x, self.test_y, self.n_test,
                                                                        self.__test_batch_counter, batch_size)
        self.__test_batch_counter = new_batch_counter
        return batch_x, batch_y

    @staticmethod
    def __process_next_batch(x, y, n, batch_counter, batch_size):
        start = batch_counter
        end = start + batch_size
        batch_x = x[start:end]
        batch_y = y[start:end]
        new_batch_counter = end if end < n else 0

        return batch_x, batch_y, new_batch_counter

    @staticmethod
    def __one_hot_encode_labels(raw_labels):
        if raw_labels.size == 0:
            return raw_labels
        else:
            n_labels = np.max(raw_labels) + 1
            n = raw_labels.shape[0]
            labels = np.zeros((n, n_labels), dtype=np.uint8)
            for i in range(n):
                labels[i, raw_labels[i]] = 1

            return labels


dl = DataLoader()

if __name__ == '__main__':
    dl.prepare_train_val_data(0.9)