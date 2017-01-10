import numpy as np
import binvox_rw
import os


input_dir = '../ModelNet10_binvox_60'

def extract_data(path_prefix, set_name):
    data = []
    folder_name = path_prefix + set_name
    for filename in os.listdir(folder_name):
        with open(folder_name + '/' + filename, 'rb') as f:
            voxels = binvox_rw.read_as_3d_array(f)
            data.append(voxels.data.tolist())
    return data


def save(set_type, filename):
    label_names = os.listdir(input_dir)
    n_label = len(label_names)
    print(n_label)
    x = []
    y = []
    for i in range(n_label):
        print(label_names[i])
        path_prefix = input_dir + '/' + label_names[i] + '/'

        d = extract_data(path_prefix, set_type)
        print(len(d))
        x.extend(d)
        y.extend([i] * len(d))

    x = np.asarray(x)
    y = np.asarray(y)
    print(x.shape)
    print(y.shape)

    np.savez_compressed(filename, x=x, y=y)


if __name__ == '__main__':
    save('train', input_dir + '_train.npz')
    save('test', input_dir + '_test.npz')