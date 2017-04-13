import numpy as np
import mayavi.mlab as mlab


def load_data(filename, i):
    data = np.load(filename)
    x = data['x'][i]
    return x


def visualize(data):
    print(data.shape)
    xx, yy, zz = np.where(data > 0)
    ss = data[xx, yy, zz]
    print(ss[0])
    mlab.points3d(xx, yy, zz, ss,
                  mode="cube",
                  color=(0, 1, 0),
                  scale_factor=1,
                  scale_mode='scalar')
    mlab.show()

if __name__ == '__main__':
    i = 0
    visualize(load_data('ModelNet10_binvox_60_test_resized_15.npz', i))
    visualize(load_data('ModelNet10_binvox_60_test.npz', i))
