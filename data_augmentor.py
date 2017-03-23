import numpy as np
import mayavi.mlab as mlab
import matplotlib.pyplot as plt


train_filename = 'ModelNet40_binvox_30_train.npz'
test_filename = 'ModelNet40_binvox_30_test.npz'


def get_voxel_image(data):
    mlab.figure(size=(480, 340))
    xx, yy, zz = np.where(data == 1)
    mlab.points3d(xx, yy, zz,
                  mode="cube",
                  color=(0, 1, 0),
                  scale_factor=1)

    img = mlab.screenshot()
    mlab.close()
    return img


def augment_data(filename):

    data = np.load(filename)
    x = data['x']
    y = data['y']

    print(x.shape)
    print(y.shape)

    for i in range(0, x.shape[0]):
        x = np.append(x, [np.flipud(x[i]), np.fliplr(x[i]), np.flipud(np.fliplr(x[i]))], axis=0)
        y = np.append(y, [y[i], y[i], y[i]])

        print(i)

    print(x.shape)
    print(y.shape)

    if False:
        data = x[5]

        fig = plt.figure()

        img = get_voxel_image(data)
        ax1 = fig.add_subplot(1, 4, 1)
        ax1.imshow(img)

        img = get_voxel_image(np.flipud(data))
        ax2 = fig.add_subplot(1, 4, 2)
        ax2.imshow(img)

        img = get_voxel_image(np.fliplr(data))
        ax3 = fig.add_subplot(1, 4, 3)
        ax3.imshow(img)

        img = get_voxel_image(np.flipud(np.fliplr(data)))
        ax3 = fig.add_subplot(1, 4, 4)
        ax3.imshow(img)

        plt.tight_layout()
        plt.show()

    np.savez_compressed('aug_' + filename, x=x, y=y)


if __name__ == '__main__':
    augment_data(train_filename)
    augment_data(test_filename)