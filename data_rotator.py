import sys
import random
import numpy as np
import mayavi.mlab as mlab
import matplotlib.pyplot as plt

SHOW_FIGURES = True
NUM_FIGURES = 5

TRAIN_FILENAME = 'ModelNet10_binvox_60_train.npz'
TEST_FILENAME = 'ModelNet10_binvox_60_test.npz'


def rotate_data(filename):

    data = np.load(filename)
    x = data['x']
    y = data['y']

    if SHOW_FIGURES:
        fig = plt.figure()

        for i in range(1, NUM_FIGURES + 1):
            j = random.choice(range(0, x.shape[0]))
            data = x[j]

            img = get_voxel_image(data)
            ax1 = fig.add_subplot(2, NUM_FIGURES, i)
            ax1.imshow(img)

            img = get_voxel_image(rotate(data, y[j]))
            ax2 = fig.add_subplot(2, NUM_FIGURES, i + NUM_FIGURES)
            ax2.imshow(img)

        plt.tight_layout()
        plt.show()

    rotated_x = []
    for i in range(0, x.shape[0]):
        rotated_x.append(rotate(x[i], y[i]))

    np.savez_compressed('rotated_' + filename, x=rotated_x, y=y)


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


def rotate(data, label):
    points = []
    for xx in range(0, data.shape[0]):
        for yy in range(0, data.shape[1]):
            for zz in range(0, data.shape[2]):
                if data[xx, yy, zz] == 1:
                    points.append([xx, yy, zz])

    point_cloud = np.transpose(np.asarray(points))
    cov = np.cov(point_cloud)
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    rotation_matrix = np.transpose(eigenvectors)
    rotated_point_cloud = np.dot(rotation_matrix, point_cloud)
    # Make mins 0
    rotated_point_cloud = np.subtract(rotated_point_cloud.transpose(), rotated_point_cloud.min(axis=1).transpose()).transpose()
    # Make max in any dimension 30
    maxes = np.max(rotated_point_cloud, axis=1)
    maxmax = np.max(maxes)
    rotated_point_cloud *= (data.shape[0] - 1) / maxmax

    rotated_data = np.zeros(data.shape)
    for p in np.transpose(rotated_point_cloud):
        try:
            rotated_data[int(p[0]), int(p[1]), int(p[2])] = 1
        except:
            print(p)
            print(maxes)
            print(label)
            #
            # fig = plt.figure()
            #
            # img = get_voxel_image(data)
            # ax1 = fig.add_subplot(2, 1, 1)
            # ax1.imshow(img)
            #
            # img = get_voxel_image(rotated_data)
            # ax2 = fig.add_subplot(2, 1, 2)
            # ax2.imshow(img)
            #
            # plt.tight_layout()
            # plt.show()
            # break

    return rotated_data

if __name__ == '__main__':
    rotate_data(TRAIN_FILENAME)
    rotate_data(TEST_FILENAME)