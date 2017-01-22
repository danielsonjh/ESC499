import sys
import numpy as np
import mayavi.mlab as mlab
import matplotlib.pyplot as plt

predict_filename = './' + sys.argv[1]


def main():
    data = np.load(predict_filename)
    weights = data['weights'][()]
    c1_weights = np.squeeze(np.transpose(weights['c1'], (4, 0, 1, 2, 3)))
    print(c1_weights.shape)
    c1_range = c1_weights.max() - c1_weights.min()

    fig = plt.figure()
    n_shown = c1_weights.shape[0]
    subplot_i = 1
    grid_size = np.ceil(np.sqrt(n_shown))
    for i in range(0, c1_weights.shape[0]):
    # for i in range(0, 1):
        mlab.figure(size=(480, 340))
        for x in range(0, c1_weights.shape[1]):
            for y in range(0, c1_weights.shape[2]):
                for z in range(0, c1_weights.shape[3]):
                    alpha = float((c1_weights[i, x, y, z] - c1_weights.min()) / c1_range)
                    mlab.points3d(x, y, z,
                                  mode="cube",
                                  color=(0, 1, 1),
                                  opacity=alpha,
                                  scale_factor=1)
        img = mlab.screenshot()
        mlab.close()

        ax1 = fig.add_subplot(grid_size, grid_size, subplot_i)
        ax1.imshow(img)
        ax1.set_axis_off()

        subplot_i += 1

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()