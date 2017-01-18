import sys
import random
import numpy as np
import mayavi.mlab as mlab
import matplotlib.pyplot as plt
from data_loader import dl

test_filename = sys.argv[1]
label_filename = sys.argv[2]
incorrect_pred_filename = sys.argv[3]

n_shown = 5

dl.prepare_test_data(test_filename)
labels = dl.load_label_file(label_filename)
data = np.load(incorrect_pred_filename)
x = data['x']
y = data['y']
pred = data['pred']

length = x.shape[0]
random_indices = random.sample(range(0, length), n_shown)

fig = plt.figure()
subplot_i = 1
for i in random_indices:
    mlab.figure(size=(480, 340))
    xx, yy, zz = np.where(x[i] == 1)
    mlab.points3d(xx, yy, zz,
                  mode="cube",
                  color=(0, 1, 0),
                  scale_factor=1)
    img = mlab.screenshot()
    mlab.close()

    ax1 = fig.add_subplot(2, n_shown, subplot_i)
    ax1.imshow(img)
    ax1.set_title(labels[np.argmax(y[i])])
    ax1.set_axis_off()

    ax2 = fig.add_subplot(2, n_shown, subplot_i + n_shown)
    sorted_pred_indices = np.argsort(pred[i])[::-1]
    x_pos = np.arange(len(labels))
    ax2.bar(x_pos, pred[i][sorted_pred_indices], align='center')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(np.asarray(labels)[sorted_pred_indices], rotation='vertical')

    subplot_i += 1

plt.tight_layout()
plt.show()
