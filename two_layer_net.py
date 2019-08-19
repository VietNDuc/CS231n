import sys
import numpy as np
import matplotlib.pyplot as plt
from neural_net import TwoLayerNet
from math import sqrt, ceil
sys.path.append('E:/AI/ML/CS231n/')
from load_data import load_all


def load_CIFAR10_data():
    path_data = 'E:/DATASETS/cifar-10-batches-py'
    x_train, y_train, x_test, y_test = load_all(path_data)

    # Split data into train, validation and test set
    num_train = 9000
    num_val = 1000
    num_test = 1000

    # Validation set will be num_val points from the original training set
    mask = range(num_train, num_train + num_val)
    x_val = x_train[mask]
    y_val = y_train[mask]

    # Training set will be the first num_train points from the original training set
    mask = range(num_train)
    x_train = x_train[mask]
    y_train = y_train[mask]

    # User the first num_test points of the original test set as out test set
    mask = range(num_test)
    x_test = x_test[mask]
    y_test = y_test[mask]

    # Preprocessing : reshape the image data into rows
    x_train = x_train.reshape(num_train, -1)
    x_val = x_val.reshape(num_val, -1)
    x_test = x_test.reshape(num_test, -1)

    # Preprocessing : subtract the mean image
    # Compute the image mean based on the training data
    mean_img = np.mean(x_train, axis=0)  # (9000,3072) -> (3072) all training data

    # Subtract
    x_train -= mean_img
    x_test -= mean_img
    x_val -= mean_img

    return x_train, y_train, x_val, y_val, x_test, y_test


x_train, y_train, x_val, y_val, x_test, y_test = load_CIFAR10_data()

input_size = 32*32*3
hidden_size = 50
num_classes = 10
net = TwoLayerNet(input_size, hidden_size, num_classes)

# Training the network using SGD
stats = net.train(x_train, y_train, x_val,
                  y_val, num_iters=1200, verbose=True)

val_acc = np.mean(net.predict(x_val) == y_val)
print("Validation accuracy: ", val_acc)  # 44.5%

test_acc = (net.predict(x_test) == y_test).mean()
print('Test accuracy: ', test_acc)


# Plot the loss function and train / validation accuracies
def visualize_loss_acc(stats):
    plt.subplot(2, 1, 1)
    plt.plot(stats['loss_his'])
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(stats['train_accuracy'], label='train')
    plt.plot(stats['val_acc_history'], label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Classification accuracy')
    plt.legend()
    plt.show()


def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.

    Inputs:
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    """
    (N, H, W, C) = Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                # grid[y0:y1, x0:x1] = Xs[next_idx]
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    # grid_max = np.max(grid)
    # grid_min = np.min(grid)
    # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
    return grid


def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()


def finding_best_net():
    best_net = None
    result = {}
    best_val = -1

    hidden_sizes = [40, 60, 80, 100, 120]
    lrs = [1e-4, 5e-4, 1e-3, 5e-3]
    regs = [1e-6, 5e-6, 1e-5, 5e-5]
    lr_decays = [0.95, 0.99, 0.999]

    input_size = 32*32*3
    num_classes = 10

    nets = {}
    i = 0
    grid_search = [(x, y, z) for x in lrs for y in regs for z in lr_decays]

    for hidden_size in hidden_sizes:
        for lr, reg, lr_decay in grid_search:
            print('hidden {} -- lr {} -- reg {} -- lr_decay {}'.format(hidden_size, lr, reg, lr_decay))
            print('Done {:2} out of {}'.format(i+1, len(grid_search) * len(hidden_sizes)))
            net = TwoLayerNet(input_size, hidden_size, num_classes)
            # Training
            net.train(x_train, y_train, x_val, y_val,
                      num_iters=2000, lr=lr, lr_decay=lr_decay,
                      reg=reg, verbose=False)
            # Predict
            y_train_pred = net.predict(x_train)
            y_val_pred = net.predict(x_val)
            # Scoring
            train_accu = np.mean(y_train_pred == y_train)
            val_accu = np.mean(y_val_pred == y_val)
            # Store results
            result[(hidden_size, lr, reg, lr_decay)] = (train_accu, val_accu)
            nets[(hidden_size, lr, reg, lr_decay)] = net

            if val_accu > best_val:
                best_val = val_accu
                best_net = net
            i += 1
