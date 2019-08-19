import numpy as np
# import matplotlib.pyplot as plt


class TwoLayerNet(object):
    '''
    A two-layer fully connected neural network
    Inputs:
     - an input dimension of N
     - a hidden layer dimension of H
     - C-classes
    We train the network with a softmax loss function and L2 regularization on
    the weight matrices. The net uses a ReLu nonlinearity after the first fc layer
    Architecture:
    Inputs -> FC layer -> ReLU -> FC layer -> Softmax
    Outputs:
     - Scores for each class
    '''
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        '''
        Intialization weights and biases which are stored in the variable self.params,
        which is a dictionary with the following keys:
        W1 (D, H): 1st layer weights
        b1 (H, ) : 1st layer biases
        W2 (H, C): 2nd layer weights
        b2 (C, ) : 2nd layer biases

        Inputs:
         - input_size : The dimension D of the input data
         - hidden_size : The num of neurons H in the hidden layer
         - output_size : The number of classes C
        '''
        self.params = {}
        self.params['W1'] = std*np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std*np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def ReLU(self, x):
        return np.maximum(0, x)

    def softmax(self, scores):
        scores = scores - np.max(scores, axis=1, keepdims=True)
        softmaxes = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape([-1, 1])
        return softmaxes

    def softmax_loss(self, x, y):
        scores = self.softmax(x)
        correct_score = scores[np.arange(x.shape[0]), y]
        loss_array = -correct_score + np.log(np.sum(np.exp(scores), axis=1))
        return np.sum(loss_array)

    def softmax_grad(self, x, y):
        dscores = self.softmax(x)
        dscores[np.arange(x.shape[0]), y] -= 1
        return dscores

    def loss(self, x, y=None, reg=0.0):
        '''
        Compute the loss and gradient for a 2 layer fc neural network
        Inputs:
         - x (N, D): each x[i] is a training sample
         - y (N, ) : vector of training labels. y[i] is the label for x[i] and each y[i] is an integer in the 
        range 0 <= y[i] < C.This param is optional;if it's passed the we return loss& gradients, if not we
        return only scores(last layer)
         - reg
        Outputs:
        If y is None, return a matrix scores (N, C)
        Else, return a tuple of:
         - loss : loss value for this batch of training data
         - grads: dictionary mapping param names to gradients of those params with respect to the loss
        function; has the same keys as self.params
        '''
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        num_train, dims = x.shape
        '''
        Compute the forward pass
         z1 = x*W1 + b1
         z2 = ReLU(z1)
         scores = z2*W2 + b2
         scores = softmax(scores)
        '''
        fc1 = x.dot(W1) + b1            # (N, H)
        hidden1 = self.ReLU(fc1)        # (N, H)
        scores = hidden1.dot(W2) + b2   # (N, C)

        if y is None:
            return scores
        # Cumpute softmax loss
        loss = self.softmax_loss(scores, y)
        loss /= num_train
        loss += .5*reg*np.sum(W2*W2) + .5*reg*np.sum(W1*W1)

        grads = {}
        # Derivation of softmax function
        dscores = self.softmax_grad(scores, y)  # (N, C)
        dscores /= num_train

        '''
        Derivative W2 & b2:
         scores = softmax(scores)
         scores = hidden1*W2 + b2
        '''
        dW2 = hidden1.T.dot(dscores)  # (H, N)*(N, C) = (H, C)
        db2 = np.sum(dscores, axis=0)

        '''
        Derivation of ReLU, W1 & b1:
         scores = hidden1*W2 + b2
         hidden1 = ReLU(fc1)
         fc1 = x*W1 + b1
        '''
        dhidden1 = dscores.dot(W2.T)  # (N, C)*(C, H) = (N, H)
        # dhidden1[hidden1 == 0] = 0
        drelu = dhidden1 * (hidden1 > 0)
        dW1 = x.T.dot(drelu)  # (D, N)*(N, H) = (D, H)
        db1 = drelu.sum(axis=0)
        # Regularization
        dW2 += 2*reg*W2
        dW1 += 2*reg*W1

        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

        return loss, grads

    def train(self, x, y, x_val, y_val,
              reg=5e-6, lr=1e-3, lr_decay=0.95,
              batch_size=200, num_iters=100, verbose=False):
        '''
        Training this neural network using SGD
        Inputs:
         - x (N, D) : training data
         - y (N,) : training labels; y[i] = c : x[i] has labels c
         - x_val (N_val, D) : validation data
         - y_val (N_val,) : validation labels
         - reg : regularization strength
         - lr : learning rate for optimaztion
         - lr_decay : used to decay learning rate after each epoch
         - batch_size : no of training samples to use per step
         - num_iters : no of steps to take when optimizing
        Outputs:

        '''
        num_train = x.shape[0]
        iters_per_epoch = max(num_train / batch_size, 1)

        loss_his = []
        train_acc_his = []
        val_acc_his = []

        for it in range(num_iters):
            # Create a random minibatch of training and label
            idx = np.random.choice(num_train, batch_size)
            x_batch = x[idx, :]
            y_batch = y[idx]

            # Compute loss & gradients using the current minibatch(size = batch_size)
            loss, grads = self.loss(x_batch, y_batch, reg)
            loss_his.append(loss)

            # Update the parameters of the network using SGD
            self.params['W1'] -= lr * grads['W1']
            self.params['W2'] -= lr * grads['W2']
            self.params['b1'] -= lr * grads['b1']
            self.params['b2'] -= lr * grads['b2']

            if verbose & (it % 100 == 0):
                print("Iteration :%d/%d -- loss : %f" % (it, num_iters, loss))

            # For every epoch, check train/validation accuracy and decay lr_rate
            if it % iters_per_epoch == 0:
                train_acc = np.mean(self.predict(x_batch) == y_batch)
                val_acc = np.mean(self.predict(x_val) == y_val)
                train_acc_his.append(train_acc)
                val_acc_his.append(val_acc)

                # Decay learning rate
                lr *= lr_decay

        return {
            'loss_his': loss_his,
            'train_accuracy': train_acc_his,
            'val_acc_history': val_acc_his,
        }

    def predict(self, x):
        fc1 = x.dot(self.params['W1']) + self.params['b1']
        hidden1 = self.ReLU(fc1)
        scores = hidden1.dot(self.params['W2']) + self.params['b2']
        y_pred = np.argmax(self.softmax(scores), axis=1)
        return y_pred
