import os
import gzip
import numpy as np
from collections import OrderedDict


URL = "https://web.archive.org/web/20150906081542/http://yann.lecun.com/exdb/mnist/"
DICT_FILES = {
    "train_image": "train-images-idx3-ubyte.gz",
    "train_label": "train-labels-idx1-ubyte.gz",
    "test_image": "t10k-images-idx3-ubyte.gz",
    "test_label": "t10k-labels-idx1-ubyte.gz"
}
IMAGE_OFFSET = 16
LABEL_OFFSET = 8
DATA_DIR = "./data"
IMAGE_SIZE = 28


def load_data(filename):
    """ Load mnist data
        Files are supposed to be current dirctory.
    Args:
        filename: string: File name.
        offset: int: Offset of file.
    Return:
        : np.array: mnist image data
    """
    if "images" in filename:
        offset = IMAGE_OFFSET
        length = IMAGE_SIZE * IMAGE_SIZE
    else:
        offset = LABEL_OFFSET
        length = 1

    with gzip.open(os.sep.join((DATA_DIR, filename)), "rb") as file_:
        data = np.frombuffer(file_.read(), np.uint8, offset=offset)
    return data.reshape(-1, length)


def xavier_init(shape):
    """ xavier initialization
    """
    std = np.sqrt(1/shape[0])
    if len(shape) == 1:
        x = std * np.random.randn(shape[0])
    else:
        x = std * np.random.randn(shape[0], shape[1])
    return x


def he_init(shape):
    """ he initialization
    """
    std = np.sqrt(2/shape[0])
    if len(shape) == 1:
        x = std * np.random.randn(shape[0])
    else:
        x = std * np.random.randn(shape[0], shape[1])
    return x


class Relu:
    """ relu layer
    """
    def __init__(self):
        self.mask = None

    def forward(self, x):
        """ forward
        """
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, diff):
        """ backward
        """
        diff[self.mask] = 0
        return diff


class Softmax:
    """ softmax layer
    """
    def __init__(self):
        self.y = None

    def forward(self, x):
        """ forward
        """
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            self.y = y
            return y.T

        x = x - np.max(x)
        y = np.exp(x) / np.sum(np.exp(x))
        self.y = y
        return y

    def backward(self, diff):
        """ backward
        """
        # diff = self.y - (diff * self.y)
        return diff


class FC:
    """ full connected layer
    """
    def __init__(self, input_shape, n_hidden, initializer=he_init):
        self.input_shape = input_shape
        self.output_shape = (input_shape[0], n_hidden)
        self.params = dict()
        self.params["W"] = initializer([input_shape[1], n_hidden])
        self.params["b"] = initializer([n_hidden])
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        """ forward
        """
        self.x = x
        out = np.dot(x, self.params["W"]) + self.params["b"]
        return out

    def backward(self, diff):
        """ backward
        """
        dx = np.dot(diff, self.params["W"].T)
        self.dW = np.dot(self.x.T, diff)
        self.db = np.sum(diff, axis=0)
        return dx

    def update(self, learning_rate):
        """ update
        """
        self.params["W"] -= learning_rate * self.dW
        self.params["b"] -= learning_rate * self.db


class NN3Layer:
    """ 3-layer Neural Network
    """
    def __init__(self, n_hidden1, n_hidden2, n_hidden3):
        self.input_shape = [None, IMAGE_SIZE*IMAGE_SIZE]
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_hidden3 = n_hidden3

        self.layers = OrderedDict()
        self._network_init()

    def _network_init(self):
        """ network init
        """
        self.layers["FC1"] = FC(self.input_shape,
                                self.n_hidden1,
                                he_init)
        self.layers["Activation1"] = Relu()
        self.layers["FC2"] = FC(self.layers["FC1"].output_shape,
                                self.n_hidden2,
                                he_init)
        self.layers["Activation2"] = Relu()
        self.layers["FC3"] = FC(self.layers["FC2"].output_shape,
                                self.n_hidden3,
                                he_init)
        self.layers["Activation3"] = Softmax()

    def feedforward(self, x):
        """ feedforward
        """
        out = x
        for layer in self.layers.values():
            out = layer.forward(out)
        return out

    def loss(self, y_pred, y_true):
        """ cross entropy loss
        """
        batch_size = y_pred.shape[0]
        y_true = y_true.reshape([1, batch_size])
        cross_entropy = -np.sum(np.log(y_pred[np.arange(batch_size), y_true]))
        diff = y_pred.copy()
        diff[np.arange(batch_size), y_true] -= 1
        diff = diff / batch_size
        return cross_entropy, diff

    def feedbackward(self, x):
        """ back propagation
        """
        diff = x
        list_layer = list(self.layers.values())
        for layer in list_layer[::-1]:
            diff = layer.backward(diff)
        return diff

    def update(self, learning_rate):
        """ update
        """
        for layer in self.layers.values():
            if hasattr(layer, "update"):
                layer.update(learning_rate)

    def accuracy(self, y_pred, y_true):
        """ accuracy
        """
        batch_size = y_pred.shape[0]
        answer_t = y_true.reshape(batch_size)
        answer_p = y_pred.argmax(axis=1)
        acc = np.sum(answer_t == answer_p)/batch_size
        return acc

    def train(self, X, Y, val_X=None, val_Y=None,
              batch_size=1, epoch=1,
              shuffle=True, learning_rate=0.0001):
        """ train
        """
        train_length = len(X)
        train_indexes = np.arange(0, train_length)
        ranges = [(i, i+batch_size)
                  for i in range(0, train_length, batch_size)]
        # epochs
        for ep in range(epoch):
            if shuffle:
                np.random.shuffle(train_indexes)
            list_acc = list()
            list_loss = list()
            # batches
            for i, (from_, to_) in enumerate(ranges):
                x_batch = X[train_indexes[from_: to_]]
                y_true = Y[train_indexes[from_: to_]]
                y_pred = self.feedforward(x_batch)
                loss_batch, diff_batch = self.loss(y_pred, y_true)
                self.feedbackward(diff_batch)
                self.update(learning_rate)
                list_loss.append(loss_batch)
                list_acc.append(self.accuracy(y_pred, y_true))

            train_message = \
                "epoch:{}, train_loss:{:.4f}, acc:{:.4f}".format(
                    ep, np.mean(list_loss), np.mean(list_acc)
                )

            if val_X is not None and val_Y is not None:
                y_val_pred = self.feedforward(val_X)
                loss_val, diff_val = self.loss(y_val_pred, val_Y)
                acc_val = self.accuracy(y_val_pred, val_Y)
                train_message += \
                    "  val_loss:{:.4f}, val_acc:{:.4f}".format(
                        loss_val, acc_val
                    )
            print(train_message)

    def save(self):
        """ save
        """
        pass


if __name__ == "__main__":

    # read data
    train_images = load_data(DICT_FILES["train_image"])
    train_labels = load_data(DICT_FILES["train_label"])

    # create network architecture
    network = NN3Layer(n_hidden1=128, n_hidden2=64, n_hidden3=10)

    # train network
    sep_ind = int(len(train_images) * 0.8)
    train_X = train_images[:sep_ind]
    train_Y = train_labels[:sep_ind]
    val_X = train_images[sep_ind:]
    val_Y = train_labels[sep_ind:]
    network.train(train_X, train_Y, val_X=val_X, val_Y=val_Y,
                  batch_size=16, epoch=30)

    # save model
    network.save()
