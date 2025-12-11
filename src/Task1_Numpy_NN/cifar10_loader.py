import pickle
import numpy as np

class Cifar10Loader:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def _unpickle(self, file):
        """
        Unpickle the given file and return the data.
        :param file: Path to the file to unpickle.
        :return: Unpickled data.
        """
        with open(file, "rb") as fo:
            data = pickle.load(fo, encoding="bytes")
        return data


    def _load_batch(self, batch_name):
        """
        Loads a single batch of CIFAR-10 data.
        :param batch_name: Name of the batch file to load.
        :return: A tuple (images, labels) where images is a NumPy array of shape (N, 3072)
                 and labels is a NumPy array of shape (N,).
        """
        batch_path = f"{self.root_dir}/{batch_name}"
        batch_data = self._unpickle(batch_path)
        images = batch_data[b"data"]              # (N, 3072)
        labels = np.array(batch_data[b"labels"])  # convert list -> np.array here
        return images, labels

    def _load_label_names(self):
        """
        Loads the label names for CIFAR-10.
        :return: A list of label names.
        """
        meta_path = f"{self.root_dir}/batches.meta"
        meta_data = self._unpickle(meta_path)
        label_names = [name.decode("utf-8") for name in meta_data[b"label_names"]]
        return label_names

    def _reshape_images(self, X):
        """
        Reshapes the flat image data into (N, 32, 32, 3) format.
        :param X: NumPy array of shape (N, 3072)
        :return: NumPy array of shape (N, 32, 32, 3)
        """
        X = X.reshape(-1, 3, 32, 32)
        X = X.transpose(0, 2, 3, 1)
        return X

    def load_train(self):
        """
        Loads the entire training set.
        :return: A tuple (images, labels, label_names) where images is a NumPy array of shape (50000, 32, 32, 3),
                 labels is a NumPy array of shape (50000,), and label_names is a list of label names.
        """
        X_list = []
        y_list = []

        for i in range(1, 6):
            X_batch, y_batch = self._load_batch("data_batch_%d" % i)
            X_list.append(X_batch)
            y_list.append(y_batch)

        X = np.vstack(X_list)              # (50000, 3072)
        y = np.concatenate(y_list)         # (50000,)
        X = self._reshape_images(X)
        labels = self._load_label_names()
        return X, y, labels

    def load_test(self):
        """
        Loads the test set.
        :return: A tuple (images, labels, label_names) where images is a NumPy array of shape (10000, 32, 32, 3),
                 labels is a NumPy array of shape (10000,), and label_names is a list of label names.
        """
        X, y = self._load_batch("test_batch")  # y is already a NumPy array now
        X = self._reshape_images(X)
        labels = self._load_label_names()
        return X, y, labels

    
loader = Cifar10Loader("dataset/cifar10")
X_train, y_train, label_names = loader.load_train()
X_test, y_test, _ = loader.load_test()

print(X_train.shape, y_train.shape)  # (50000, 32, 32, 3) (50000,)
print(X_test.shape, y_test.shape)    # (10000, 32, 32, 3) (10000,)
print(label_names)