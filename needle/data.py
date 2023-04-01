import numpy as np
from .autograd import Tensor

import gzip
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        return np.flip(img, axis=1) if flip_img else img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        pad_width = ((self.padding, self.padding), (self.padding, self.padding), (0, 0))
        padded_img = np.pad(img, pad_width, mode='constant', constant_values=0)
        return padded_img[self.padding + shift_x : self.padding + shift_x + img.shape[0], 
                          self.padding + shift_y : self.padding + shift_y + img.shape[1], :]
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x
    

def parse_mnist(image_filesname, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(label_filename, 'rb') as f:
        fin = f.read()
        msb = int.from_bytes(fin[0:4], byteorder='big')
        num_items = int.from_bytes(fin[4:8], byteorder='big')
        # read the rest of the file as a 1D array of unsigned bytes.
        y = np.frombuffer(fin[8:], dtype=np.uint8)
        assert y.shape[0] == num_items

    with gzip.open(image_filesname, 'rb') as f:
        fin = f.read()
        msb = int.from_bytes(fin[0:4], byteorder='big')
        num_items = int.from_bytes(fin[4:8], byteorder='big')
        num_rows = int.from_bytes(fin[8:12], byteorder='big')
        num_cols = int.from_bytes(fin[12:16], byteorder='big')
        # read the rest of the file as a 1D array of unsigned bytes.
        X = np.frombuffer(fin[16:], dtype=np.uint8)
        assert X.shape[0] == num_items * num_rows * num_cols
        X = X.reshape(num_items, num_rows * num_cols)
        X = X.astype(np.float32)
        X = X / 255.0

    return X, y
    ### END YOUR SOLUTION



class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.calls = 0
        if self.shuffle:
            self.ordering = np.array_split(np.random.permutation(len(self.dataset)),
                                            range(self.batch_size, len(self.dataset), self.batch_size))
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.calls == len(self.ordering):
            raise StopIteration
        else:
            idx = self.ordering[self.calls]
            self.calls += 1
            batch = self.dataset[idx]
            return tuple(map(lambda x: Tensor(x, dtype=x.dtype, requires_grad=False), batch))
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.transforms = transforms
        self.X, self.y = parse_mnist(image_filename, label_filename)
        self.X = self.X.reshape(-1, 28, 28, 1)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        res_X = self.X[index]
        if self.transforms is not None:
            for tform in self.transforms:
                res_X = tform(res_X)
        return res_X, self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.X)
        ### END YOUR SOLUTION

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
