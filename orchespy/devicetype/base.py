from abc import ABC, abstractmethod
from .. import numpy as _numpy


class Base(ABC):
    @abstractmethod
    def can_transfer(self, obj):
        pass

    @abstractmethod
    def can_transfer_to(self, obj, target):
        pass

    @abstractmethod
    def create_ndarray_on_device(self, ary):
        pass

    @property
    @abstractmethod
    def numpy_class(self):
        pass

    @classmethod
    @abstractmethod
    def get_device(self, obj):
        pass

    def __enter__(self):
        _numpy.push_current_numpy(self.numpy_class)
        return self.numpy_class

    def __exit__(self, exc_type, exc_value, traceback):
        _numpy.pop_current_numpy()

    @abstractmethod
    def transfer_array_content(self, dst, src):
        return None

    @abstractmethod
    def transfer_array_content_to(self, dst, src):
        return None
