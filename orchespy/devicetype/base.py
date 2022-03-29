from abc import ABC, abstractmethod


class Base(ABC):
    @abstractmethod
    def get_ndarray_on_host(self, ary):
        pass

    @abstractmethod
    def get_ndarray_on_device(self, ary):
        pass

    @property
    @abstractmethod
    def numpy_class(self):
        pass

    @classmethod
    @abstractmethod
    def find_device(self, ary):
        pass

    def __enter__(self):
        return self.numpy_class

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def func_to_transfer_ndarray_from(self, srcdev):
        return None

    def func_to_transfer_ndarray_to(self, dstdev):
        return None
