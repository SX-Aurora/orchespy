import sys
import threading
import numpy as _numpy


_thread_local = threading.local()
_default_numpy_package = _numpy


class _NumPy:
    @staticmethod
    def push_current_numpy(np_):
        try:
            _thread_local.current_numpy.append(np_)
        except AttributeError:
            _thread_local.current_numpy = [np_]

    @staticmethod
    def pop_current_numpy():
        _thread_local.current_numpy.pop()

    @staticmethod
    def get_current_numpy():
        try:
            return _thread_local.current_numpy[-1]
        except AttributeError:
            _thread_local.current_numpy = []
            return _default_numpy_package
        except IndexError:
            return _default_numpy_package

    @staticmethod
    def __getattr__(name):
        return getattr(_NumPy.get_current_numpy(), name)


sys.modules[__name__] = _NumPy()
