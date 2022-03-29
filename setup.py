from setuptools import setup, Extension
from Cython.Build import cythonize

import os.path

_veo_path = '/opt/nec/ve/veos'
_veo_include_dir = _veo_path + '/include'
_veo_lib_dir = _veo_path + '/lib64'

_cuda_path = '/usr/local/cuda'
_cuda_include_dir = _cuda_path + '/include'
_cuda_lib_dir = _cuda_path + '/lib64'

ext_modules = {
    'cupy-nlcpy': [
        'orchespy.devicetype._transfer.cunlc',
    ],
}
include_dirs = {
    'cupy-nlcpy': [_cuda_include_dir, _veo_include_dir, ],
}
libraries = {
    'cupy-nlcpy': ['veo', 'cudart', ],
}
library_dirs = {
    'cupy-nlcpy': [_cuda_lib_dir, _veo_lib_dir, ],
}

extensions = []
for key in ext_modules:
    for module in ext_modules[key]:
        src = os.path.join(*module.split('.')) + '.pyx'
        ext = Extension(
            module,
            [src, ],
            libraries=libraries[key],
            library_dirs=library_dirs[key],
            include_dirs=include_dirs[key],
        )
        extensions.append(ext)
setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={'embedsignature': True},
        language_level=3
    ),
)
