# OrchesPy: Device-independent library for NumPy program on heterogeneous system
OrchesPy is a library for NumPy programs to execute part of the program
on accelerators by decorating functions.

## Prerequisites
Using OrchesPy requires the following packages.

- Python 3: tested with Python 3.8.
- NumPy 1.23.2: since CuPy and NLCPy require numpy>=1.17 and NLCPy requires
  numpy<=1.23.2.
- To run programs on VE, install NLCPy = 2.2.0 and its dependencies
  such as veoffload.
- To run programs on CUDA GPU, install CuPy = 11.0.0 and its dependencies such
  as CUDA toolkit working with your GPU.
- Install Inter-Device Copy Library = 0.1.0b1 for GPU-VE transfer.

To build OrchesPy, see also the section "Install from source".

## Installation
You can install OrchesPy from PyPI or from source.

### Install from PyPI
Execute the following command.

```
$ pip install orchespy
```

### Install from source
To build OrchesPy, install CUDA toolkit and veoffload (VEO) for CuPy and NLCPy.
Download the source tree from GitHub.

```
$ git clone https://github.com/SX-Aurora/orchespy.git
```
Execute the following command.

```
$ cd orchespy
$ pip install .
```

PIP will install dependencies automatically, and build and install OrchesPy on
your environment.

## Documentation
- [OrchesPy User's Guide](https://sxauroratsubasa.sakura.ne.jp/documents/orchespy/en/index.html)

## License

The BSD-3-Clause license (see `LICENSE` file).

