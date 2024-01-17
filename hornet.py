"""
Haai :3

So I got bored and decided to make this(pain, but I loved it). So I've been wanting to learn to integrate python and
c++ like forever, and I think this is the best place to start (Oh, and did I mention it has CUDA to?). I'll add more
stuff when I feel like it/am bored. Bye~ """


import ctypes
import numpy as np

lib = ctypes.CDLL("Dependencies/vecadd.dll")

lib.add.argtypes = (ctypes.c_int, ctypes.c_int)
lib.add.restype = ctypes.c_int

lib.add_vec.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int
]
lib.add_vec.restype = None

lib.cdot.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int
]

lib.cdot.restype = ctypes.c_float

lib.cmagnitude.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int
]

lib.cmagnitude.restype = ctypes.c_float


def add_vectors(a, b):
    size = len(a)

    if len(a) != len(b):
        raise IndexError("Size of the two vectors must be the same. >:(")

    result = np.empty_like(a)
    lib.add_vec(a, b, result, size)
    return result


def dot(a, b):
    size = len(a)

    if len(a) != len(b):
        raise IndexError("Size of the two vectors must be the same. >:(")

    result = np.empty_like(a)
    d = lib.cdot(a, b, result, size)
    return d


def magnitude(a):
    size = len(a)

    result = np.empty_like(a)
    m = lib.cmagnitude(a, result, size)
    return m
