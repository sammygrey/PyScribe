#adapted from https://github.com/jiaaro/pydub/blob/master/pydub/pyaudioop.py
#replacement for audioop

import math
import struct
from ctypes import create_string_buffer


class error(Exception):
    pass


def _check_size(size):
    if size != 1 and size != 2 and size != 4:
        raise error("Size should be 1, 2 or 4")


def _check_params(length, size):
    _check_size(size)
    if length % size != 0:
        raise error("not a whole number of frames")


def _sample_count(cp, size):
    return len(cp) / size


def _get_samples(cp, size, signed=True):
    for i in range(_sample_count(cp, size)):
        yield _get_sample(cp, size, i, signed)


def _struct_format(size, signed):
    if size == 1:
        return "b" if signed else "B"
    elif size == 2:
        return "h" if signed else "H"
    elif size == 4:
        return "i" if signed else "I"


def _get_sample(cp, size, i, signed=True):
    fmt = _struct_format(size, signed)
    start = i * size
    end = start + size
    return struct.unpack_from(fmt, memoryview(cp)[start:end])[0]


def _put_sample(cp, size, i, val, signed=True):
    fmt = _struct_format(size, signed)
    struct.pack_into(fmt, cp, i * size, val)


def _get_maxval(size, signed=True):
    if signed and size == 1:
        return 0x7f
    elif size == 1:
        return 0xff
    elif signed and size == 2:
        return 0x7fff
    elif size == 2:
        return 0xffff
    elif signed and size == 4:
        return 0x7fffffff
    elif size == 4:
        return 0xffffffff


def _get_minval(size, signed=True):
    if not signed:
        return 0
    elif size == 1:
        return -0x80
    elif size == 2:
        return -0x8000
    elif size == 4:
        return -0x80000000


def _get_clipfn(size, signed=True):
    maxval = _get_maxval(size, signed)
    minval = _get_minval(size, signed)
    return lambda val: max(min(val, maxval), minval)


def getsample(cp, size, i):
    _check_params(len(cp), size)
    if not (0 <= i < len(cp) / size):
        raise error("Index out of range")
    return _get_sample(cp, size, i)


def rms(cp, size):
    _check_params(len(cp), size)

    sample_count = _sample_count(cp, size)
    if sample_count == 0:
        return 0

    sum_squares = sum(sample**2 for sample in _get_samples(cp, size))
    return int(math.sqrt(sum_squares / sample_count))


def add(cp1, cp2, size):
    _check_params(len(cp1), size)

    if len(cp1) != len(cp2):
        raise error("Lengths should be the same")

    clip = _get_clipfn(size)
    sample_count = _sample_count(cp1, size)
    result = create_string_buffer(len(cp1))

    for i in range(sample_count):
        sample1 = getsample(cp1, size, i)
        sample2 = getsample(cp2, size, i)

        sample = clip(sample1 + sample2)

        _put_sample(result, size, i, sample)

    return result.raw


def reverse(cp, size):
    _check_params(len(cp), size)
    sample_count = _sample_count(cp, size)

    result = create_string_buffer(len(cp))
    for i, sample in enumerate(_get_samples(cp, size)):
        _put_sample(result, size, sample_count - i - 1, sample)

    return result.raw

