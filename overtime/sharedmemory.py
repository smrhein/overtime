from __future__ import print_function
import contextlib
import mmap
import posix_ipc
import sys
import pickle
import uuid
import warnings

import numpy as np

from overtime.contextual import exiting


def _debug(*args, **kwargs):
    pass
    # k = multiprocessing.current_process().ident, threading.current_thread().ident
    # with _lck:
    # print(k, *args, **kwargs)


class shmmap(mmap.mmap):
    @staticmethod
    def _ipc_name(prefix='shmmap', suffix=''):
        return ''.join((prefix, str(uuid.uuid4()), suffix))

    @classmethod
    def fromlength(cls, length):
        return cls(None, None, length)

    def __new__(cls, semname, shmname, length, access=mmap.ACCESS_WRITE, offset=0):
        try:
            if semname and shmname:
                flags = 0
            elif not semname and not shmname:
                semname = cls._ipc_name()
                shmname = cls._ipc_name()
                flags = posix_ipc.O_CREX
            else:
                raise ValueError('semname if and only if shmname')

            with contextlib.closing(posix_ipc.Semaphore(name=semname, flags=flags)), \
                 exiting(posix_ipc.SharedMemory(name=shmname, flags=flags, size=length),
                         exiter=posix_ipc.SharedMemory.close_fd) as shm:
                self = super(shmmap, cls).__new__(cls, shm.fd, length, access=access, offset=offset)
            self.shmname = shmname
            self.semname = semname
        except:
            exc = sys.exc_info()
            try:
                cls._cleanup(semname, shmname)
            finally:
                raise exc[0], exc[1], exc[2]
        else:
            self.length = length
            self.access = access
            self.offset = offset
            return self

    def __reduce_ex__(self, protocol):
        return self.__reduce__()

    def __reduce__(self):
        if self.access == mmap.ACCESS_COPY:
            raise pickle.PicklingError('copy on write memory cannot be pickled')
        copy = self.__class__, (self.semname, self.shmname, self.length, self.access, self.offset)
        with contextlib.closing(posix_ipc.Semaphore(self.semname, 0)) as sem:
            sem.release()
        return copy

    def __del__(self):
        self.close()
        self._cleanup(self.semname, self.shmname)

    @staticmethod
    def _cleanup(semname, shmname):
        try:
            with contextlib.closing(posix_ipc.Semaphore(semname, 0)) as sem:
                try:
                    sem.acquire(0)
                except posix_ipc.BusyError:
                    posix_ipc.unlink_semaphore(semname)
                    posix_ipc.unlink_shared_memory(shmname)
        except Exception as e:
            warnings.warn('possible resource leakage of {} and/or {} caused by {}'.format(semname, shmname, e),
                          RuntimeWarning)


class ndshm(np.ndarray):
    @classmethod
    def fromndarray(cls, a, dtype=None, order=None, subok=False):
        b = cls.empty_like(a, dtype, order, subok)
        b[...] = a
        return b

    @classmethod
    def zeros(cls, shape, dtype=None, order='C'):
        a = cls.empty(shape, dtype, order)
        a[...] = 0
        return a

    @classmethod
    def empty_like(cls, a, dtype=None, order='K', subok=True):
        b = (cls if subok else ndshm).empty(a.shape, dtype, order)
        return b

    @classmethod
    def empty(cls, shape, dtype=float, order='C'):
        return cls(shape=shape, dtype=dtype, order=order)

    def __new__(cls, shape, dtype=float, shm=None, offset=0, strides=None, order=None):
        if shm is None:
            shm = shmmap.fromlength(np.prod(shape) * np.dtype(dtype).itemsize)
        self = super(ndshm, cls).__new__(cls, shape, dtype, shm, offset, strides, order)
        self._shm = shm
        self.offset = offset
        return self

    def __array_finalize__(self, obj):
        if np.may_share_memory(self, obj):
            try:
                self._shm = obj._shm
                self.offset = obj.offset
                if self._shm is not None:
                    self.offset += np.byte_bounds(self)[0] - np.byte_bounds(obj)[0]
                return
            except AttributeError:
                pass
        self._shm = None
        self.offset = None

    def __reduce_ex__(self, protocol):
        return self.__reduce__()

    def __reduce__(self):
        if self._shm is None:
            raise pickle.PicklingError('array is not backed by shared memory')
        if self.flags['C_CONTIGUOUS']:
            order = 'C'
        elif self.flags['F_CONTIGUOUS']:
            order = 'F'
        else:
            order = None
        return self.__class__, (self.shape, self.dtype, self._shm, self.offset, self.strides, order)



